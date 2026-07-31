"""Run the no-training automated Label-2 mechanism identification audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize

from thesis_exp.exp54_rar_sft import REPO_ROOT


ARMS = ("R3", "P1_FIELD_DPO")
SEEDS = (42, 43, 44)
SCORES = (1, 2, 3, 4, 5)
REGULARIZATION_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
AUTOMATED_HIERARCHY = (
    "measurement_ambiguous",
    "decoder_failure",
    "prior_recoverable",
    "calibration_recoverable",
    "support_deficient",
    "preference_coverage_deficient",
    "residual",
)
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_PRIVATE_ROOT = (
    DEFAULT_ARTIFACT_ROOT
    / "label2_identification_audit/private/score_probabilities"
)
DEFAULT_TRAIN = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
)
DEFAULT_DEV = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
DEFAULT_PAIRS = (
    DEFAULT_ARTIFACT_ROOT / "preference_pairs/private/score_pairs_hybrid.jsonl"
)
DEFAULT_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "label2_identification_audit/automated_mechanism_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def deterministic_group_fold(group: str, *, namespace: str, folds: int) -> int:
    if folds < 2 or not group:
        raise ValueError("valid group and at least two folds are required")
    payload = f"{namespace}|{group}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % folds


def _softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    terms = np.exp(shifted)
    return terms / terms.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(selected, 1e-300)).mean())


def fit_vector_scaler(
    log_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 5:
        raise ValueError("expected an n-by-5 log-probability matrix")
    if labels.shape != (len(log_probabilities),):
        raise ValueError("calibration labels differ in length")
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        scales = parameters[:5]
        biases = parameters[5:]
        scores = log_probabilities * scales[None, :] + biases[None, :]
        probabilities = _softmax(scores)
        errors = probabilities.copy()
        errors[np.arange(len(labels)), labels] -= 1.0
        value = _nll(probabilities, labels) + 0.5 * regularization * float(
            np.square(scales - 1.0).sum() + np.square(biases).sum()
        )
        scale_gradient = (
            (errors * log_probabilities).mean(axis=0)
            + regularization * (scales - 1.0)
        )
        bias_gradient = errors.mean(axis=0) + regularization * biases
        return value, np.concatenate([scale_gradient, bias_gradient])

    initial = np.concatenate([np.ones(5), np.zeros(5)])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.05, 10.0)] * 5 + [(-10.0, 10.0)] * 5,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"vector scaling did not converge: {result.message}")
    return result.x[:5], result.x[5:]


def apply_vector_scaler(
    log_probabilities: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    return _softmax(log_probabilities * scales[None, :] + biases[None, :])


def nested_question_group_vector_scaling(
    log_probabilities: np.ndarray,
    labels: np.ndarray,
    question_keys: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(log_probabilities) != len(labels) or len(labels) != len(question_keys):
        raise ValueError("calibration inputs differ in length")
    outer_folds = np.asarray(
        [
            deterministic_group_fold(
                key,
                namespace="exp54-label2-audit-v1",
                folds=5,
            )
            for key in question_keys
        ],
        dtype=np.int64,
    )
    calibrated = np.full_like(log_probabilities, np.nan, dtype=np.float64)
    selected_regularization: dict[str, float] = {}
    inner_scores: dict[str, dict[str, float]] = {}
    for outer in range(5):
        outer_test = outer_folds == outer
        outer_train = ~outer_test
        if not outer_test.any() or not outer_train.any():
            raise ValueError(f"outer calibration fold {outer} is empty")
        train_indices = np.flatnonzero(outer_train)
        inner_fold = np.asarray(
            [
                deterministic_group_fold(
                    question_keys[index],
                    namespace=f"exp54-label2-audit-v1-inner-{outer}",
                    folds=4,
                )
                for index in train_indices
            ],
            dtype=np.int64,
        )
        scores: dict[float, float] = {}
        for regularization in REGULARIZATION_GRID:
            fold_nlls = []
            for inner in range(4):
                validation_local = inner_fold == inner
                fitting_local = ~validation_local
                if not validation_local.any() or not fitting_local.any():
                    raise ValueError(
                        f"inner calibration fold {outer}/{inner} is empty"
                    )
                fitting = train_indices[fitting_local]
                validation = train_indices[validation_local]
                scales, biases = fit_vector_scaler(
                    log_probabilities[fitting],
                    labels[fitting],
                    regularization=regularization,
                )
                probabilities = apply_vector_scaler(
                    log_probabilities[validation], scales, biases
                )
                fold_nlls.append(_nll(probabilities, labels[validation]))
            scores[regularization] = float(np.mean(fold_nlls))
        best = min(REGULARIZATION_GRID, key=lambda value: (scores[value], value))
        selected_regularization[str(outer)] = best
        inner_scores[str(outer)] = {
            str(value): scores[value] for value in REGULARIZATION_GRID
        }
        scales, biases = fit_vector_scaler(
            log_probabilities[outer_train],
            labels[outer_train],
            regularization=best,
        )
        calibrated[outer_test] = apply_vector_scaler(
            log_probabilities[outer_test], scales, biases
        )
    if not np.all(np.isfinite(calibrated)):
        raise RuntimeError("cross-fitted calibration left missing predictions")
    return calibrated, {
        "outer_folds": 5,
        "inner_folds": 4,
        "selected_regularization": selected_regularization,
        "inner_mean_nll": inner_scores,
    }


def measurement_ambiguous(row: dict[str, Any]) -> bool:
    scores = [float(row[field]) for field in ("human_1_5", "human_2_5", "human_3_5")]
    return max(scores) - min(scores) >= 2.0 or {2.0, 3.0}.issubset(set(scores))


def cluster_fraction_interval(
    rows: list[dict[str, Any]],
    *,
    flag: str,
    seed: int,
    replicates: int = 10000,
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot bootstrap an empty failure cohort")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_key"])].append(row)
    groups = sorted(grouped)
    observed = sum(bool(row[flag]) for row in rows) / len(rows)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = [row for group in sampled for row in grouped[str(group)]]
        values[index] = sum(bool(row[flag]) for row in selected) / len(selected)
    return {
        "fraction": float(observed),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def support_risk_difference(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int = 10000,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_key"])].append(row)
    groups = sorted(grouped)

    def difference(selected: list[dict[str, Any]]) -> float | None:
        low = [row for row in selected if row["low_support"]]
        high = [row for row in selected if not row["low_support"]]
        if not low or not high:
            return None
        return sum(row["failure"] for row in low) / len(low) - sum(
            row["failure"] for row in high
        ) / len(high)

    observed = difference(rows)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = [row for group in sampled for row in grouped[str(group)]]
        value = difference(selected)
        if value is not None:
            values.append(value)
    if observed is None or not values:
        return {
            "identifiable": False,
            "risk_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_valid_replicates": len(values),
            "adverse_association_gate": False,
        }
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return {
        "identifiable": True,
        "risk_difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_valid_replicates": len(values),
        "adverse_association_gate": bool(observed > 0 and low > 0),
    }


def _probability_path(private_root: Path, arm: str, seed: int) -> Path:
    return private_root / arm.lower() / f"seed_{seed}/score_probabilities.jsonl"


def _log_probability_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    probabilities = np.asarray(
        [
            [
                float(row["canonical_score_option_probabilities"][str(score)])
                for score in SCORES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    return np.log(np.maximum(probabilities, 1e-300))


def analyze(
    *,
    train_path: Path,
    dev_path: Path,
    pair_path: Path,
    private_root: Path,
) -> dict[str, Any]:
    expected_hashes = {
        train_path: "0a1733b9209984c5c4291d205d1ac6057bed341717903b9de075d07de44a878e",
        dev_path: "a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0",
        pair_path: "aeb079c56be8a8afbfdbec386e72faad748fef848014761bf602fbc40c97a1bb",
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise ValueError(f"locked input hash differs: {path}")
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    pair_rows = read_jsonl(pair_path)
    if len(train_rows) != 2654 or len(dev_rows) != 664 or len(pair_rows) != 838:
        raise ValueError("locked input row count differs")

    train_label_counts = Counter(int(row["label_5"]) for row in train_rows)
    train_priors = np.asarray(
        [train_label_counts[score] / len(train_rows) for score in SCORES],
        dtype=np.float64,
    )
    support_counts = Counter(
        (str(row["metric_id"]), str(row["language"]))
        for row in train_rows
        if int(row["label_5"]) == 2
    )
    direct_pair_counts = Counter(
        (str(row["metric_id"]), str(row["language"]))
        for row in pair_rows
        if int(row["chosen"]["score"]) == 2
        and int(row["rejected"]["score"]) == 3
    )
    direct_pair_records = {
        str(row["record_id"])
        for row in pair_rows
        if int(row["chosen"]["score"]) == 2
        and int(row["rejected"]["score"]) == 3
    }
    if sum(direct_pair_counts.values()) != 22:
        raise ValueError("direct 2-over-3 pair count differs")

    public: dict[str, Any] = {}
    primary_private: dict[int, list[dict[str, Any]]] = {}
    for arm in ARMS:
        public[arm] = {}
        for seed in SEEDS:
            path = _probability_path(private_root, arm, seed)
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            if len(rows) != len(dev_rows):
                raise ValueError(f"{arm}/{seed}: probability row count differs")
            for source, dev in zip(rows, dev_rows, strict=True):
                if source["record_id"] != dev["record_id"]:
                    raise ValueError(f"{arm}/{seed}: dev order differs")
            log_probabilities = _log_probability_matrix(rows)
            labels = np.asarray([int(row["label_5"]) - 1 for row in rows])
            question_keys = [str(row["question_key"]) for row in rows]
            calibrated, calibration_details = nested_question_group_vector_scaling(
                log_probabilities, labels, question_keys
            )
            calibrated_predictions = np.argmax(calibrated, axis=1) + 1
            prior_scores = log_probabilities - np.log(train_priors)[None, :]
            prior_predictions = np.argmax(prior_scores, axis=1) + 1

            label2_rows: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                if int(row["label_5"]) != 2:
                    continue
                stratum = (str(row["metric_id"]), str(row["language"]))
                parsed_failure = int(row["parsed_score"]) != 2
                decoder = parsed_failure and int(row["forced_choice_score"]) == 2
                prior = parsed_failure and int(prior_predictions[index]) == 2
                calibration = parsed_failure and int(calibrated_predictions[index]) == 2
                low_support = support_counts[stratum] < 5
                pair_deficient = (
                    str(row["record_id"]) not in direct_pair_records
                    and direct_pair_counts[stratum] == 0
                )
                label2_rows.append(
                    {
                        "question_key": str(row["question_key"]),
                        "record_id": str(row["record_id"]),
                        "failure": parsed_failure,
                        "measurement_ambiguous": measurement_ambiguous(row),
                        "decoder_failure": decoder,
                        "prior_recoverable": prior,
                        "calibration_recoverable": calibration,
                        "low_support": low_support,
                        "support_deficient": False,
                        "preference_coverage_deficient": pair_deficient,
                    }
                )
            support_result = support_risk_difference(
                label2_rows,
                seed=20260731 + seed,
            )
            for row in label2_rows:
                row["support_deficient"] = bool(
                    row["low_support"]
                    and support_result["adverse_association_gate"]
                )

            failures = [row for row in label2_rows if row["failure"]]
            for row in failures:
                category = "residual"
                for candidate in AUTOMATED_HIERARCHY[:-1]:
                    if row[candidate]:
                        category = candidate
                        break
                row["automated_exclusive_category"] = category
                for candidate in AUTOMATED_HIERARCHY:
                    row[f"exclusive_{candidate}"] = category == candidate
            sensitivity = {
                flag: cluster_fraction_interval(
                    failures,
                    flag=flag,
                    seed=20260731 + seed * 101 + index,
                )
                for index, flag in enumerate(AUTOMATED_HIERARCHY[:-1])
            }
            exclusive = {
                category: cluster_fraction_interval(
                    failures,
                    flag=f"exclusive_{category}",
                    seed=20260731 + seed * 211 + index,
                )
                for index, category in enumerate(AUTOMATED_HIERARCHY)
            }
            calibrated_label2 = calibrated_predictions[labels == 1]
            public[arm][str(seed)] = {
                "label_2_rows": len(label2_rows),
                "label_2_failures": len(failures),
                "measurement_ambiguous_label_2_rows": sum(
                    row["measurement_ambiguous"] for row in label2_rows
                ),
                "prior_corrected_label_2_prediction_counts": dict(
                    sorted(Counter(str(int(value)) for value in prior_predictions[labels == 1]).items())
                ),
                "prior_recovered_label_2_failures": sum(
                    row["prior_recoverable"] for row in failures
                ),
                "calibrated_label_2_prediction_counts": dict(
                    sorted(Counter(str(int(value)) for value in calibrated_label2).items())
                ),
                "calibration_recovered_label_2_failures": sum(
                    row["calibration_recoverable"] for row in failures
                ),
                "raw_nll": _nll(_softmax(log_probabilities), labels),
                "cross_fitted_calibrated_nll": _nll(calibrated, labels),
                "calibration": calibration_details,
                "support": support_result,
                "low_support_label_2_rows": sum(row["low_support"] for row in label2_rows),
                "pair_coverage_deficient_label_2_rows": sum(
                    row["preference_coverage_deficient"] for row in label2_rows
                ),
                "sensitivity_failure_fractions": sensitivity,
                "provisional_automated_exclusive_fractions": exclusive,
            }
            if arm == "P1_FIELD_DPO":
                primary_private[seed] = failures

    dominance: dict[str, Any] = {}
    for category in AUTOMATED_HIERARCHY:
        per_seed = [
            public["P1_FIELD_DPO"][str(seed)][
                "provisional_automated_exclusive_fractions"
            ][category]
            for seed in SEEDS
        ]
        passing = sum(
            item["fraction"] >= 0.6 and item["ci95_low"] > 0.5
            for item in per_seed
        )
        dominance[category] = {
            "seeds_passing_fraction_and_ci_gate": passing,
            "passes_two_seed_gate": passing >= 2,
        }

    stratum_report = {
        f"{metric}|{language}": {
            "train_label_2_support": support_counts[(metric, language)],
            "direct_2_over_3_pairs": direct_pair_counts[(metric, language)],
        }
        for metric, language in sorted(
            {
                (str(row["metric_id"]), str(row["language"]))
                for row in dev_rows
                if int(row["label_5"]) == 2
            }
        )
    }
    return {
        "schema_version": "exp54-label2-automated-mechanism-report-v1",
        "status": "AUTOMATED_MECHANISM_AUDIT_COMPLETE_HUMAN_RUBRIC_PENDING",
        "primary_arm": "P1_FIELD_DPO",
        "comparator_arm": "R3",
        "arms": public,
        "p1_provisional_dominance_gate": dominance,
        "label_2_stratum_coverage": stratum_report,
        "global_direct_2_over_3_pair_count": sum(direct_pair_counts.values()),
        "automated_hierarchy_excludes_pending_rubric_category": True,
        "final_exclusive_attribution_allowed": False,
        "new_method_development_allowed": False,
        "human_rubric_audit_pending": True,
        "private_row_level_flags_published": False,
        "bootstrap_replicates": 10000,
        "bootstrap_cluster": "question_key",
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "pair_source_sha256": sha256_file(pair_path),
        "training_started": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(
        train_path=args.train,
        dev_path=args.dev,
        pair_path=args.pairs,
        private_root=args.private_root,
    )
    write_json(args.output, result)
    print("AUTOMATED_MECHANISM_AUDIT_COMPLETE_HUMAN_RUBRIC_PENDING")


if __name__ == "__main__":
    main()
