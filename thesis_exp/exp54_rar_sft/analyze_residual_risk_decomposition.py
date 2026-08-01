"""Run the CPU-only Exp54 residual-risk direction-selection diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from thesis_exp.exp54_rar_sft.analyze_label2_identification import (
    nested_question_group_vector_scaling,
)


SCHEMA_VERSION = "exp54-residual-risk-decomposition-v1"
ARMS = ("S0", "R2", "R3", "P1_FIELD_DPO")
SEEDS = (42, 43, 44)
SCORES = (1, 2, 3, 4, 5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prediction_score(row: dict[str, Any]) -> int:
    if not row.get("parse_success", False):
        raise ValueError(f"unparsed frozen prediction at row {row.get('row_position')}")
    prediction = row["prediction"]
    if isinstance(prediction, dict):
        for key in ("score", "score_5", "label_5"):
            if key in prediction:
                score = int(prediction[key])
                break
        else:
            raise ValueError("prediction object has no score field")
    else:
        score = int(prediction)
    if score not in SCORES:
        raise ValueError(f"prediction score is outside 1..5: {score}")
    return score


def _ambiguity(row: dict[str, Any]) -> str:
    values = [float(row[f"human_{index}_5"]) for index in (1, 2, 3)]
    span = max(values) - min(values)
    if span == 0:
        return "H0"
    if span <= 1:
        return "H1"
    return "H2"


def _adjacent_boundaries(label: int) -> tuple[int, ...]:
    return tuple(k for k in (label - 1, label) if 1 <= k <= 4)


def annotate_dev(
    train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    train_responses = {(row["question_key"], row["answer_key"]) for row in train}
    train_questions = {row["question_key"] for row in train}
    train_answers = {row["answer_key"] for row in train}
    train_edges = {
        (row["question_key"], row["answer_key"], row["metric_id"], row["language"])
        for row in train
    }
    stratum_labels: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    scenario_labels: dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    for row in train:
        stratum_labels[(row["metric_id"], row["language"])][int(row["label_5"])] += 1
        scenario_labels[
            (row["metric_id"], row["language"], row["scenario_canonical"])
        ][int(row["label_5"])] += 1

    result = []
    for row in dev:
        label = int(row["label_5"])
        response = (row["question_key"], row["answer_key"])
        stratum = (row["metric_id"], row["language"])
        counts = stratum_labels[stratum]
        boundary = {}
        for k in range(1, 5):
            lower = sum(counts[value] for value in range(1, k + 1))
            upper = sum(counts[value] for value in range(k + 1, 6))
            boundary[str(k)] = {
                "n_le": lower,
                "n_gt": upper,
                "support": min(lower, upper),
            }
        adjacent = _adjacent_boundaries(label)
        adjacent_support = min(boundary[str(k)]["support"] for k in adjacent)
        metric_prior = [counts[value] for value in SCORES]
        scenario_counts = scenario_labels[
            (row["metric_id"], row["language"], row["scenario_canonical"])
        ]
        result.append(
            {
                "record_id": row["record_id"],
                "question_key": row["question_key"],
                "answer_key": row["answer_key"],
                "metric_id": row["metric_id"],
                "language": row["language"],
                "scenario": row["scenario_canonical"],
                "label": label,
                "response_seen": response in train_responses,
                "question_seen": row["question_key"] in train_questions,
                "answer_seen": row["answer_key"] in train_answers,
                "exact_response_metric_edge_seen": (
                    *response,
                    row["metric_id"],
                    row["language"],
                )
                in train_edges,
                "ambiguity": _ambiguity(row),
                "boundary": boundary,
                "adjacent_support": adjacent_support,
                "supported": all(boundary[str(k)]["support"] >= 20 for k in adjacent),
                "metric_language_prior_counts": metric_prior,
                "metric_language_scenario_prior_counts": [
                    scenario_counts[value] for value in SCORES
                ],
            }
        )
    return result


def _qwk(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    confusion = np.zeros((5, 5), dtype=np.float64)
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[int(label) - 1, int(prediction) - 1] += 1
    if confusion.sum() == 0:
        return None
    observed = confusion / confusion.sum()
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0))
    expected /= confusion.sum() ** 2
    indices = np.arange(5, dtype=np.float64)
    weights = np.square(indices[:, None] - indices[None, :]) / 16.0
    denominator = float((weights * expected).sum())
    return None if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def score_metrics(
    labels: Sequence[int], predictions: Sequence[int], probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if len(y) == 0 or y.shape != p.shape:
        return {"n": 0}
    errors = p - y
    tau = kendalltau(y, p).statistic if len(set(y)) > 1 and len(set(p)) > 1 else None
    result: dict[str, Any] = {
        "n": int(len(y)),
        "MAE": float(np.abs(errors).mean()),
        "Exact": float((errors == 0).mean()),
        "Signed_Bias": float(errors.mean()),
        "Kendall_tau_b": None if tau is None or math.isnan(float(tau)) else float(tau),
        "QWK": _qwk(y, p),
        "L2H_numerator": int(((y <= 2) & (p >= 4)).sum()),
        "L2H_denominator": int((y <= 2).sum()),
        "H2L_numerator": int(((y >= 4) & (p <= 2)).sum()),
        "H2L_denominator": int((y >= 4).sum()),
    }
    result["L2H"] = (
        result["L2H_numerator"] / result["L2H_denominator"]
        if result["L2H_denominator"]
        else None
    )
    result["H2L"] = (
        result["H2L_numerator"] / result["H2L_denominator"]
        if result["H2L_denominator"]
        else None
    )
    for label in (1, 2, 5):
        denominator = int((y == label).sum())
        numerator = int(((y == label) & (p == label)).sum())
        result[f"Recall_{label}_numerator"] = numerator
        result[f"Recall_{label}_denominator"] = denominator
        result[f"Recall_{label}"] = numerator / denominator if denominator else None
    for k in range(1, 5):
        result[f"boundary_error_{k}"] = float(((y > k) != (p > k)).mean())
    result["mean_boundary_error"] = float(
        np.mean([result[f"boundary_error_{k}"] for k in range(1, 5)])
    )
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(y), 5) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("probability matrix is invalid")
        one_hot = np.eye(5)[y - 1]
        cumulative_p = np.cumsum(probs, axis=1)[:, :4]
        cumulative_y = np.asarray([[float(label <= k) for k in range(1, 5)] for label in y])
        def binary_ece(confidence: np.ndarray, outcome: np.ndarray) -> float:
            total = len(confidence)
            value = 0.0
            for lower in np.linspace(0.0, 0.9, 10):
                upper = lower + 0.1
                mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
                if mask.any():
                    value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(outcome[mask].mean()))
            return value

        result.update(
            {
                "NLL": float(-np.log(np.maximum(probs[np.arange(len(y)), y - 1], 1e-300)).mean()),
                "multiclass_Brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
                "RPS": float(np.square(cumulative_p - cumulative_y).sum(axis=1).mean() / 4),
                "classwise_ECE": {
                    str(label): binary_ece(probs[:, label - 1], (y == label).astype(np.float64))
                    for label in SCORES
                },
                "low_tail_ECE": binary_ece(probs[:, :2].sum(axis=1), (y <= 2).astype(np.float64)),
                "high_tail_ECE": binary_ece(probs[:, 3:].sum(axis=1), (y >= 4).astype(np.float64)),
            }
        )
    return result


def _prediction_map(rows: Sequence[dict[str, Any]], expected_ids: Sequence[str]) -> dict[str, int]:
    if len(rows) != len(expected_ids):
        raise ValueError("prediction row count differs from dev")
    result: dict[str, int] = {}
    for expected_position, (row, record_id) in enumerate(zip(rows, expected_ids, strict=True)):
        if row["record_id"] != record_id or int(row["row_position"]) != expected_position:
            raise ValueError("prediction order/identity differs from dev")
        if record_id in result:
            raise ValueError("duplicate prediction record")
        result[record_id] = _prediction_score(row)
    return result


def _probability_matrix(
    rows: Sequence[dict[str, Any]], expected_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) != len(expected_ids):
        raise ValueError("probability row count differs from dev")
    probabilities, logprobabilities = [], []
    for position, (row, record_id) in enumerate(zip(rows, expected_ids, strict=True)):
        if row["record_id"] != record_id or int(row["row_position"]) != position:
            raise ValueError("probability order/identity differs from dev")
        probabilities.append([float(row["canonical_score_option_probabilities"][str(k)]) for k in SCORES])
        logprobabilities.append([float(row["canonical_score_option_logprobs"][str(k)]) for k in SCORES])
    matrix = np.asarray(probabilities, dtype=np.float64)
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("canonical score probabilities are not normalized")
    return matrix, np.asarray(logprobabilities, dtype=np.float64)


def clustered_gap_bootstrap(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[int],
    group: Callable[[dict[str, Any]], bool],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[row["question_key"]].append(index)
    keys = sorted(clusters)
    stats = []
    for key in keys:
        index = np.asarray(clusters[key], dtype=np.int64)
        mask = np.asarray([group(rows[i]) for i in index], dtype=bool)
        errors = np.abs(
            np.asarray([predictions[i] for i in index])
            - np.asarray([rows[i]["label"] for i in index])
        )
        stats.append((float(errors[mask].sum()), int(mask.sum()), float(errors[~mask].sum()), int((~mask).sum())))
    values = np.asarray(stats, dtype=np.float64)
    if values[:, 1].sum() == 0 or values[:, 3].sum() == 0:
        return {"estimate": None, "ci95": [None, None], "p_two_sided": None}
    estimate = values[:, 0].sum() / values[:, 1].sum() - values[:, 2].sum() / values[:, 3].sum()
    rng = np.random.default_rng(seed)
    sampled_values = []
    attempts = 0
    while len(sampled_values) < replicates and attempts < replicates * 20:
        attempts += 1
        aggregate = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        if aggregate[1] == 0 or aggregate[3] == 0:
            continue
        sampled_values.append(aggregate[0] / aggregate[1] - aggregate[2] / aggregate[3])
    if len(sampled_values) != replicates:
        raise RuntimeError("cluster bootstrap could not retain both comparison groups")
    sampled = np.asarray(sampled_values, dtype=np.float64)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    pvalue = min(1.0, 2 * min(float((sampled <= 0).mean()), float((sampled >= 0).mean())))
    return {"estimate": float(estimate), "ci95": [float(lower), float(upper)], "p_two_sided": pvalue}


def holm_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((value, key) for key, value in pvalues.items() if value is not None)
    adjusted: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    total = len(valid)
    for rank, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return adjusted


def _indices(rows: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[int]:
    return [index for index, row in enumerate(rows) if predicate(row)]


def _stratum_metrics(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[int],
    predicate: Callable[[dict[str, Any]], bool],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    index = _indices(rows, predicate)
    selected_probabilities = probabilities[index] if probabilities is not None else None
    return score_metrics(
        [rows[i]["label"] for i in index],
        [predictions[i] for i in index],
        selected_probabilities,
    )


def leave_one_group_out_gaps(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[int],
    *,
    field: str,
    adverse_group: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    values = {}
    for held_out in sorted({str(row[field]) for row in rows}):
        retained = [i for i, row in enumerate(rows) if str(row[field]) != held_out]
        adverse = [i for i in retained if adverse_group(rows[i])]
        reference = [i for i in retained if not adverse_group(rows[i])]
        if not adverse or not reference:
            values[held_out] = None
            continue
        values[held_out] = float(
            np.mean([abs(predictions[i] - rows[i]["label"]) for i in adverse])
            - np.mean([abs(predictions[i] - rows[i]["label"]) for i in reference])
        )
    finite = [value for value in values.values() if value is not None]
    return {
        "held_out_values": values,
        "minimum_gap": min(finite) if finite else None,
        "maximum_gap": max(finite) if finite else None,
        "all_same_positive_direction": bool(finite) and all(value > 0 for value in finite),
    }


def boundary_cell_support_association(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[int],
    *,
    exposure: bool,
) -> dict[str, Any]:
    """Associate support with cumulative-boundary error at the cell level.

    A cell is metric_id x language x boundary k. Aggregating before computing
    the association prevents a large cell from being counted once per row and
    makes the comparison explicitly boundary-aware.
    """
    cells: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    supports: dict[tuple[str, str, int], int] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row["response_seen"] != exposure:
            continue
        for k in range(1, 5):
            key = (row["metric_id"], row["language"], k)
            cells[key].append(int((prediction > k) != (row["label"] > k)))
            supports[key] = int(row["boundary"][str(k)]["support"])
    values = [
        (key, supports[key], float(np.mean(errors)), len(errors))
        for key, errors in sorted(cells.items())
        if errors
    ]
    if len(values) < 3 or len({value[1] for value in values}) < 2:
        return {
            "cell_n": len(values),
            "raw_spearman_rho": None,
            "raw_pvalue": None,
            "boundary_adjusted_spearman_rho": None,
            "boundary_adjusted_pvalue": None,
            "by_boundary": {},
            "adverse_strata": 0,
        }
    statistic = spearmanr([value[1] for value in values], [value[2] for value in values])
    by_boundary_values: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (_, _, k), support, error, _ in values:
        by_boundary_values[k].append((support, error))
    by_boundary = {}
    boundary_means = {}
    for k, local in sorted(by_boundary_values.items()):
        boundary_means[k] = (
            float(np.mean([item[0] for item in local])),
            float(np.mean([item[1] for item in local])),
        )
        local_x, local_y = [item[0] for item in local], [item[1] for item in local]
        local_statistic = spearmanr(local_x, local_y) if len(set(local_x)) >= 2 and len(set(local_y)) >= 2 else None
        by_boundary[str(k)] = {
            "cell_n": len(local),
            "spearman_rho": None if local_statistic is None or math.isnan(float(local_statistic.statistic)) else float(local_statistic.statistic),
            "pvalue": None if local_statistic is None or math.isnan(float(local_statistic.pvalue)) else float(local_statistic.pvalue),
        }
    residual_support = [support - boundary_means[key[2]][0] for key, support, _, _ in values]
    residual_error = [error - boundary_means[key[2]][1] for key, _, error, _ in values]
    adjusted_statistic = spearmanr(residual_support, residual_error) if len(set(residual_support)) >= 2 and len(set(residual_error)) >= 2 else None
    adverse = 0
    by_stratum: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (metric, language, k), support, error, _ in values:
        by_stratum[(metric, language)].append(
            (support - boundary_means[k][0], error - boundary_means[k][1])
        )
    for local in by_stratum.values():
        if len(local) >= 3 and len({item[0] for item in local}) >= 2 and len({item[1] for item in local}) >= 2:
            local_statistic = spearmanr([item[0] for item in local], [item[1] for item in local])
            if not math.isnan(float(local_statistic.statistic)) and local_statistic.statistic < 0:
                adverse += 1
    return {
        "cell_n": len(values),
        "raw_spearman_rho": None if math.isnan(float(statistic.statistic)) else float(statistic.statistic),
        "raw_pvalue": None if math.isnan(float(statistic.pvalue)) else float(statistic.pvalue),
        "boundary_adjusted_spearman_rho": None if adjusted_statistic is None or math.isnan(float(adjusted_statistic.statistic)) else float(adjusted_statistic.statistic),
        "boundary_adjusted_pvalue": None if adjusted_statistic is None or math.isnan(float(adjusted_statistic.pvalue)) else float(adjusted_statistic.pvalue),
        "by_boundary": by_boundary,
        "adverse_strata": adverse,
    }


def _prior_neutralize(probabilities: np.ndarray, rows: Sequence[dict[str, Any]]) -> np.ndarray:
    adjusted = np.empty_like(probabilities)
    for index, row in enumerate(rows):
        counts = np.asarray(row["metric_language_prior_counts"], dtype=np.float64)
        priors = (counts + 0.5) / (counts.sum() + 2.5)
        value = probabilities[index] / priors
        adjusted[index] = value / value.sum()
    return adjusted


def landing_report(
    rows: Sequence[dict[str, Any]], r3: np.ndarray, p1: np.ndarray
) -> dict[str, Any]:
    selected = [
        i
        for i, row in enumerate(rows)
        if row["label"] <= 2
        and row["response_seen"]
        and row["supported"]
        and row["ambiguity"] in {"H0", "H1"}
        and -((p1[i, 3] + p1[i, 4]) - (r3[i, 3] + r3[i, 4])) > 0
    ]
    if not selected:
        return {"eligible_n": 0, "positive_delta_high_n": 0, "sum_delta_high": 0.0, "GLE": None, "MCR": None}
    delta_high, delta_gold, delta_middle = [], [], []
    strata: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for i in selected:
        dh = -((p1[i, 3] + p1[i, 4]) - (r3[i, 3] + r3[i, 4]))
        dg = p1[i, rows[i]["label"] - 1] - r3[i, rows[i]["label"] - 1]
        dm = p1[i, 2] - r3[i, 2]
        delta_high.append(dh); delta_gold.append(dg); delta_middle.append(dm)
        strata[(rows[i]["metric_id"], rows[i]["language"])].append((dh, dg, dm))
    total_high = float(sum(delta_high))
    stratum_reports = []
    for (metric, language), values in sorted(strata.items()):
        high = sum(v[0] for v in values)
        stratum_reports.append(
            {
                "metric_id": metric,
                "language": language,
                "n": len(values),
                "sum_delta_high": high,
                "GLE": sum(v[1] for v in values) / high,
                "MCR": sum(v[2] for v in values) / high,
            }
        )
    return {
        "eligible_n": len(selected),
        "positive_delta_high_n": len(selected),
        "sum_delta_high": total_high,
        "sum_delta_gold": float(sum(delta_gold)),
        "sum_delta_middle": float(sum(delta_middle)),
        "GLE": float(sum(delta_gold) / total_high),
        "MCR": float(sum(delta_middle) / total_high),
        "metric_language_strata": stratum_reports,
    }


def _arm_path(root: Path, arm: str, seed: int) -> Path:
    if arm == "P1_FIELD_DPO":
        return root / "preference_dev/p1_field_dpo" / f"seed_{seed}/predictions.jsonl"
    return root / "dev_runs" / arm.lower() / f"seed{seed}/epoch3/predictions.jsonl"


def run_diagnostic(
    private_root: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError("protocol schema differs")
    forbidden = tuple(part.lower() for part in private_root.parts)
    if any("test" in part or "holdout" in part for part in forbidden):
        raise ValueError("private root must not point at test/holdout artifacts")
    train_path = private_root / "data/train.jsonl"
    dev_path = private_root / "data/dev.jsonl"
    train, dev_raw = read_jsonl(train_path), read_jsonl(dev_path)
    if len(train) != 2654 or len(dev_raw) != 664:
        raise ValueError("locked train/dev row counts differ")
    if len({row["record_id"] for row in dev_raw}) != len(dev_raw):
        raise ValueError("duplicate dev record")
    rows = annotate_dev(train, dev_raw)
    expected_ids = [row["record_id"] for row in rows]
    predictions: dict[tuple[str, int], list[int]] = {}
    input_hashes: dict[str, str] = {
        "train": sha256_file(train_path),
        "dev": sha256_file(dev_path),
        "protocol": sha256_file(protocol_path),
    }
    for arm in ARMS:
        for seed in SEEDS:
            path = _arm_path(private_root, arm, seed)
            mapped = _prediction_map(read_jsonl(path), expected_ids)
            predictions[arm, seed] = [mapped[record_id] for record_id in expected_ids]
            input_hashes[f"prediction:{arm}:{seed}"] = sha256_file(path)

    probabilities: dict[tuple[str, int], np.ndarray] = {}
    logprobabilities: dict[tuple[str, int], np.ndarray] = {}
    for arm, dirname in (("R3", "r3"), ("P1_FIELD_DPO", "p1_field_dpo")):
        for seed in SEEDS:
            path = private_root / "score_probabilities" / dirname / f"seed_{seed}/score_probabilities.jsonl"
            probabilities[arm, seed], logprobabilities[arm, seed] = _probability_matrix(read_jsonl(path), expected_ids)
            input_hashes[f"probability:{arm}:{seed}"] = sha256_file(path)

    exposure_counts = Counter(row["response_seen"] for row in rows)
    support_counts = Counter(row["supported"] for row in rows)
    ambiguity_counts = Counter(row["ambiguity"] for row in rows)
    exact_edges = sum(row["exact_response_metric_edge_seen"] for row in rows)
    replicates = bootstrap_replicates or int(protocol["statistics"]["bootstrap_replicates"])
    seed_base = int(protocol["statistics"]["bootstrap_seed"])

    per_seed: dict[str, Any] = {}
    raw_pvalues: dict[str, float | None] = {}
    exposure_seed_passes, support_seed_passes, landing_seed_passes = 0, 0, 0
    exposure_strata_positive: set[tuple[str, str]] = set()
    support_strata_positive: set[tuple[str, str]] = set()
    landing_strata_pass: set[tuple[str, str]] = set()
    low_failure_total = low_failure_unsupported = 0

    for seed in SEEDS:
        seed_report: dict[str, Any] = {"arms": {}}
        calibrated: dict[str, np.ndarray] = {}
        for arm in ("R3", "P1_FIELD_DPO"):
            calibrated[arm], _ = nested_question_group_vector_scaling(
                logprobabilities[arm, seed],
                np.asarray([row["label"] - 1 for row in rows], dtype=np.int64),
                [row["question_key"] for row in rows],
            )
        for arm in ARMS:
            pred = predictions[arm, seed]
            arm_probabilities = probabilities.get((arm, seed))
            arm_report: dict[str, Any] = {
                "all": score_metrics([row["label"] for row in rows], pred, arm_probabilities),
                "response_seen": _stratum_metrics(rows, pred, lambda row: row["response_seen"], arm_probabilities),
                "response_unseen": _stratum_metrics(rows, pred, lambda row: not row["response_seen"], arm_probabilities),
                "boundary_supported": _stratum_metrics(rows, pred, lambda row: row["supported"], arm_probabilities),
                "boundary_unsupported": _stratum_metrics(rows, pred, lambda row: not row["supported"], arm_probabilities),
                "ambiguity": {
                    level: _stratum_metrics(rows, pred, lambda row, level=level: row["ambiguity"] == level, arm_probabilities)
                    for level in ("H0", "H1", "H2")
                },
            }
            exposure_gap = clustered_gap_bootstrap(
                rows, pred, lambda row: not row["response_seen"], replicates=replicates, seed=seed_base + seed * 17 + ARMS.index(arm)
            )
            support_gap = clustered_gap_bootstrap(
                rows, pred, lambda row: not row["supported"], replicates=replicates, seed=seed_base + seed * 31 + ARMS.index(arm)
            )
            arm_report["unseen_minus_seen_MAE_bootstrap"] = exposure_gap
            arm_report["unsupported_minus_supported_MAE_bootstrap"] = support_gap
            raw_pvalues[f"{seed}:{arm}:exposure"] = exposure_gap["p_two_sided"]
            raw_pvalues[f"{seed}:{arm}:support"] = support_gap["p_two_sided"]
            associations = {}
            absolute_error = np.abs(np.asarray(pred) - np.asarray([row["label"] for row in rows]))
            for exposure in (True, False):
                idx = [i for i, row in enumerate(rows) if row["response_seen"] == exposure]
                statistic = spearmanr(
                    [rows[i]["adjacent_support"] for i in idx], absolute_error[idx]
                )
                associations["seen" if exposure else "unseen"] = {
                    "n": len(idx),
                    "spearman_rho": None if math.isnan(float(statistic.statistic)) else float(statistic.statistic),
                    "pvalue": None if math.isnan(float(statistic.pvalue)) else float(statistic.pvalue),
                }
            arm_report["continuous_support_association"] = associations
            arm_report["boundary_cell_support_association"] = {
                "seen": boundary_cell_support_association(rows, pred, exposure=True),
                "unseen": boundary_cell_support_association(rows, pred, exposure=False),
            }
            arm_report["leave_one_out"] = {
                "exposure_by_metric": leave_one_group_out_gaps(rows, pred, field="metric_id", adverse_group=lambda row: not row["response_seen"]),
                "exposure_by_language": leave_one_group_out_gaps(rows, pred, field="language", adverse_group=lambda row: not row["response_seen"]),
                "support_by_metric": leave_one_group_out_gaps(rows, pred, field="metric_id", adverse_group=lambda row: not row["supported"]),
                "support_by_language": leave_one_group_out_gaps(rows, pred, field="language", adverse_group=lambda row: not row["supported"]),
            }
            seed_report["arms"][arm] = arm_report

        for metric, language in sorted({(row["metric_id"], row["language"]) for row in rows}):
            local = [i for i, row in enumerate(rows) if (row["metric_id"], row["language"]) == (metric, language)]
            if sum(rows[i]["response_seen"] for i in local) >= 3 and sum(not rows[i]["response_seen"] for i in local) >= 3:
                pred = predictions["R3", seed]
                seen_mae = np.mean([abs(pred[i] - rows[i]["label"]) for i in local if rows[i]["response_seen"]])
                unseen_mae = np.mean([abs(pred[i] - rows[i]["label"]) for i in local if not rows[i]["response_seen"]])
                if unseen_mae > seen_mae:
                    exposure_strata_positive.add((metric, language))
        for exposure in (True, False):
            association = seed_report["arms"]["R3"]["boundary_cell_support_association"]["seen" if exposure else "unseen"]
            # The public report records only the count. Reconstruct the set
            # conservatively below from strata with an adverse local relation.
            if association["adverse_strata"]:
                for metric, language in sorted({(row["metric_id"], row["language"]) for row in rows}):
                    idx = [
                        i for i, row in enumerate(rows)
                        if row["metric_id"] == metric
                        and row["language"] == language
                        and row["response_seen"] == exposure
                    ]
                    if not idx:
                        continue
                    cell_values = []
                    pred = predictions["R3", seed]
                    for k in range(1, 5):
                        cell_values.append((rows[idx[0]]["boundary"][str(k)]["support"], float(np.mean([int((pred[i] > k) != (rows[i]["label"] > k)) for i in idx]))))
                    if len({item[0] for item in cell_values}) >= 2 and len({item[1] for item in cell_values}) >= 2:
                        local_statistic = spearmanr([item[0] for item in cell_values], [item[1] for item in cell_values])
                        if not math.isnan(float(local_statistic.statistic)) and local_statistic.statistic < 0:
                            support_strata_positive.add((metric, language))

        r3_raw, p1_raw = probabilities["R3", seed], probabilities["P1_FIELD_DPO", seed]
        seed_report["landing"] = {
            "raw": landing_report(rows, r3_raw, p1_raw),
            "prior_neutralized": landing_report(rows, _prior_neutralize(r3_raw, rows), _prior_neutralize(p1_raw, rows)),
            "cross_fitted_calibrated": landing_report(rows, calibrated["R3"], calibrated["P1_FIELD_DPO"]),
        }
        for item in seed_report["landing"]["raw"].get("metric_language_strata", []):
            if item["n"] >= 2 and item["MCR"] >= 0.50 and item["GLE"] <= 0.35:
                landing_strata_pass.add((item["metric_id"], item["language"]))

        r3_all = seed_report["arms"]["R3"]["all"]
        p1_all = seed_report["arms"]["P1_FIELD_DPO"]["all"]
        raw_landing = seed_report["landing"]["raw"]
        controls = seed_report["landing"]
        landing_conditions = [
            value["eligible_n"] > 0 and value["MCR"] is not None and value["MCR"] >= 0.50 and value["GLE"] <= 0.35
            for value in controls.values()
        ]
        small_improvement = (p1_all["Exact"] - r3_all["Exact"] <= 0.05) and (
            (p1_all["Recall_2"] or 0) - (r3_all["Recall_2"] or 0) <= 0.05
        )
        if raw_landing["sum_delta_high"] > 0 and all(landing_conditions) and small_improvement:
            landing_seed_passes += 1

        r3_exposure = seed_report["arms"]["R3"]
        gap = r3_exposure["unseen_minus_seen_MAE_bootstrap"]
        unseen = r3_exposure["response_unseen"]
        seen = r3_exposure["response_seen"]
        boundary_relative = (unseen["mean_boundary_error"] - seen["mean_boundary_error"]) / max(seen["mean_boundary_error"], 1e-12)
        ci_positive = gap["ci95"][0] is not None and gap["ci95"][0] > 0
        if ci_positive and (gap["estimate"] >= 0.05 or boundary_relative >= 0.20):
            exposure_seed_passes += 1

        r3_support = seed_report["arms"]["R3"]
        sgap = r3_support["unsupported_minus_supported_MAE_bootstrap"]
        unsupported = r3_support["boundary_unsupported"]
        supported = r3_support["boundary_supported"]
        support_relative = (unsupported["mean_boundary_error"] - supported["mean_boundary_error"]) / max(supported["mean_boundary_error"], 1e-12)
        adverse = all(
            r3_support["boundary_cell_support_association"][key]["boundary_adjusted_spearman_rho"] is not None
            and r3_support["boundary_cell_support_association"][key]["boundary_adjusted_spearman_rho"] < 0
            for key in ("seen", "unseen")
        )
        if adverse and (sgap["estimate"] >= 0.05 or support_relative >= 0.20):
            support_seed_passes += 1

        for i, row in enumerate(rows):
            if row["label"] <= 2 and predictions["R3", seed][i] >= 4:
                low_failure_total += 1
                low_failure_unsupported += not row["supported"]

        seed_report["rationale_semantic_R2_minus_R3"] = {
            stratum: (
                seed_report["arms"]["R2"][stratum]["MAE"]
                - seed_report["arms"]["R3"][stratum]["MAE"]
            )
            for stratum in ("response_seen", "response_unseen", "boundary_supported", "boundary_unsupported")
        }
        per_seed[str(seed)] = seed_report

    adjusted = holm_adjust(raw_pvalues)
    for key, value in adjusted.items():
        seed, arm, contrast = key.split(":")
        field = "unseen_minus_seen_MAE_bootstrap" if contrast == "exposure" else "unsupported_minus_supported_MAE_bootstrap"
        per_seed[seed]["arms"][arm][field]["holm_adjusted_p"] = value

    s0_r3_seen = []
    s0_r3_unseen = []
    r3_p1_seen = []
    r3_p1_unseen = []
    for seed in SEEDS:
        report = per_seed[str(seed)]["arms"]
        s0_r3_seen.append(report["S0"]["response_seen"]["MAE"] - report["R3"]["response_seen"]["MAE"])
        s0_r3_unseen.append(report["S0"]["response_unseen"]["MAE"] - report["R3"]["response_unseen"]["MAE"])
        r3_p1_seen.append(report["R3"]["response_seen"]["MAE"] - report["P1_FIELD_DPO"]["response_seen"]["MAE"])
        r3_p1_unseen.append(report["R3"]["response_unseen"]["MAE"] - report["P1_FIELD_DPO"]["response_unseen"]["MAE"])
    benefits_mainly_seen = np.mean(s0_r3_seen) > np.mean(s0_r3_unseen) and np.mean(r3_p1_seen) > np.mean(r3_p1_unseen)
    low_failure_fraction = low_failure_unsupported / low_failure_total if low_failure_total else 0.0
    gates = {
        "A_EVALUATION_UNIT": exposure_seed_passes >= 2 and len(exposure_strata_positive) >= 3 and benefits_mainly_seen,
        "B_BOUNDARY_SUPPORT": support_seed_passes >= 2 and len(support_strata_positive) >= 3 and low_failure_fraction >= 0.60,
        "C_RELATIVE_TO_ABSOLUTE": landing_seed_passes >= 2 and len(landing_strata_pass) >= 3,
    }
    passed = [name for name, value in gates.items() if value]
    direction = passed[0] if len(passed) == 1 else "D_STOP_METHOD_EXPANSION"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "direction": direction,
        "gate_results": gates,
        "passed_primary_gates": passed,
        "decision_reason": (
            "exactly_one_preregistered_direction_gate_passed"
            if len(passed) == 1
            else "no_unique_preregistered_direction_gate_passed"
        ),
        "data_scope": {
            "train_rows": len(train),
            "dev_rows": len(rows),
            "test_accessed": False,
            "new_training": False,
            "new_api_calls": False,
            "gpu_used": False,
            "input_sha256": input_hashes,
        },
        "diagnostic_inventory": {
            "response_seen": exposure_counts[True],
            "response_unseen": exposure_counts[False],
            "boundary_supported": support_counts[True],
            "boundary_unsupported": support_counts[False],
            "ambiguity": {key: ambiguity_counts[key] for key in ("H0", "H1", "H2")},
            "exact_response_metric_edges_seen": exact_edges,
            "label_by_support": {
                str(label): {
                    "supported": sum(row["label"] == label and row["supported"] for row in rows),
                    "unsupported": sum(row["label"] == label and not row["supported"] for row in rows),
                }
                for label in SCORES
            },
            "landing_base_population_by_seed": {
                str(seed): sum(
                    row["label"] <= 2
                    and row["response_seen"]
                    and row["supported"]
                    and row["ambiguity"] in {"H0", "H1"}
                    for row in rows
                )
                for seed in SEEDS
            },
        },
        "gate_evidence": {
            "exposure_seed_passes": exposure_seed_passes,
            "exposure_positive_metric_language_strata": len(exposure_strata_positive),
            "benefits_mainly_seen": bool(benefits_mainly_seen),
            "support_seed_passes": support_seed_passes,
            "support_positive_metric_language_strata": len(support_strata_positive),
            "low_score_failure_unsupported_numerator": int(low_failure_unsupported),
            "low_score_failure_unsupported_denominator": int(low_failure_total),
            "low_score_failure_unsupported_fraction": float(low_failure_fraction),
            "landing_seed_passes": landing_seed_passes,
            "landing_metric_language_strata_passes": len(landing_strata_pass),
        },
        "per_seed": per_seed,
        "limitations": [
            "Observational direction-selection diagnostic; it does not prove a causal root mechanism.",
            "Decision thresholds are project rules rather than universal statistical constants.",
            "The public report contains aggregate results and hashes only.",
        ],
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    report = run_diagnostic(
        args.private_root,
        args.protocol,
        args.output,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(report["direction"])


if __name__ == "__main__":
    main()
