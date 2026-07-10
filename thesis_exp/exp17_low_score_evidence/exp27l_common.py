"""Shared, train-only utilities for the Exp27L cross-fitted calibration audit.

The Exp27J adjudicated score is a *silver evaluation target*.  This module
keeps that target out of every teacher/risk feature vector and centralizes the
question-key grouped split logic used by preparation, fitting, and analysis.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_EXP27I = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_EXP27J = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)
DEFAULT_EXP27K = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42"
)
DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27l_group_crossfit_calibration_seed42"
)
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
REPRESENTATIVE_VIEW = "representative"
RISK_ENRICHED_VIEW = "risk_enriched"
SCORE_LABELS = (1, 2, 3, 4, 5)
TAUS = (0.25, 0.5, 0.75, 1.0, 1.5)
SIMPLEX = tuple(
    (a / 10.0, b / 10.0, c / 10.0)
    for a in range(11)
    for b in range(11 - a)
    for c in (10 - a - b,)
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def score(value: Any) -> int:
    try:
        return min(5, max(1, int(round(float(value)))))
    except (TypeError, ValueError):
        raise ValueError(f"Expected a score in 1..5, received {value!r}")


def bool_value(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def sample_id(row: dict[str, Any]) -> str:
    return clean(row.get("sample_id") or row.get("record_id") or row.get("id"))


def question_key(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def metadata(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(row.get("language") or "unknown") or "unknown",
        clean(row.get("metric_group") or row.get("metric_canonical") or "unknown") or "unknown",
        clean(row.get("subject_canonical") or row.get("subject_raw") or "unknown") or "unknown",
        clean(row.get("metric_canonical") or row.get("metric_raw") or "unknown") or "unknown",
    )


def severe_conflict(row: dict[str, Any]) -> int:
    return int(abs(int(row["original_score"]) - int(row["silver_final_score"])) >= 2)


def canonical_rows(
    train_path: Path = DEFAULT_TRAIN,
    exp27j_dir: Path = DEFAULT_EXP27J,
    exp27k_dir: Path = DEFAULT_EXP27K,
) -> list[dict[str, Any]]:
    """Join the locked 180 Exp27J rows to train-only structural metadata."""
    train = {sample_id(row): row for row in read_jsonl(train_path)}
    design = {row["sample_id"]: row for row in read_csv(exp27j_dir / "tables" / "exp27j_sampling_design.csv")}
    manifest = {row["sample_id"]: row for row in read_csv(exp27j_dir / "tables" / "exp27j_adjudicated_manifest.csv")}
    cases = {row["sample_id"]: row for row in read_csv(exp27k_dir / "tables" / "exp27k_case_level_comparison.csv")}
    expected = set(design)
    if len(expected) != 180 or set(manifest) != expected or set(cases) != expected:
        raise ValueError("Exp27J/Exp27K 180-row artifacts are not aligned")

    rows: list[dict[str, Any]] = []
    for sid in sorted(expected):
        source = train.get(sid)
        if source is None:
            raise ValueError(f"Exp27J sample is absent from train: {sid}")
        d, m, c = design[sid], manifest[sid], cases[sid]
        language, metric_group, subject, metric = metadata(source)
        original_score = score(c["original_score"])
        silver_score = score(c["final_score"])
        qwen_score = score(c["qwen_score"])
        deepseek_score = score(c["deepseek_score"])
        exp27i_score = score(c["exp27i_v1_score"])
        design_weight = float(d["design_weight"]) if clean(d.get("design_weight")) else 1.0
        row = {
            "sample_id": sid,
            "question_key": question_key(source),
            "question_key_hash": d["question_key_hash"],
            "view": d["view"],
            "score_stratum": d["score_stratum"],
            "design_weight": design_weight,
            "analysis_weight": design_weight if d["view"] == REPRESENTATIVE_VIEW else 1.0,
            "original_score": original_score,
            "silver_final_score": silver_score,
            "qwen_score": qwen_score,
            "deepseek_score": deepseek_score,
            "teacher_mean_score": int(round((qwen_score + deepseek_score) / 2.0)),
            "human_teacher_median_score": score(c["human_teacher_median_score"]),
            "exp27i_v1_score": exp27i_score,
            "exp27i_v1_tier": clean(c["exp27i_v1_tier"]),
            "target_issue_flag": bool_value(c["target_issue_flag"]),
            "teacher_gap": abs(qwen_score - deepseek_score),
            "max_three_way_gap": max(original_score, qwen_score, deepseek_score) - min(original_score, qwen_score, deepseek_score),
            "agreement_pattern": clean(c["agreement_pattern"]) or "unknown",
            "language": language,
            "metric_group": metric_group,
            "subject": subject,
            "metric": metric,
            "qwen_span_valid": bool_value(c["qwen_span_valid"]),
            "qwen_score_failure_consistent": bool_value(c["qwen_score_failure_consistent"]),
            "qwen_score_cap_consistent": bool_value(c["qwen_score_cap_consistent"]),
            "qwen_low_score_rubric_clause_present": bool_value(c["qwen_low_score_rubric_clause_present"]),
            "deepseek_span_valid": bool_value(c["deepseek_span_valid"]),
            "deepseek_score_failure_consistent": bool_value(c["deepseek_score_failure_consistent"]),
            "deepseek_score_cap_consistent": bool_value(c["deepseek_score_cap_consistent"]),
            "deepseek_low_score_rubric_clause_present": bool_value(c["deepseek_low_score_rubric_clause_present"]),
        }
        row["severe_human_silver_conflict"] = severe_conflict(row)
        rows.append(row)
    return rows


def allocation_key(row: dict[str, Any]) -> str:
    return "|".join(
        [str(row["view"]), str(row["score_stratum"]), str(row["severe_human_silver_conflict"])]
    )


def grouped_folds(rows: list[dict[str, Any]], n_splits: int, seed: int) -> dict[str, int]:
    """Deterministically allocate complete question-key clusters to balanced folds."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_key"])].append(row)
    if len(clusters) < n_splits:
        raise ValueError(f"Only {len(clusters)} question clusters for {n_splits} folds")
    strata = sorted({allocation_key(row) for row in rows})
    totals = Counter(allocation_key(row) for row in rows)
    expected = {key: totals[key] / n_splits for key in strata}
    rng = random.Random(seed)
    keyed_clusters = []
    for key, values in clusters.items():
        vector = Counter(allocation_key(row) for row in values)
        keyed_clusters.append((key, values, vector, rng.random()))
    keyed_clusters.sort(key=lambda item: (-len(item[1]), -item[2].get("representative|low_1_2|1", 0), item[3], item[0]))
    fold_counts = [Counter() for _ in range(n_splits)]
    assignments: dict[str, int] = {}
    # Seed each fold with a full question cluster before greedy balancing. This
    # prevents an empty inner validation fold when representative strata are
    # sparse, while retaining question-key isolation.
    for fold, (key, _values, vector, _tie) in enumerate(keyed_clusters[:n_splits]):
        fold_counts[fold].update(vector)
        assignments[key] = fold
    for key, values, vector, _ in keyed_clusters[n_splits:]:
        choices: list[tuple[float, int]] = []
        for fold in range(n_splits):
            trial = fold_counts[fold] + vector
            imbalance = sum(abs(trial[s] - expected[s]) for s in strata)
            size_penalty = 0.05 * sum(trial.values())
            choices.append((imbalance + size_penalty, fold))
        _, chosen = min(choices)
        fold_counts[chosen].update(vector)
        assignments[key] = chosen
    return assignments


def soft_score_distribution(row: dict[str, Any], lambdas: tuple[float, float, float], tau: float) -> list[float]:
    h, q, d = row["original_score"], row["qwen_score"], row["deepseek_score"]
    values = [
        math.exp(-((lambdas[0] * abs(k - h)) + (lambdas[1] * abs(k - q)) + (lambdas[2] * abs(k - d))) / tau)
        for k in SCORE_LABELS
    ]
    total = sum(values)
    return [value / total for value in values]


def expected_score(probs: list[float]) -> float:
    return sum(label * prob for label, prob in zip(SCORE_LABELS, probs))


def rounded_score(probs: list[float]) -> int:
    value = int(math.floor(expected_score(probs) + 0.5))
    return min(5, max(1, value))


def entropy(probs: list[float]) -> float:
    return -sum(prob * math.log(prob) for prob in probs if prob > 0)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float | None:
    values_l, weights_l = list(values), list(weights)
    den = sum(weights_l)
    return sum(value * weight for value, weight in zip(values_l, weights_l)) / den if den else None


def weighted_mae(gold: list[int], pred: list[int], weights: list[float]) -> float | None:
    return weighted_mean([abs(a - b) for a, b in zip(gold, pred)], weights)


def weighted_rate(labels: list[int], weights: list[float]) -> float | None:
    return weighted_mean(labels, weights)


def weighted_qwk(gold: list[int], pred: list[int], weights: list[float]) -> float | None:
    if not gold or len(gold) != len(pred) or len(gold) != len(weights):
        return None
    observed = np.zeros((5, 5), dtype=float)
    for a, b, weight in zip(gold, pred, weights):
        observed[a - 1, b - 1] += weight
    total = observed.sum()
    if total <= 0:
        return None
    hist_a, hist_b = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / total
    distances = np.square(np.subtract.outer(np.arange(5), np.arange(5))) / 16.0
    den = float((distances * expected).sum())
    return 1.0 - float((distances * observed).sum()) / den if den else None


def weighted_average_precision(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    positives = sum(weight for label, weight in zip(labels, weights) if label)
    if positives <= 0:
        return None
    ordered = sorted(zip(scores, labels, weights), key=lambda item: item[0], reverse=True)
    tp = 0.0
    total = 0.0
    last_recall = 0.0
    ap = 0.0
    for _, label, weight in ordered:
        total += weight
        if label:
            tp += weight
            recall = tp / positives
            ap += (recall - last_recall) * (tp / total)
            last_recall = recall
    return ap


def brier(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    return weighted_mean([(label - value) ** 2 for label, value in zip(labels, scores)], weights)


def ece(labels: list[int], scores: list[float], weights: list[float], bins: int = 5) -> float | None:
    total = sum(weights)
    if total <= 0:
        return None
    value = 0.0
    for idx in range(bins):
        low, high = idx / bins, (idx + 1) / bins
        selected = [i for i, score_value in enumerate(scores) if low <= score_value < high or (idx == bins - 1 and score_value == 1.0)]
        if not selected:
            continue
        local_w = [weights[i] for i in selected]
        mass = sum(local_w)
        value += (mass / total) * abs(
            weighted_mean([labels[i] for i in selected], local_w) - weighted_mean([scores[i] for i in selected], local_w)
        )
    return value


def top_fraction_indices(scores: list[float], weights: list[float], fraction: float, population_weight: bool) -> set[int]:
    if not scores:
        return set()
    ordered = sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)
    if not population_weight:
        return set(ordered[: max(1, int(math.ceil(len(scores) * fraction)))])
    target = fraction * sum(weights)
    selected: set[int] = set()
    running = 0.0
    for idx in ordered:
        selected.add(idx)
        running += weights[idx]
        if running >= target:
            break
    return selected


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None], seed: int, resamples: int = 2000
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_key"])].append(row)
    keys = sorted(clusters)
    if not keys:
        return None, None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample: list[dict[str, Any]] = []
        for key in rng.choices(keys, k=len(keys)):
            sample.extend(clusters[key])
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def risk_feature_rows(rows: list[dict[str, Any]], category_maps: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Build only features authorized by the Exp27L protocol."""
    categorical = ("agreement_pattern", "score_stratum", "language", "metric_group")
    if category_maps is None:
        category_maps = {field: sorted({str(row[field]) for row in rows}) for field in categorical}
    features: list[list[float]] = []
    for row in rows:
        vector = [
            abs(row["qwen_score"] - row["original_score"]),
            abs(row["deepseek_score"] - row["original_score"]),
            abs(row["qwen_score"] - row["deepseek_score"]),
            row["max_three_way_gap"],
            float(row["target_issue_flag"]),
            float(row["qwen_span_valid"]),
            float(row["qwen_score_failure_consistent"]),
            float(row["qwen_score_cap_consistent"]),
            float(row["qwen_low_score_rubric_clause_present"]),
            float(row["deepseek_span_valid"]),
            float(row["deepseek_score_failure_consistent"]),
            float(row["deepseek_score_cap_consistent"]),
            float(row["deepseek_low_score_rubric_clause_present"]),
        ]
        for field in categorical:
            vector.extend(float(str(row[field]) == category) for category in category_maps[field])
        features.append(vector)
    return np.asarray(features, dtype=float), category_maps


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Exp27L requires scikit-learn for the locked LogisticRegression risk calibrator. "
            "Install scikit-learn before fitting."
        ) from exc
