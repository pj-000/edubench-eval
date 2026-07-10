"""Shared train-only utilities for Exp27L-R1.

Exp27L-R1 deliberately does not modify Exp27L.  It replaces the old local
greedy fold allocator with validated StratifiedGroupKFold splits and uses a
tie-safe weighted average precision implementation for discrete disagreement
signals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_EXP27I = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42")
DEFAULT_EXP27J = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42")
DEFAULT_EXP27K = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42")
DEFAULT_EXP27L = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27l_group_crossfit_calibration_seed42")
DEFAULT_OUT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27lr1_balanced_crossfit_review_seed42")

SEED = 42
OUTER_SEEDS = (42, 43, 44)
OUTER_SPLITS = 5
INNER_SPLITS = 3
REPRESENTATIVE_VIEW = "representative"
RISK_VIEW = "risk_enriched"
LABELS = (1, 2, 3, 4, 5)
TAUS = (0.25, 0.5, 0.75, 1.0, 1.5)
SIMPLEX = tuple(
    (a / 10.0, b / 10.0, (10 - a - b) / 10.0)
    for a in range(11)
    for b in range(11 - a)
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
        return min(5, max(1, int(math.floor(float(value) + 0.5))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ordinal score: {value!r}") from exc


def half_up(value: float) -> int:
    return min(5, max(1, int(math.floor(value + 0.5))))


def bool_value(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def sample_id(row: dict[str, Any]) -> str:
    return clean(row.get("sample_id") or row.get("record_id") or row.get("id"))


def question_key(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def question_key_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def row_metadata(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(row.get("language") or "unknown") or "unknown",
        clean(row.get("metric_group") or row.get("metric_canonical") or "unknown") or "unknown",
        clean(row.get("subject_canonical") or row.get("subject_raw") or "unknown") or "unknown",
        clean(row.get("metric_canonical") or row.get("metric_raw") or "unknown") or "unknown",
    )


def canonical_rows(
    train_path: Path = DEFAULT_TRAIN,
    exp27j_dir: Path = DEFAULT_EXP27J,
    exp27k_dir: Path = DEFAULT_EXP27K,
) -> list[dict[str, Any]]:
    """Join the locked 180 Exp27J/K rows to train-only source metadata."""
    train = {sample_id(row): row for row in read_jsonl(train_path)}
    design = {row["sample_id"]: row for row in read_csv(exp27j_dir / "tables" / "exp27j_sampling_design.csv")}
    cases = {row["sample_id"]: row for row in read_csv(exp27k_dir / "tables" / "exp27k_case_level_comparison.csv")}
    if len(design) != 180 or set(design) != set(cases):
        raise ValueError("Exp27J and Exp27K case tables must align to 180 rows")
    rows: list[dict[str, Any]] = []
    for sid in sorted(design):
        source = train.get(sid)
        if source is None:
            raise ValueError(f"Exp27J sample missing from train: {sid}")
        d, c = design[sid], cases[sid]
        language, metric_group, subject, metric = row_metadata(source)
        original, final = score(c["original_score"]), score(c["final_score"])
        qwen, deepseek = score(c["qwen_score"]), score(c["deepseek_score"])
        continuous_mean = (qwen + deepseek) / 2.0
        design_weight = float(d["design_weight"]) if clean(d.get("design_weight")) else 1.0
        row = {
            "sample_id": sid,
            "question_key": question_key(source),
            "question_key_hash": d["question_key_hash"],
            "view": d["view"],
            "score_stratum": d["score_stratum"],
            "design_weight": design_weight,
            "analysis_weight": design_weight if d["view"] == REPRESENTATIVE_VIEW else 1.0,
            "original_score": original,
            "silver_final_score": final,
            "qwen_score": qwen,
            "deepseek_score": deepseek,
            "continuous_teacher_mean": continuous_mean,
            "half_up_teacher_mean_score": half_up(continuous_mean),
            "human_teacher_median_score": score(c["human_teacher_median_score"]),
            "exp27i_v1_score": score(c["exp27i_v1_score"]),
            "exp27i_v1_tier": clean(c["exp27i_v1_tier"]),
            "target_issue_flag": bool_value(c["target_issue_flag"]),
            "teacher_gap": abs(qwen - deepseek),
            "max_three_way_gap": max(original, qwen, deepseek) - min(original, qwen, deepseek),
            "agreement_pattern": clean(c["agreement_pattern"]) or "unknown",
            "language": language,
            "metric_group": metric_group,
            "subject": subject,
            "metric": metric,
        }
        for prefix in ("qwen", "deepseek"):
            for suffix in (
                "span_valid",
                "score_failure_consistent",
                "score_cap_consistent",
                "low_score_rubric_clause_present",
            ):
                row[f"{prefix}_{suffix}"] = bool_value(c[f"{prefix}_{suffix}"])
        row["severe_human_silver_conflict"] = int(abs(original - final) >= 2)
        rows.append(row)
    return rows


def outer_target(row: dict[str, Any]) -> str:
    return f"{row['view']}|{row['score_stratum']}|{row['severe_human_silver_conflict']}"


def inner_target(row: dict[str, Any]) -> str:
    return f"{row['score_stratum']}|{row['severe_human_silver_conflict']}"


def stratified_group_assignments(rows: list[dict[str, Any]], seed: int, inner: bool = False) -> dict[str, int]:
    """Return validated sklearn StratifiedGroupKFold assignments; never fallback."""
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise RuntimeError("Exp27L-R1 requires scikit-learn StratifiedGroupKFold") from exc
    n_splits = INNER_SPLITS if inner else OUTER_SPLITS
    labels = [inner_target(row) if inner else outer_target(row) for row in rows]
    groups = [row["question_key_hash"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments: dict[str, int] = {}
    for fold, (_train_idx, held_idx) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        for index in held_idx:
            sid = rows[int(index)]["sample_id"]
            if sid in assignments:
                raise ValueError("StratifiedGroupKFold assigned a sample twice")
            assignments[sid] = fold
    if len(assignments) != len(rows):
        raise ValueError("StratifiedGroupKFold did not assign every sample")
    return assignments


def validate_outer(rows: list[dict[str, Any]], assignments: dict[str, int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_to_fold: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group_to_fold[row["question_key_hash"]].add(assignments[row["sample_id"]])
    leaked = [group for group, folds in group_to_fold.items() if len(folds) != 1]
    if leaked:
        raise ValueError(f"Question-key crosses outer folds: {leaked[:5]}")
    for fold in range(OUTER_SPLITS):
        held = [row for row in rows if assignments[row["sample_id"]] == fold]
        representative = [row for row in held if row["view"] == REPRESENTATIVE_VIEW]
        risk = [row for row in held if row["view"] == RISK_VIEW]
        row = {
            "outer_fold": fold,
            "total_rows": len(held),
            "unique_question_keys": len({item["question_key_hash"] for item in held}),
            "representative_rows": len(representative),
            "risk_enriched_rows": len(risk),
            "representative_severe_positive": sum(item["severe_human_silver_conflict"] for item in representative),
            "representative_severe_negative": sum(not item["severe_human_silver_conflict"] for item in representative),
        }
        checks = (
            30 <= row["total_rows"] <= 42,
            row["unique_question_keys"] >= 12,
            20 <= row["representative_rows"] <= 28,
            8 <= row["risk_enriched_rows"] <= 16,
            row["representative_severe_positive"] >= 3,
            row["representative_severe_negative"] >= 15,
        )
        row["constraints_pass"] = all(checks)
        output.append(row)
    if not all(row["constraints_pass"] for row in output):
        raise ValueError(f"Outer StratifiedGroupKFold constraints failed: {output}")
    return output


def validate_inner(rows: list[dict[str, Any]], assignments: dict[str, int], outer_fold: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fold in range(INNER_SPLITS):
        held = [row for row in rows if assignments[row["sample_id"]] == fold]
        positives = sum(row["severe_human_silver_conflict"] for row in held)
        negatives = len(held) - positives
        record = {
            "outer_fold": outer_fold,
            "inner_fold": fold,
            "held_rows": len(held),
            "unique_question_keys": len({row["question_key_hash"] for row in held}),
            "severe_positive": positives,
            "severe_negative": negatives,
        }
        record["constraints_pass"] = bool(
            record["held_rows"] >= 15
            and record["unique_question_keys"] >= 5
            and positives > 0
            and negatives > 0
        )
        output.append(record)
    if not all(row["constraints_pass"] for row in output):
        raise ValueError(f"Inner StratifiedGroupKFold constraints failed for outer fold {outer_fold}: {output}")
    return output


def soft_distribution(row: dict[str, Any], lambdas: tuple[float, float, float], tau: float) -> list[float]:
    values = []
    for label in LABELS:
        distance = (
            lambdas[0] * abs(label - row["original_score"])
            + lambdas[1] * abs(label - row["qwen_score"])
            + lambdas[2] * abs(label - row["deepseek_score"])
        )
        values.append(math.exp(-distance / tau))
    total = sum(values)
    return [value / total for value in values]


def expected_score(probs: list[float]) -> float:
    return sum(label * probability for label, probability in zip(LABELS, probs))


def rps(probs: list[float], gold: int) -> float:
    cumulative = 0.0
    value = 0.0
    for index, probability in enumerate(probs[:-1], 1):
        cumulative += probability
        indicator = float(gold <= index)
        value += (cumulative - indicator) ** 2
    return value / 4.0


def nll(probs: list[float], gold: int) -> float:
    return -math.log(max(probs[gold - 1], 1e-12))


def entropy(probs: list[float]) -> float:
    return -sum(value * math.log(value) for value in probs if value > 0.0)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float | None:
    values_l, weights_l = list(values), list(weights)
    total = sum(weights_l)
    return sum(value * weight for value, weight in zip(values_l, weights_l)) / total if total else None


def weighted_mae(gold: list[float], pred: list[float], weights: list[float]) -> float | None:
    return weighted_mean([abs(left - right) for left, right in zip(gold, pred)], weights)


def weighted_qwk(gold: list[int], pred: list[int], weights: list[float]) -> float | None:
    if not gold or not (len(gold) == len(pred) == len(weights)):
        return None
    observed = np.zeros((5, 5), dtype=float)
    for left, right, weight in zip(gold, pred, weights):
        observed[left - 1, right - 1] += weight
    total = observed.sum()
    if total <= 0:
        return None
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    distances = np.square(np.subtract.outer(np.arange(5), np.arange(5))) / 16.0
    denominator = float((distances * expected).sum())
    return 1.0 - float((distances * observed).sum()) / denominator if denominator else None


def tie_safe_average_precision(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    """Weighted AP with tied scores aggregated before updating the PR curve."""
    positive_mass = sum(weight for label, weight in zip(labels, weights) if label)
    if positive_mass <= 0:
        return None
    buckets: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for label, score_value, weight in zip(labels, scores, weights):
        buckets[float(score_value)].append((int(label), float(weight)))
    seen = 0.0
    true_positive = 0.0
    previous_recall = 0.0
    area = 0.0
    for score_value in sorted(buckets, reverse=True):
        bucket = buckets[score_value]
        seen += sum(weight for _label, weight in bucket)
        true_positive += sum(weight for label, weight in bucket if label)
        recall = true_positive / positive_mass
        precision = true_positive / seen if seen else 0.0
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def brier(labels: list[int], scores: list[float], weights: list[float]) -> float | None:
    return weighted_mean([(label - value) ** 2 for label, value in zip(labels, scores)], weights)


def ece(labels: list[int], scores: list[float], weights: list[float], bins: int = 5) -> float | None:
    total = sum(weights)
    if total <= 0:
        return None
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        positions = [
            pos for pos, value in enumerate(scores)
            if low <= value < high or (index == bins - 1 and value == 1.0)
        ]
        if not positions:
            continue
        local_weights = [weights[pos] for pos in positions]
        mass = sum(local_weights)
        observed = weighted_mean([labels[pos] for pos in positions], local_weights) or 0.0
        predicted = weighted_mean([scores[pos] for pos in positions], local_weights) or 0.0
        result += mass / total * abs(observed - predicted)
    return result


def top_fraction_indices(scores: list[float], weights: list[float], fraction: float, population_weight: bool) -> set[int]:
    ordering = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)
    if not population_weight:
        return set(ordering[: max(1, math.ceil(len(ordering) * fraction))])
    target = sum(weights) * fraction
    running = 0.0
    selected: set[int] = set()
    for index in ordering:
        selected.add(index)
        running += weights[index]
        if running >= target:
            break
    return selected


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]], statistic: Callable[[list[dict[str, Any]]], float | None], seed: int, resamples: int = 2000
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["question_key_hash"]].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = [entry for key in rng.choices(keys, k=len(keys)) for entry in clusters[key]]
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def full_feature_matrix(rows: list[dict[str, Any]], categories: dict[str, list[str]] | None = None) -> tuple[np.ndarray, dict[str, list[str]]]:
    categorical = ("agreement_pattern", "score_stratum", "language", "metric_group")
    if categories is None:
        categories = {field: sorted({str(row[field]) for row in rows}) for field in categorical}
    vectors = []
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
            vector.extend(float(str(row[field]) == item) for item in categories[field])
        vectors.append(vector)
    return np.asarray(vectors, dtype=float), categories


def core_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                abs(row["qwen_score"] - row["original_score"]),
                abs(row["deepseek_score"] - row["original_score"]),
                abs(row["qwen_score"] - row["deepseek_score"]),
                row["max_three_way_gap"],
                float(row["target_issue_flag"]),
                row["score_stratum"],
            ]
            for row in rows
        ],
        dtype=object,
    )


def safe_split_ids(path: Path) -> tuple[set[str], set[str]]:
    """Read only identifiers from a split for an overlap guard; labels are ignored."""
    ids: set[str] = set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(sample_id(row))
            keys.add(question_key(row))
    return ids, keys
