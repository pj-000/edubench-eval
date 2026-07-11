"""Shared input, metric, and I/O helpers for Exp27P."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis_exp.src.edujudge.exp03.templates import make_prompt


TEMPLATE_NAME = "A4_question_answer_metric_rubric_metadata"
MODEL_INPUT_SOURCE_FIELDS = (
    "question",
    "answer",
    "metric_canonical",
    "rubric",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
)
FORBIDDEN_INPUT_FIELDS = (
    "label_5",
    "original_label_5",
    "human_1",
    "human_2",
    "human_3",
    "judge_scores",
    "qwen_score",
    "deepseek_score",
    "calibrated_score",
    "soft_target_5",
    "sample_weight",
    "exp27o_audit_tier",
    "exp27o_training_tier",
    "exp27o_target_source",
    "exp27o_reference_status",
)
LABELS = (1, 2, 3, 4, 5)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("record_id") or row.get("id") or "")


def build_model_text(row: dict[str, Any]) -> str:
    """Render only the locked A4 whitelist; supervision fields are inaccessible."""
    whitelisted = {field: row.get(field) for field in MODEL_INPUT_SOURCE_FIELDS}
    return make_prompt(whitelisted, TEMPLATE_NAME)


def input_source_hash(row: dict[str, Any]) -> str:
    return stable_hash({field: row.get(field) for field in MODEL_INPUT_SOURCE_FIELDS})


def qwk(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = list(y_true)
    pred = list(y_pred)
    if not true:
        return float("nan")
    observed = np.zeros((5, 5), dtype=float)
    for left, right in zip(true, pred):
        observed[int(left) - 1, int(right) - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5), dtype=float)
    denominator = float((weights * expected).sum())
    return float("nan") if denominator == 0 else 1.0 - float((weights * observed).sum()) / denominator


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    """Kendall tau-b for two tied five-level ordinal score vectors."""
    if len(gold) < 2:
        return float("nan")
    table = np.zeros((5, 5), dtype=int)
    for gold_value, pred_value in zip(gold, pred):
        table[int(gold_value) - 1, int(pred_value) - 1] += 1
    concordant = 0
    discordant = 0
    for i in range(5):
        for j in range(5):
            count = int(table[i, j])
            if count == 0:
                continue
            concordant += count * int(table[i + 1 :, j + 1 :].sum())
            discordant += count * int(table[i + 1 :, :j].sum())
    tied_gold = sum(int(count) * (int(count) - 1) // 2 for count in table.sum(axis=1))
    tied_pred = sum(int(count) * (int(count) - 1) // 2 for count in table.sum(axis=0))
    tied_both = sum(int(count) * (int(count) - 1) // 2 for count in table.reshape(-1))
    only_gold = tied_gold - tied_both
    only_pred = tied_pred - tied_both
    denominator = math.sqrt(
        (concordant + discordant + only_gold)
        * (concordant + discordant + only_pred)
    )
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def score_bin(values: np.ndarray) -> np.ndarray:
    """Map 1-2/3/4-5 to the low/mid/high bins used by the paper."""
    return np.where(values <= 2, 0, np.where(values == 3, 1, 2))


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    gold = np.asarray([int(row["gold_label_5"]) for row in rows], dtype=int)
    pred = np.asarray([int(row["pred_label_5"]) for row in rows], dtype=int)
    expected = np.asarray([float(row["pred_score_expected"]) for row in rows], dtype=float)
    probs = np.asarray([[float(row[f"prob_{label}"]) for label in LABELS] for row in rows], dtype=float)
    low = gold <= 2
    high = gold >= 4
    exact_match = float(np.mean(pred == gold))
    metrics: dict[str, Any] = {
        "n": len(rows),
        "MAE_argmax": float(np.mean(np.abs(pred - gold))),
        "MAE_expected": float(np.mean(np.abs(expected - gold))),
        "QWK": qwk(gold.tolist(), pred.tolist()),
        "Accuracy": exact_match,
        "Exact_Match": exact_match,
        "Kendall_tau": kendall_tau_b(gold.tolist(), pred.tolist()),
        "Bin_Agreement": float(np.mean(score_bin(gold) == score_bin(pred))),
        "Signed_Bias_argmax": float(np.mean(pred - gold)),
        "Signed_Bias_expected": float(np.mean(expected - gold)),
        "low_n": int(low.sum()),
        "low_to_high_count": int(np.sum(low & (pred >= 4))),
        "low_to_high_rate": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "high_n": int(high.sum()),
        "high_to_low_count": int(np.sum(high & (pred <= 2))),
        "high_to_low_rate": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
        "low_mean_p_score_ge_4": float(np.mean(probs[low, 3:].sum(axis=1))) if low.any() else float("nan"),
        "high_mean_p_score_le_2": float(np.mean(probs[high, :2].sum(axis=1))) if high.any() else float("nan"),
    }
    pred_counts = Counter(pred.tolist())
    for label in LABELS:
        metrics[f"label{label}_recall"] = (
            float(np.mean(pred[gold == label] == label))
            if np.any(gold == label)
            else float("nan")
        )
        metrics[f"pred_count_{label}"] = pred_counts[label]
        metrics[f"expected_mass_{label}"] = float(probs[:, label - 1].sum())
    return metrics


def stratified_metrics(rows: list[dict[str, Any]], variant: str, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = {
        "gold_label": lambda row: str(row["gold_label_5"]),
        "language": lambda row: str(row.get("language") or "unknown"),
        "metric_group": lambda row: str(row.get("metric_group") or "unknown"),
        "subject": lambda row: str(row.get("subject_canonical") or "unknown"),
        "gold_region": lambda row: "low" if int(row["gold_label_5"]) <= 2 else (
            "high" if int(row["gold_label_5"]) >= 4 else "mid"
        ),
    }
    for dimension, getter in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[getter(row)].append(row)
        for value, subset in sorted(grouped.items()):
            output.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "dimension": dimension,
                    "value": value,
                    **prediction_metrics(subset),
                }
            )
    return output


def select_checkpoint(history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    if not history:
        raise ValueError("Cannot select a checkpoint without epoch metrics")
    minimum = min(float(row["MAE_argmax"]) for row in history)
    eligible = [row for row in history if float(row["MAE_argmax"]) <= minimum + 0.005 + 1e-12]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["low_to_high_rate"]),
            -float(row["QWK"]),
            float(row["MAE_expected"]),
            int(row["epoch"]),
        ),
    )
    pure = min(history, key=lambda row: (float(row["MAE_argmax"]), int(row["epoch"])))
    return selected, pure, [int(row["epoch"]) for row in eligible]


def finite(value: float) -> bool:
    return math.isfinite(float(value))
