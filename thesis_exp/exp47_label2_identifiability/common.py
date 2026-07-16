"""Shared paths and aggregate-only helpers for Exp47A."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis_exp.exp43_rubimor.common import (
    canonical_metric,
    human_stats,
    prediction_metrics,
    read_jsonl,
    sample_id,
    sha256_file,
    write_csv,
    write_json,
)


ROOT = Path("thesis_exp/exp47_label2_identifiability/outputs/exp47a_label2_audit")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
EXP43_ROOT = Path("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered")
E4_PATH = EXP43_ROOT / "private/data/exp43_train_E4.jsonl"
FOLD_PATH = EXP43_ROOT / "private/data/exp43_groupcv_fold_assignment.csv"
EXP44_RUN_ROOT = Path("thesis_exp/runs/exp44_taco_score")
EXP46_RUN_ROOT = Path("thesis_exp/runs/exp46_hato_kd")
EXP46_PUBLIC_ROOT = Path("thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42")
EXPECTED_ROWS = 2654
EXPECTED_FOLDS = tuple(range(5))

MODEL_06B = "M0_0.6B_E4"
MODEL_4B = "M1_4B_teacher"


def ensure_dirs(root: Path = ROOT) -> None:
    for name in ("configs", "tables", "reports", "decision", "hashes", "state"):
        (root / name).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fold_assignments(path: Path = FOLD_PATH) -> dict[str, int]:
    rows = read_csv(path)
    mapping = {row["sample_id"]: int(row["fold"]) for row in rows}
    if len(mapping) != EXPECTED_ROWS or set(mapping.values()) != set(EXPECTED_FOLDS):
        raise RuntimeError(f"Invalid Exp47 fold assignment: rows={len(mapping)} folds={sorted(set(mapping.values()))}")
    return mapping


def _raw_to_canonical(row: dict[str, Any]) -> dict[str, Any]:
    stats = human_stats(row)
    return {
        "sample_id": sample_id(row),
        "question_key": str(row["question_key"]),
        "gold_label_5": stats["gold_label_5"],
        "human_scores": stats["human_scores"],
        "human_distribution_5": stats["human_distribution_5"],
        "expected_human_score": stats["expected_human_score"],
        "human_entropy": stats["human_entropy"],
        "human_score_range": stats["human_score_range"],
        "metric": canonical_metric(row),
        "subject": row.get("subject_canonical") or row.get("subject_raw") or "unknown",
        "language": row.get("language") or "unknown",
        "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "unknown",
        "education_level": row.get("education_level_canonical") or row.get("education_level_raw") or "unknown",
        "generator_model": row.get("generator_model") or row.get("answer_model") or "unknown",
    }


def load_canonical_rows(e4_path: Path = E4_PATH, train_path: Path = TRAIN_PATH) -> list[dict[str, Any]]:
    if e4_path.exists():
        rows = read_jsonl(e4_path)
        raw_by_id = {sample_id(row): row for row in read_jsonl(train_path)}
        canonical = [
            {
                **row,
                "generator_model": raw_by_id[row["sample_id"]].get("generator_model")
                or raw_by_id[row["sample_id"]].get("answer_model")
                or "unknown",
            }
            for row in rows
        ]
    else:
        canonical = [_raw_to_canonical(row) for row in read_jsonl(train_path)]
    if len(canonical) != EXPECTED_ROWS or len({row["sample_id"] for row in canonical}) != EXPECTED_ROWS:
        raise RuntimeError(f"Invalid Exp47 canonical train rows: {len(canonical)}")
    return canonical


def label2_subtype(scores: Iterable[int | float]) -> str:
    values = sorted(int(float(value)) for value in scores)
    if values == [2, 2, 2]:
        return "strict_222"
    count_two = values.count(2)
    score_range = max(values) - min(values)
    if count_two >= 2 and score_range < 3:
        return "stable_majority_non_strict"
    return "ambiguous"


def label2_flags(row: dict[str, Any]) -> dict[str, Any]:
    scores = [int(float(value)) for value in row["human_scores"]]
    subtype = label2_subtype(scores) if int(row["gold_label_5"]) == 2 else "not_label2"
    count_two = scores.count(2)
    return {
        "label2_subtype": subtype,
        "strict_label2": subtype == "strict_222",
        "stable_label2": subtype in {"strict_222", "stable_majority_non_strict"},
        "ambiguous_label2": subtype == "ambiguous",
        "annotator_two_count": count_two,
        "human_score_pattern": "/".join(str(value) for value in sorted(scores)),
    }


def rows_by_id() -> dict[str, dict[str, Any]]:
    output = {}
    for row in load_canonical_rows():
        output[row["sample_id"]] = {**row, **label2_flags(row)}
    return output


def softmax(values: Iterable[float]) -> list[float]:
    data = np.asarray(list(values), dtype=float)
    shifted = data - np.max(data)
    result = np.exp(shifted)
    result /= result.sum()
    return result.tolist()


def enrich_prediction(
    source: dict[str, Any],
    metadata: dict[str, Any],
    model: str,
    role: str,
    fold: int,
    logits: list[float] | None = None,
) -> dict[str, Any]:
    probabilities = softmax(logits) if logits is not None else [float(source[f"prob_{label}"]) for label in range(1, 6)]
    values = list(logits) if logits is not None else [math.log(max(value, 1e-12)) for value in probabilities]
    predicted = int(np.argmax(probabilities)) + 1
    output = {
        "model": model,
        "role": role,
        "fold": fold,
        "sample_id": metadata["sample_id"],
        "question_key": metadata["question_key"],
        "gold_label_5": int(metadata["gold_label_5"]),
        "human_distribution_5": metadata["human_distribution_5"],
        "pred_label_5": predicted,
        "pred_score_expected": sum(label * probabilities[label - 1] for label in range(1, 6)),
        "label2_subtype": metadata["label2_subtype"],
        "stable_label2": metadata["stable_label2"],
        "strict_label2": metadata["strict_label2"],
        "ambiguous_label2": metadata["ambiguous_label2"],
        "logit_source": "native" if logits is not None else "log_probability_equivalent",
    }
    for label in range(1, 6):
        output[f"prob_{label}"] = probabilities[label - 1]
        output[f"logit_{label}"] = values[label - 1]
    return output


def load_4b_predictions(run_root: Path = EXP46_RUN_ROOT) -> list[dict[str, Any]]:
    metadata = rows_by_id()
    assignments = fold_assignments()
    output = []
    for fold in EXPECTED_FOLDS:
        fold_root = run_root / f"groupcv/T1_4B_teacher/seed_42/fold_{fold}"
        all_rows = read_jsonl(fold_root / "teacher_logits_all.jsonl")
        if len(all_rows) != EXPECTED_ROWS:
            raise RuntimeError(f"Invalid 4B logit row count for fold {fold}: {len(all_rows)}")
        for row in all_rows:
            sample = metadata[row["sample_id"]]
            expected_role = "heldout" if assignments[row["sample_id"]] == fold else "outer_train"
            if row.get("role") != expected_role:
                raise RuntimeError(f"Role mismatch for 4B fold {fold}")
            if expected_role == "outer_train":
                output.append(enrich_prediction(row, sample, MODEL_4B, expected_role, fold, list(row["teacher_logits"])))
        # Use the official heldout-only export for OOF metrics. The all-row export
        # is retained solely for train-side diagnosis because batch composition can
        # otherwise produce a small evaluation discrepancy.
        heldout_rows = read_jsonl(fold_root / "heldout_predictions.jsonl")
        for row in heldout_rows:
            sample = metadata[row["sample_id"]]
            if assignments[row["sample_id"]] != fold:
                raise RuntimeError(f"4B official heldout fold mismatch for fold {fold}")
            logits = [float(row[f"logit_{label}"]) for label in range(1, 6)]
            output.append(enrich_prediction(row, sample, MODEL_4B, "heldout", fold, logits))
    return output


def load_06b_oof_predictions(run_root: Path = EXP44_RUN_ROOT) -> list[dict[str, Any]]:
    metadata = rows_by_id()
    assignments = fold_assignments()
    output = []
    for fold in EXPECTED_FOLDS:
        path = run_root / f"groupcv/C0_E4_baseline/seed_42/fold_{fold}/heldout_predictions.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            sample = metadata[row["sample_id"]]
            if assignments[row["sample_id"]] != fold:
                raise RuntimeError(f"0.6B heldout fold mismatch for fold {fold}")
            output.append(enrich_prediction(row, sample, MODEL_06B, "heldout", fold))
    if len(output) != EXPECTED_ROWS or len({row["sample_id"] for row in output}) != EXPECTED_ROWS:
        raise RuntimeError(f"Invalid 0.6B OOF coverage: {len(output)}")
    return output


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = prediction_metrics(rows)
    correct2 = sum(int(row["gold_label_5"]) == 2 and int(row["pred_label_5"]) == 2 for row in rows)
    pred2 = sum(int(row["pred_label_5"]) == 2 for row in rows)
    metrics.update(
        {
            "label2_correct": correct2,
            "label2_total": sum(int(row["gold_label_5"]) == 2 for row in rows),
            "label2_precision": correct2 / pred2 if pred2 else 0.0,
            "class2_prediction_count": pred2,
        }
    )
    return metrics


def quantile(values: Iterable[float], q: float) -> float:
    data = list(values)
    return float(np.quantile(data, q)) if data else float("nan")


def finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sanitize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: finite_or_none(value) for key, value in row.items()} for row in rows]


def input_hashes() -> dict[str, Any]:
    paths = [TRAIN_PATH, FOLD_PATH, EXP46_PUBLIC_ROOT / "decision/exp46a_teacher_capacity_decision.json"]
    return {str(path): sha256_file(path) for path in paths if path.exists()}


__all__ = [
    "E4_PATH",
    "EXPECTED_FOLDS",
    "EXPECTED_ROWS",
    "EXP44_RUN_ROOT",
    "EXP46_PUBLIC_ROOT",
    "EXP46_RUN_ROOT",
    "FOLD_PATH",
    "MODEL_06B",
    "MODEL_4B",
    "ROOT",
    "TRAIN_PATH",
    "ensure_dirs",
    "fold_assignments",
    "input_hashes",
    "label2_flags",
    "label2_subtype",
    "load_06b_oof_predictions",
    "load_4b_predictions",
    "load_canonical_rows",
    "metric_summary",
    "quantile",
    "read_csv",
    "rows_by_id",
    "sanitize_rows",
    "sha256_file",
    "write_csv",
    "write_json",
]
