"""Locked paths, variants, and helpers for Exp46A HATO-KD."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp43_rubimor.common import (
    prediction_metrics,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


ROOT = Path("thesis_exp/exp46_hato_kd/outputs/exp46a_hato_seed42")
RUN_ROOT = Path("thesis_exp/runs/exp46_hato_kd")
EXP43_ROOT = Path("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered")
EXP43_RUN_ROOT = Path("thesis_exp/runs/exp43_rubimor")
EXP44_RUN_ROOT = Path("thesis_exp/runs/exp44_taco_score")
DATA_PATH = EXP43_ROOT / "private/data/exp43_train_E4.jsonl"
FOLD_PATH = EXP43_ROOT / "private/data/exp43_groupcv_fold_assignment.csv"

TEACHER_VARIANT = "T1_4B_teacher"
STUDENT_VARIANTS = ("K1_standard_kd", "K2_hato_kd", "K3_shuffled_hato_control")
ALL_VARIANTS = (TEACHER_VARIANT, *STUDENT_VARIANTS)
EXPECTED_ROWS = 2654
EXPECTED_FOLDS = tuple(range(5))


def ensure_dirs(root: Path = ROOT) -> None:
    for name in ("configs", "tables", "reports", "decision", "hashes", "state", "logs_private"):
        (root / name).mkdir(parents=True, exist_ok=True)


def fold_assignments(path: Path = FOLD_PATH) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if set(rows.values()) != set(EXPECTED_FOLDS):
        raise RuntimeError(f"Unexpected Exp46 fold values: {sorted(set(rows.values()))}")
    return rows


def split_rows(rows: list[dict[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assigned = fold_assignments()
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in assigned]
    if missing:
        raise RuntimeError(f"Missing fold assignment for {len(missing)} rows")
    train = [row for row in rows if assigned[row["sample_id"]] != fold]
    heldout = [row for row in rows if assigned[row["sample_id"]] == fold]
    overlap = {row["question_key"] for row in train} & {row["question_key"] for row in heldout}
    if overlap:
        raise RuntimeError(f"Exp46 question-key leakage in fold {fold}: {len(overlap)}")
    return train, heldout


def run_dir(variant: str, fold: int, run_root: Path = RUN_ROOT) -> Path:
    return run_root / "groupcv" / variant / "seed_42" / f"fold_{fold}"


def teacher_logit_path(fold: int, run_root: Path = RUN_ROOT) -> Path:
    return run_dir(TEACHER_VARIANT, fold, run_root) / "teacher_logits_all.jsonl"


def baseline_prediction_paths(variant: str) -> list[Path]:
    if variant == "K0_E4":
        return [EXP44_RUN_ROOT / f"groupcv/C0_E4_baseline/seed_42/fold_{fold}/heldout_predictions.jsonl" for fold in EXPECTED_FOLDS]
    if variant == "C1_strongest_point":
        return [EXP44_RUN_ROOT / f"groupcv/C1_balanced_plain_contrastive/seed_42/fold_{fold}/heldout_predictions.jsonl" for fold in EXPECTED_FOLDS]
    raise ValueError(f"Unknown Exp46 baseline: {variant}")


def load_predictions(variant: str, run_root: Path = RUN_ROOT) -> list[dict[str, Any]]:
    if variant in {"K0_E4", "C1_strongest_point"}:
        paths = baseline_prediction_paths(variant)
    else:
        paths = [run_dir(variant, fold, run_root) / "heldout_predictions.jsonl" for fold in EXPECTED_FOLDS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp46 prediction inputs: {missing}")
    rows = [row for path in paths for row in read_jsonl(path)]
    if len(rows) != EXPECTED_ROWS or len({row["sample_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError(f"Invalid OOF coverage for {variant}: rows={len(rows)} unique={len({row['sample_id'] for row in rows})}")
    return rows


def metric_row(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = prediction_metrics(rows)
    correct2 = sum(int(row["gold_label_5"]) == 2 and int(row["pred_label_5"]) == 2 for row in rows)
    pred2 = sum(int(row["pred_label_5"]) == 2 for row in rows)
    metrics.update(
        {
            "variant": name,
            "label2_correct": correct2,
            "label2_precision": correct2 / pred2 if pred2 else 0.0,
            "mean_pred": sum(int(row["pred_label_5"]) for row in rows) / len(rows),
        }
    )
    return metrics


def validate_no_eval_access(summary: dict[str, Any]) -> bool:
    return (
        summary.get("status") == "COMPLETED"
        and summary.get("question_key_overlap") == 0
        and summary.get("dev_access_count") == 0
        and summary.get("test_access_count") == 0
        and summary.get("nan_count") == 0
        and summary.get("oom_count") == 0
    )


def write_protocol_locks(root: Path = ROOT) -> None:
    ensure_dirs(root)
    write_json(
        root / "configs/exp46a_method_lock.json",
        {
            "method": "HATO-KD",
            "teacher": {
                "variant": TEACHER_VARIANT,
                "model_family": "Qwen3-Reranker-4B",
                "supervision": "human_distribution_plus_0.5_ordinal_cdf_mse",
                "external_teacher_labels": False,
                "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            },
            "student": {
                "model_family": "Qwen3-Reranker-0.6B",
                "natural_human_anchor": True,
                "temperature": 2.0,
                "kd_coefficient": 0.25,
                "ordinal_kd_coefficient": 0.25,
                "balanced_distillation_classes": [1, 2, 3, 4, 5],
                "balanced_distillation_share_each": 0.2,
            },
            "test_access_allowed": False,
        },
    )
    write_json(
        root / "configs/exp46a_training_lock.json",
        {
            "seed": 42,
            "folds": list(EXPECTED_FOLDS),
            "epochs": 10,
            "fixed_checkpoint": "epoch_10",
            "batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation": 32,
            "max_length": 2048,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "teacher_learning_rate": 1e-4,
            "student_learning_rate": 2e-5,
            "precision": "bf16",
            "teacher_gradient_checkpointing": True,
        },
    )
    write_json(
        root / "configs/exp46a_gate_lock.json",
        {
            "teacher": {"label2_recall_min": 0.10, "label2_correct_min": 6, "label2_precision_min": 0.10, "l2h_relative_max": 0.90, "mae_improvement_min": 0.01, "qwk_improvement_min": 0.015, "kendall_improvement_min": 0.01, "exact_drop_max": 0.01, "label5_recall_drop_max": 0.02, "high_to_low_increase_max": 0.01, "abs_bias_increase_max": 0.01},
            "student": {"label2_recall_min": 0.05, "label2_correct_min": 3, "low_to_high_max": 0.7368421052631579, "mae_improvement_min": 0.003, "qwk_improvement_min": 0.005, "kendall_improvement_min": 0.005, "exact_drop_max": 0.005, "label5_recall_drop_max": 0.02, "high_to_low_increase_max": 0.01, "mean_score_downshift_max": 0.10, "abs_bias_increase_max": 0.01, "shuffled_donor_change_min": 0.80},
        },
    )


__all__ = [
    "ALL_VARIANTS", "DATA_PATH", "EXPECTED_FOLDS", "EXPECTED_ROWS", "FOLD_PATH", "ROOT", "RUN_ROOT",
    "STUDENT_VARIANTS", "TEACHER_VARIANT", "ensure_dirs", "fold_assignments", "load_predictions", "metric_row",
    "prediction_metrics", "read_jsonl", "run_dir", "sha256_file", "split_rows", "stable_hash", "teacher_logit_path",
    "validate_no_eval_access", "write_csv", "write_json", "write_jsonl", "write_protocol_locks",
]
