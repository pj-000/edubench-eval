"""Shared constants and helpers for the locked Exp42A experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp41_rubric_bridge.common import (
    canonical_model_text,
    human_distribution_metrics,
    human_stats,
    prediction_metrics,
    raw_rubric_text,
    read_jsonl,
    sample_id,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

ROOT = Path("thesis_exp/exp42_rubidist/outputs/exp42a_rubidist_multiseed")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
PROCESSED_PATH = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
EXP41_ROOT = Path("thesis_exp/exp41_rubric_bridge/outputs/exp41a_rubric_bridge_groupcv_seed42")
RUN_ROOT = Path("thesis_exp/runs/exp42_rubidist")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp42_rubidist")

VARIANTS = (
    "v00_hard_no_rubric",
    "v01_soft_no_rubric",
    "v10_hard_raw_rubric",
    "v11_soft_raw_rubric",
)
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("v11_soft_raw_rubric", "v01_soft_no_rubric", "rubric_effect_with_soft"),
    ("v11_soft_raw_rubric", "v10_hard_raw_rubric", "soft_effect_with_rubric"),
    ("v10_hard_raw_rubric", "v00_hard_no_rubric", "rubric_effect_with_hard"),
    ("v01_soft_no_rubric", "v00_hard_no_rubric", "soft_effect_without_rubric"),
    ("v11_soft_raw_rubric", "v00_hard_no_rubric", "combined_effect"),
)
METRICS = (
    "MAE",
    "QWK",
    "Exact_Match",
    "Kendall_tau",
    "Signed_Bias",
    "abs_Signed_Bias",
    "expected_score_MAE",
    "human_CE",
    "human_Brier",
    "human_RPS",
    "Bin_Agreement",
    "low_to_high_rate",
    "high_to_low_rate",
    "label1_recall",
    "label2_recall",
    "label3_recall",
    "label4_recall",
    "label5_recall",
)


def ensure_output_dirs(root: Path = ROOT) -> None:
    for name in ("configs", "tables", "reports", "decision", "hashes", "private/data", "logs_private"):
        (root / name).mkdir(parents=True, exist_ok=True)


def read_fold_assignment(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {"sample_id": row["sample_id"], "question_key": row["question_key"], "fold": int(row["fold"])}
            for row in csv.DictReader(handle)
        ]


def prediction_path(run_root: Path, variant: str, seed: int, fold: int) -> Path:
    return run_root / variant / f"seed_{seed}" / f"fold_{fold}" / "heldout_predictions.jsonl"


def run_summary_path(run_root: Path, variant: str, seed: int, fold: int) -> Path:
    return run_root / variant / f"seed_{seed}" / f"fold_{fold}" / "run_summary.json"


def load_oof_predictions(run_root: Path, variant: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        rows.extend(read_jsonl(prediction_path(run_root, variant, seed, fold)))
    if len(rows) != 2654 or len({row["sample_id"] for row in rows}) != 2654:
        raise ValueError(f"Expected 2654 unique OOF rows for {variant} seed {seed}")
    return rows


def all_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**prediction_metrics(rows), **human_distribution_metrics(rows)}


def entropy_band(value: float) -> str:
    if value < 1e-9:
        return "zero"
    return "low" if value <= 0.64 else "high"


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ARTIFACT_ROOT", "COMPARISONS", "EXP41_ROOT", "METRICS", "PROCESSED_PATH", "ROOT",
    "RUN_ROOT", "SEEDS", "TRAIN_PATH", "VARIANTS", "all_metrics", "canonical_model_text",
    "ensure_output_dirs", "entropy_band", "finite", "human_stats", "load_oof_predictions",
    "mean_std", "prediction_path", "raw_rubric_text", "read_fold_assignment", "read_jsonl",
    "run_summary_path", "sample_id", "sha256_file", "stable_hash", "write_csv", "write_json",
    "write_jsonl",
]
