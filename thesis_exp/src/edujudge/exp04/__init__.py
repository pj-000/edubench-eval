"""Exp4 target-objective comparison utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP04_NAME = "exp04_target_objectives"

EXP04_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP04_NAME
EXP04_DATASETS_DIR = EXP04_OUTPUT_DIR / "datasets"
EXP04_RUNS_DIR = EXP04_OUTPUT_DIR / "runs"
EXP04_TABLES_DIR = EXP04_OUTPUT_DIR / "tables"
EXP04_FIGURES_DIR = EXP04_OUTPUT_DIR / "figures"
EXP04_PREDICTIONS_DIR = EXP04_OUTPUT_DIR / "predictions"
EXP04_ARRAYS_DIR = EXP04_OUTPUT_DIR / "arrays"
EXP04_LOGS_DIR = EXP04_OUTPUT_DIR / "logs"
EXP04_REPORTS_DIR = EXP04_OUTPUT_DIR / "reports"

EXP04_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP04_NAME
EXP04_CHECKPOINTS_DIR = EXP04_ARTIFACTS_DIR / "checkpoints"
EXP04_HF_CACHE_DIR = EXP04_ARTIFACTS_DIR / "hf_cache"

EXP03_A4_TEMPLATE = "A4_question_answer_metric_rubric_metadata"
EXP03_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp03_input_ablation"
EXP03_A4_DATASET_DIR = EXP03_OUTPUT_DIR / "datasets" / EXP03_A4_TEMPLATE
EXP03_A4_RUN_DIR = EXP03_OUTPUT_DIR / "runs" / EXP03_A4_TEMPLATE

A4_FIXED_DATASET_NAME = "A4_fixed_question_answer_metric_rubric_metadata"
A4_FIXED_DATASET_DIR = EXP04_DATASETS_DIR / A4_FIXED_DATASET_NAME

EXPECTED_SPLIT_ROWS = {"train": 2654, "dev": 664, "test": 2218}

OBJECTIVE_IDS = [
    "O1_classification",
    "O2_regression_smoothl1",
    "O3_ordinal",
]

OBJECTIVE_LABELS = {
    "O1_classification": "classification",
    "O2_regression_smoothl1": "regression_smoothl1",
    "O3_ordinal": "ordinal",
}


def run_dir(objective_id: str) -> Path:
    return EXP04_RUNS_DIR / objective_id


def checkpoint_dir(objective_id: str) -> Path:
    return EXP04_CHECKPOINTS_DIR / objective_id


def ensure_exp04_dirs() -> None:
    for path in [
        EXP04_OUTPUT_DIR,
        EXP04_DATASETS_DIR,
        EXP04_RUNS_DIR,
        EXP04_TABLES_DIR,
        EXP04_FIGURES_DIR,
        EXP04_PREDICTIONS_DIR,
        EXP04_ARRAYS_DIR,
        EXP04_LOGS_DIR,
        EXP04_REPORTS_DIR,
        EXP04_CHECKPOINTS_DIR,
        EXP04_HF_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
