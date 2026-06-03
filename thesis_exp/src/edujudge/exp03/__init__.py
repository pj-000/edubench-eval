"""Exp3 rubric-aware input ablation utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP03_NAME = "exp03_input_ablation"

EXP03_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP03_NAME
EXP03_DATASETS_DIR = EXP03_OUTPUT_DIR / "datasets"
EXP03_RUNS_DIR = EXP03_OUTPUT_DIR / "runs"
EXP03_TABLES_DIR = EXP03_OUTPUT_DIR / "tables"
EXP03_FIGURES_DIR = EXP03_OUTPUT_DIR / "figures"
EXP03_PREDICTIONS_DIR = EXP03_OUTPUT_DIR / "predictions"
EXP03_ARRAYS_DIR = EXP03_OUTPUT_DIR / "arrays"
EXP03_SAMPLES_DIR = EXP03_OUTPUT_DIR / "samples"
EXP03_LOGS_DIR = EXP03_OUTPUT_DIR / "logs"
EXP03_REPORTS_DIR = EXP03_OUTPUT_DIR / "reports"
EXP03_TEMPLATES_DIR = EXP03_OUTPUT_DIR / "templates"
EXP03_SMOKE_DIR = EXP03_OUTPUT_DIR / "smoke_test"

EXP03_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP03_NAME
EXP03_CHECKPOINTS_DIR = EXP03_ARTIFACTS_DIR / "checkpoints"
EXP03_HF_CACHE_DIR = EXP03_ARTIFACTS_DIR / "hf_cache"

SPLIT_DIR = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42"
TRAIN_SPLIT_PATH = SPLIT_DIR / "train.jsonl"
DEV_SPLIT_PATH = SPLIT_DIR / "dev.jsonl"
TEST_SPLIT_PATH = SPLIT_DIR / "test.jsonl"
SPLIT_PATHS = {
    "train": TRAIN_SPLIT_PATH,
    "dev": DEV_SPLIT_PATH,
    "test": TEST_SPLIT_PATH,
}

EXPECTED_SPLIT_ROWS = {"train": 2654, "dev": 664, "test": 2218}

TEMPLATE_NAMES = [
    "A0_answer_only",
    "A1_question_answer",
    "A2_question_answer_metric",
    "A3_question_answer_metric_rubric",
    "A4_question_answer_metric_rubric_metadata",
]

CORE_TRAIN_TEMPLATES = [
    "A3_question_answer_metric_rubric",
    "A4_question_answer_metric_rubric_metadata",
]

EXP02_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp02_ce_baseline"


def template_dataset_dir(template_name: str) -> Path:
    return EXP03_DATASETS_DIR / template_name


def template_run_dir(template_name: str) -> Path:
    return EXP03_RUNS_DIR / template_name


def template_checkpoint_dir(template_name: str) -> Path:
    return EXP03_CHECKPOINTS_DIR / template_name


def ensure_exp03_dirs() -> None:
    for path in [
        EXP03_OUTPUT_DIR,
        EXP03_DATASETS_DIR,
        EXP03_RUNS_DIR,
        EXP03_TABLES_DIR,
        EXP03_FIGURES_DIR,
        EXP03_PREDICTIONS_DIR,
        EXP03_ARRAYS_DIR,
        EXP03_SAMPLES_DIR,
        EXP03_LOGS_DIR,
        EXP03_REPORTS_DIR,
        EXP03_TEMPLATES_DIR,
        EXP03_SMOKE_DIR,
        EXP03_CHECKPOINTS_DIR,
        EXP03_HF_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
