"""Exp2 CE baseline training utilities for EduBenchEvaluator 0.6B."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP02_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp02_ce_baseline"
EXP02_DATA_DIR = EXP02_OUTPUT_DIR / "data"
EXP02_TABLES_DIR = EXP02_OUTPUT_DIR / "tables"
EXP02_PREDICTIONS_DIR = EXP02_OUTPUT_DIR / "predictions"
EXP02_ARRAYS_DIR = EXP02_OUTPUT_DIR / "arrays"
EXP02_LOG_DIR = EXP02_OUTPUT_DIR / "logs"
EXP02_FIGURES_DIR = EXP02_OUTPUT_DIR / "figures"
EXP02_REPORT_DIR = EXP02_OUTPUT_DIR / "report"
EXP02_SUMMARIES_DIR = EXP02_OUTPUT_DIR / "summaries"
EXP02_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / "exp02_ce_baseline"
EXP02_CHECKPOINT_DIR = EXP02_ARTIFACTS_DIR / "checkpoints" / "edubench_evaluator_0_6b_ce"

SPLIT_DIR = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42"
TRAIN_SPLIT_PATH = SPLIT_DIR / "train.jsonl"
DEV_SPLIT_PATH = SPLIT_DIR / "dev.jsonl"
TEST_SPLIT_PATH = SPLIT_DIR / "test.jsonl"

EXPECTED_SPLIT_ROWS = {"train": 2654, "dev": 664, "test": 2218}


def ensure_exp02_dirs() -> None:
    for path in [
        EXP02_OUTPUT_DIR,
        EXP02_DATA_DIR,
        EXP02_TABLES_DIR,
        EXP02_PREDICTIONS_DIR,
        EXP02_ARRAYS_DIR,
        EXP02_LOG_DIR,
        EXP02_FIGURES_DIR,
        EXP02_REPORT_DIR,
        EXP02_SUMMARIES_DIR,
        EXP02_CHECKPOINT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
