"""Exp2 CE baseline training utilities for EduBenchEvaluator 0.6B."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP02_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp02_ce_baseline"
EXP02_DATA_DIR = EXP02_OUTPUT_DIR / "data"
EXP02_TABLES_DIR = EXP02_OUTPUT_DIR / "tables"
EXP02_MODEL_DIR = EXP02_OUTPUT_DIR / "models" / "edubench_evaluator_0_6b_ce"
EXP02_LOG_DIR = EXP02_OUTPUT_DIR / "logs"

SPLIT_DIR = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42"
TRAIN_SPLIT_PATH = SPLIT_DIR / "train.jsonl"
DEV_SPLIT_PATH = SPLIT_DIR / "dev.jsonl"
TEST_SPLIT_PATH = SPLIT_DIR / "test.jsonl"

EXPECTED_SPLIT_ROWS = {"train": 2654, "dev": 664, "test": 2218}


def ensure_exp02_dirs() -> None:
    for path in [EXP02_OUTPUT_DIR, EXP02_DATA_DIR, EXP02_TABLES_DIR, EXP02_MODEL_DIR, EXP02_LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

