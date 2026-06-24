"""Exp15 post-hoc ordinal calibration paths and defaults."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import REPO_ROOT


EXP15_NAME = "exp15_posthoc_ordinal_calibration"
EXP15_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / EXP15_NAME
EXP15_TABLES_DIR = EXP15_OUTPUT_DIR / "tables"
EXP15_REPORTS_DIR = EXP15_OUTPUT_DIR / "reports"
EXP15_CONFIG_DIR = REPO_ROOT / "thesis_exp" / "configs" / EXP15_NAME

DEFAULT_SELECTION_RULE = "pava_mae_guard_low_to_high_then_label2_then_calibration"
DEFAULT_SELECTION_DELTA = 0.005
DEFAULT_ECE_BINS = 10


def ensure_exp15_dirs() -> None:
    for path in [EXP15_OUTPUT_DIR, EXP15_TABLES_DIR, EXP15_REPORTS_DIR, EXP15_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
