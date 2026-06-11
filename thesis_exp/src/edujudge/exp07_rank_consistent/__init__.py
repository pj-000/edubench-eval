"""Exp7 rank-consistent ordinal scorer utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP07_NAME = "exp07_rank_consistent_ordinal"
EXP07_RUN_ID = "QD-R1_CORAL_human_only"
EXP07_DATASET_ID = "QD-S0_human_only"

EXP07_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP07_NAME
EXP07_RUNS_DIR = EXP07_OUTPUT_DIR / "runs"
EXP07_TABLES_DIR = EXP07_OUTPUT_DIR / "tables"
EXP07_REPORTS_DIR = EXP07_OUTPUT_DIR / "reports"
EXP07_SMOKE_DIR = EXP07_OUTPUT_DIR / "smoke_test"
EXP07_LOGS_DIR = EXP07_OUTPUT_DIR / "logs"

EXP07_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP07_NAME
EXP07_CHECKPOINTS_DIR = EXP07_ARTIFACTS_DIR / "checkpoints"
EXP07_SMOKE_CHECKPOINTS_DIR = EXP07_ARTIFACTS_DIR / "smoke_test"

EXP07_CONFIG_DIR = THESIS_DIR / "configs" / "exp07_rank_consistent"
EXP07_SRC_DIR = THESIS_DIR / "src" / "edujudge" / "exp07_rank_consistent"

EXP06_SYNTH_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_synthetic_low_score"
EXP07_DATASET_DIR = EXP06_SYNTH_OUTPUT_DIR / "training_datasets" / EXP07_DATASET_ID

QD_BASELINE_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_question_disjoint_baselines"
QD_BASELINE_RUNS_DIR = QD_BASELINE_OUTPUT_DIR / "runs"
QD_BASELINE_TABLES_DIR = QD_BASELINE_OUTPUT_DIR / "tables"
QD_B0_RUN_ID = "QD-B0_human_only_ordinary_ordinal"
QD_B1_RUN_ID = "QD-B1_human_only_L1_weighted_ordinal"

EXPECTED_SPLIT_ROWS = {"train": 3326, "dev": 1107, "test": 1103}
ORDINAL_THRESHOLDS = [1, 2, 3, 4]
LABELS = [1, 2, 3, 4, 5]


def exp07_run_dir(smoke: bool = False) -> Path:
    base = EXP07_SMOKE_DIR if smoke else EXP07_RUNS_DIR
    return base / EXP07_RUN_ID


def exp07_checkpoint_dir(smoke: bool = False) -> Path:
    base = EXP07_SMOKE_CHECKPOINTS_DIR if smoke else EXP07_CHECKPOINTS_DIR
    return base / EXP07_RUN_ID


def ensure_exp07_dirs() -> None:
    for path in [
        EXP07_OUTPUT_DIR,
        EXP07_RUNS_DIR,
        EXP07_TABLES_DIR,
        EXP07_REPORTS_DIR,
        EXP07_SMOKE_DIR,
        EXP07_LOGS_DIR,
        EXP07_ARTIFACTS_DIR,
        EXP07_CHECKPOINTS_DIR,
        EXP07_SMOKE_CHECKPOINTS_DIR,
        EXP07_CONFIG_DIR,
        exp07_run_dir(smoke=False),
        exp07_run_dir(smoke=True),
    ]:
        path.mkdir(parents=True, exist_ok=True)
