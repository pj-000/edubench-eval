"""Exp8 EduRisk ordinal loss utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP08_NAME = "exp08_edurisk_loss"
EXP08_RUN_ID = "QD-ER1_EduRisk_human_only"
EXP08_DATASET_ID = "QD-S0_human_only"

EXP08_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP08_NAME
EXP08_RUNS_DIR = EXP08_OUTPUT_DIR / "runs"
EXP08_TABLES_DIR = EXP08_OUTPUT_DIR / "tables"
EXP08_REPORTS_DIR = EXP08_OUTPUT_DIR / "reports"
EXP08_LOGS_DIR = EXP08_OUTPUT_DIR / "logs"
EXP08_SMOKE_DIR = EXP08_OUTPUT_DIR / "smoke_test"

EXP08_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP08_NAME
EXP08_CHECKPOINTS_DIR = EXP08_ARTIFACTS_DIR / "checkpoints"
EXP08_SMOKE_CHECKPOINTS_DIR = EXP08_ARTIFACTS_DIR / "smoke_test"

EXP08_CONFIG_DIR = THESIS_DIR / "configs" / "exp08_edurisk"
EXP08_SRC_DIR = THESIS_DIR / "src" / "edujudge" / "exp08_edurisk"

EXP08_DATASET_DIR = (
    THESIS_DIR
    / "outputs"
    / "exp06_synthetic_low_score"
    / "training_datasets"
    / EXP08_DATASET_ID
)

QD_B0_RUN_ID = "QD-B0_human_only_ordinary_ordinal"
QD_B1_RUN_ID = "QD-B1_human_only_L1_weighted_ordinal"
QD_R1_RUN_ID = "QD-R1_CORAL_human_only"
QD_BASELINE_RUNS_DIR = THESIS_DIR / "outputs" / "exp06_question_disjoint_baselines" / "runs"
QD_R1_RUN_DIR = THESIS_DIR / "outputs" / "exp07_rank_consistent_ordinal" / "runs" / QD_R1_RUN_ID

EXPECTED_SPLIT_ROWS = {"train": 3326, "dev": 1107, "test": 1103}
LABELS = [1, 2, 3, 4, 5]
ORDINAL_THRESHOLDS = [1, 2, 3, 4]

DEFAULT_TAU = 0.7
DEFAULT_ALPHA_RISK = 0.3
DEFAULT_BETA_BCE = 0.5
DEFAULT_LAMBDA_LH = 2.0
DEFAULT_LAMBDA_HL = 0.5
DEFAULT_CLASS_BALANCE_BETA = 0.99


def exp08_run_dir(smoke: bool = False) -> Path:
    base = EXP08_SMOKE_DIR if smoke else EXP08_RUNS_DIR
    return base / EXP08_RUN_ID


def exp08_checkpoint_dir(smoke: bool = False) -> Path:
    base = EXP08_SMOKE_CHECKPOINTS_DIR if smoke else EXP08_CHECKPOINTS_DIR
    return base / EXP08_RUN_ID


def ensure_exp08_dirs() -> None:
    for path in [
        EXP08_OUTPUT_DIR,
        EXP08_RUNS_DIR,
        EXP08_TABLES_DIR,
        EXP08_REPORTS_DIR,
        EXP08_LOGS_DIR,
        EXP08_SMOKE_DIR,
        EXP08_ARTIFACTS_DIR,
        EXP08_CHECKPOINTS_DIR,
        EXP08_SMOKE_CHECKPOINTS_DIR,
        EXP08_CONFIG_DIR,
        exp08_run_dir(smoke=False),
        exp08_run_dir(smoke=True),
        exp08_run_dir(smoke=False) / "tables",
        exp08_run_dir(smoke=False) / "logs",
        exp08_run_dir(smoke=False) / "predictions",
        exp08_run_dir(smoke=False) / "arrays",
    ]:
        path.mkdir(parents=True, exist_ok=True)
