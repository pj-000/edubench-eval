"""Exp9 risk-aware pairwise ordinal training utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP09_NAME = "exp09_pairwise_ordinal"
EXP09_RUN_ID = "QD-PR1_PairwiseRiskOrdinal_human_only"
EXP09_DATASET_ID = "QD-S0_human_only"

EXP09_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP09_NAME
EXP09_TABLES_DIR = EXP09_OUTPUT_DIR / "tables"
EXP09_REPORTS_DIR = EXP09_OUTPUT_DIR / "reports"
EXP09_PAIRS_DIR = EXP09_OUTPUT_DIR / "pairs"
EXP09_LOGS_DIR = EXP09_OUTPUT_DIR / "logs"
EXP09_RUNS_DIR = EXP09_OUTPUT_DIR / "runs"
EXP09_SMOKE_DIR = EXP09_OUTPUT_DIR / "smoke_test"

EXP09_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP09_NAME
EXP09_CHECKPOINTS_DIR = EXP09_ARTIFACTS_DIR / "checkpoints"
EXP09_SMOKE_CHECKPOINTS_DIR = EXP09_ARTIFACTS_DIR / "smoke_test"

EXP09_CONFIG_DIR = THESIS_DIR / "configs" / "exp09_pairwise_ordinal"
EXP09_SRC_DIR = THESIS_DIR / "src" / "edujudge" / "exp09_pairwise_ordinal"

EXP09_DATASET_DIR = (
    THESIS_DIR
    / "outputs"
    / "exp06_synthetic_low_score"
    / "training_datasets"
    / EXP09_DATASET_ID
)

QD_BASELINE_RUNS_DIR = THESIS_DIR / "outputs" / "exp06_question_disjoint_baselines" / "runs"
QD_B0_RUN_ID = "QD-B0_human_only_ordinary_ordinal"
QD_B1_RUN_ID = "QD-B1_human_only_L1_weighted_ordinal"
QD_R1_RUN_ID = "QD-R1_CORAL_human_only"
QD_R1_RUN_DIR = THESIS_DIR / "outputs" / "exp07_rank_consistent_ordinal" / "runs" / QD_R1_RUN_ID
QD_ER1_RUN_ID = "QD-ER1_EduRisk_human_only"
QD_ER1_RUN_DIR = THESIS_DIR / "outputs" / "exp08_edurisk_loss" / "runs" / QD_ER1_RUN_ID

EXPECTED_SPLIT_ROWS = {"train": 3326, "dev": 1107, "test": 1103}
LABELS = [1, 2, 3, 4, 5]
ORDINAL_THRESHOLDS = [1, 2, 3, 4]

PAIR_TYPE_PROPORTIONS = {
    "low_high": 0.40,
    "low_mid": 0.20,
    "adjacent": 0.30,
    "random_ordinal": 0.10,
}
PAIR_PRIORITIES = ["same_question", "same_metric_language", "same_metric", "any_valid"]
DEFAULT_TRAIN_PAIR_COUNT = 20_000
DEFAULT_DEV_PAIR_COUNT = 5_000
DEFAULT_PAIR_SAMPLING_SEED = 42
DEFAULT_MAX_PAIRS_PER_RECORD = 80
DEFAULT_MAX_PAIRS_PER_LOW_RECORD = 240

DEFAULT_MARGIN_SCALE = 1.0
DEFAULT_LOW_HIGH_MARGIN = 0.25
DEFAULT_LOW_HIGH_WEIGHT = 1.0
DEFAULT_GAP_WEIGHT = 0.5
DEFAULT_LAMBDA_PAIR = 0.3
DEFAULT_W_MIN = 0.5
DEFAULT_W_MAX = 3.0


def exp09_run_dir(smoke: bool = False) -> Path:
    return (EXP09_SMOKE_DIR if smoke else EXP09_RUNS_DIR) / EXP09_RUN_ID


def exp09_checkpoint_dir(smoke: bool = False) -> Path:
    return (EXP09_SMOKE_CHECKPOINTS_DIR if smoke else EXP09_CHECKPOINTS_DIR) / EXP09_RUN_ID


def ensure_exp09_dirs() -> None:
    for path in [
        EXP09_OUTPUT_DIR,
        EXP09_TABLES_DIR,
        EXP09_REPORTS_DIR,
        EXP09_PAIRS_DIR,
        EXP09_LOGS_DIR,
        EXP09_RUNS_DIR,
        EXP09_SMOKE_DIR,
        EXP09_ARTIFACTS_DIR,
        EXP09_CHECKPOINTS_DIR,
        EXP09_SMOKE_CHECKPOINTS_DIR,
        EXP09_CONFIG_DIR,
        exp09_run_dir(False),
        exp09_run_dir(True),
        exp09_run_dir(False) / "tables",
        exp09_run_dir(False) / "logs",
        exp09_run_dir(False) / "predictions",
        exp09_run_dir(False) / "arrays",
    ]:
        path.mkdir(parents=True, exist_ok=True)
