"""Exp5 low-score loss ablation utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP05_NAME = "exp05_low_score_loss"
EXP05_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP05_NAME
EXP05_TABLES_DIR = EXP05_OUTPUT_DIR / "tables"
EXP05_FIGURES_DIR = EXP05_OUTPUT_DIR / "figures"
EXP05_REPORTS_DIR = EXP05_OUTPUT_DIR / "reports"
EXP05_RUNS_DIR = EXP05_OUTPUT_DIR / "runs"
EXP05_SMOKE_DIR = EXP05_OUTPUT_DIR / "smoke_test"
EXP05_SAMPLES_DIR = EXP05_OUTPUT_DIR / "samples"
EXP05_LOGS_DIR = EXP05_OUTPUT_DIR / "logs"

EXP05_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP05_NAME
EXP05_CHECKPOINTS_DIR = EXP05_ARTIFACTS_DIR / "checkpoints"
EXP05_HF_CACHE_DIR = EXP05_ARTIFACTS_DIR / "hf_cache"

L1_RUN_ID = "L1_weighted_ordinal"
L2A_RUN_ID = "L2a_asymmetric_ordinal_lambda03_margin0"
L2B_RUN_ID = "L2b_asymmetric_ordinal_lambda05_margin0"
L3B_RUN_ID = "L3b_weighted_threshold_mu03"
L2_RUN_CONFIGS = {
    L2A_RUN_ID: {"lambda_low": 0.3, "margin": 0.0},
    L2B_RUN_ID: {"lambda_low": 0.5, "margin": 0.0},
}
L3B_RUN_CONFIG = {"mu_thr": 0.3}
LOSS_RUN_IDS = [
    "L0_exp04_o3_ordinal",
    L1_RUN_ID,
    L2A_RUN_ID,
    L2B_RUN_ID,
    L3B_RUN_ID,
]

EXP04_NAME = "exp04_target_objectives"
EXP04_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP04_NAME
EXP04_TABLES_DIR = EXP04_OUTPUT_DIR / "tables"
EXP04_O3_RUN_DIR = EXP04_OUTPUT_DIR / "runs" / "O3_ordinal"
EXP04_A4_DATASET_DIR = (
    EXP04_OUTPUT_DIR
    / "datasets"
    / "A4_fixed_question_answer_metric_rubric_metadata"
)

EXPECTED_SPLIT_ROWS = {"train": 2654, "dev": 664, "test": 2218}
LABELS = [1, 2, 3, 4, 5]
ORDINAL_THRESHOLDS = [1, 2, 3, 4]
DEFAULT_W_MIN = 0.5
DEFAULT_W_MAX = 3.0


def l1_run_dir(smoke: bool = False) -> Path:
    base = EXP05_SMOKE_DIR if smoke else EXP05_RUNS_DIR
    return base / L1_RUN_ID


def l2_run_dir(run_id: str, smoke: bool = False) -> Path:
    base = EXP05_SMOKE_DIR if smoke else EXP05_RUNS_DIR
    return base / run_id


def l3b_run_dir(smoke: bool = False) -> Path:
    base = EXP05_SMOKE_DIR if smoke else EXP05_RUNS_DIR
    return base / L3B_RUN_ID


def l1_checkpoint_dir(smoke: bool = False) -> Path:
    base = EXP05_ARTIFACTS_DIR / "smoke_test" if smoke else EXP05_CHECKPOINTS_DIR
    return base / L1_RUN_ID


def l2_checkpoint_dir(run_id: str, smoke: bool = False) -> Path:
    base = EXP05_ARTIFACTS_DIR / "smoke_test" if smoke else EXP05_CHECKPOINTS_DIR
    return base / run_id


def l3b_checkpoint_dir(smoke: bool = False) -> Path:
    base = EXP05_ARTIFACTS_DIR / "smoke_test" if smoke else EXP05_CHECKPOINTS_DIR
    return base / L3B_RUN_ID


def ensure_exp05_dirs() -> None:
    for path in [
        EXP05_OUTPUT_DIR,
        EXP05_TABLES_DIR,
        EXP05_FIGURES_DIR,
        EXP05_REPORTS_DIR,
        EXP05_RUNS_DIR,
        EXP05_SMOKE_DIR,
        EXP05_SAMPLES_DIR,
        EXP05_LOGS_DIR,
        EXP05_CHECKPOINTS_DIR,
        EXP05_HF_CACHE_DIR,
        l1_run_dir(smoke=True),
        *[l2_run_dir(run_id, smoke=True) for run_id in L2_RUN_CONFIGS],
        l3b_run_dir(smoke=True),
    ]:
        path.mkdir(parents=True, exist_ok=True)
