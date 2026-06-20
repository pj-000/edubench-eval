"""Exp11 checkpoint-selection sensitivity utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP11_NAME = "exp11_checkpoint_selection_sensitivity"
EXP11_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP11_NAME
EXP11_TABLES_DIR = EXP11_OUTPUT_DIR / "tables"
EXP11_REPORTS_DIR = EXP11_OUTPUT_DIR / "reports"
EXP11_LOCAL_RUNS_DIR = THESIS_DIR / "runs" / EXP11_NAME
EXP11_CONFIG_DIR = THESIS_DIR / "configs" / EXP11_NAME

DEFAULT_GAMMA = 4.0
DEFAULT_MAE_GUARD_DELTA = 0.005
DEFAULT_MONO_BETA = 0.2

SELECTION_RULES = [
    "dev_mae_min",
    "dev_qwk_max",
    "dev_low_to_high_min_diagnostic",
    "mae_guard_soft_risk",
    "mae_guard_label2_soft_risk",
    "mae_guard_p_gt_3_low_mean",
    "mae_guard_soft_risk_mono",
    "last_epoch_diagnostic",
]

DIAGNOSTIC_RULES = {"dev_low_to_high_min_diagnostic", "last_epoch_diagnostic"}


def ensure_exp11_dirs() -> None:
    for path in [EXP11_OUTPUT_DIR, EXP11_TABLES_DIR, EXP11_REPORTS_DIR, EXP11_LOCAL_RUNS_DIR, EXP11_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def seed_run_dir(seed: int | str, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP11_LOCAL_RUNS_DIR / f"{prefix}_{seed}" / "run"


def seed_checkpoint_dir(seed: int | str, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP11_LOCAL_RUNS_DIR / f"{prefix}_{seed}" / "checkpoints"
