"""Exp12 monotonic projection / MAP-OC paths and defaults."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import REPO_ROOT


EXP12_NAME = "exp12_monotonic_projection_map_oc"
EXP12_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / EXP12_NAME
EXP12_TABLES_DIR = EXP12_OUTPUT_DIR / "tables"
EXP12_REPORTS_DIR = EXP12_OUTPUT_DIR / "reports"
EXP12_LOCAL_RUNS_DIR = REPO_ROOT / "thesis_exp" / "runs" / EXP12_NAME
EXP12_CONFIG_DIR = REPO_ROOT / "thesis_exp" / "configs" / EXP12_NAME

EXP11_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / "exp11_checkpoint_selection_sensitivity"
EXP11_TABLES_DIR = EXP11_OUTPUT_DIR / "tables"
EXP11_LOCAL_RUNS_DIR = REPO_ROOT / "thesis_exp" / "runs" / "exp11_checkpoint_selection_sensitivity"

DEFAULT_SELECTION_RULE = "mae_guard_p_gt_3_low_mean"
DEFAULT_SELECTION_DELTA = 0.005
DEFAULT_SOFT_RISK_GAMMA = 4.0
DEFAULT_PROJECTION_METHOD = "pava"
DEFAULT_GPU_LIST = "6 7"
DEFAULT_SEEDS = "42"
DEFAULT_EPOCHS = "3"

EXP12B_RUNS = [
    "train_projection_score",
    "train_projection_point_pair",
    "map_oc_full",
]

CONFIG_BY_RUN = {
    "qdpr2_raw_selected": EXP12_CONFIG_DIR / "exp12_qdpr2_raw_selected.yaml",
    "decode_projection_only": EXP12_CONFIG_DIR / "exp12_decode_projection_only.yaml",
    "train_projection_score": EXP12_CONFIG_DIR / "exp12_train_projection_score.yaml",
    "train_projection_point_pair": EXP12_CONFIG_DIR / "exp12_train_projection_point_pair.yaml",
    "map_oc_full": EXP12_CONFIG_DIR / "exp12_map_oc_full.yaml",
}


def ensure_exp12_dirs() -> None:
    for path in [EXP12_OUTPUT_DIR, EXP12_TABLES_DIR, EXP12_REPORTS_DIR, EXP12_LOCAL_RUNS_DIR, EXP12_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def exp12a_eval_dir(seed: str | int, epoch: str | int) -> Path:
    return EXP12_LOCAL_RUNS_DIR / "exp12a_decode_projection" / f"seed_{seed}" / f"eval_epoch_{int(epoch):02d}"


def exp12b_run_dir(run_name: str, seed: str | int, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP12_LOCAL_RUNS_DIR / run_name / f"{prefix}_{seed}" / "run"


def exp12b_checkpoint_dir(run_name: str, seed: str | int, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP12_LOCAL_RUNS_DIR / run_name / f"{prefix}_{seed}" / "checkpoints"
