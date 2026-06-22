"""Exp13 risk-boundary MAP-OC paths and defaults."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import REPO_ROOT


EXP13_NAME = "exp13_risk_boundary_map_oc"
EXP13_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / EXP13_NAME
EXP13_TABLES_DIR = EXP13_OUTPUT_DIR / "tables"
EXP13_REPORTS_DIR = EXP13_OUTPUT_DIR / "reports"
EXP13_LOCAL_RUNS_DIR = REPO_ROOT / "thesis_exp" / "runs" / EXP13_NAME
EXP13_CONFIG_DIR = REPO_ROOT / "thesis_exp" / "configs" / EXP13_NAME

DEFAULT_SELECTION_RULE = "mae_guard_p_gt_3_low_mean"
DEFAULT_SELECTION_DELTA = 0.005
DEFAULT_SOFT_RISK_GAMMA = 4.0
DEFAULT_GPU_LIST = "6 7"
DEFAULT_SEEDS = "42"
DEFAULT_EPOCHS = "3"
DEFAULT_MODE = "scout"
DEFAULT_EVAL_TEST = "0"

EXP13_RUNS = [
    "point_pair_proj_l2h_lam0p10",
    "point_pair_proj_l2h_lam0p20",
    "point_pair_proj_l2h_lam0p40",
    "point_pair_proj_l2h_label2_w1p5_lam0p20",
    "point_pair_proj_l2h_lam0p20_no_mono",
    "score_proj_l2h_lam0p20",
    "map_oc_full_l2h_lam0p20",
    "score_proj_t3_brier_lam0p05",
]

CONFIG_BY_RUN = {
    "point_pair_proj_l2h_lam0p10": EXP13_CONFIG_DIR / "exp13_point_pair_proj_l2h_lam0p10.yaml",
    "point_pair_proj_l2h_lam0p20": EXP13_CONFIG_DIR / "exp13_point_pair_proj_l2h_lam0p20.yaml",
    "point_pair_proj_l2h_lam0p40": EXP13_CONFIG_DIR / "exp13_point_pair_proj_l2h_lam0p40.yaml",
    "point_pair_proj_l2h_label2_w1p5_lam0p20": EXP13_CONFIG_DIR
    / "exp13_point_pair_proj_l2h_label2_w1p5_lam0p20.yaml",
    "point_pair_proj_l2h_lam0p20_no_mono": EXP13_CONFIG_DIR / "exp13_point_pair_proj_l2h_lam0p20_no_mono.yaml",
    "score_proj_l2h_lam0p20": EXP13_CONFIG_DIR / "exp13_score_proj_l2h_lam0p20.yaml",
    "map_oc_full_l2h_lam0p20": EXP13_CONFIG_DIR / "exp13_map_oc_full_l2h_lam0p20.yaml",
    "score_proj_t3_brier_lam0p05": EXP13_CONFIG_DIR / "exp13_score_proj_t3_brier_lam0p05.yaml",
}


def ensure_exp13_dirs() -> None:
    for path in [EXP13_OUTPUT_DIR, EXP13_TABLES_DIR, EXP13_REPORTS_DIR, EXP13_LOCAL_RUNS_DIR, EXP13_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def exp13_run_dir(run_name: str, seed: str | int, mode: str = DEFAULT_MODE, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP13_LOCAL_RUNS_DIR / mode / run_name / f"{prefix}_{seed}" / "run"


def exp13_checkpoint_dir(run_name: str, seed: str | int, mode: str = DEFAULT_MODE, smoke: bool = False) -> Path:
    prefix = "smoke_seed" if smoke else "seed"
    return EXP13_LOCAL_RUNS_DIR / mode / run_name / f"{prefix}_{seed}" / "checkpoints"
