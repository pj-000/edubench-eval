"""Exp14 logit-margin tail-risk ordinal calibration paths and defaults."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import REPO_ROOT


EXP14_NAME = "exp14_logit_margin_tail_risk_oc"
EXP14_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / EXP14_NAME
EXP14_TABLES_DIR = EXP14_OUTPUT_DIR / "tables"
EXP14_REPORTS_DIR = EXP14_OUTPUT_DIR / "reports"
EXP14_LOCAL_RUNS_DIR = REPO_ROOT / "thesis_exp" / "runs" / EXP14_NAME
EXP14_CONFIG_DIR = REPO_ROOT / "thesis_exp" / "configs" / EXP14_NAME

DEFAULT_SELECTION_RULE = "mae_guard_low_to_high_then_p_gt_3"
DEFAULT_SELECTION_DELTA = 0.005
DEFAULT_SOFT_RISK_GAMMA = 4.0
DEFAULT_GPU_LIST = "6 7"
DEFAULT_SEEDS = "42"
DEFAULT_EPOCHS = "3"
DEFAULT_MODE = "scout"
DEFAULT_EVAL_TEST = "0"

EXP14_RUNS = [
    "score_logit_margin_lam0p01_alllow",
    "score_logit_margin_lam0p02_alllow",
    "score_logit_margin_lam0p05_alllow",
    "score_tail_logit_margin_lam0p02_top0p50",
    "score_tail_logit_margin_lam0p05_top0p50",
    "score_tail_logit_margin_lam0p02_top0p25",
    "point_pair_tail_logit_margin_lam0p02_top0p50",
]

CONFIG_BY_RUN = {
    "score_logit_margin_lam0p01_alllow": EXP14_CONFIG_DIR / "exp14_score_logit_margin_lam0p01_alllow.yaml",
    "score_logit_margin_lam0p02_alllow": EXP14_CONFIG_DIR / "exp14_score_logit_margin_lam0p02_alllow.yaml",
    "score_logit_margin_lam0p05_alllow": EXP14_CONFIG_DIR / "exp14_score_logit_margin_lam0p05_alllow.yaml",
    "score_tail_logit_margin_lam0p02_top0p50": EXP14_CONFIG_DIR
    / "exp14_score_tail_logit_margin_lam0p02_top0p50.yaml",
    "score_tail_logit_margin_lam0p05_top0p50": EXP14_CONFIG_DIR
    / "exp14_score_tail_logit_margin_lam0p05_top0p50.yaml",
    "score_tail_logit_margin_lam0p02_top0p25": EXP14_CONFIG_DIR
    / "exp14_score_tail_logit_margin_lam0p02_top0p25.yaml",
    "point_pair_tail_logit_margin_lam0p02_top0p50": EXP14_CONFIG_DIR
    / "exp14_point_pair_tail_logit_margin_lam0p02_top0p50.yaml",
}


def ensure_exp14_dirs() -> None:
    for path in [EXP14_OUTPUT_DIR, EXP14_TABLES_DIR, EXP14_REPORTS_DIR, EXP14_LOCAL_RUNS_DIR, EXP14_CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
