"""Exp10 QD-PR2 module ablation utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP10_NAME = "exp10_qdpr2_module_ablation"
EXP10_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP10_NAME
EXP10_TABLES_DIR = EXP10_OUTPUT_DIR / "tables"
EXP10_REPORTS_DIR = EXP10_OUTPUT_DIR / "reports"
EXP10_RUNS_DIR = EXP10_OUTPUT_DIR / "runs"
EXP10_LOGS_DIR = EXP10_OUTPUT_DIR / "logs"
EXP10_CONFIG_SNAPSHOT_DIR = EXP10_OUTPUT_DIR / "configs"
EXP10_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP10_NAME
EXP10_CHECKPOINTS_DIR = EXP10_ARTIFACTS_DIR / "checkpoints"
EXP10_CONFIG_DIR = THESIS_DIR / "configs" / EXP10_NAME

ABLATION_ORDER = [
    "full_qdpr2",
    "no_pair",
    "no_anchor",
    "no_mono",
    "point_only",
    "no_point_diagnostic",
]

PRIMARY_ABLATIONS = [
    "full_qdpr2",
    "no_pair",
    "no_anchor",
    "no_mono",
    "point_only",
]

ABLATION_LAMBDAS = {
    "full_qdpr2": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.1},
    "no_pair": {"lambda_point": 1.0, "lambda_pair": 0.0, "lambda_anchor": 0.5, "lambda_mono": 0.1},
    "no_anchor": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.0, "lambda_mono": 0.1},
    "no_mono": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.0},
    "point_only": {"lambda_point": 1.0, "lambda_pair": 0.0, "lambda_anchor": 0.0, "lambda_mono": 0.0},
    "no_point_diagnostic": {"lambda_point": 0.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.1},
}


def exp10_run_id(ablation_name: str) -> str:
    return f"EXP10_{ablation_name}"


def exp10_run_dir(ablation_name: str) -> Path:
    return EXP10_RUNS_DIR / ablation_name


def exp10_checkpoint_dir(ablation_name: str) -> Path:
    return EXP10_CHECKPOINTS_DIR / ablation_name


def ensure_exp10_dirs() -> None:
    for path in [
        EXP10_OUTPUT_DIR,
        EXP10_TABLES_DIR,
        EXP10_REPORTS_DIR,
        EXP10_RUNS_DIR,
        EXP10_LOGS_DIR,
        EXP10_CONFIG_SNAPSHOT_DIR,
        EXP10_ARTIFACTS_DIR,
        EXP10_CHECKPOINTS_DIR,
        EXP10_CONFIG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for name in ABLATION_ORDER:
        (exp10_run_dir(name) / "tables").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "logs").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "predictions").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "arrays").mkdir(parents=True, exist_ok=True)
