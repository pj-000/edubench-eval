"""Exp10R QD-PR2 reproducibility diagnosis utilities."""

from __future__ import annotations

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP10R_NAME = "exp10r_qdpr2_repro_diagnosis"
EXP10R_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP10R_NAME
EXP10R_TABLES_DIR = EXP10R_OUTPUT_DIR / "tables"
EXP10R_REPORTS_DIR = EXP10R_OUTPUT_DIR / "reports"


def ensure_exp10r_dirs() -> None:
    for path in [EXP10R_OUTPUT_DIR, EXP10R_TABLES_DIR, EXP10R_REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
