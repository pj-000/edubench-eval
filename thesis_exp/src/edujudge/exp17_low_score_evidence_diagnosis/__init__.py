"""Exp17-D0 low-score evidence diagnosis paths."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP17_NAME = "exp17_low_score_evidence_diagnosis"
EXP17_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP17_NAME
EXP17_TABLES_DIR = EXP17_OUTPUT_DIR / "tables"
EXP17_REPORTS_DIR = EXP17_OUTPUT_DIR / "reports"


def ensure_exp17_dirs() -> None:
    for path in [EXP17_OUTPUT_DIR, EXP17_TABLES_DIR, EXP17_REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
