"""Exp16A rubric-conditioned ordinal boundary linking."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP16_NAME = "exp16_boundary_linking"
EXP16_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP16_NAME
EXP16_TABLES_DIR = EXP16_OUTPUT_DIR / "tables"
EXP16_REPORTS_DIR = EXP16_OUTPUT_DIR / "reports"
EXP16_RUNS_DIR = EXP16_OUTPUT_DIR / "runs"
EXP16_SANITY_DIR = EXP16_OUTPUT_DIR / "sanity"
EXP16_DEFAULT_DATA_DIR = THESIS_DIR / "data" / "splits" / "question_seed42"

BOUNDARY_VARIANTS = ("global", "metric_rubric", "qmr", "qmr_meta")


def ensure_exp16_dirs() -> None:
    for path in [
        EXP16_OUTPUT_DIR,
        EXP16_TABLES_DIR,
        EXP16_REPORTS_DIR,
        EXP16_RUNS_DIR,
        EXP16_SANITY_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def default_split_path(split: str) -> Path:
    return EXP16_DEFAULT_DATA_DIR / f"{split}.jsonl"
