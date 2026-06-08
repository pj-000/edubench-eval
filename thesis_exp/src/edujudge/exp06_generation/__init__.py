"""Exp6-2 train-only synthetic low-score generation plan utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP06_GENERATION_NAME = "generation_plan"
EXP06_GENERATION_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_synthetic_low_score" / EXP06_GENERATION_NAME
EXP06_GENERATION_TABLES_DIR = EXP06_GENERATION_OUTPUT_DIR / "tables"
EXP06_GENERATION_PROMPTS_DIR = EXP06_GENERATION_OUTPUT_DIR / "prompt_templates"
EXP06_GENERATION_SAMPLES_DIR = EXP06_GENERATION_OUTPUT_DIR / "samples"

EXP06_GENERATION_SRC_DIR = THESIS_DIR / "src" / "edujudge" / "exp06_generation"
SPLIT_DIR = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42"

DEFAULT_GENERATION_MODEL = "deepseek-v4-pro"
PROMPT_VERSION = "exp06_low_score_v1"

ERROR_TYPES = [
    "factual_error",
    "reasoning_gap",
    "instruction_violation",
    "scenario_mismatch",
    "rubric_violation",
    "superficial_fluency",
    "overconfident_wrong",
]

TARGET_LABEL_COUNTS_PER_METRIC_LANGUAGE = {
    1: 7,
    2: 7,
    3: 2,
}


def ensure_generation_dirs() -> None:
    for path in [
        EXP06_GENERATION_OUTPUT_DIR,
        EXP06_GENERATION_TABLES_DIR,
        EXP06_GENERATION_PROMPTS_DIR,
        EXP06_GENERATION_SAMPLES_DIR,
        EXP06_GENERATION_SRC_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
