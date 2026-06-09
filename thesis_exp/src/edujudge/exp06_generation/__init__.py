"""Exp6 train-only synthetic low-score generation planning utilities."""

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
MINI_BATCH_PROMPT_VERSION = "exp06_mini_batch_v1"
BATCH96_PROMPT_VERSION = "exp06_batch96_hardened_v1"

EXP06_MINI_BATCH_NAME = "mini_batch_generation"
EXP06_MINI_BATCH_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_synthetic_low_score" / EXP06_MINI_BATCH_NAME
EXP06_MINI_BATCH_TABLES_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "tables"
EXP06_MINI_BATCH_SAMPLES_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "samples"
EXP06_MINI_BATCH_PROMPTS_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "prompts"
EXP06_MINI_BATCH_GENERATED_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "generated"
EXP06_MINI_BATCH_FILTERED_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "filtered"
EXP06_MINI_BATCH_LEAKAGE_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "leakage"
EXP06_MINI_BATCH_SPOTCHECK_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "spotcheck"
EXP06_MINI_BATCH_REPORTS_DIR = EXP06_MINI_BATCH_OUTPUT_DIR / "reports"

MINI_BATCH_TARGET_LABEL_COUNTS = {1: 10, 2: 10, 3: 4}
MINI_BATCH_TOTAL_TARGET = sum(MINI_BATCH_TARGET_LABEL_COUNTS.values())

EXP06_BATCH96_NAME = "batch96_generation"
EXP06_BATCH96_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_synthetic_low_score" / EXP06_BATCH96_NAME
EXP06_BATCH96_CURATED_DIR = EXP06_BATCH96_OUTPUT_DIR / "curated"
EXP06_BATCH96_TABLES_DIR = EXP06_BATCH96_OUTPUT_DIR / "tables"
EXP06_BATCH96_PROMPT_TEMPLATES_DIR = EXP06_BATCH96_OUTPUT_DIR / "prompt_templates"
EXP06_BATCH96_PROMPTS_DIR = EXP06_BATCH96_OUTPUT_DIR / "prompts"
EXP06_BATCH96_GENERATED_DIR = EXP06_BATCH96_OUTPUT_DIR / "generated"
EXP06_BATCH96_FILTERED_DIR = EXP06_BATCH96_OUTPUT_DIR / "filtered"
EXP06_BATCH96_LEAKAGE_DIR = EXP06_BATCH96_OUTPUT_DIR / "leakage"
EXP06_BATCH96_SPOTCHECK_DIR = EXP06_BATCH96_OUTPUT_DIR / "spotcheck"
EXP06_BATCH96_REPORTS_DIR = EXP06_BATCH96_OUTPUT_DIR / "reports"

BATCH96_TARGET_LABEL_COUNTS = {1: 40, 2: 40, 3: 16}
BATCH96_TOTAL_TARGET = sum(BATCH96_TARGET_LABEL_COUNTS.values())
BATCH96_LANGUAGE_COUNTS = {"en": 48, "zh": 48}

GENERATION_SPLIT_MODES = {
    "paper_like_strict": {
        "split_dir": THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42",
        "require_question_disjoint": True,
        "require_triple_disjoint": True,
        "allowed_for_training": False,
        "risk_level": "BLOCKED",
        "purpose": "demonstrate strict source unavailability under paper-like split",
    },
    "paper_like_triple_pilot": {
        "split_dir": THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42",
        "require_question_disjoint": False,
        "require_triple_disjoint": True,
        "allowed_for_training": False,
        "risk_level": "HIGH",
        "purpose": "high-risk prompt/debug pilot only",
    },
    "question_disjoint_formal": {
        "split_dir": THESIS_DIR / "data" / "splits" / "question_seed42",
        "require_question_disjoint": True,
        "require_triple_disjoint": True,
        "allowed_for_training": True,
        "risk_level": "LOW",
        "purpose": "formal leakage-safe synthetic generation and augmentation",
    },
}

DEFAULT_GENERATION_SPLIT_MODE = "question_disjoint_formal"

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


def ensure_mini_batch_dirs() -> None:
    for path in [
        EXP06_MINI_BATCH_OUTPUT_DIR,
        EXP06_MINI_BATCH_TABLES_DIR,
        EXP06_MINI_BATCH_SAMPLES_DIR,
        EXP06_MINI_BATCH_PROMPTS_DIR,
        EXP06_MINI_BATCH_GENERATED_DIR,
        EXP06_MINI_BATCH_FILTERED_DIR,
        EXP06_MINI_BATCH_LEAKAGE_DIR,
        EXP06_MINI_BATCH_SPOTCHECK_DIR,
        EXP06_MINI_BATCH_REPORTS_DIR,
        EXP06_GENERATION_PROMPTS_DIR,
        EXP06_GENERATION_SRC_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_split_mode_dirs(mode: str) -> None:
    base = EXP06_MINI_BATCH_OUTPUT_DIR / mode
    for name in ["tables", "samples", "prompts", "generated", "filtered", "leakage", "spotcheck", "reports"]:
        (base / name).mkdir(parents=True, exist_ok=True)


def ensure_batch96_dirs() -> None:
    for path in [
        EXP06_BATCH96_OUTPUT_DIR,
        EXP06_BATCH96_CURATED_DIR,
        EXP06_BATCH96_TABLES_DIR,
        EXP06_BATCH96_PROMPT_TEMPLATES_DIR,
        EXP06_BATCH96_PROMPTS_DIR,
        EXP06_BATCH96_GENERATED_DIR,
        EXP06_BATCH96_FILTERED_DIR,
        EXP06_BATCH96_LEAKAGE_DIR,
        EXP06_BATCH96_SPOTCHECK_DIR,
        EXP06_BATCH96_REPORTS_DIR,
        EXP06_GENERATION_SRC_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
