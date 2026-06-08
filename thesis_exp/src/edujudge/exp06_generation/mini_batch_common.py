"""Shared helpers for Exp6-3 mini-batch generation audit.

The mini-batch pipeline is dry-run by default. Helpers here only read train/dev/test
splits and Exp6 generation-plan artifacts, then write audit artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.src.edujudge.exp06_generation import (
    ERROR_TYPES,
    EXP06_GENERATION_OUTPUT_DIR,
    EXP06_GENERATION_PROMPTS_DIR,
    EXP06_GENERATION_TABLES_DIR,
    EXP06_MINI_BATCH_FILTERED_DIR,
    EXP06_MINI_BATCH_GENERATED_DIR,
    EXP06_MINI_BATCH_LEAKAGE_DIR,
    EXP06_MINI_BATCH_OUTPUT_DIR,
    EXP06_MINI_BATCH_PROMPTS_DIR,
    EXP06_MINI_BATCH_REPORTS_DIR,
    EXP06_MINI_BATCH_SAMPLES_DIR,
    EXP06_MINI_BATCH_SPOTCHECK_DIR,
    EXP06_MINI_BATCH_TABLES_DIR,
    MINI_BATCH_PROMPT_VERSION,
    ensure_generation_dirs,
    ensure_mini_batch_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.common import (
    load_split,
    planned_error_types,
    qa_key,
    question_key,
    source_record_id,
    split_key_sets,
    synthetic_id_from_parts,
)
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_jsonl, write_text
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


PRIORITY_METRICS = [
    "Reasoning Process Rigor",
    "Scenario Element Integration",
    "Personalization, Adaptation & Learning Support",
    "Higher-Order Thinking & Skill Development",
    "Error Identification & Correction Precision",
    "Instruction Following & Task Completion",
    "Basic Factual Accuracy",
    "Domain Knowledge Accuracy",
    "Role & Tone Consistency",
    "Content Relevance & Scope Control",
    "Clarity, Simplicity & Inspiration",
    "Motivation, Guidance & Positive Feedback",
]

METRIC_ERROR_SEQUENCE = {
    "Reasoning Process Rigor": ["reasoning_gap", "overconfident_wrong"],
    "Scenario Element Integration": ["scenario_mismatch", "instruction_violation"],
    "Personalization, Adaptation & Learning Support": ["scenario_mismatch", "rubric_violation"],
    "Higher-Order Thinking & Skill Development": ["reasoning_gap", "superficial_fluency"],
    "Error Identification & Correction Precision": ["factual_error", "reasoning_gap"],
    "Instruction Following & Task Completion": ["instruction_violation", "rubric_violation"],
    "Basic Factual Accuracy": ["factual_error", "overconfident_wrong"],
    "Domain Knowledge Accuracy": ["factual_error", "rubric_violation"],
    "Role & Tone Consistency": ["scenario_mismatch", "superficial_fluency"],
    "Content Relevance & Scope Control": ["scenario_mismatch", "rubric_violation"],
    "Clarity, Simplicity & Inspiration": ["superficial_fluency", "rubric_violation"],
    "Motivation, Guidance & Positive Feedback": ["scenario_mismatch", "superficial_fluency"],
}

ARTIFACT_PHRASES = [
    "intentionally flawed",
    "intentionally wrong",
    "low-score answer",
    "low score answer",
    "synthetic",
    "generated for",
    "故意错误",
    "低分回答",
    "合成",
]


def mini_table_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_TABLES_DIR / name


def mini_report_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_REPORTS_DIR / name


def mini_prompt_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_PROMPTS_DIR / name


def mini_generated_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_GENERATED_DIR / name


def mini_filtered_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_FILTERED_DIR / name


def mini_leakage_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_LEAKAGE_DIR / name


def mini_spotcheck_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_SPOTCHECK_DIR / name


def mini_sample_path(name: str) -> Path:
    ensure_mini_batch_dirs()
    return EXP06_MINI_BATCH_SAMPLES_DIR / name


def generation_table_path(name: str) -> Path:
    ensure_generation_dirs()
    return EXP06_GENERATION_TABLES_DIR / name


def write_mini_table(name: str, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    write_csv(mini_table_path(name), rows, fieldnames)


def read_mini_table(name: str) -> list[dict[str, str]]:
    path = mini_table_path(name)
    if not path.exists():
        return []
    return read_csv(path)


def write_mini_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def read_generation_table(name: str) -> list[dict[str, str]]:
    path = generation_table_path(name)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_train_by_source_id() -> dict[str, dict[str, Any]]:
    return {source_record_id(row): row for row in load_split("train")}


def split_key_cache() -> dict[str, dict[str, set[str]]]:
    return split_key_sets()


def parse_bool(value: Any) -> bool:
    return stringify(value).lower() in {"true", "1", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(stringify(value)))
    except (TypeError, ValueError):
        return default


def has_complete_source_fields(row: dict[str, Any]) -> bool:
    return all(
        stringify(row.get(field)).strip()
        for field in ["question", "answer", "rubric", "metric_canonical", "language", "question_key", "triple_key"]
    )


def source_has_dev_test_overlap(row: dict[str, Any], keys: dict[str, dict[str, set[str]]]) -> bool:
    return (
        stringify(row.get("question_key")) in keys["dev"]["source_question_key"]
        or stringify(row.get("question_key")) in keys["test"]["source_question_key"]
        or stringify(row.get("triple_key")) in keys["dev"]["source_triple_key"]
        or stringify(row.get("triple_key")) in keys["test"]["source_triple_key"]
        or question_key(row.get("question")) in keys["dev"]["question_key"]
        or question_key(row.get("question")) in keys["test"]["question_key"]
    )


def make_synthetic_id(plan_id: str, source_id: str, target_label: Any, error_type: str) -> str:
    return synthetic_id_from_parts(plan_id, source_id, target_label, error_type, MINI_BATCH_PROMPT_VERSION)


def metric_error_type(metric: str, slot: int) -> str:
    planned = set(planned_error_types(metric))
    preferred = METRIC_ERROR_SEQUENCE.get(metric, planned_error_types(metric))
    aligned = [err for err in preferred if err in planned and err in ERROR_TYPES]
    if not aligned:
        aligned = [err for err in planned_error_types(metric) if err in ERROR_TYPES]
    if not aligned:
        aligned = ["rubric_violation"]
    return aligned[slot % len(aligned)]


def source_excerpt(row: dict[str, Any], field: str, max_len: int = 700) -> str:
    return truncate_text(row.get(field, ""), max_len)


def answer_hash(answer: Any) -> str:
    return sha1_text(normalize_text(answer))


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def target_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(stringify(row.get("target_label_5")) for row in rows)
    langs = Counter(stringify(row.get("language")) for row in rows)
    metrics = Counter(stringify(row.get("metric_canonical")) for row in rows)
    errors = Counter(stringify(row.get("error_type")) for row in rows)
    return {
        "labels": dict(labels),
        "languages": dict(langs),
        "metrics": dict(metrics),
        "error_types": dict(errors),
    }


def path_list_for_report(paths: Iterable[Path]) -> str:
    return "\n".join(f"- `{relpath(path)}`" for path in paths)


def write_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def generated_rows_available() -> bool:
    raw_path = mini_generated_path("raw_generations.jsonl")
    return raw_path.exists() and count_jsonl_lines(raw_path) > 0


def report_line(name: str, value: Any) -> str:
    return f"- {name}: **{value}**"

