"""Shared helpers for Exp6-2 generation planning.

This package intentionally contains no API calls. It prepares train-only source
sampling, prompt templates, schema docs, and validation/filtering scaffolds.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.src.edujudge.exp06_generation import (
    EXP06_GENERATION_OUTPUT_DIR,
    EXP06_GENERATION_TABLES_DIR,
    SPLIT_DIR,
    ensure_generation_dirs,
)
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_csv, write_json, write_text
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


METRIC_ERROR_TYPE_MAP = {
    "Instruction Following & Task Completion": ["instruction_violation", "rubric_violation", "superficial_fluency"],
    "Role & Tone Consistency": ["scenario_mismatch", "superficial_fluency", "instruction_violation"],
    "Content Relevance & Scope Control": ["scenario_mismatch", "rubric_violation", "superficial_fluency"],
    "Scenario Element Integration": ["scenario_mismatch", "instruction_violation", "rubric_violation"],
    "Basic Factual Accuracy": ["factual_error", "overconfident_wrong", "reasoning_gap"],
    "Domain Knowledge Accuracy": ["factual_error", "overconfident_wrong", "rubric_violation"],
    "Reasoning Process Rigor": ["reasoning_gap", "overconfident_wrong", "factual_error"],
    "Error Identification & Correction Precision": ["factual_error", "reasoning_gap", "overconfident_wrong"],
    "Clarity, Simplicity & Inspiration": ["superficial_fluency", "rubric_violation", "scenario_mismatch"],
    "Motivation, Guidance & Positive Feedback": ["scenario_mismatch", "rubric_violation", "superficial_fluency"],
    "Personalization, Adaptation & Learning Support": ["scenario_mismatch", "rubric_violation", "instruction_violation"],
    "Higher-Order Thinking & Skill Development": ["reasoning_gap", "superficial_fluency", "rubric_violation"],
}


def table_path(name: str) -> Path:
    ensure_generation_dirs()
    return EXP06_GENERATION_TABLES_DIR / name


def output_path(name: str) -> Path:
    ensure_generation_dirs()
    return EXP06_GENERATION_OUTPUT_DIR / name


def load_split(split: str) -> list[dict[str, Any]]:
    return read_jsonl(SPLIT_DIR / f"{split}.jsonl")


def write_table(name: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    write_csv(table_path(name), rows, fieldnames)


def read_table(name: str) -> list[dict[str, str]]:
    path = table_path(name)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_record_id(row: dict[str, Any]) -> str:
    return stringify(row.get("record_id") or sha1_text(row.get("source_file"), row.get("source_row_index"), row.get("triple_key"))[:16])


def planned_error_types(metric: str) -> list[str]:
    return METRIC_ERROR_TYPE_MAP.get(metric, ["rubric_violation", "superficial_fluency", "instruction_violation"])


def planned_target_labels() -> list[int]:
    return [1, 2, 3]


def qa_key(question: Any, answer: Any) -> str:
    return sha1_text(normalize_text(question), normalize_text(answer))


def question_key(question: Any) -> str:
    return sha1_text(normalize_text(question))


def synthetic_id_from_parts(*parts: Any) -> str:
    return "syn_" + sha1_text(*parts)[:16]


def split_key_sets() -> dict[str, dict[str, set[str]]]:
    out = {}
    for split in ["train", "dev", "test"]:
        rows = load_split(split)
        out[split] = {
            "source_question_key": {stringify(row.get("question_key")) for row in rows if row.get("question_key")},
            "source_triple_key": {stringify(row.get("triple_key")) for row in rows if row.get("triple_key")},
            "question_key": {question_key(row.get("question")) for row in rows},
            "qa_key": {qa_key(row.get("question"), row.get("answer")) for row in rows},
            "answer_key": {sha1_text(normalize_text(row.get("answer"))) for row in rows},
        }
    return out


def read_context_report(path: Path) -> str:
    if path.exists():
        return truncate_text(path.read_text(encoding="utf-8"), 2500)
    return ""


def markdown_json_schema() -> str:
    fields = [
        ("synthetic_id", "string; stable generated id"),
        ("source_record_id", "string; train split source record id"),
        ("source_question_key", "string; source train question key"),
        ("source_triple_key", "string; source train triple key"),
        ("source_split", "literal train"),
        ("question", "string; copied from train source question"),
        ("answer_synthetic", "string; generated plausible flawed answer"),
        ("metric_canonical", "string; one of the 12 EduBench metrics"),
        ("rubric_text", "string; source rubric"),
        ("language", "en or zh"),
        ("scenario_canonical", "string"),
        ("subject_canonical", "string"),
        ("education_level_canonical", "string"),
        ("target_label_5", "integer 1-5; synthetic design pseudo-label"),
        ("label_source", "literal synthetic_design"),
        ("error_type", "one of the Exp6-2 error types"),
        ("generation_model", "string; planned model, e.g. deepseek-v4-pro"),
        ("generation_prompt_version", "string"),
        ("generation_timestamp", "ISO timestamp from generation runner"),
        ("generation_status", "dry_run/planned/generated/failed"),
        ("raw_generation", "raw model JSON/text if generated"),
        ("filter_status", "pending/pass/fail"),
        ("filter_reasons", "array of strings"),
    ]
    lines = ["# Exp6 Synthetic Low-Score Output Schema", "", "| field | specification |", "| --- | --- |"]
    lines.extend(f"| `{name}` | {spec} |" for name, spec in fields)
    lines.extend(
        [
            "",
            "Synthetic labels are pseudo labels from experimental design. They must not be described as human labels.",
            "Generated rows may only be used for train-side augmentation after leakage and filtering checks pass.",
        ]
    )
    return "\n".join(lines)
