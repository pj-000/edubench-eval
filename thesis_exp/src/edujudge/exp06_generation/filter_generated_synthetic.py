"""Filtering scaffold for generated synthetic Exp6 data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ensure_generation_dirs
from thesis_exp.src.edujudge.exp06_generation.common import qa_key, write_table
from thesis_exp.src.edujudge.utils.text_norm import detect_language_from_text, normalize_text, stringify


RULE_FIELDS = ["rule_id", "description", "required", "failure_reason"]
FILTER_FIELDS = ["synthetic_id", "filter_status", "filter_reasons"]


FILTERING_RULES = [
    ("valid_json", "Generated row is valid JSON.", True, "invalid_json"),
    ("non_empty_answer", "answer_synthetic is non-empty.", True, "empty_answer"),
    ("length_within_range", "answer_synthetic length is between 20 and 4000 characters.", True, "length_out_of_range"),
    ("language_match", "answer_synthetic language matches row language.", True, "language_mismatch"),
    ("no_intentional_wrong_phrase", "Answer does not reveal that it is intentionally flawed or synthetic.", True, "contains_generation_meta"),
    ("no_copied_original_answer", "answer_synthetic is not a direct copy of original/source answer.", True, "copied_original_answer"),
    ("valid_target_label", "target_label_5 is in 1..5.", True, "invalid_target_label"),
    ("metric_rubric_exists", "metric_canonical and rubric_text are non-empty.", True, "missing_metric_or_rubric"),
    ("source_split_train", "source_split is train.", True, "not_train_source"),
    ("not_duplicate", "synthetic answer is unique after normalization.", True, "duplicate_synthetic_answer"),
    ("judge_consistency_placeholder", "Optional judge/human consistency check is available but not run in scaffold.", False, "judge_consistency_not_checked"),
]


META_PATTERNS = [
    "intentionally wrong",
    "intentionally flawed",
    "synthetic",
    "low-score",
    "低分",
    "故意错误",
    "有缺陷",
    "实验",
]


def rule_rows() -> list[dict[str, Any]]:
    return [
        {"rule_id": rule_id, "description": desc, "required": required, "failure_reason": reason}
        for rule_id, desc, required, reason in FILTERING_RULES
    ]


def filter_row(row: dict[str, Any], seen_answers: set[str]) -> dict[str, Any]:
    answer = stringify(row.get("answer_synthetic"))
    reasons = []
    if not answer.strip():
        reasons.append("empty_answer")
    if not (20 <= len(answer) <= 4000):
        reasons.append("length_out_of_range")
    expected_lang = stringify(row.get("language"))
    if expected_lang in {"en", "zh"} and detect_language_from_text(answer) != expected_lang:
        reasons.append("language_mismatch")
    folded = normalize_text(answer)
    if any(pattern in folded for pattern in META_PATTERNS):
        reasons.append("contains_generation_meta")
    original = normalize_text(row.get("answer") or row.get("source_answer") or "")
    if original and normalize_text(answer) == original:
        reasons.append("copied_original_answer")
    if stringify(row.get("target_label_5")) not in {"1", "2", "3", "4", "5"}:
        reasons.append("invalid_target_label")
    if not row.get("metric_canonical") or not row.get("rubric_text"):
        reasons.append("missing_metric_or_rubric")
    if row.get("source_split") != "train":
        reasons.append("not_train_source")
    answer_key = normalize_text(answer)
    if answer_key in seen_answers:
        reasons.append("duplicate_synthetic_answer")
    seen_answers.add(answer_key)
    return {
        "synthetic_id": row.get("synthetic_id", ""),
        "filter_status": "pass" if not reasons else "fail",
        "filter_reasons": reasons,
    }


def filter_file(input_path: Path) -> list[dict[str, Any]]:
    out = []
    seen_answers: set[str] = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            out.append(filter_row(json.loads(line), seen_answers))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args, _ = parser.parse_known_args()
    ensure_generation_dirs()
    write_table("filtering_rules.csv", rule_rows(), RULE_FIELDS)
    if args.input_jsonl:
        rows = filter_file(args.input_jsonl)
    else:
        rows = [{"synthetic_id": "", "filter_status": "DRY_RUN", "filter_reasons": ["no generated JSONL supplied"]}]
    write_table("generated_synthetic_filter_results.csv", rows, FILTER_FIELDS)
    print("Wrote filtering_rules.csv and generated_synthetic_filter_results.csv")


if __name__ == "__main__":
    main()
