"""Inventory existing synthetic/sampled/augmented Exp6 candidate sources."""

from __future__ import annotations

from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_TABLES_DIR, INVENTORY_FIELDS, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.common import (
    ANSWER_PATTERNS,
    ERROR_TYPE_PATTERNS,
    GENERATION_PATTERNS,
    LABEL_PATTERNS,
    LANGUAGE_PATTERNS,
    METRIC_PATTERNS,
    QUESTION_PATTERNS,
    REASON_PATTERNS,
    RUBRIC_PATTERNS,
    SCORE_PATTERNS,
    SOURCE_ID_PATTERNS,
    contains_from_keys,
    file_type,
    iter_candidate_paths,
    key_profile,
    likely_role_for_path,
    record_count,
    risk_and_usable,
    sample_records,
    source_rel,
    synthetic_marker_for_path,
    write_rows,
)


def inventory_row(path: Any) -> dict[str, Any]:
    exists = path.exists()
    keys, nested, error = key_profile(path) if exists and not path.is_dir() else ([], [], "")
    all_keys = keys + nested
    sample = sample_records(path, limit=3)
    sample_text = " ".join(str(value) for row in sample for value in row.values())
    contains = {
        "question": contains_from_keys(all_keys, QUESTION_PATTERNS) or "对话" in sample_text,
        "answer": contains_from_keys(all_keys, ANSWER_PATTERNS),
        "metric": contains_from_keys(all_keys, METRIC_PATTERNS),
        "label": contains_from_keys(all_keys, LABEL_PATTERNS),
        "score": contains_from_keys(all_keys, SCORE_PATTERNS),
        "rubric": contains_from_keys(all_keys, RUBRIC_PATTERNS),
        "reasoning": contains_from_keys(all_keys, REASON_PATTERNS),
        "error_type": contains_from_keys(all_keys, ERROR_TYPE_PATTERNS),
        "source_question_id": contains_from_keys(all_keys, SOURCE_ID_PATTERNS),
        "language": contains_from_keys(all_keys, LANGUAGE_PATTERNS),
        "synthetic_marker": synthetic_marker_for_path(path),
    }
    if contains_from_keys(all_keys, GENERATION_PATTERNS):
        contains["synthetic_marker"] = True
    role = likely_role_for_path(path)
    risk, usable, notes = risk_and_usable(path, role, contains)
    if error:
        notes = f"{notes}; profile_error={error}"
    return {
        "source_path": source_rel(path),
        "exists": exists,
        "file_type": file_type(path),
        "num_records": record_count(path),
        "likely_role": role,
        "contains_question": contains["question"],
        "contains_answer": contains["answer"],
        "contains_metric": contains["metric"],
        "contains_label": contains["label"],
        "contains_score": contains["score"],
        "contains_rubric": contains["rubric"],
        "contains_reasoning": contains["reasoning"],
        "contains_error_type": contains["error_type"],
        "contains_source_question_id": contains["source_question_id"],
        "contains_language": contains["language"],
        "contains_synthetic_marker": contains["synthetic_marker"],
        "usable_for_exp06": usable,
        "risk_level": risk,
        "notes": notes,
    }


def build_inventory() -> list[dict[str, Any]]:
    return [inventory_row(path) for path in iter_candidate_paths()]


def main() -> None:
    ensure_exp06_dirs()
    rows = build_inventory()
    write_rows(EXP06_TABLES_DIR / "synthetic_source_inventory.csv", rows, INVENTORY_FIELDS)
    print(f"Wrote {len(rows)} rows to {EXP06_TABLES_DIR / 'synthetic_source_inventory.csv'}")


if __name__ == "__main__":
    main()
