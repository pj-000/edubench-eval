"""Profile schema fields for Exp6 synthetic candidate sources."""

from __future__ import annotations

import json
from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_SAMPLES_DIR, EXP06_TABLES_DIR, SCHEMA_PROFILE_FIELDS, ensure_exp06_dirs
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
    fields_like,
    file_type,
    iter_candidate_paths,
    key_profile,
    likely_role_for_path,
    record_count,
    sample_records,
    source_rel,
    write_rows,
)
from thesis_exp.src.edujudge.utils.io import write_json


def profile_row(path: Any) -> dict[str, Any]:
    keys, nested, error = key_profile(path) if path.exists() and not path.is_dir() else ([], [], "")
    all_keys = keys + nested
    samples = sample_records(path, limit=5)
    return {
        "source_path": source_rel(path),
        "exists": path.exists(),
        "file_type": file_type(path),
        "num_records": record_count(path),
        "likely_role": likely_role_for_path(path),
        "all_keys": keys,
        "nested_keys": nested,
        "question_like_fields": fields_like(all_keys, QUESTION_PATTERNS),
        "answer_like_fields": fields_like(all_keys, ANSWER_PATTERNS),
        "metric_like_fields": fields_like(all_keys, METRIC_PATTERNS),
        "label_score_fields": fields_like(all_keys, LABEL_PATTERNS + SCORE_PATTERNS),
        "rubric_fields": fields_like(all_keys, RUBRIC_PATTERNS),
        "reason_rationale_fields": fields_like(all_keys, REASON_PATTERNS),
        "source_id_fields": fields_like(all_keys, SOURCE_ID_PATTERNS),
        "generation_method_fields": fields_like(all_keys, GENERATION_PATTERNS),
        "language_fields": fields_like(all_keys, LANGUAGE_PATTERNS),
        "error_type_fields": fields_like(all_keys, ERROR_TYPE_PATTERNS),
        "sample_rows": json.dumps(samples, ensure_ascii=False),
        "profile_error": error,
    }


def build_schema_profile() -> list[dict[str, Any]]:
    return [profile_row(path) for path in iter_candidate_paths()]


def main() -> None:
    ensure_exp06_dirs()
    rows = build_schema_profile()
    write_rows(EXP06_TABLES_DIR / "synthetic_schema_profile.csv", rows, SCHEMA_PROFILE_FIELDS)
    sample_payload = {row["source_path"]: json.loads(row["sample_rows"] or "[]") for row in rows}
    write_json(EXP06_SAMPLES_DIR / "synthetic_schema_sample_rows.json", sample_payload)
    print(f"Wrote {len(rows)} rows to {EXP06_TABLES_DIR / 'synthetic_schema_profile.csv'}")


if __name__ == "__main__":
    main()
