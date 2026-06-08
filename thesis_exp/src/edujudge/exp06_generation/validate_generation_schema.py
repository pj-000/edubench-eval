"""Validate generated synthetic JSONL schema, or write schema docs in dry-run mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ensure_generation_dirs
from thesis_exp.src.edujudge.exp06_generation.common import markdown_json_schema, output_path, synthetic_id_from_parts, write_table
from thesis_exp.src.edujudge.utils.io import write_text


REQUIRED_FIELDS = [
    "synthetic_id",
    "source_record_id",
    "source_question_key",
    "source_triple_key",
    "source_split",
    "question",
    "answer_synthetic",
    "metric_canonical",
    "rubric_text",
    "language",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "target_label_5",
    "label_source",
    "error_type",
    "generation_model",
    "generation_prompt_version",
    "generation_timestamp",
    "generation_status",
    "raw_generation",
    "filter_status",
    "filter_reasons",
]

VALIDATION_FIELDS = ["row_index", "synthetic_id", "status", "missing_fields", "invalid_fields", "notes"]


def validate_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    invalid = []
    if row.get("source_split") != "train":
        invalid.append("source_split must be train")
    if row.get("label_source") != "synthetic_design":
        invalid.append("label_source must be synthetic_design")
    if row.get("target_label_5") not in {1, 2, 3, 4, 5, "1", "2", "3", "4", "5"}:
        invalid.append("target_label_5 must be 1-5")
    if row.get("language") not in {"en", "zh"}:
        invalid.append("language must be en or zh")
    if not row.get("answer_synthetic"):
        invalid.append("answer_synthetic empty")
    return {
        "row_index": row_index,
        "synthetic_id": row.get("synthetic_id", ""),
        "status": "PASS" if not missing and not invalid else "FAIL",
        "missing_fields": missing,
        "invalid_fields": invalid,
        "notes": "schema validation only; no model call performed",
    }


def validate_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            rows.append(validate_row(json.loads(line), idx))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args, _ = parser.parse_known_args()
    ensure_generation_dirs()
    write_text(output_path("synthetic_schema.md"), markdown_json_schema())
    if args.input_jsonl:
        rows = validate_file(args.input_jsonl)
    else:
        rows = [
            {
                "row_index": "",
                "synthetic_id": "",
                "status": "DRY_RUN",
                "missing_fields": [],
                "invalid_fields": [],
                "notes": "No generated JSONL supplied; wrote schema document only.",
            }
        ]
    write_table("generation_schema_validation.csv", rows, VALIDATION_FIELDS)
    print("Wrote synthetic_schema.md and generation_schema_validation.csv")


if __name__ == "__main__":
    main()
