"""Profile processed Excel candidate source schemas for Exp6-1."""

from __future__ import annotations

from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06.processed_excel_common import (
    PROCESSED_EXCEL_SOURCES,
    all_candidate_rows,
    load_source_records,
    source_paths,
    write_processed_csv,
)
from thesis_exp.src.edujudge.utils.io import flatten_keys, relpath
from thesis_exp.src.edujudge.utils.text_norm import truncate_text


SCHEMA_FIELDS = [
    "source_file",
    "exists",
    "num_records",
    "score_rows",
    "top_level_fields",
    "nested_score_fields",
    "records_with_scores",
    "score_count_distribution",
    "metric_count",
    "language_distribution",
    "has_question",
    "has_response",
    "has_model",
    "has_reason",
    "sample_record_id",
    "sample_question_preview",
    "sample_response_preview",
]


def build_schema_profile() -> list[dict[str, Any]]:
    candidate_rows = all_candidate_rows()
    rows: list[dict[str, Any]] = []
    for path in source_paths():
        records = load_source_records(path)
        source = relpath(path)
        source_candidates = [row for row in candidate_rows if row["source_file"] == source]
        top_fields = sorted({key for record in records for key in record})
        nested_score_fields = sorted(
            {
                key
                for record in records
                for item in record.get("scores", [])
                if isinstance(item, dict)
                for key in item
            }
        )
        score_counts = Counter(len(record.get("scores", []) or []) for record in records)
        languages = Counter(row.get("language", "unknown") for row in source_candidates)
        sample = records[0] if records else {}
        rows.append(
            {
                "source_file": source,
                "exists": path.exists(),
                "num_records": len(records),
                "score_rows": len(source_candidates),
                "top_level_fields": top_fields,
                "nested_score_fields": nested_score_fields,
                "records_with_scores": sum(1 for record in records if record.get("scores")),
                "score_count_distribution": dict(score_counts),
                "metric_count": len({row.get("metric_canonical") for row in source_candidates}),
                "language_distribution": dict(languages),
                "has_question": "question" in top_fields,
                "has_response": "response" in top_fields,
                "has_model": "model" in top_fields,
                "has_reason": "reason" in nested_score_fields,
                "sample_record_id": sample.get("id", ""),
                "sample_question_preview": truncate_text(sample.get("question", ""), 240),
                "sample_response_preview": truncate_text(sample.get("response", ""), 240),
            }
        )
    return rows


def main() -> None:
    rows = build_schema_profile()
    write_processed_csv("processed_excel_schema_profile.csv", rows, SCHEMA_FIELDS)
    print(f"Wrote {len(rows)} rows to processed_excel_schema_profile.csv")


if __name__ == "__main__":
    main()
