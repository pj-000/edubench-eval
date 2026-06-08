"""Compare processed Excel merged/en/zh sources and deduplicate candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from thesis_exp.src.edujudge.exp06.processed_excel_common import (
    PROCESSED_EXCEL_SOURCES,
    all_candidate_rows,
    canonical_record_key,
    count_duplicates,
    load_source_records,
    records_by_source,
    rows_by_source,
    source_paths,
    write_processed_csv,
)
from thesis_exp.src.edujudge.utils.io import relpath


COMPARISON_FIELDS = [
    "source_a",
    "source_b",
    "records_a",
    "records_b",
    "score_rows_a",
    "score_rows_b",
    "record_overlap",
    "question_overlap",
    "triple_overlap",
    "qa_overlap",
    "candidate_overlap",
    "merge_equals_en_plus_zh",
    "recommended_non_duplicate_use",
    "notes",
]

DEDUP_FIELDS = [
    "scope",
    "source_file",
    "total_candidate_rows",
    "unique_candidate_keys",
    "duplicate_candidate_groups",
    "duplicate_candidate_rows",
    "unique_qa_keys",
    "duplicate_qa_groups",
    "duplicate_qa_rows",
    "unique_triple_keys",
    "duplicate_triple_groups",
    "duplicate_triple_rows",
    "dedup_recommendation",
]


def source_sets(records: dict[str, list[dict[str, Any]]], rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, set[str]]]:
    out = {}
    for source in PROCESSED_EXCEL_SOURCES:
        source_records = records.get(source, [])
        source_rows = rows.get(source, [])
        out[source] = {
            "record": {canonical_record_key(record) for record in source_records},
            "question": {row["question_key"] for row in source_rows},
            "qa": {row["qa_key"] for row in source_rows},
            "triple": {row["triple_key"] for row in source_rows},
            "candidate": {row["candidate_key"] for row in source_rows},
        }
    return out


def merge_equals_split(records: dict[str, list[dict[str, Any]]]) -> bool:
    merged = {canonical_record_key(record) for record in records.get(PROCESSED_EXCEL_SOURCES[0], [])}
    split_union = {
        canonical_record_key(record)
        for source in PROCESSED_EXCEL_SOURCES[1:]
        for record in records.get(source, [])
    }
    return bool(merged) and merged == split_union and len(records.get(PROCESSED_EXCEL_SOURCES[0], [])) == sum(
        len(records.get(source, [])) for source in PROCESSED_EXCEL_SOURCES[1:]
    )


def build_source_comparison() -> list[dict[str, Any]]:
    records = records_by_source()
    rows = rows_by_source(all_candidate_rows())
    sets = source_sets(records, rows)
    merge_equal = merge_equals_split(records)
    out: list[dict[str, Any]] = []
    for left, right in combinations(PROCESSED_EXCEL_SOURCES, 2):
        out.append(
            {
                "source_a": left,
                "source_b": right,
                "records_a": len(records.get(left, [])),
                "records_b": len(records.get(right, [])),
                "score_rows_a": len(rows.get(left, [])),
                "score_rows_b": len(rows.get(right, [])),
                "record_overlap": len(sets[left]["record"] & sets[right]["record"]),
                "question_overlap": len(sets[left]["question"] & sets[right]["question"]),
                "triple_overlap": len(sets[left]["triple"] & sets[right]["triple"]),
                "qa_overlap": len(sets[left]["qa"] & sets[right]["qa"]),
                "candidate_overlap": len(sets[left]["candidate"] & sets[right]["candidate"]),
                "merge_equals_en_plus_zh": merge_equal,
                "recommended_non_duplicate_use": "use merged only OR en+zh only, never both" if merge_equal else "deduplicate by candidate_key before use",
                "notes": "merged source equals union of en and zh split files" if merge_equal else "merged source differs from split union",
            }
        )
    return out


def dedup_row(scope: str, source_file: str, rows: list[dict[str, Any]], recommendation: str) -> dict[str, Any]:
    candidate_groups, candidate_dupes = count_duplicates(rows, "candidate_key")
    qa_groups, qa_dupes = count_duplicates(rows, "qa_key")
    triple_groups, triple_dupes = count_duplicates(rows, "triple_key")
    return {
        "scope": scope,
        "source_file": source_file,
        "total_candidate_rows": len(rows),
        "unique_candidate_keys": len({row["candidate_key"] for row in rows}),
        "duplicate_candidate_groups": candidate_groups,
        "duplicate_candidate_rows": candidate_dupes,
        "unique_qa_keys": len({row["qa_key"] for row in rows}),
        "duplicate_qa_groups": qa_groups,
        "duplicate_qa_rows": qa_dupes,
        "unique_triple_keys": len({row["triple_key"] for row in rows}),
        "duplicate_triple_groups": triple_groups,
        "duplicate_triple_rows": triple_dupes,
        "dedup_recommendation": recommendation,
    }


def build_dedup_report() -> list[dict[str, Any]]:
    rows = all_candidate_rows()
    by_source = rows_by_source(rows)
    merge_equal = merge_equals_split(records_by_source())
    report = []
    for source in PROCESSED_EXCEL_SOURCES:
        report.append(dedup_row("within_source", source, by_source.get(source, []), "deduplicate repeated candidate_key rows within source"))
    recommendation = "if used, choose merged only as canonical; en/zh are duplicate split views" if merge_equal else "deduplicate across all three files by candidate_key"
    report.append(dedup_row("all_sources", "ALL", rows, recommendation))
    return report


def main() -> None:
    comparison = build_source_comparison()
    dedup = build_dedup_report()
    write_processed_csv("processed_excel_source_comparison.csv", comparison, COMPARISON_FIELDS)
    write_processed_csv("processed_excel_dedup_report.csv", dedup, DEDUP_FIELDS)
    print(f"Wrote {len(comparison)} source comparison rows and {len(dedup)} dedup rows")


if __name__ == "__main__":
    main()
