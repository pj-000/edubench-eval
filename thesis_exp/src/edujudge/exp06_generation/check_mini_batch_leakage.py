"""Leakage checks for Exp6-3 mini-batch filtered synthetic candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ensure_mini_batch_dirs
from thesis_exp.src.edujudge.exp06_generation.common import qa_key, question_key
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    answer_hash,
    mini_filtered_path,
    mini_leakage_path,
    read_jsonl_if_exists,
    split_key_cache,
    write_mini_table,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


DETAIL_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "source_record_id",
    "source_question_key_in_dev",
    "source_question_key_in_test",
    "source_triple_key_in_dev",
    "source_triple_key_in_test",
    "synthetic_question_key_in_dev",
    "synthetic_question_key_in_test",
    "synthetic_qa_key_in_dev",
    "synthetic_qa_key_in_test",
    "duplicate_synthetic_id",
    "duplicate_synthetic_answer",
    "duplicate_with_human_test_answer",
    "source_split_not_train",
    "leakage_status",
    "notes",
]

SUMMARY_FIELDS = ["check_name", "status", "count", "notes"]


def check_rows(input_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_mini_batch_dirs()
    input_path = input_path or mini_filtered_path("filtered_synthetic_candidates.jsonl")
    rows = read_jsonl_if_exists(input_path)
    keys = split_key_cache()
    seen_ids: set[str] = set()
    seen_answers: set[str] = set()
    detail_rows: list[dict[str, Any]] = []

    for row in rows:
        synthetic_id = stringify(row.get("synthetic_id"))
        answer_key = answer_hash(row.get("answer_synthetic"))
        duplicate_id = synthetic_id in seen_ids
        duplicate_answer = answer_key in seen_answers
        seen_ids.add(synthetic_id)
        seen_answers.add(answer_key)
        synthetic_question = question_key(row.get("question"))
        synthetic_qa = qa_key(row.get("question"), row.get("answer_synthetic"))
        checks = {
            "source_question_key_in_dev": row.get("source_question_key") in keys["dev"]["source_question_key"],
            "source_question_key_in_test": row.get("source_question_key") in keys["test"]["source_question_key"],
            "source_triple_key_in_dev": row.get("source_triple_key") in keys["dev"]["source_triple_key"],
            "source_triple_key_in_test": row.get("source_triple_key") in keys["test"]["source_triple_key"],
            "synthetic_question_key_in_dev": synthetic_question in keys["dev"]["question_key"],
            "synthetic_question_key_in_test": synthetic_question in keys["test"]["question_key"],
            "synthetic_qa_key_in_dev": synthetic_qa in keys["dev"]["qa_key"],
            "synthetic_qa_key_in_test": synthetic_qa in keys["test"]["qa_key"],
            "duplicate_synthetic_id": duplicate_id,
            "duplicate_synthetic_answer": duplicate_answer,
            "duplicate_with_human_test_answer": answer_key in keys["test"]["answer_key"],
            "source_split_not_train": row.get("source_split") != "train",
        }
        blocked = any(bool(value) for value in checks.values())
        detail_rows.append(
            {
                "synthetic_id": synthetic_id,
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "source_record_id": row.get("source_record_id", ""),
                **checks,
                "leakage_status": "BLOCKED" if blocked else "PASS",
                "notes": "generated mini-batch leakage and dedup check",
            }
        )

    blocked_rows = [row for row in detail_rows if row["leakage_status"] == "BLOCKED"]
    counter: Counter[str] = Counter()
    for row in blocked_rows:
        for key, value in row.items():
            if key.endswith("_in_dev") or key.endswith("_in_test") or key.startswith("duplicate_") or key == "source_split_not_train":
                if stringify(value).lower() == "true":
                    counter[key] += 1

    summary_rows = [
        {"check_name": "total_filtered_rows", "status": "INFO", "count": len(rows), "notes": "input rows after filters"},
        {"check_name": "passed_leakage_rows", "status": "PASS" if not blocked_rows else "INFO", "count": len(rows) - len(blocked_rows), "notes": "rows without leakage/dedup hits"},
        {"check_name": "blocked_leakage_rows", "status": "FAIL" if blocked_rows else "PASS", "count": len(blocked_rows), "notes": "any blocked row prevents training use"},
    ]
    if not rows:
        summary_rows.append({"check_name": "generation_presence", "status": "DRY_RUN", "count": 0, "notes": "no filtered generated rows available"})
    for key, count in sorted(counter.items()):
        summary_rows.append({"check_name": key, "status": "FAIL", "count": count, "notes": "leakage detail hit count"})
    return summary_rows, detail_rows


def write_leakage_report(summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    blocked = sum(1 for row in detail_rows if row.get("leakage_status") == "BLOCKED")
    status = "DRY_RUN_NO_GENERATED_ROWS" if not detail_rows else ("BLOCKED" if blocked else "PASS")
    report = f"""# Exp6-3 Mini-batch Leakage Report

Status: **{status}**

- Filtered input path: `{relpath(mini_filtered_path("filtered_synthetic_candidates.jsonl"))}`
- Leakage summary path: `{relpath(mini_leakage_path("leakage_summary.csv"))}`
- Leakage details path: `{relpath(mini_leakage_path("leakage_details.csv"))}`
- Filtered rows checked: **{len(detail_rows)}**
- Blocked rows: **{blocked}**

Checks:

- source question/triple keys must not occur in dev/test.
- synthetic question and question+answer keys must not occur in dev/test.
- synthetic answers must not duplicate human test answers.
- synthetic ids and answers must be unique within the batch.
- source split must be train.

Any dev/test leakage blocks the row and blocks Exp6 training use.
"""
    write_text(mini_leakage_path("leakage_report.md"), report)


def run_leakage(input_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows, detail_rows = check_rows(input_path)
    write_mini_table("leakage_summary.csv", summary_rows, SUMMARY_FIELDS)
    # write_mini_table targets the mini tables directory; leakage files are required
    # under the leakage subdirectory, so write them explicitly as well.
    from thesis_exp.src.edujudge.utils.io import write_csv

    write_csv(mini_leakage_path("leakage_summary.csv"), summary_rows, SUMMARY_FIELDS)
    write_csv(mini_leakage_path("leakage_details.csv"), detail_rows, DETAIL_FIELDS)
    write_leakage_report(summary_rows, detail_rows)
    return summary_rows, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args = parser.parse_args()
    summary_rows, detail_rows = run_leakage(args.input_jsonl)
    print(f"Wrote leakage summary rows={len(summary_rows)} detail rows={len(detail_rows)}")


if __name__ == "__main__":
    main()
