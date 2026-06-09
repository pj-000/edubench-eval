"""Filter normalized Exp6-3 mini-batch synthetic candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ERROR_TYPES, ensure_mini_batch_dirs
from thesis_exp.src.edujudge.exp06_generation.hardened_filter import HARDENED_FILTER_FIELDS, apply_hardened_checks
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    ARTIFACT_PHRASES,
    answer_hash,
    mini_filtered_path,
    mini_generated_path,
    mini_table_path,
    read_jsonl_if_exists,
    write_empty_jsonl,
    write_mini_jsonl,
    write_mini_table,
)
from thesis_exp.src.edujudge.exp06_generation.build_mini_batch_generation_plan import mode_key_sets
from thesis_exp.src.edujudge.utils.text_norm import detect_language_from_text, normalize_text, stringify


REPORT_FIELDS = ["check_name", "status", "count", "notes"]
DETAIL_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "passed_filter",
    "filter_reasons",
    "filter_stage",
    "filter_status",
    "failure_reasons",
    *HARDENED_FILTER_FIELDS,
    "notes",
]


def artifact_phrase_hits(answer: str) -> list[str]:
    folded = normalize_text(answer)
    return [phrase for phrase in ARTIFACT_PHRASES if phrase in folded]


def filter_candidate(row: dict[str, Any], seen_answers: set[str], human_test_answer_keys: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    answer = stringify(row.get("answer_synthetic"))
    if row.get("normalization_status") != "pass":
        reasons.append("normalization_failed")
    if not answer.strip():
        reasons.append("empty_answer")
    if not (20 <= len(answer) <= 4000):
        reasons.append("length_out_of_range")
    expected_language = stringify(row.get("language"))
    detected_language = detect_language_from_text(answer)
    if expected_language in {"en", "zh"} and detected_language != expected_language:
        reasons.append(f"language_mismatch:{detected_language}")
    hits = artifact_phrase_hits(answer)
    if hits:
        reasons.append("artifact_phrase:" + ",".join(hits))
    if stringify(row.get("target_label_5")) not in {"1", "2", "3"}:
        reasons.append("invalid_target_label")
    if row.get("source_split") != "train":
        reasons.append("not_train_source")
    if not row.get("metric_canonical") or not row.get("rubric_text"):
        reasons.append("missing_metric_or_rubric")
    if normalize_text(answer) == normalize_text(row.get("source_answer")):
        reasons.append("copied_source_answer")
    answer_norm = normalize_text(answer)
    if answer_norm in seen_answers:
        reasons.append("duplicate_synthetic_answer")
    if answer:
        seen_answers.add(answer_norm)
    if answer_hash(answer) in human_test_answer_keys:
        reasons.append("duplicate_human_test_answer")
    if not stringify(row.get("rationale_for_label")).strip():
        reasons.append("missing_rationale_for_label")
    if stringify(row.get("error_type")) not in ERROR_TYPES:
        reasons.append("invalid_error_type")
    return not reasons, reasons


def build_report(rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = sum(1 for row in detail_rows if row["filter_status"] == "pass")
    failed = sum(1 for row in detail_rows if row["filter_status"] == "fail")
    reason_counts: Counter[str] = Counter()
    for row in detail_rows:
        for reason in stringify(row.get("failure_reasons")).split("; "):
            if reason:
                reason_counts[reason.split(":")[0]] += 1
    report = [
        {"check_name": "total_normalized_rows", "status": "INFO", "count": len(rows), "notes": "input candidate count"},
        {"check_name": "passed_filter_rows", "status": "PASS" if passed else "INFO", "count": passed, "notes": "rows written to filtered_synthetic_candidates.jsonl"},
        {"check_name": "failed_filter_rows", "status": "FAIL" if failed else "PASS", "count": failed, "notes": "rows blocked by one or more required filters"},
    ]
    if not rows:
        report.append({"check_name": "generation_presence", "status": "DRY_RUN", "count": 0, "notes": "no generated candidates available"})
    for reason, count in sorted(reason_counts.items()):
        report.append({"check_name": f"failure_reason:{reason}", "status": "FAIL", "count": count, "notes": "filter failure reason count"})
    return report


def filter_candidates(input_path: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_mini_batch_dirs()
    input_path = input_path or mini_generated_path("normalized_synthetic_candidates.jsonl")
    rows = read_jsonl_if_exists(input_path)
    mode_keys: dict[str, dict[str, dict[str, set[str]]]] = {}
    seen_answers: set[str] = set()
    passed_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        mode = stringify(row.get("generation_split_mode") or "question_disjoint_formal")
        if mode not in mode_keys:
            keys = mode_key_sets(mode)
            from thesis_exp.src.edujudge.exp06_generation import GENERATION_SPLIT_MODES
            from thesis_exp.src.edujudge.utils.io import read_jsonl

            # mode_key_sets covers ids/keys; add answer hashes for duplicate answer checks.
            test_rows = read_jsonl(GENERATION_SPLIT_MODES[mode]["split_dir"] / "test.jsonl")
            keys["test"]["answer_key"] = {answer_hash(test_row.get("answer")) for test_row in test_rows}
            mode_keys[mode] = keys
        passed, reasons = filter_candidate(row, seen_answers, mode_keys[mode]["test"]["answer_key"])
        hardened = apply_hardened_checks(row)
        row.update(hardened)
        if hardened["artifact_phrase_status"] == "fail" and "artifact_phrase_hardened" not in reasons:
            reasons.append("artifact_phrase_hardened")
        if (
            stringify(row.get("target_label_5")) in {"1", "2"}
            and hardened["rubric_failure_visibility"] == "missing_clear_failure"
            and "missing_clear_rubric_failure_hardened" not in reasons
        ):
            reasons.append("missing_clear_rubric_failure_hardened")
        passed = not reasons
        row["filter_status"] = "pass" if passed else "fail"
        row["filter_reasons"] = reasons
        if passed:
            passed_rows.append(row)
        detail_rows.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "passed_filter": passed,
                "filter_reasons": "; ".join(reasons),
                "filter_stage": "required_generated_row_filters",
                "filter_status": "pass" if passed else "fail",
                "failure_reasons": "; ".join(reasons),
                **{field: hardened.get(field, "") for field in HARDENED_FILTER_FIELDS},
                "notes": "required Exp6-3 generated-row filter",
            }
        )
    write_mini_jsonl(mini_filtered_path("filtered_synthetic_candidates.jsonl"), passed_rows)
    report_rows = build_report(rows, detail_rows)
    write_mini_table("filter_report.csv", report_rows, REPORT_FIELDS)
    write_mini_table("filter_failure_details.csv", detail_rows, DETAIL_FIELDS)
    if not rows and not mini_filtered_path("filtered_synthetic_candidates.jsonl").exists():
        write_empty_jsonl(mini_filtered_path("filtered_synthetic_candidates.jsonl"))
    return passed_rows, report_rows, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args = parser.parse_args()
    passed_rows, report_rows, _ = filter_candidates(args.input_jsonl)
    print(f"Wrote {len(passed_rows)} passed candidates and {len(report_rows)} filter report rows to {mini_table_path('filter_report.csv')}")


if __name__ == "__main__":
    main()
