"""Build audit-only train-side candidates for processed Excel sources."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_TABLES_DIR, PROCESSED_EXCEL_AUDIT_DIR
from thesis_exp.src.edujudge.exp06.compare_processed_excel_en_zh_merge import merge_equals_split
from thesis_exp.src.edujudge.exp06.processed_excel_common import (
    PROCESSED_EXCEL_SOURCES,
    all_candidate_rows,
    load_split_indexes,
    output_path,
    read_csv_rows,
    records_by_source,
    small_preview,
    summarize_low_score,
    write_processed_csv,
)
from thesis_exp.src.edujudge.utils.text_norm import stringify


OVERLAP_FIELDS = [
    "candidate_id",
    "source_file",
    "source_record_id",
    "source_record_index",
    "source_score_index",
    "question_key",
    "qa_key",
    "triple_key",
    "in_train",
    "in_dev",
    "in_test",
    "triple_in_dev",
    "triple_in_test",
    "qa_in_dev",
    "qa_in_test",
    "any_dev_test_overlap",
    "duplicate_status",
    "metric_canonical",
    "target_label_5",
    "language",
    "question_preview",
    "answer_preview",
]

TRAIN_FIELDS = [
    "candidate_id",
    "source_file",
    "source_record_id",
    "source_record_index",
    "source_score_index",
    "question",
    "answer",
    "metric_raw",
    "metric_canonical",
    "score_raw",
    "target_label_5",
    "reason",
    "language",
    "model",
    "level",
    "label_provenance_status",
    "candidate_use_status",
    "question_key",
    "qa_key",
    "triple_key",
    "candidate_key",
]

LOW_FIELDS = TRAIN_FIELDS

FINAL_FIELDS = [
    "source_file",
    "usable_for_exp6_training",
    "recommended_use",
    "reason",
    "label_provenance_status",
    "train_only_rows",
    "low_score_rows",
    "duplicate_risk",
    "leakage_risk",
    "next_action",
]


def provenance_by_source() -> dict[str, str]:
    rows = read_csv_rows(output_path("processed_excel_label_provenance.csv"))
    return {row.get("source_file", ""): row.get("provenance_status", "unknown") for row in rows}


def source_dev_test_leakage(row: dict[str, Any], indexes: dict[str, dict[str, set[str]]]) -> dict[str, bool]:
    return {
        "in_train": row["question_key"] in indexes["train"]["question_key"],
        "in_dev": row["question_key"] in indexes["dev"]["question_key"],
        "in_test": row["question_key"] in indexes["test"]["question_key"],
        "triple_in_dev": row["triple_key"] in indexes["dev"]["triple_key"],
        "triple_in_test": row["triple_key"] in indexes["test"]["triple_key"],
        "qa_in_dev": row["qa_key"] in indexes["dev"]["qa_key"],
        "qa_in_test": row["qa_key"] in indexes["test"]["qa_key"],
    }


def duplicate_statuses(rows: list[dict[str, Any]]) -> dict[str, str]:
    merge_equal = merge_equals_split(records_by_source())
    canonical_source = PROCESSED_EXCEL_SOURCES[0] if merge_equal else ""
    seen: set[str] = set()
    statuses: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (0 if item["source_file"] == canonical_source else 1, item["source_file"], item["candidate_key"])):
        if merge_equal and row["source_file"] != canonical_source:
            statuses[row["candidate_id"]] = "duplicate_split_view_excluded"
            continue
        key = row["candidate_key"]
        if key in seen:
            statuses[row["candidate_id"]] = "duplicate_within_canonical_excluded"
        else:
            seen.add(key)
            statuses[row["candidate_id"]] = "keep_canonical"
    return statuses


def exp60_large_detail_presence() -> dict[str, Any]:
    candidate_path = EXP06_TABLES_DIR / "synthetic_candidate_rows.csv"
    leakage_path = EXP06_TABLES_DIR / "synthetic_leakage_details.csv"
    out = {
        "synthetic_candidate_rows_exists": candidate_path.exists(),
        "synthetic_leakage_details_exists": leakage_path.exists(),
        "processed_excel_candidate_rows_seen": 0,
        "processed_excel_leakage_detail_rows_seen": 0,
    }
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_file") in PROCESSED_EXCEL_SOURCES:
                    out["processed_excel_candidate_rows_seen"] += 1
    if leakage_path.exists():
        with leakage_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_file") in PROCESSED_EXCEL_SOURCES:
                    out["processed_excel_leakage_detail_rows_seen"] += 1
    return out


def build_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = all_candidate_rows()
    indexes = load_split_indexes()
    provenance = provenance_by_source()
    duplicate_status = duplicate_statuses(rows)
    overlap_rows: list[dict[str, Any]] = []
    train_candidates: list[dict[str, Any]] = []
    for row in rows:
        overlap = source_dev_test_leakage(row, indexes)
        any_dev_test = any(overlap[key] for key in ["in_dev", "in_test", "triple_in_dev", "triple_in_test", "qa_in_dev", "qa_in_test"])
        status = duplicate_status[row["candidate_id"]]
        overlap_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "source_file": row["source_file"],
                "source_record_id": row["source_record_id"],
                "source_record_index": row["source_record_index"],
                "source_score_index": row["source_score_index"],
                "question_key": row["question_key"],
                "qa_key": row["qa_key"],
                "triple_key": row["triple_key"],
                **overlap,
                "any_dev_test_overlap": any_dev_test,
                "duplicate_status": status,
                "metric_canonical": row["metric_canonical"],
                "target_label_5": row["target_label_5"],
                "language": row["language"],
                "question_preview": small_preview(row["question"]),
                "answer_preview": small_preview(row["answer"]),
            }
        )
        prov = provenance.get(row["source_file"], "unknown")
        passes = (
            not any_dev_test
            and row["valid_label"]
            and row["valid_metric"]
            and row["valid_question_answer"]
            and status == "keep_canonical"
            and prov not in {"unknown"}
        )
        if passes:
            use_status = "pilot_only_not_human_confirmed" if prov != "human_confirmed" else "train_only_human_confirmed"
            train_candidates.append({**row, "label_provenance_status": prov, "candidate_use_status": use_status})

    low_candidates = [
        row for row in train_candidates if stringify(row.get("target_label_5")) in {"1", "2"} or row.get("target_label_5") in {1, 2}
    ]
    final = build_final_recommendation(overlap_rows, train_candidates, low_candidates, provenance)
    return overlap_rows, train_candidates, low_candidates, final


def source_counts(rows: list[dict[str, Any]], key: str = "source_file") -> Counter[str]:
    return Counter(stringify(row.get(key)) for row in rows)


def build_final_recommendation(
    overlap_rows: list[dict[str, Any]],
    train_candidates: list[dict[str, Any]],
    low_candidates: list[dict[str, Any]],
    provenance: dict[str, str],
) -> list[dict[str, Any]]:
    train_counts = source_counts(train_candidates)
    low_counts = source_counts(low_candidates)
    overlap_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlap_rows:
        overlap_by_source[row["source_file"]].append(row)
    merge_equal = merge_equals_split(records_by_source())
    final = []
    for source in PROCESSED_EXCEL_SOURCES:
        overlaps = overlap_by_source.get(source, [])
        dev_test = sum(1 for row in overlaps if row["any_dev_test_overlap"])
        prov = provenance.get(source, "unknown")
        train_rows = train_counts.get(source, 0)
        low_rows = low_counts.get(source, 0)
        if source != PROCESSED_EXCEL_SOURCES[0] and merge_equal:
            duplicate_risk = "DUPLICATE_OF_MERGED"
            recommended = "do_not_use_with_merged"
            reason = "en/zh split source is duplicate of processed_excel_data_1.jsonl"
        elif prov != "human_confirmed":
            duplicate_risk = "HIGH_IF_COMBINED_WITH_EN_ZH" if source == PROCESSED_EXCEL_SOURCES[0] and merge_equal else "LOW_AFTER_DEDUP"
            recommended = "pilot_only_after_manual_approval"
            reason = "no dev/test overlap after exact checks, but labels are not human-confirmed"
        else:
            duplicate_risk = "HIGH_IF_COMBINED_WITH_EN_ZH" if source == PROCESSED_EXCEL_SOURCES[0] and merge_equal else "LOW_AFTER_DEDUP"
            recommended = "train_only_after_final_manual_confirmation"
            reason = "human-confirmed labels and no dev/test overlap"
        if dev_test:
            recommended = "blocked_due_to_dev_test_overlap"
            reason = "exact dev/test overlap found"
        enough = low_rows >= 50
        final.append(
            {
                "source_file": source,
                "usable_for_exp6_training": "NO" if prov != "human_confirmed" or dev_test or not enough else "YES",
                "recommended_use": recommended,
                "reason": reason,
                "label_provenance_status": prov,
                "train_only_rows": train_rows,
                "low_score_rows": low_rows,
                "duplicate_risk": duplicate_risk,
                "leakage_risk": "BLOCKED" if dev_test else "LOW",
                "next_action": "confirm label provenance; use only as pseudo-label pilot if approved; generate/obtain more train-only low-score data for meaningful augmentation"
                if low_rows < 50
                else "manual approval before train-only pilot",
            }
        )
    return final


def main() -> None:
    overlap_rows, train_candidates, low_candidates, final = build_outputs()
    write_processed_csv("processed_excel_train_dev_test_overlap.csv", overlap_rows, OVERLAP_FIELDS)
    write_processed_csv("processed_excel_train_only_candidates.csv", train_candidates, TRAIN_FIELDS)
    write_processed_csv("processed_excel_low_score_candidates.csv", low_candidates, LOW_FIELDS)
    write_processed_csv("processed_excel_final_recommendation.csv", final, FINAL_FIELDS)
    large = exp60_large_detail_presence()
    write_processed_csv("processed_excel_exp60_large_detail_crosscheck.csv", [large], list(large))
    print(
        f"Wrote processed Excel overlap/candidate/recommendation tables: "
        f"train_only={len(train_candidates)} low_score={len(low_candidates)}"
    )


if __name__ == "__main__":
    main()
