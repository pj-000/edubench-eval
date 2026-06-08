"""Check synthetic candidates against Exp0 human train/dev/test splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP0_SPLIT_DIR, EXP06_TABLES_DIR, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.common import (
    read_csv_rows,
    stable_qa_hash,
    stable_question_hash,
    stable_triple_hash,
    write_rows,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl
from thesis_exp.src.edujudge.utils.text_norm import truncate_text


SUMMARY_FIELDS = [
    "source_file",
    "total_candidates",
    "source_question_key_in_train",
    "source_question_key_in_dev",
    "source_question_key_in_test",
    "question_key_in_dev",
    "question_key_in_test",
    "triple_key_in_dev",
    "triple_key_in_test",
    "normalized_qa_in_dev",
    "normalized_qa_in_test",
    "from_dev_test_question",
    "any_dev_test_leakage",
    "leakage_risk",
    "notes",
]

DETAIL_FIELDS = [
    "synthetic_id",
    "source_file",
    "source_row_index",
    "check_type",
    "matched_split",
    "matched_key_type",
    "matched_key_value",
    "question_preview",
    "answer_preview",
    "metric_canonical",
    "target_label_5",
    "recommended_action",
]


def load_split_indexes() -> dict[str, dict[str, set[str]]]:
    indexes: dict[str, dict[str, set[str]]] = {}
    for split in ["train", "dev", "test"]:
        rows = read_jsonl(EXP0_SPLIT_DIR / f"{split}.jsonl")
        indexes[split] = {
            "source_question_key": {str(row.get("question_key", "")) for row in rows if row.get("question_key")},
            "source_triple_key": {str(row.get("triple_key", "")) for row in rows if row.get("triple_key")},
            "question_hash": {stable_question_hash(row.get("question")) for row in rows},
            "qa_hash": {stable_qa_hash(row.get("question"), row.get("answer")) for row in rows},
            "triple_hash": {
                stable_triple_hash(row.get("question"), row.get("answer"), row.get("metric_canonical") or row.get("metric_raw"))
                for row in rows
            },
        }
    return indexes


def add_detail(details: list[dict[str, Any]], row: dict[str, str], check_type: str, split: str, key_type: str, key_value: str) -> None:
    details.append(
        {
            "synthetic_id": row.get("synthetic_id", ""),
            "source_file": row.get("source_file", ""),
            "source_row_index": row.get("source_row_index", ""),
            "check_type": check_type,
            "matched_split": split,
            "matched_key_type": key_type,
            "matched_key_value": key_value,
            "question_preview": truncate_text(row.get("question", ""), 220),
            "answer_preview": truncate_text(row.get("answer", ""), 220),
            "metric_canonical": row.get("metric_canonical", ""),
            "target_label_5": row.get("target_label_5", ""),
            "recommended_action": "BLOCK_FOR_EXP6_TRAINING_UNTIL_MANUAL_REVIEW" if split in {"dev", "test"} else "TRAIN_OVERLAP_REVIEW",
        }
    )


def run_leakage_check() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = read_csv_rows(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv")
    indexes = load_split_indexes()
    details: list[dict[str, Any]] = []
    per_source: dict[str, Counter[str]] = defaultdict(Counter)

    for row in candidates:
        source = str(row.get("source_file") or "")
        per_source[source]["total_candidates"] += 1
        question_hash = stable_question_hash(row.get("question", ""))
        qa_hash = stable_qa_hash(row.get("question", ""), row.get("answer", ""))
        triple_hash = stable_triple_hash(row.get("question", ""), row.get("answer", ""), row.get("metric_canonical") or row.get("metric_raw"))
        source_q = row.get("source_question_key", "")
        source_t = row.get("source_triple_key", "")

        for split, split_indexes in indexes.items():
            if source_q and source_q in split_indexes["source_question_key"]:
                key = f"source_question_key_in_{split}"
                per_source[source][key] += 1
                add_detail(details, row, key, split, "source_question_key", source_q)
            if source_q and source_q in split_indexes["question_hash"]:
                key = f"source_question_key_hash_in_{split}"
                per_source[source][key] += 1
                add_detail(details, row, key, split, "question_hash", source_q)
            if source_t and source_t in split_indexes["source_triple_key"]:
                key = f"source_triple_key_in_{split}"
                per_source[source][key] += 1
                add_detail(details, row, key, split, "source_triple_key", source_t)

        for split in ["dev", "test"]:
            split_indexes = indexes[split]
            if question_hash in split_indexes["question_hash"]:
                key = f"question_key_in_{split}"
                per_source[source][key] += 1
                per_source[source]["from_dev_test_question"] += 1
                add_detail(details, row, key, split, "question_hash", question_hash)
            if triple_hash in split_indexes["triple_hash"] or source_t in split_indexes["source_triple_key"]:
                key = f"triple_key_in_{split}"
                per_source[source][key] += 1
                add_detail(details, row, key, split, "triple_hash/source_triple_key", source_t or triple_hash)
            if qa_hash in split_indexes["qa_hash"]:
                key = f"normalized_qa_in_{split}"
                per_source[source][key] += 1
                add_detail(details, row, key, split, "qa_hash", qa_hash)

    summary: list[dict[str, Any]] = []
    for source, counts in sorted(per_source.items(), key=lambda item: str(item[0] or "")):
        dev_test = sum(
            counts.get(key, 0)
            for key in [
                "source_question_key_in_dev",
                "source_question_key_in_test",
                "source_question_key_hash_in_dev",
                "source_question_key_hash_in_test",
                "source_triple_key_in_dev",
                "source_triple_key_in_test",
                "question_key_in_dev",
                "question_key_in_test",
                "triple_key_in_dev",
                "triple_key_in_test",
                "normalized_qa_in_dev",
                "normalized_qa_in_test",
            ]
        )
        train_overlap = counts.get("source_question_key_in_train", 0) + counts.get("source_question_key_hash_in_train", 0)
        summary.append(
            {
                "source_file": source,
                "total_candidates": counts.get("total_candidates", 0),
                "source_question_key_in_train": counts.get("source_question_key_in_train", 0) + counts.get("source_question_key_hash_in_train", 0),
                "source_question_key_in_dev": counts.get("source_question_key_in_dev", 0) + counts.get("source_question_key_hash_in_dev", 0),
                "source_question_key_in_test": counts.get("source_question_key_in_test", 0) + counts.get("source_question_key_hash_in_test", 0),
                "question_key_in_dev": counts.get("question_key_in_dev", 0),
                "question_key_in_test": counts.get("question_key_in_test", 0),
                "triple_key_in_dev": counts.get("triple_key_in_dev", 0) + counts.get("source_triple_key_in_dev", 0),
                "triple_key_in_test": counts.get("triple_key_in_test", 0) + counts.get("source_triple_key_in_test", 0),
                "normalized_qa_in_dev": counts.get("normalized_qa_in_dev", 0),
                "normalized_qa_in_test": counts.get("normalized_qa_in_test", 0),
                "from_dev_test_question": counts.get("from_dev_test_question", 0),
                "any_dev_test_leakage": dev_test > 0,
                "leakage_risk": "BLOCKED" if dev_test > 0 else ("MEDIUM" if train_overlap > 0 else "LOW"),
                "notes": "dev/test overlap found" if dev_test > 0 else ("train question overlap only" if train_overlap > 0 else "no exact split overlap detected"),
            }
        )
    return summary, details


def main() -> None:
    ensure_exp06_dirs()
    summary, details = run_leakage_check()
    write_rows(EXP06_TABLES_DIR / "synthetic_leakage_summary.csv", summary, SUMMARY_FIELDS)
    write_rows(EXP06_TABLES_DIR / "synthetic_leakage_details.csv", details, DETAIL_FIELDS)
    print(f"Wrote leakage summary/details to {EXP06_TABLES_DIR}")


if __name__ == "__main__":
    main()
