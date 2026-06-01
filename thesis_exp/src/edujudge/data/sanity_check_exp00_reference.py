"""Reference sanity checks for Exp 0.1."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.data.build_dataset import PROCESSED_PATH
from thesis_exp.src.edujudge.data.normalize_fields import convert_score_to_five, detect_score_scale, round_half_up
from thesis_exp.src.edujudge.data.reference_contract import (
    EXPECTED_GENERATOR_MODELS,
    EXPECTED_LANGUAGES,
    EXPECTED_SCENARIOS,
    EXPECTED_METRICS,
    EXPECTED_SUBJECTS,
    PDF_AUDIT_TOTAL,
    PDF_HELDOUT_TEST_ROWS,
    PDF_TRAIN_POOL_ROWS,
)
from thesis_exp.src.edujudge.utils.io import OUTPUT_DIR, SPLITS_DIR, TABLES_DIR, ensure_exp_dirs, read_jsonl, write_csv, write_text


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_rows(split_name: str, split: str) -> list[dict[str, Any]]:
    return read_jsonl(SPLITS_DIR / split_name / f"{split}.jsonl")


def _overlap(split_name: str, key: str) -> int:
    sets = {split: {row.get(key) for row in _split_rows(split_name, split)} for split in ["train", "dev", "test"]}
    return len(sets["train"] & sets["dev"]) + len(sets["train"] & sets["test"]) + len(sets["dev"] & sets["test"])


def _verify_score(row: dict[str, Any]) -> bool:
    raw_scores = [row.get("human_1_raw"), row.get("human_2_raw"), row.get("human_3_raw")]
    raw_scores = [float(score) for score in raw_scores if score is not None]
    scale = detect_score_scale(raw_scores)
    if scale != row.get("raw_score_scale_detected"):
        return False
    if scale == "1-10":
        mapped = [convert_score_to_five(score) for score in raw_scores]
        if any(score is None for score in mapped):
            return False
        expected_mean = sum(float(score) for score in mapped if score is not None) / len(mapped)
    elif scale == "1-5":
        expected_mean = sum(raw_scores) / len(raw_scores)
    else:
        return False
    expected_label = max(1, min(5, round_half_up(expected_mean)))
    return int(row.get("label_5")) == expected_label


def run_checks() -> tuple[str, list[dict[str, Any]]]:
    ensure_exp_dirs()
    rows = read_jsonl(PROCESSED_PATH)
    results: list[dict[str, Any]] = []

    def add(check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
        results.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})

    add("processed dataset rows", "PASS" if len(rows) == PDF_AUDIT_TOTAL else "FAIL", len(rows), PDF_AUDIT_TOTAL)
    labels = sorted({row.get("label_5") for row in rows})
    add("label_5 domain", "PASS" if labels == [1, 2, 3, 4, 5] else "FAIL", labels, [1, 2, 3, 4, 5])
    models = sorted({row.get("generator_model") for row in rows})
    add("generator model count", "PASS" if set(models) == set(EXPECTED_GENERATOR_MODELS) else "FAIL", models, EXPECTED_GENERATOR_MODELS)
    metrics = sorted({row.get("metric_canonical") for row in rows if row.get("metric_canonical")})
    add("canonical metric count", "PASS" if set(metrics) == set(EXPECTED_METRICS) else "FAIL", len(metrics), len(EXPECTED_METRICS))
    scenarios = sorted({row.get("scenario_canonical") for row in rows if row.get("scenario_canonical")})
    add("canonical scenario count", "PASS" if set(scenarios) == set(EXPECTED_SCENARIOS) else "FAIL", len(scenarios), len(EXPECTED_SCENARIOS))
    missing_metric = sum(1 for row in rows if not row.get("metric_canonical") or row.get("metric_canonical") == "unknown")
    missing_scenario = sum(1 for row in rows if not row.get("scenario_canonical") or row.get("scenario_canonical") == "unknown")
    add("all rows have metric_canonical", "PASS" if missing_metric == 0 else "FAIL", missing_metric, 0)
    add("all rows have scenario_canonical", "PASS" if missing_scenario == 0 else "FAIL", missing_scenario, 0)
    synthetic_rows = sum(1 for row in rows if "sampled_merge_50_new" in str(row.get("source_file")))
    add("synthetic files excluded", "PASS" if synthetic_rows == 0 else "FAIL", synthetic_rows, 0)
    train_pool = len(_split_rows("paper_like_triple_seed42", "train")) + len(_split_rows("paper_like_triple_seed42", "dev"))
    test_rows = len(_split_rows("paper_like_triple_seed42", "test"))
    add("paper-like train+dev rows", "PASS" if train_pool == PDF_TRAIN_POOL_ROWS else "FAIL", train_pool, PDF_TRAIN_POOL_ROWS)
    add("paper-like test rows", "PASS" if test_rows == PDF_HELDOUT_TEST_ROWS else "FAIL", test_rows, PDF_HELDOUT_TEST_ROWS)
    triple_overlap = _overlap("paper_like_triple_seed42", "triple_key")
    question_overlap = _overlap("question_seed42", "question_key")
    add("paper-like triple_key cross-split overlap", "PASS" if triple_overlap == 0 else "FAIL", triple_overlap, 0)
    add("question split question_key cross-split overlap", "PASS" if question_overlap == 0 else "FAIL", question_overlap, 0)
    score_bad = sum(1 for row in rows if not _verify_score(row))
    add("score scale mapping matches 5-grades.py logic", "PASS" if score_bad == 0 else "FAIL", score_bad, 0)
    subjects = sorted({row.get("subject_canonical") for row in rows if row.get("subject_canonical") and row.get("subject_canonical") != "unknown"})
    subject_status = "PASS" if set(subjects) == set(EXPECTED_SUBJECTS) else "WARNING"
    add("canonical subject count", subject_status, len(subjects), len(EXPECTED_SUBJECTS), "WARNING is non-blocking for global sanity.")
    leakage_summary = _read_csv(TABLES_DIR / "leakage_summary.csv")
    duplicate_across = 0
    for row in leakage_summary:
        if row.get("check") == "duplicate full scored item across split files":
            duplicate_across += int(row.get("overlap_count") or 0)
    add(
        "duplicate full scored item across split files",
        "PASS" if duplicate_across == 0 else "FAIL",
        duplicate_across,
        0,
        "If non-zero, see tables/leakage_details.csv.",
    )

    status_rank = {"PASS": 0, "INFO": 0, "WARNING": 1, "FAIL": 2}
    overall = "PASS"
    for row in results:
        if status_rank[row["status"]] > status_rank[overall]:
            overall = row["status"]
    if overall == "WARNING" and all(row["status"] != "FAIL" for row in results):
        overall = "WARNING"
    return overall, results


def main() -> None:
    status, rows = run_checks()
    write_csv(TABLES_DIR / "sanity_check_results.csv", rows, ["check", "status", "observed", "expected", "notes"])
    lines = [
        "# Exp 0.1 Reference Sanity Check",
        "",
        f"Overall status: **{status}**",
        "",
        "| check | status | observed | expected | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['status']} | {row['observed']} | {row['expected']} | {row.get('notes', '')} |")
    write_text(OUTPUT_DIR / "sanity_check_exp00_reference.md", "\n".join(lines))
    print(f"Reference sanity check status: {status}")


if __name__ == "__main__":
    main()
