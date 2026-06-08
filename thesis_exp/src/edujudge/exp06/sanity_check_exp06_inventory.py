"""Sanity checks for Exp6 synthetic inventory outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06 import (
    EXP0_SPLIT_DIR,
    EXP06_OUTPUT_DIR,
    EXP06_TABLES_DIR,
    INVENTORY_FIELDS,
    NORMALIZED_FIELDS,
    SCHEMA_PROFILE_FIELDS,
    ensure_exp06_dirs,
)
from thesis_exp.src.edujudge.exp06.common import read_csv_rows, write_rows
from thesis_exp.src.edujudge.utils.io import md_table, write_text


REQUIRED_TABLES = {
    "synthetic_source_inventory.csv": INVENTORY_FIELDS,
    "synthetic_schema_profile.csv": SCHEMA_PROFILE_FIELDS,
    "synthetic_candidate_rows.csv": NORMALIZED_FIELDS,
    "synthetic_leakage_summary.csv": [
        "source_file",
        "total_candidates",
        "any_dev_test_leakage",
        "leakage_risk",
    ],
    "synthetic_leakage_details.csv": [
        "synthetic_id",
        "source_file",
        "check_type",
        "matched_split",
    ],
    "synthetic_score_distribution.csv": ["source_file", "target_label_5", "count"],
    "synthetic_metric_distribution.csv": ["source_file", "metric_canonical", "target_label_5", "count"],
    "synthetic_language_distribution.csv": ["source_file", "language", "target_label_5", "count"],
    "synthetic_error_type_distribution.csv": ["source_file", "error_type", "target_label_5", "count"],
    "synthetic_filter_recommendation.csv": [
        "source_file",
        "recommended_use",
        "allowed_split",
        "blocked_reasons",
    ],
}


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def run_checks(strict: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, required in REQUIRED_TABLES.items():
        path = EXP06_TABLES_DIR / name
        header = csv_header(path)
        add(rows, f"{name} exists", "PASS" if path.exists() else "FAIL", path.exists(), True)
        missing = sorted(set(required) - set(header))
        add(rows, f"{name} required columns", "PASS" if not missing else "FAIL", missing, [])

    for name in ["report.md", "review_package.md", "notion_exp06_inventory_summary.md"]:
        path = EXP06_OUTPUT_DIR / name
        add(rows, f"{name} exists", "PASS" if path.exists() else "FAIL", path.exists(), True)

    inventory = read_csv_rows(EXP06_TABLES_DIR / "synthetic_source_inventory.csv")
    inv_by_path = {row.get("source_path", ""): row for row in inventory}
    for name in ["sampled_merge_50_new.json", "sampled_merge_50_new_swift.json"]:
        row = inv_by_path.get(name, {})
        ok = row.get("risk_level") == "HIGH"
        add(rows, f"{name} default HIGH risk", "PASS" if ok else "FAIL", row.get("risk_level"), "HIGH")
    for name in [
        "merge_model_metric.jsonl",
        "deepseek-r1_merged.jsonl",
        "groupby_metric_qwq_eval_en.jsonl",
        "groupby_metric_qwq_eval_zh.jsonl",
        "groupby_metric_r1_eval_en.jsonl",
        "groupby_metric_r1_eval_zh.jsonl",
        "groupby_metric_v3_eval_en.jsonl",
        "groupby_metric_v3_eval_zh.jsonl",
    ]:
        row = inv_by_path.get(name, {})
        ok = row.get("likely_role") == "model_judge_output" and row.get("usable_for_exp06", "").startswith("NO")
        add(rows, f"{name} blocked judge role", "PASS" if ok else "FAIL", (row.get("likely_role"), row.get("usable_for_exp06")), "model_judge_output / NO*")

    candidates = read_csv_rows(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv")
    labels = sum(1 for row in candidates if row.get("target_label_5"))
    low = sum(1 for row in candidates if row.get("target_label_5") in {"1", "2"})
    add(rows, "candidate rows generated", "PASS" if candidates else "FAIL", len(candidates), ">0")
    add(rows, "target_label_5 rows profiled", "PASS" if labels else "WARNING", labels, ">0", "warning allows no-label repositories but Exp6 needs labels")
    add(rows, "low-score rows profiled", "PASS" if low else "WARNING", low, ">0", "Exp6 low-score augmentation requires low labels")

    forbidden_split_outputs = [
        EXP06_OUTPUT_DIR / "train.jsonl",
        EXP06_OUTPUT_DIR / "dev.jsonl",
        EXP06_OUTPUT_DIR / "test.jsonl",
        EXP06_OUTPUT_DIR / "datasets",
    ]
    existing_forbidden = [str(path) for path in forbidden_split_outputs if path.exists()]
    add(rows, "no Exp6 synthetic train/dev/test generated", "PASS" if not existing_forbidden else "FAIL", existing_forbidden, [])

    expected_counts = {"train": 2654, "dev": 664, "test": 2218}
    for split, expected in expected_counts.items():
        path = EXP0_SPLIT_DIR / f"{split}.jsonl"
        observed = sum(1 for line in path.open("r", encoding="utf-8") if line.strip()) if path.exists() else 0
        add(rows, f"Exp0 split {split} row count unchanged", "PASS" if observed == expected else "FAIL", observed, expected)

    if strict and any(row["status"] == "FAIL" for row in rows):
        return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    ensure_exp06_dirs()
    rows = run_checks(strict=args.strict)
    write_rows(EXP06_TABLES_DIR / "sanity_check_exp06_inventory.csv", rows, ["check", "status", "observed", "expected", "notes"])
    write_text(EXP06_OUTPUT_DIR / "sanity_check_exp06_inventory.md", "# Exp6 Inventory Sanity Check\n\n" + md_table(rows, ["check", "status", "observed", "expected", "notes"], max_rows=200))
    print(f"Wrote Exp6 sanity check with {len(rows)} checks")
    if args.strict and any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
