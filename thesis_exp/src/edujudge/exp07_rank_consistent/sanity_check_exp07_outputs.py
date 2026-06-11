"""Sanity checks for completed Exp7 QD-R1 outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXPECTED_SPLIT_ROWS,
    EXP07_OUTPUT_DIR,
    EXP07_RUN_ID,
    EXP07_TABLES_DIR,
    ensure_exp07_dirs,
    exp07_run_dir,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.data import tracked_weight_files
from thesis_exp.src.edujudge.utils.io import count_json_records, read_csv, relpath, write_csv, write_text


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def _shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    return tuple(np.load(path).shape)


def _split_row(path: Path, split: str) -> dict[str, str]:
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("split") == split:
            return row
    return {}


def main() -> None:
    ensure_exp07_dirs()
    run_dir = exp07_run_dir()
    rows: list[dict[str, Any]] = []

    pred_test = run_dir / "predictions" / "predictions_test.jsonl"
    add(rows, "predictions_test rows = 1103", count_json_records(pred_test) == EXPECTED_SPLIT_ROWS["test"], count_json_records(pred_test))
    add(rows, "logits_test shape = [1103,4]", _shape(run_dir / "arrays" / "logits_test.npy") == (EXPECTED_SPLIT_ROWS["test"], 4), _shape(run_dir / "arrays" / "logits_test.npy"))
    add(rows, "probs_test shape = [1103,4]", _shape(run_dir / "arrays" / "probs_test.npy") == (EXPECTED_SPLIT_ROWS["test"], 4), _shape(run_dir / "arrays" / "probs_test.npy"))
    add(rows, "monotonicity_metrics.csv exists", (run_dir / "tables" / "monotonicity_metrics.csv").exists())
    mono = _split_row(run_dir / "tables" / "monotonicity_metrics.csv", "test")
    try:
        mono_rate = float(mono.get("monotonic_violation_rate", "nan"))
    except ValueError:
        mono_rate = float("nan")
    add(rows, "monotonic_violation_rate near 0", mono_rate <= 1e-6, mono.get("monotonic_violation_rate", "missing"))
    for rel in [
        "tables/metrics_summary.csv",
        "tables/low_score_metrics.csv",
        "tables/high_score_metrics.csv",
        "tables/per_bin_metrics.csv",
        "tables/metric_level_metrics.csv",
        "tables/scenario_level_metrics.csv",
        "tables/language_level_metrics.csv",
    ]:
        add(rows, f"{rel} exists", (run_dir / rel).exists())
    for rel in [
        "tables/exp07_r1_comparison.csv",
        "tables/exp07_monotonicity_comparison.csv",
        "tables/exp07_low_score_comparison.csv",
        "tables/exp07_high_score_comparison.csv",
    ]:
        add(rows, f"global {rel} exists", (EXP07_OUTPUT_DIR / rel).exists())
    weights = tracked_weight_files()
    add(rows, "no checkpoint/weights tracked", not weights, ", ".join(weights))
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    add(rows, "run_summary metadata correct objective", metadata.get("objective") == "CORAL_rank_consistent_ordinal", metadata)
    add(rows, "run_metadata synthetic/class/calibration disabled", metadata.get("synthetic_used") is False and metadata.get("class_weights_used") is False and metadata.get("calibration_used") is False)

    write_csv(EXP07_TABLES_DIR / "sanity_check_exp07_outputs.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp7 Output Sanity Check",
        "",
        f"Run ID: `{EXP07_RUN_ID}`",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP07_OUTPUT_DIR / "sanity_check_exp07_outputs.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp7 output sanity failed. See {relpath(EXP07_OUTPUT_DIR)}")
    print("Exp7 output sanity PASS")


if __name__ == "__main__":
    main()
