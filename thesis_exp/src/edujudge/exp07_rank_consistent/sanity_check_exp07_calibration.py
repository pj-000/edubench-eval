"""Sanity checks for Exp7-C calibration outputs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_OUTPUT_DIR,
    EXP07_CALIBRATION_PREDICTIONS_DIR,
    EXP07_CALIBRATION_REPORTS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    EXP07_OUTPUT_DIR,
    EXP07_RUN_ID,
    ensure_exp07_calibration_dirs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.data import tracked_weight_files
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


REQUIRED_TABLES = [
    "calibration_base_inventory.csv",
    "raw_baseline_metrics.csv",
    "temperature_scaling_dev.csv",
    "temperature_scaling_test.csv",
    "global_threshold_calibration_dev.csv",
    "global_threshold_calibration_test.csv",
    "risk_aware_threshold_calibration_dev.csv",
    "risk_aware_threshold_calibration_test.csv",
    "risk_aware_threshold_best_configs.csv",
    "calibration_summary.csv",
    "calibration_delta_vs_raw.csv",
    "low_score_calibration_comparison.csv",
    "high_score_calibration_comparison.csv",
    "rejection_analysis.csv",
    "server_calibration_artifact_inventory.csv",
    "unified_raw_baseline_metrics.csv",
    "unified_temperature_scaling_test.csv",
    "unified_global_threshold_test.csv",
    "unified_risk_aware_threshold_test.csv",
    "unified_calibration_summary.csv",
    "unified_calibration_delta_vs_raw.csv",
    "unified_rejection_analysis.csv",
    "best_calibrated_model_selection.csv",
]
CHECKPOINT_SUFFIXES = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def _rows(name: str) -> list[dict[str, str]]:
    path = EXP07_CALIBRATION_TABLES_DIR / name
    return read_csv(path) if path.exists() else []


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Any) -> bool:
    number = _float(value)
    return True if number is None else math.isfinite(number)


def _test_row(rows: list[dict[str, str]], base_model: str, method: str) -> dict[str, str]:
    for row in rows:
        if row.get("base_model") == base_model and row.get("method") == method and row.get("split") == "test":
            return row
    return {}


def check_required_files(rows: list[dict[str, Any]]) -> None:
    for name in REQUIRED_TABLES:
        path = EXP07_CALIBRATION_TABLES_DIR / name
        add(rows, f"{name} exists", path.exists(), relpath(path))
    for name in [
        "exp07_calibration_report.md",
        "exp07_calibration_review_package.md",
        "notion_exp07_calibration_summary.md",
        "exp07_c2_unified_calibration_report.md",
        "exp07_c2_review_package.md",
        "notion_exp07_c2_unified_calibration_summary.md",
    ]:
        path = EXP07_CALIBRATION_REPORTS_DIR / name
        add(rows, f"{name} exists", path.exists(), relpath(path))


def check_inventory(rows: list[dict[str, Any]]) -> None:
    inventory = _rows("calibration_base_inventory.csv")
    ready = [row for row in inventory if row.get("can_threshold_calibrate") == "yes"]
    add(rows, "at least one local base calibration-ready", bool(ready), [row.get("base_model") for row in ready])
    qdr1 = next((row for row in inventory if row.get("base_model") == EXP07_RUN_ID), {})
    add(rows, "QD-R1 local threshold calibration-ready", qdr1.get("can_threshold_calibrate") == "yes", qdr1)


def check_dev_only_selection(rows: list[dict[str, Any]]) -> None:
    checked = []
    for name in [
        "temperature_scaling_dev.csv",
        "temperature_scaling_test.csv",
        "global_threshold_calibration_dev.csv",
        "global_threshold_calibration_test.csv",
        "risk_aware_threshold_calibration_dev.csv",
        "risk_aware_threshold_calibration_test.csv",
        "risk_aware_threshold_best_configs.csv",
        "unified_temperature_scaling_test.csv",
        "unified_global_threshold_test.csv",
        "unified_risk_aware_threshold_test.csv",
        "best_calibrated_model_selection.csv",
    ]:
        for row in _rows(name):
            selection_split = row.get("selection_split")
            objective = row.get("selection_objective", "")
            if selection_split:
                checked.append((name, selection_split, objective))
                if selection_split != "dev" or (objective and not objective.startswith("dev_")):
                    add(rows, "selection uses dev only", False, f"{name}: {row}")
                    return
    add(rows, "selection uses dev only", True, checked[:5])


def check_thresholds(rows: list[dict[str, Any]]) -> None:
    for name in [
        "global_threshold_calibration_dev.csv",
        "global_threshold_calibration_test.csv",
        "risk_aware_threshold_calibration_dev.csv",
        "risk_aware_threshold_calibration_test.csv",
        "risk_aware_threshold_best_configs.csv",
    ]:
        for row in _rows(name):
            values = [_float(row.get(f"theta_{idx}")) for idx in range(1, 5)]
            if any(value is None for value in values):
                continue
            valid = all(0.0 <= value <= 1.0 for value in values if value is not None) and values == sorted(values)
            if not valid:
                add(rows, "threshold values valid and monotonic", False, f"{name}: {values}")
                return
    add(rows, "threshold values valid and monotonic", True)


def check_metrics_finite(rows: list[dict[str, Any]]) -> None:
    for name in REQUIRED_TABLES:
        for row in _rows(name):
            for key, value in row.items():
                if key in {"base_model", "method", "split", "source", "blocking_reason", "selection_type", "selection_objective", "selection_split"}:
                    continue
                if not _is_finite(value):
                    add(rows, "all numeric outputs finite", False, f"{name} {key}={value}")
                    return
    add(rows, "all numeric outputs finite", True)


def check_raw_matches_existing(rows: list[dict[str, Any]]) -> None:
    summary = _rows("raw_baseline_metrics.csv")
    raw = _test_row(summary, EXP07_RUN_ID, "raw")
    existing_path = EXP07_OUTPUT_DIR / "tables" / "exp07_r1_comparison.csv"
    existing = {}
    if existing_path.exists():
        for row in read_csv(existing_path):
            if row.get("run_id") == EXP07_RUN_ID and row.get("split") == "test":
                existing = row
                break
    checks = {
        "Accuracy": "Accuracy",
        "MAE_label": "MAE_label",
        "QWK": "Quadratic Weighted Kappa",
        "Kendall tau": "Kendall tau",
        "low_to_high_rate": "low_to_high_rate",
        "Acc@5": "Acc@5",
    }
    diffs = {}
    for left, right in checks.items():
        a = _float(raw.get(left))
        b = _float(existing.get(right))
        if a is None or b is None:
            add(rows, "raw baseline matches existing QD-R1 summary", False, f"missing {left}/{right}")
            return
        diffs[left] = abs(a - b)
    add(rows, "raw baseline matches existing QD-R1 summary", all(value <= 1e-6 for value in diffs.values()), diffs)


def check_predictions(rows: list[dict[str, Any]]) -> None:
    files = sorted(EXP07_CALIBRATION_PREDICTIONS_DIR.glob("*.csv"))
    if not files:
        add(rows, "calibrated predictions written", False)
        return
    for path in files:
        for row in read_csv(path):
            pred = _float(row.get("pred_label_calibrated"))
            if pred is None or pred < 1 or pred > 5:
                add(rows, "calibrated pred_label in 1-5", False, f"{path}: {pred}")
                return
    add(rows, "calibrated predictions written", True, [relpath(path) for path in files])
    add(rows, "calibrated pred_label in 1-5", True)


def check_no_training_api_or_weights(rows: list[dict[str, Any]]) -> None:
    output_files = [path for path in EXP07_CALIBRATION_OUTPUT_DIR.rglob("*") if path.is_file()]
    checkpoints = [relpath(path) for path in output_files if path.suffix.lower() in CHECKPOINT_SUFFIXES]
    add(rows, "no checkpoint/weight files in Exp7-C outputs", not checkpoints, checkpoints)
    tracked = tracked_weight_files()
    add(rows, "no tracked checkpoint/weights", not tracked, ", ".join(tracked))
    suspicious = []
    api_env_name = "OPENAI" + "_API_KEY"
    for path in output_files:
        if path.suffix.lower() not in {".py", ".md", ".csv", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if api_env_name in text or re.search(r"\bsk-[A-Za-z0-9_-]{20,}", text):
            suspicious.append(relpath(path))
    add(rows, "no API key strings in Exp7-C outputs", not suspicious, suspicious)
    add(rows, "no training command output generated", not (EXP07_CALIBRATION_OUTPUT_DIR / "checkpoints").exists())


def main() -> None:
    ensure_exp07_calibration_dirs()
    rows: list[dict[str, Any]] = []
    check_required_files(rows)
    check_inventory(rows)
    check_dev_only_selection(rows)
    check_thresholds(rows)
    check_metrics_finite(rows)
    check_raw_matches_existing(rows)
    check_predictions(rows)
    check_no_training_api_or_weights(rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "sanity_check_exp07_calibration.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp7-C Calibration Sanity Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP07_CALIBRATION_OUTPUT_DIR / "sanity_check_exp07_calibration.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp7-C calibration sanity failed. See {relpath(EXP07_CALIBRATION_OUTPUT_DIR)}")
    print("Exp7-C calibration sanity PASS")


if __name__ == "__main__":
    main()
