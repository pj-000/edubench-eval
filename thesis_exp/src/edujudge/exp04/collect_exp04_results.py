"""Collect available Exp4 objective-comparison outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04 import (
    EXP04_ARRAYS_DIR,
    EXP04_PREDICTIONS_DIR,
    EXP04_TABLES_DIR,
    OBJECTIVE_IDS,
    ensure_exp04_dirs,
    run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_row(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("split") == split), {})


def metadata(run_path: Path) -> dict[str, Any]:
    path = run_path / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_status(run_path: Path) -> str:
    meta = metadata(run_path)
    if meta.get("status"):
        return str(meta["status"])
    if (run_path / "tables" / "metrics_summary.csv").exists():
        return "completed"
    return "pending"


def copy_available_artifacts(objective_id: str, run_path: Path) -> None:
    for split in ["dev", "test"]:
        src = run_path / "predictions" / f"predictions_{split}.jsonl"
        if src.exists():
            dst = EXP04_PREDICTIONS_DIR / f"{objective_id}_predictions_{split}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    arrays_src = run_path / "arrays" / "dev_test_arrays.npz"
    if arrays_src.exists():
        arrays_dst = EXP04_ARRAYS_DIR / f"{objective_id}_dev_test_arrays.npz"
        arrays_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arrays_src, arrays_dst)


def collect_table(objective_id: str, status: str, run_path: Path, table_name: str) -> list[dict[str, Any]]:
    path = run_path / "tables" / table_name
    if not path.exists():
        return []
    meta = metadata(run_path)
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "objective_id": objective_id,
                "objective_type": meta.get("objective_type", ""),
                "objective_label": meta.get("objective_label", objective_id),
                "status": status,
                **row,
            }
        )
    return rows


def summarize_objective(objective_id: str) -> dict[str, Any]:
    path = run_dir(objective_id)
    status = run_status(path)
    meta = metadata(path)
    metrics_path = path / "tables" / "metrics_summary.csv"
    if not metrics_path.exists():
        return {
            "objective_id": objective_id,
            "objective_type": meta.get("objective_type", ""),
            "objective_label": meta.get("objective_label", objective_id),
            "status": status,
            "loss": meta.get("loss", ""),
            "checkpoint_selection": meta.get("checkpoint_selection", ""),
            "test_accuracy": None,
            "test_MAE_label": None,
            "test_MAE_expected": None,
            "test_MAE_score": None,
            "test_signed_bias_label": None,
            "test_qwk": None,
            "test_kendall_tau": None,
            "test_severe_error_rate": None,
            "test_low_to_high_rate": None,
            "test_low_signed_bias": None,
            "test_monotonic_violation_rate": None,
            "run_dir": relpath(path),
        }

    copy_available_artifacts(objective_id, path)
    metrics = read_csv(metrics_path)
    test = first_row(metrics, "test")
    return {
        "objective_id": objective_id,
        "objective_type": meta.get("objective_type", ""),
        "objective_label": meta.get("objective_label", objective_id),
        "status": status,
        "loss": meta.get("loss", ""),
        "checkpoint_selection": meta.get("checkpoint_selection", ""),
        "test_accuracy": float_or_none(test.get("Accuracy") or test.get("Exact Match")),
        "test_MAE_label": float_or_none(test.get("MAE_label")),
        "test_MAE_expected": float_or_none(test.get("MAE_expected")),
        "test_MAE_score": float_or_none(test.get("MAE_score")),
        "test_signed_bias_label": float_or_none(test.get("Signed Bias label")),
        "test_qwk": float_or_none(test.get("Quadratic Weighted Kappa")),
        "test_kendall_tau": float_or_none(test.get("Kendall tau")),
        "test_severe_error_rate": float_or_none(test.get("severe_error_rate")),
        "test_low_to_high_rate": float_or_none(test.get("low_to_high_rate")),
        "test_low_signed_bias": float_or_none(test.get("low_signed_bias")),
        "test_monotonic_violation_rate": float_or_none(test.get("monotonic_violation_rate")),
        "run_dir": relpath(path),
    }


def collect_exp04_results() -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    summary_rows = []
    per_bin_rows: list[dict[str, Any]] = []
    low_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for objective_id in OBJECTIVE_IDS:
        path = run_dir(objective_id)
        summary = summarize_objective(objective_id)
        summary_rows.append(summary)
        status = str(summary["status"])
        if status in {"completed", "reused_exp03_a4", "eval_only"}:
            low_rows.extend(collect_table(objective_id, status, path, "low_score_metrics.csv"))
            high_rows.extend(collect_table(objective_id, status, path, "high_score_metrics.csv"))
            per_bin_rows.extend(collect_table(objective_id, status, path, "per_bin_metrics.csv"))
            metric_rows.extend(collect_table(objective_id, status, path, "metric_level_metrics.csv"))
            scenario_rows.extend(collect_table(objective_id, status, path, "scenario_level_metrics.csv"))
            distribution_rows.extend(collect_table(objective_id, status, path, "score_distribution.csv"))

    write_csv(EXP04_TABLES_DIR / "target_objective_summary.csv", summary_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_low_score.csv", low_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_high_score.csv", high_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_per_bin.csv", per_bin_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_metric_level.csv", metric_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_scenario_level.csv", scenario_rows)
    write_csv(EXP04_TABLES_DIR / "target_objective_score_distribution.csv", distribution_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp4 target objective outputs.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = collect_exp04_results()
    statuses = {row["objective_id"]: row["status"] for row in rows}
    print(f"Exp4 collected objectives: {statuses}")
    print(f"Output: {relpath(EXP04_TABLES_DIR / 'target_objective_summary.csv')}")


if __name__ == "__main__":
    main()
