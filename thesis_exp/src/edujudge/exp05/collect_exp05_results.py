"""Collect Exp5 L0/L1/L2 loss-ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import (
    EXP04_TABLES_DIR,
    EXP05_TABLES_DIR,
    L1_RUN_ID,
    L2_RUN_CONFIGS,
    ensure_exp05_dirs,
    l1_run_dir,
    l2_run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_row(rows: list[dict[str, Any]], **filters: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in filters.items()):
            return row
    return {}


def l0_summary_row() -> dict[str, Any]:
    summary_path = EXP04_TABLES_DIR / "target_objective_summary.csv"
    row = first_row(read_csv(summary_path), objective_id="O3_ordinal") if summary_path.exists() else {}
    return {
        "loss_id": "L0_exp04_o3_ordinal",
        "source": "Exp4 O3 ordinal",
        "status": row.get("status", "missing"),
        "is_retrained": "NO",
        "test_accuracy": float_or_none(row.get("test_accuracy")),
        "test_MAE_label": float_or_none(row.get("test_MAE_label")),
        "test_MAE_expected": float_or_none(row.get("test_MAE_expected")),
        "test_qwk": float_or_none(row.get("test_qwk")),
        "test_kendall_tau": float_or_none(row.get("test_kendall_tau")),
        "test_severe_error_rate": float_or_none(row.get("test_severe_error_rate")),
        "test_low_to_high_rate": float_or_none(row.get("test_low_to_high_rate")),
        "test_low_signed_bias": float_or_none(row.get("test_low_signed_bias")),
        "test_monotonic_violation_rate": float_or_none(row.get("test_monotonic_violation_rate")),
        "run_dir": row.get("run_dir", "thesis_exp/outputs/exp04_target_objectives/runs/O3_ordinal"),
    }


def l1_summary_row() -> dict[str, Any]:
    run_path = l1_run_dir(smoke=False)
    metadata = load_json(run_path / "run_metadata.json")
    metrics_path = run_path / "tables" / "metrics_summary.csv"
    test = first_row(read_csv(metrics_path), split="test") if metrics_path.exists() else {}
    status = metadata.get("status") or ("completed" if test else "pending")
    return {
        "loss_id": L1_RUN_ID,
        "source": "Exp5 L1 weighted ordinal",
        "status": status,
        "is_retrained": "YES",
        "test_accuracy": float_or_none(test.get("Accuracy") or test.get("Exact Match")),
        "test_MAE_label": float_or_none(test.get("MAE_label")),
        "test_MAE_expected": float_or_none(test.get("MAE_expected")),
        "test_qwk": float_or_none(test.get("Quadratic Weighted Kappa")),
        "test_kendall_tau": float_or_none(test.get("Kendall tau")),
        "test_severe_error_rate": float_or_none(test.get("severe_error_rate")),
        "test_low_to_high_rate": float_or_none(test.get("low_to_high_rate")),
        "test_low_signed_bias": float_or_none(test.get("low_signed_bias")),
        "test_monotonic_violation_rate": float_or_none(test.get("monotonic_violation_rate")),
        "run_dir": relpath(run_path),
    }


def run_summary_row(run_id: str, source: str, run_path: Path) -> dict[str, Any]:
    metadata = load_json(run_path / "run_metadata.json")
    metrics_path = run_path / "tables" / "metrics_summary.csv"
    test = first_row(read_csv(metrics_path), split="test") if metrics_path.exists() else {}
    status = metadata.get("status") or ("completed" if test else "pending")
    return {
        "loss_id": run_id,
        "source": source,
        "status": status,
        "is_retrained": "YES",
        "test_accuracy": float_or_none(test.get("Accuracy") or test.get("Exact Match")),
        "test_MAE_label": float_or_none(test.get("MAE_label")),
        "test_MAE_expected": float_or_none(test.get("MAE_expected")),
        "test_qwk": float_or_none(test.get("Quadratic Weighted Kappa")),
        "test_kendall_tau": float_or_none(test.get("Kendall tau")),
        "test_severe_error_rate": float_or_none(test.get("severe_error_rate")),
        "test_low_to_high_rate": float_or_none(test.get("low_to_high_rate")),
        "test_low_signed_bias": float_or_none(test.get("low_signed_bias")),
        "test_monotonic_violation_rate": float_or_none(test.get("monotonic_violation_rate")),
        "run_dir": relpath(run_path),
    }


def l2_summary_rows() -> list[dict[str, Any]]:
    return [
        run_summary_row(run_id, f"Exp5 {run_id}", l2_run_dir(run_id, smoke=False))
        for run_id in L2_RUN_CONFIGS
    ]


def collect_table_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    low_rows: list[dict[str, Any]] = []
    per_bin_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []

    exp04_low = EXP04_TABLES_DIR / "target_objective_low_score.csv"
    exp04_per_bin = EXP04_TABLES_DIR / "target_objective_per_bin.csv"
    exp04_high = EXP04_TABLES_DIR / "target_objective_high_score.csv"
    for row in read_csv(exp04_low) if exp04_low.exists() else []:
        if row.get("objective_id") == "O3_ordinal" and row.get("split") == "test":
            low_rows.append({"loss_id": "L0_exp04_o3_ordinal", **row})
    for row in read_csv(exp04_per_bin) if exp04_per_bin.exists() else []:
        if row.get("objective_id") == "O3_ordinal" and row.get("split") == "test":
            per_bin_rows.append({"loss_id": "L0_exp04_o3_ordinal", **row})
    for row in read_csv(exp04_high) if exp04_high.exists() else []:
        if row.get("objective_id") == "O3_ordinal" and row.get("split") == "test":
            high_rows.append({"loss_id": "L0_exp04_o3_ordinal", **row})

    run_path = l1_run_dir(smoke=False)
    for name, target in [
        ("low_score_metrics.csv", low_rows),
        ("per_bin_metrics.csv", per_bin_rows),
        ("high_score_metrics.csv", high_rows),
    ]:
        path = run_path / "tables" / name
        for row in read_csv(path) if path.exists() else []:
            if row.get("split") == "test":
                target.append({"loss_id": L1_RUN_ID, **row})
    for run_id in L2_RUN_CONFIGS:
        run_path = l2_run_dir(run_id, smoke=False)
        for name, target in [
            ("low_score_metrics.csv", low_rows),
            ("per_bin_metrics.csv", per_bin_rows),
            ("high_score_metrics.csv", high_rows),
        ]:
            path = run_path / "tables" / name
            for row in read_csv(path) if path.exists() else []:
                if row.get("split") == "test":
                    target.append({"loss_id": run_id, **row})
    return low_rows, per_bin_rows, high_rows


def make_delta_row(
    summary_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    baseline_loss_id: str,
    comparison_loss_id: str,
) -> dict[str, Any]:
    baseline = first_row(summary_rows, loss_id=baseline_loss_id)
    comparison = first_row(summary_rows, loss_id=comparison_loss_id)
    baseline_low = first_row(low_rows, loss_id=baseline_loss_id)
    comparison_low = first_row(low_rows, loss_id=comparison_loss_id)
    baseline_high = first_row(high_rows, loss_id=baseline_loss_id)
    comparison_high = first_row(high_rows, loss_id=comparison_loss_id)

    def delta(a: Any, b: Any) -> float | None:
        left = float_or_none(a)
        right = float_or_none(b)
        if left is None or right is None:
            return None
        return right - left

    return {
        "baseline_loss_id": baseline_loss_id,
        "comparison_loss_id": comparison_loss_id,
        "delta_accuracy": delta(baseline.get("test_accuracy"), comparison.get("test_accuracy")),
        "delta_MAE_label": delta(baseline.get("test_MAE_label"), comparison.get("test_MAE_label")),
        "delta_MAE_expected": delta(baseline.get("test_MAE_expected"), comparison.get("test_MAE_expected")),
        "delta_kendall_tau": delta(baseline.get("test_kendall_tau"), comparison.get("test_kendall_tau")),
        "delta_qwk": delta(baseline.get("test_qwk"), comparison.get("test_qwk")),
        "delta_low_to_high_rate": delta(baseline_low.get("low_to_high_rate"), comparison_low.get("low_to_high_rate")),
        "delta_low_signed_bias": delta(baseline_low.get("low_signed_bias"), comparison_low.get("low_signed_bias")),
        "delta_Acc@1": delta(baseline_low.get("Acc@1"), comparison_low.get("Acc@1")),
        "delta_Acc@2": delta(baseline_low.get("Acc@2"), comparison_low.get("Acc@2")),
        "delta_Acc@5": delta(baseline_high.get("Acc@5"), comparison_high.get("Acc@5")),
        "delta_high_to_mid_or_low_rate": delta(
            baseline_high.get("high_to_mid_or_low_rate"),
            comparison_high.get("high_to_mid_or_low_rate"),
        ),
    }


def delta_rows(
    summary_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    baseline_loss_id: str,
) -> list[dict[str, Any]]:
    return [
        make_delta_row(summary_rows, low_rows, high_rows, baseline_loss_id, row["loss_id"])
        for row in summary_rows
        if row["loss_id"] != baseline_loss_id
    ]


def tradeoff_rows(
    summary_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        low = first_row(low_rows, loss_id=row["loss_id"])
        high = first_row(high_rows, loss_id=row["loss_id"])
        rows.append(
            {
                "method": row["loss_id"],
                "status": row.get("status"),
                "accuracy": row.get("test_accuracy"),
                "kendall_tau": row.get("test_kendall_tau"),
                "qwk": row.get("test_qwk"),
                "MAE_label": row.get("test_MAE_label"),
                "low_to_high_rate": row.get("test_low_to_high_rate"),
                "low_signed_bias": row.get("test_low_signed_bias"),
                "Acc@1": float_or_none(low.get("Acc@1")),
                "Acc@2": float_or_none(low.get("Acc@2")),
                "Acc@5": float_or_none(high.get("Acc@5")),
                "high_to_mid_or_low_rate": float_or_none(high.get("high_to_mid_or_low_rate")),
                "severe_error_rate": row.get("test_severe_error_rate"),
                "is_pareto_candidate": row.get("status") == "completed",
            }
        )
    return rows


def collect_exp05_results() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    summary_rows = [l0_summary_row(), l1_summary_row(), *l2_summary_rows()]
    low_rows, per_bin_rows, high_rows = collect_table_rows()
    write_csv(EXP05_TABLES_DIR / "loss_ablation_summary.csv", summary_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_low_score.csv", low_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_per_bin.csv", per_bin_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_high_score.csv", high_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_delta_vs_L0.csv", delta_rows(summary_rows, low_rows, high_rows, "L0_exp04_o3_ordinal"))
    write_csv(
        EXP05_TABLES_DIR / "loss_ablation_delta_vs_L1.csv",
        [
            make_delta_row(summary_rows, low_rows, high_rows, L1_RUN_ID, run_id)
            for run_id in L2_RUN_CONFIGS
        ],
    )
    write_csv(EXP05_TABLES_DIR / "loss_ablation_tradeoff.csv", tradeoff_rows(summary_rows, low_rows, high_rows))
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp5 L0/L1 result tables.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = collect_exp05_results()
    print(f"Exp5 collected losses: {[row['loss_id'] for row in rows]}")
    print(f"Output: {relpath(EXP05_TABLES_DIR / 'loss_ablation_summary.csv')}")


if __name__ == "__main__":
    main()
