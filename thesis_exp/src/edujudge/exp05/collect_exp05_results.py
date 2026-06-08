"""Collect Exp5 L0/L1 loss-ablation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import (
    EXP04_TABLES_DIR,
    EXP05_TABLES_DIR,
    L1_RUN_ID,
    ensure_exp05_dirs,
    l1_run_dir,
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
    return low_rows, per_bin_rows, high_rows


def delta_row(summary_rows: list[dict[str, Any]], low_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]]) -> dict[str, Any]:
    l0 = first_row(summary_rows, loss_id="L0_exp04_o3_ordinal")
    l1 = first_row(summary_rows, loss_id=L1_RUN_ID)
    l0_low = first_row(low_rows, loss_id="L0_exp04_o3_ordinal")
    l1_low = first_row(low_rows, loss_id=L1_RUN_ID)
    l0_high = first_row(high_rows, loss_id="L0_exp04_o3_ordinal")
    l1_high = first_row(high_rows, loss_id=L1_RUN_ID)

    def delta(a: Any, b: Any) -> float | None:
        left = float_or_none(a)
        right = float_or_none(b)
        if left is None or right is None:
            return None
        return right - left

    return {
        "baseline_loss_id": "L0_exp04_o3_ordinal",
        "comparison_loss_id": L1_RUN_ID,
        "delta_accuracy": delta(l0.get("test_accuracy"), l1.get("test_accuracy")),
        "delta_MAE_label": delta(l0.get("test_MAE_label"), l1.get("test_MAE_label")),
        "delta_MAE_expected": delta(l0.get("test_MAE_expected"), l1.get("test_MAE_expected")),
        "delta_kendall_tau": delta(l0.get("test_kendall_tau"), l1.get("test_kendall_tau")),
        "delta_qwk": delta(l0.get("test_qwk"), l1.get("test_qwk")),
        "delta_low_to_high_rate": delta(l0_low.get("low_to_high_rate"), l1_low.get("low_to_high_rate")),
        "delta_low_signed_bias": delta(l0_low.get("low_signed_bias"), l1_low.get("low_signed_bias")),
        "delta_Acc@1": delta(l0_low.get("Acc@1"), l1_low.get("Acc@1")),
        "delta_Acc@2": delta(l0_low.get("Acc@2"), l1_low.get("Acc@2")),
        "delta_Acc@5": delta(l0_high.get("Acc@5"), l1_high.get("Acc@5")),
        "delta_high_to_mid_or_low_rate": delta(
            l0_high.get("high_to_mid_or_low_rate"),
            l1_high.get("high_to_mid_or_low_rate"),
        ),
    }


def collect_exp05_results() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    summary_rows = [l0_summary_row(), l1_summary_row()]
    low_rows, per_bin_rows, high_rows = collect_table_rows()
    write_csv(EXP05_TABLES_DIR / "loss_ablation_summary.csv", summary_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_low_score.csv", low_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_per_bin.csv", per_bin_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_high_score.csv", high_rows)
    write_csv(EXP05_TABLES_DIR / "loss_ablation_delta_vs_L0.csv", [delta_row(summary_rows, low_rows, high_rows)])
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
