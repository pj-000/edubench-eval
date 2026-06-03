"""Compute derived Exp3 input-ablation comparison tables."""

from __future__ import annotations

import argparse
from typing import Any

from thesis_exp.src.edujudge.exp03 import EXP03_TABLES_DIR, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.collect_exp03_results import collect_exp03_results, float_or_none
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv


DELTA_METRICS = [
    "test_accuracy",
    "test_MAE_label",
    "test_MAE_expected",
    "test_signed_bias",
    "test_kendall_tau",
    "test_acc_at_1",
    "test_acc_at_2",
    "test_acc_at_5",
    "test_low_to_high_rate",
]


def as_float_row(row: dict[str, Any], key: str) -> float | None:
    return float_or_none(row.get(key))


def compute_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((row for row in summary_rows if row.get("template_name") == "A2_question_answer_metric"), None)
    if not baseline:
        return []
    out = []
    for row in summary_rows:
        result = {
            "template_name": row.get("template_name"),
            "ablation_id": row.get("ablation_id"),
            "status": row.get("status"),
            "baseline_template": "A2_question_answer_metric",
        }
        for metric in DELTA_METRICS:
            value = as_float_row(row, metric)
            base_value = as_float_row(baseline, metric)
            result[f"delta_{metric}_vs_a2"] = None if value is None or base_value is None else value - base_value
        out.append(result)
    return out


def metric_delta_rows() -> list[dict[str, Any]]:
    path = EXP03_TABLES_DIR / "input_ablation_metric_level.csv"
    if not path.exists():
        return []
    rows = [row for row in read_csv(path) if row.get("split") == "test"]
    baseline = {
        row.get("metric_canonical"): row
        for row in rows
        if row.get("template_name") == "A2_question_answer_metric" and row.get("metric_canonical")
    }
    out = []
    for row in rows:
        metric = row.get("metric_canonical")
        base = baseline.get(metric)
        if not base:
            continue
        mae = float_or_none(row.get("MAE_label"))
        base_mae = float_or_none(base.get("MAE_label"))
        exact = float_or_none(row.get("Exact Match"))
        base_exact = float_or_none(base.get("Exact Match"))
        out.append(
            {
                "template_name": row.get("template_name"),
                "ablation_id": row.get("ablation_id"),
                "metric_canonical": metric,
                "n": row.get("n"),
                "delta_MAE_label_vs_a2": None if mae is None or base_mae is None else mae - base_mae,
                "delta_exact_match_vs_a2": None if exact is None or base_exact is None else exact - base_exact,
            }
        )
    return out


def compute_input_ablation_metrics() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    summary_rows = collect_exp03_results()
    write_csv(EXP03_TABLES_DIR / "input_ablation_delta_vs_a2.csv", compute_deltas(summary_rows))
    write_csv(EXP03_TABLES_DIR / "input_ablation_metric_delta_vs_a2.csv", metric_delta_rows())
    low_rows = []
    for row in summary_rows:
        low_rows.append(
            {
                "template_name": row.get("template_name"),
                "ablation_id": row.get("ablation_id"),
                "status": row.get("status"),
                "test_acc_at_1": row.get("test_acc_at_1"),
                "test_acc_at_2": row.get("test_acc_at_2"),
                "test_low_to_high_rate": row.get("test_low_to_high_rate"),
                "test_low_signed_bias": row.get("test_low_signed_bias"),
            }
        )
    write_csv(EXP03_TABLES_DIR / "input_ablation_low_score_comparison.csv", low_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Exp3 derived comparison tables.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = compute_input_ablation_metrics()
    print(f"Exp3 derived metrics computed for {len(rows)} templates.")
    print(f"Output: {relpath(EXP03_TABLES_DIR / 'input_ablation_delta_vs_a2.csv')}")


if __name__ == "__main__":
    main()
