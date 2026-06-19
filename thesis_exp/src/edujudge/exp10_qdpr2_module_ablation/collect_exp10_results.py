"""Collect Exp10 QD-PR2 module ablation metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    ABLATION_LAMBDAS,
    ABLATION_ORDER,
    EXP10_OUTPUT_DIR,
    EXP10_REPORTS_DIR,
    EXP10_TABLES_DIR,
    ensure_exp10_dirs,
    exp10_run_dir,
    exp10_run_id,
)
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_text


QD_B1_SUMMARY = Path("thesis_exp/outputs/exp06_question_disjoint_baselines/tables/qd_baseline_metrics_summary.csv")
QD_B1_RUN_SUMMARY = Path(
    "thesis_exp/outputs/exp06_question_disjoint_baselines/runs/QD-B1_human_only_L1_weighted_ordinal/tables/metrics_summary.csv"
)
FULL_NAME = "full_qdpr2"
SUMMARY_FIELDS = [
    "ablation_name",
    "run_id",
    "status",
    "split",
    "n",
    "MAE",
    "QWK",
    "Accuracy",
    "Acc@5",
    "low_to_high_rate",
    "low_to_high_count",
    "true_low_n",
    "low_to_high_display",
    "label_1_low_to_high_rate",
    "label_1_low_to_high_count",
    "label_1_n",
    "label_2_low_to_high_rate",
    "label_2_low_to_high_count",
    "label_2_n",
    "low_pred_1_count",
    "low_pred_2_count",
    "low_pred_3_count",
    "low_pred_4_count",
    "low_pred_5_count",
    "monotonic_violation_rate",
    "monotonic_violation_count",
    "p1_lt_p2_rate",
    "p1_lt_p2_count",
    "p2_lt_p3_rate",
    "p2_lt_p3_count",
    "p3_lt_p4_rate",
    "p3_lt_p4_count",
    "lambda_point",
    "lambda_pair",
    "lambda_anchor",
    "lambda_mono",
]
EFFECT_FIELDS = [
    "comparison",
    "ablation_name",
    "metric",
    "reference_value",
    "ablation_value",
    "delta",
    "direction",
    "improved",
]
METRICS_FOR_DELTAS = [
    ("low_to_high_rate", "lower_better"),
    ("low_to_high_count", "lower_better"),
    ("MAE", "lower_better"),
    ("QWK", "higher_better"),
    ("Accuracy", "higher_better"),
    ("Acc@5", "higher_better"),
    ("monotonic_violation_rate", "lower_better"),
    ("p1_lt_p2_rate", "lower_better"),
    ("p2_lt_p3_rate", "lower_better"),
    ("p3_lt_p4_rate", "lower_better"),
]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _format_count_rate(count: int, total: int) -> str:
    rate = _safe_rate(count, total)
    return f"{count} / {total} = {rate:.4f}" if math.isfinite(rate) else f"{count} / {total} = nan"


def _read_metric_summary(run_dir: Path, split: str) -> dict[str, str]:
    path = run_dir / "tables" / "metrics_summary.csv"
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("split") == split:
            return row
    return {}


def _monotonic_counts(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    probs = np.array(
        [
            [
                _float(row.get("prob_gt_1")),
                _float(row.get("prob_gt_2")),
                _float(row.get("prob_gt_3")),
                _float(row.get("prob_gt_4")),
            ]
            for row in predictions
        ],
        dtype=float,
    )
    if probs.size == 0:
        return {
            "monotonic_violation_count": 0,
            "p1_lt_p2_count": 0,
            "p2_lt_p3_count": 0,
            "p3_lt_p4_count": 0,
        }
    pair_violations = probs[:, :-1] < probs[:, 1:]
    return {
        "monotonic_violation_count": int(np.any(pair_violations, axis=1).sum()),
        "p1_lt_p2_count": int(pair_violations[:, 0].sum()),
        "p2_lt_p3_count": int(pair_violations[:, 1].sum()),
        "p3_lt_p4_count": int(pair_violations[:, 2].sum()),
    }


def _row_from_predictions(ablation_name: str) -> dict[str, Any]:
    run_dir = exp10_run_dir(ablation_name)
    metrics = _read_metric_summary(run_dir, "test")
    pred_path = run_dir / "predictions_test.jsonl"
    lambdas = ABLATION_LAMBDAS[ablation_name]
    if not metrics or not pred_path.exists():
        return {
            "ablation_name": ablation_name,
            "run_id": exp10_run_id(ablation_name),
            "status": "missing",
            "split": "test",
            **lambdas,
        }
    predictions = read_jsonl(pred_path)
    y_true = [int(row["label_5"]) for row in predictions]
    y_pred = [int(row["pred_label_5"]) for row in predictions]
    low_indices = [idx for idx, label in enumerate(y_true) if label <= 2]
    low_to_high = [idx for idx in low_indices if y_pred[idx] >= 4]
    label_rows: dict[int, dict[str, int]] = {}
    for label in [1, 2]:
        indices = [idx for idx, value in enumerate(y_true) if value == label]
        count = sum(1 for idx in indices if y_pred[idx] >= 4)
        label_rows[label] = {"n": len(indices), "count": count}
    low_pred_counts = {label: 0 for label in [1, 2, 3, 4, 5]}
    for idx in low_indices:
        low_pred_counts[y_pred[idx]] += 1
    mono = _monotonic_counts(predictions)
    n = len(predictions)
    row = {
        "ablation_name": ablation_name,
        "run_id": exp10_run_id(ablation_name),
        "status": "completed",
        "split": "test",
        "n": n,
        "MAE": _float(metrics.get("MAE_label")),
        "QWK": _float(metrics.get("Quadratic Weighted Kappa")),
        "Accuracy": _float(metrics.get("Accuracy") or metrics.get("Exact Match")),
        "Acc@5": _float(metrics.get("Acc@5")),
        "low_to_high_rate": _safe_rate(len(low_to_high), len(low_indices)),
        "low_to_high_count": len(low_to_high),
        "true_low_n": len(low_indices),
        "low_to_high_display": _format_count_rate(len(low_to_high), len(low_indices)),
        "label_1_low_to_high_rate": _safe_rate(label_rows[1]["count"], label_rows[1]["n"]),
        "label_1_low_to_high_count": label_rows[1]["count"],
        "label_1_n": label_rows[1]["n"],
        "label_2_low_to_high_rate": _safe_rate(label_rows[2]["count"], label_rows[2]["n"]),
        "label_2_low_to_high_count": label_rows[2]["count"],
        "label_2_n": label_rows[2]["n"],
        "low_pred_1_count": low_pred_counts[1],
        "low_pred_2_count": low_pred_counts[2],
        "low_pred_3_count": low_pred_counts[3],
        "low_pred_4_count": low_pred_counts[4],
        "low_pred_5_count": low_pred_counts[5],
        "monotonic_violation_rate": _safe_rate(mono["monotonic_violation_count"], n),
        "monotonic_violation_count": mono["monotonic_violation_count"],
        "p1_lt_p2_rate": _safe_rate(mono["p1_lt_p2_count"], n),
        "p1_lt_p2_count": mono["p1_lt_p2_count"],
        "p2_lt_p3_rate": _safe_rate(mono["p2_lt_p3_count"], n),
        "p2_lt_p3_count": mono["p2_lt_p3_count"],
        "p3_lt_p4_rate": _safe_rate(mono["p3_lt_p4_count"], n),
        "p3_lt_p4_count": mono["p3_lt_p4_count"],
        **lambdas,
    }
    return row


def _baseline_row() -> dict[str, Any] | None:
    if QD_B1_RUN_SUMMARY.exists():
        for row in read_csv(QD_B1_RUN_SUMMARY):
            if row.get("split") != "test":
                continue
            low_n = _int(row.get("low_n")) or 31
            count = int(round(_float(row.get("low_to_high_rate")) * low_n))
            mono_count = int(round(_float(row.get("monotonic_violation_rate")) * _int(row.get("n"))))
            return {
                "ablation_name": "QD-B1_low_score_weighted_ordinal_baseline",
                "run_id": "QD-B1_human_only_L1_weighted_ordinal",
                "status": "completed",
                "split": "test",
                "n": _int(row.get("n")),
                "MAE": _float(row.get("MAE_label")),
                "QWK": _float(row.get("Quadratic Weighted Kappa")),
                "Accuracy": _float(row.get("Accuracy") or row.get("Exact Match")),
                "Acc@5": _float(row.get("Acc@5")),
                "low_to_high_rate": _float(row.get("low_to_high_rate")),
                "low_to_high_count": count,
                "true_low_n": low_n,
                "low_to_high_display": _format_count_rate(count, low_n),
                "monotonic_violation_rate": _float(row.get("monotonic_violation_rate")),
                "monotonic_violation_count": mono_count,
            }
    if not QD_B1_SUMMARY.exists():
        return None
    for row in read_csv(QD_B1_SUMMARY):
        if row.get("run_id") == "QD-B1_human_only_L1_weighted_ordinal" and row.get("split") == "test":
            low_n = 31
            count = int(round(_float(row.get("low_to_high_rate")) * low_n))
            return {
                "ablation_name": "QD-B1_low_score_weighted_ordinal_baseline",
                "run_id": "QD-B1_human_only_L1_weighted_ordinal",
                "status": "completed",
                "split": "test",
                "n": _int(row.get("n")),
                "MAE": _float(row.get("MAE_label")),
                "QWK": _float(row.get("Quadratic Weighted Kappa")),
                "Accuracy": _float(row.get("Exact Match")),
                "Acc@5": float("nan"),
                "low_to_high_rate": _float(row.get("low_to_high_rate")),
                "low_to_high_count": count,
                "true_low_n": low_n,
                "low_to_high_display": _format_count_rate(count, low_n),
            }
    return None


def _delta_rows(summary_rows: list[dict[str, Any]], baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
    completed = {row["ablation_name"]: row for row in summary_rows if row.get("status") == "completed"}
    rows: list[dict[str, Any]] = []

    def add_delta(comparison: str, ref: dict[str, Any], current: dict[str, Any]) -> None:
        for metric, direction in METRICS_FOR_DELTAS:
            ref_value = _float(ref.get(metric))
            value = _float(current.get(metric))
            delta = value - ref_value
            if direction == "lower_better":
                improved = delta < 0
            elif direction == "higher_better":
                improved = delta > 0
            else:
                improved = False
            rows.append(
                {
                    "comparison": comparison,
                    "ablation_name": current["ablation_name"],
                    "metric": metric,
                    "reference_value": ref_value,
                    "ablation_value": value,
                    "delta": delta,
                    "direction": direction,
                    "improved": str(bool(improved)).lower() if math.isfinite(delta) else "",
                }
            )

    full = completed.get(FULL_NAME)
    if full is not None:
        for name in ABLATION_ORDER:
            current = completed.get(name)
            if current is not None:
                add_delta("vs_full_qdpr2", full, current)
    if baseline is not None:
        for name in ABLATION_ORDER:
            current = completed.get(name)
            if current is not None:
                add_delta("vs_qd_b1_low_score_weighted_ordinal", baseline, current)
    return rows


def _write_report(summary_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]]) -> None:
    completed = [row for row in summary_rows if row.get("status") == "completed"]
    pending = [row["ablation_name"] for row in summary_rows if row.get("status") != "completed"]
    lines = [
        "# Exp10 QD-PR2 Module Ablation Summary",
        "",
        f"Completed runs: `{len(completed)}` / `{len(summary_rows)}`",
        f"Pending runs: `{', '.join(pending) if pending else 'none'}`",
        "",
        "This collector uses question-disjoint test predictions only. It does not train models and does not use test labels for model selection.",
        "",
        "| ablation | MAE | QWK | Accuracy | Acc@5 | low-to-high | monotonic | p1<p2 | p2<p3 | p3<p4 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row.get("status") != "completed":
            lines.append(f"| {row['ablation_name']} | pending | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        lines.append(
            "| {name} | {mae:.4f} | {qwk:.4f} | {acc:.4f} | {acc5:.4f} | {lth} | {mono:.4f} | {p12:.4f} | {p23:.4f} | {p34:.4f} |".format(
                name=row["ablation_name"],
                mae=_float(row.get("MAE")),
                qwk=_float(row.get("QWK")),
                acc=_float(row.get("Accuracy")),
                acc5=_float(row.get("Acc@5")),
                lth=row.get("low_to_high_display", ""),
                mono=_float(row.get("monotonic_violation_rate")),
                p12=_float(row.get("p1_lt_p2_rate")),
                p23=_float(row.get("p2_lt_p3_rate")),
                p34=_float(row.get("p3_lt_p4_rate")),
            )
        )
    lines.extend(
        [
            "",
            "Required output tables:",
            "",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_metrics_summary.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_module_effects.csv')}`",
        ]
    )
    write_text(EXP10_REPORTS_DIR / "exp10_ablation_summary.md", "\n".join(lines))


def collect() -> dict[str, Any]:
    ensure_exp10_dirs()
    summary_rows = [_row_from_predictions(name) for name in ABLATION_ORDER]
    baseline = _baseline_row()
    effect_rows = _delta_rows(summary_rows, baseline)
    write_csv(EXP10_TABLES_DIR / "exp10_ablation_metrics_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(EXP10_TABLES_DIR / "exp10_ablation_module_effects.csv", effect_rows, fieldnames=EFFECT_FIELDS)
    if baseline is not None:
        write_csv(EXP10_TABLES_DIR / "exp10_qdb1_reference_metrics.csv", [baseline])
    _write_report(summary_rows, effect_rows)
    return {
        "completed": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "total": len(summary_rows),
        "summary": relpath(EXP10_TABLES_DIR / "exp10_ablation_metrics_summary.csv"),
        "effects": relpath(EXP10_TABLES_DIR / "exp10_ablation_module_effects.csv"),
    }


def main() -> None:
    result = collect()
    print(result)


if __name__ == "__main__":
    main()
