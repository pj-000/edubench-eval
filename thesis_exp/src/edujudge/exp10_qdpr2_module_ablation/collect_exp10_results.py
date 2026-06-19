"""Collect Exp10 QD-PR2 module ablation metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    ABLATION_ORDER,
    EXP10_REPORTS_DIR,
    EXP10_TABLES_DIR,
    ablation_metadata,
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
STRICT_PAIR_NAME = "no_pair_same_pair_batches"
SUMMARY_FIELDS = [
    "ablation_name",
    "display_name",
    "run_id",
    "status",
    "split",
    "dataloader_mode",
    "force_pair_training",
    "use_pair_training",
    "strict_module_ablation",
    "removed_module",
    "active_losses",
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
    "interpretation",
]
EFFECT_FIELDS = [
    "comparison",
    "ablation_name",
    "display_name",
    "removed_module",
    "dataloader_mode",
    "force_pair_training",
    "strict_module_ablation",
    "metric",
    "reference_value",
    "ablation_value",
    "delta",
    "direction",
    "improved",
    "interpretation",
]
BASELINE_FIELDS = [
    "baseline_name",
    "ablation_name",
    "display_name",
    "metric",
    "baseline_value",
    "ablation_value",
    "delta",
    "direction",
    "improved",
]
LOW_DIST_FIELDS = ["ablation_name", "display_name", "status", "true_low_n", "pred_label", "count", "rate"]
LOW_BY_LABEL_FIELDS = [
    "ablation_name",
    "display_name",
    "status",
    "label_5",
    "n",
    "low_to_high_count",
    "low_to_high_rate",
]
MONO_FIELDS = [
    "ablation_name",
    "display_name",
    "status",
    "threshold_pair",
    "n",
    "violation_count",
    "violation_rate",
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


def _base_row(ablation_name: str, status: str) -> dict[str, Any]:
    meta = ablation_metadata(ablation_name)
    return {
        "ablation_name": ablation_name,
        "display_name": meta["display_name"],
        "run_id": exp10_run_id(ablation_name),
        "status": status,
        "split": "test",
        "dataloader_mode": meta["dataloader_mode"],
        "force_pair_training": meta["force_pair_training"],
        "use_pair_training": meta["use_pair_training"],
        "strict_module_ablation": meta["strict_module_ablation"],
        "removed_module": meta["removed_module"],
        "active_losses": meta["active_losses"],
        "lambda_point": meta["lambda_point"],
        "lambda_pair": meta["lambda_pair"],
        "lambda_anchor": meta["lambda_anchor"],
        "lambda_mono": meta["lambda_mono"],
        "interpretation": meta["interpretation"],
    }


def _row_from_predictions(ablation_name: str) -> dict[str, Any]:
    run_dir = exp10_run_dir(ablation_name)
    metrics = _read_metric_summary(run_dir, "test")
    pred_path = run_dir / "predictions_test.jsonl"
    if not metrics or not pred_path.exists():
        return _base_row(ablation_name, "missing")
    predictions = read_jsonl(pred_path)
    y_true = [int(row["label_5"]) for row in predictions]
    y_pred = [int(row.get("pred_label_5", row.get("pred_label"))) for row in predictions]
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
        **_base_row(ablation_name, "completed"),
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


def _delta(metric: str, direction: str, ref: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    ref_value = _float(ref.get(metric))
    value = _float(current.get(metric))
    delta = value - ref_value
    if direction == "lower_better":
        improved = delta < 0
    elif direction == "higher_better":
        improved = delta > 0
    else:
        improved = False
    return {
        "metric": metric,
        "reference_value": ref_value,
        "ablation_value": value,
        "delta": delta,
        "direction": direction,
        "improved": str(bool(improved)).lower() if math.isfinite(delta) else "",
    }


def _module_effect_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = {row["ablation_name"]: row for row in summary_rows if row.get("status") == "completed"}
    full = completed.get(FULL_NAME)
    if full is None:
        return []
    rows: list[dict[str, Any]] = []
    for name in ABLATION_ORDER:
        current = completed.get(name)
        if current is None:
            continue
        for metric, direction in METRICS_FOR_DELTAS:
            rows.append(
                {
                    "comparison": "vs_full_qdpr2",
                    "ablation_name": name,
                    "display_name": current.get("display_name"),
                    "removed_module": current.get("removed_module"),
                    "dataloader_mode": current.get("dataloader_mode"),
                    "force_pair_training": current.get("force_pair_training"),
                    "strict_module_ablation": current.get("strict_module_ablation"),
                    "interpretation": current.get("interpretation"),
                    **_delta(metric, direction, full, current),
                }
            )
    return rows


def _baseline_delta_rows(summary_rows: list[dict[str, Any]], baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for current in summary_rows:
        if current.get("status") != "completed":
            continue
        for metric, direction in METRICS_FOR_DELTAS:
            delta = _delta(metric, direction, baseline, current)
            rows.append(
                {
                    "baseline_name": "QD-B1_low_score_weighted_ordinal",
                    "ablation_name": current["ablation_name"],
                    "display_name": current.get("display_name"),
                    "metric": metric,
                    "baseline_value": delta["reference_value"],
                    "ablation_value": delta["ablation_value"],
                    "delta": delta["delta"],
                    "direction": delta["direction"],
                    "improved": delta["improved"],
                }
            )
    return rows


def _low_score_distribution_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        total = _int(row.get("true_low_n"))
        for pred_label in [1, 2, 3, 4, 5]:
            count = _int(row.get(f"low_pred_{pred_label}_count"))
            rows.append(
                {
                    "ablation_name": row["ablation_name"],
                    "display_name": row.get("display_name"),
                    "status": row.get("status"),
                    "true_low_n": total,
                    "pred_label": pred_label,
                    "count": count,
                    "rate": _safe_rate(count, total),
                }
            )
    return rows


def _low_to_high_by_label_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        for label in [1, 2]:
            n = _int(row.get(f"label_{label}_n"))
            count = _int(row.get(f"label_{label}_low_to_high_count"))
            rows.append(
                {
                    "ablation_name": row["ablation_name"],
                    "display_name": row.get("display_name"),
                    "status": row.get("status"),
                    "label_5": label,
                    "n": n,
                    "low_to_high_count": count,
                    "low_to_high_rate": _safe_rate(count, n),
                }
            )
    return rows


def _monotonic_by_threshold_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = [
        ("p1<p2", "p1_lt_p2_count", "p1_lt_p2_rate"),
        ("p2<p3", "p2_lt_p3_count", "p2_lt_p3_rate"),
        ("p3<p4", "p3_lt_p4_count", "p3_lt_p4_rate"),
    ]
    for row in summary_rows:
        n = _int(row.get("n"))
        for label, count_key, rate_key in fields:
            rows.append(
                {
                    "ablation_name": row["ablation_name"],
                    "display_name": row.get("display_name"),
                    "status": row.get("status"),
                    "threshold_pair": label,
                    "n": n,
                    "violation_count": _int(row.get(count_key)),
                    "violation_rate": _float(row.get(rate_key)),
                }
            )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "pending"


def _write_report(summary_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]]) -> None:
    completed = [row for row in summary_rows if row.get("status") == "completed"]
    pending = [row["ablation_name"] for row in summary_rows if row.get("status") != "completed"]
    strict_row = next((row for row in summary_rows if row["ablation_name"] == STRICT_PAIR_NAME), None)
    no_pair_row = next((row for row in summary_rows if row["ablation_name"] == "no_pair"), None)
    lines = [
        "# Exp10 QD-PR2 Module Ablation Report",
        "",
        f"Completed runs: `{len(completed)}` / `{len(summary_rows)}`",
        f"Pending runs: `{', '.join(pending) if pending else 'none'}`",
        "",
        "This collector uses question-disjoint test predictions only. It does not train models and does not use test labels for model selection.",
        "",
        "## Ablation Matrix",
        "",
        "| ablation | display | loader | force pair | strict | active losses | low-to-high | MAE | QWK |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {display} | {loader} | {force} | {strict} | {losses} | {lth} | {mae} | {qwk} |".format(
                name=row["ablation_name"],
                display=row.get("display_name", ""),
                loader=row.get("dataloader_mode", ""),
                force=row.get("force_pair_training", ""),
                strict=row.get("strict_module_ablation", ""),
                losses=row.get("active_losses", ""),
                lth=row.get("low_to_high_display", "pending") if row.get("status") == "completed" else "pending",
                mae=_fmt(row.get("MAE")),
                qwk=_fmt(row.get("QWK")),
            )
        )
    lines.extend(
        [
            "",
            "## Strict L_pair Ablation with Same Pair Batches",
            "",
            "The strict run `no_pair_same_pair_batches` is the main evidence for the L_pair module because it keeps pair-batch exposure while setting `lambda_pair=0`. The older `no_pair` run is auxiliary diagnostic evidence because it removes L_pair and uses the pointwise loader.",
            "",
            "Interpretation should stay cautious: results can suggest effects under identical pair-batch exposure, and the low-score test subset is small, so count-level interpretation is necessary.",
            "",
        ]
    )
    if strict_row is not None and no_pair_row is not None:
        lines.extend(
            [
                "| run | loader | force_pair_training | lambda_pair | low-to-high | MAE |",
                "| --- | --- | ---: | ---: | --- | ---: |",
                "| no_pair | {loader} | {force} | {pair} | {lth} | {mae} |".format(
                    loader=no_pair_row.get("dataloader_mode"),
                    force=no_pair_row.get("force_pair_training"),
                    pair=no_pair_row.get("lambda_pair"),
                    lth=no_pair_row.get("low_to_high_display", "pending"),
                    mae=_fmt(no_pair_row.get("MAE")),
                ),
                "| no_pair_same_pair_batches | {loader} | {force} | {pair} | {lth} | {mae} |".format(
                    loader=strict_row.get("dataloader_mode"),
                    force=strict_row.get("force_pair_training"),
                    pair=strict_row.get("lambda_pair"),
                    lth=strict_row.get("low_to_high_display", "pending"),
                    mae=_fmt(strict_row.get("MAE")),
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Required Tables",
            "",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_metrics_summary.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_module_effects.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_low_score_distribution.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_low_to_high_by_label.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_monotonic_by_threshold.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_vs_existing_baselines.csv')}`",
            "",
            "## Module Effects",
            "",
            "The module-effects table distinguishes `no_pair` from `no_pair_same_pair_batches` through `dataloader_mode`, `force_pair_training`, and the interpretation column.",
        ]
    )
    if effect_rows:
        low_rows = [row for row in effect_rows if row["metric"] == "low_to_high_rate"]
        lines.extend(["", "| ablation | metric | delta vs full | interpretation |", "| --- | --- | ---: | --- |"])
        for row in low_rows:
            lines.append(
                f"| {row['ablation_name']} | {row['metric']} | {_fmt(row.get('delta'))} | {row.get('interpretation', '')} |"
            )
    text = "\n".join(lines)
    write_text(EXP10_REPORTS_DIR / "exp10_ablation_summary.md", text)
    write_text(EXP10_REPORTS_DIR / "exp10_qdpr2_module_ablation_report.md", text)


def _write_review_package(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp10 QD-PR2 Module Ablation Review Package",
        "",
        "## What Changed",
        "",
        "- Formal matrix now contains seven runs, including `no_pair_same_pair_batches`.",
        "- `no_pair` is retained as `no_pair_pointwise_loader` auxiliary diagnostic evidence.",
        "- `no_pair_same_pair_batches` is the strict L_pair ablation: pair loader is used, `force_pair_training=true`, and `lambda_pair=0`.",
        "",
        "## Expected Run Order",
        "",
    ]
    for idx, name in enumerate(ABLATION_ORDER, start=1):
        meta = ablation_metadata(name)
        lines.append(
            f"{idx}. `{name}`: loader `{meta['dataloader_mode']}`, active losses `{meta['active_losses']}`, strict `{meta['strict_module_ablation']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "The strict run is the main L_pair evidence under identical pair-batch exposure. The older `no_pair` run is useful for diagnosing how the pointwise-loader path behaves, but it should not be treated as the strict module comparison.",
            "",
            "Use cautious wording in the thesis discussion: the table can suggest module effects, while the low-score test subset is small and count-level interpretation is necessary.",
            "",
            "## Output Tables",
            "",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_metrics_summary.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_module_effects.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_low_score_distribution.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_low_to_high_by_label.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_monotonic_by_threshold.csv')}`",
            f"- `{relpath(EXP10_TABLES_DIR / 'exp10_ablation_vs_existing_baselines.csv')}`",
            "",
            "## Current Completion",
            "",
            "| ablation | status | loader | force pair | strict |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['ablation_name']} | {row.get('status')} | {row.get('dataloader_mode')} | {row.get('force_pair_training')} | {row.get('strict_module_ablation')} |"
        )
    write_text(EXP10_REPORTS_DIR / "exp10_qdpr2_module_ablation_review_package.md", "\n".join(lines))


def collect() -> dict[str, Any]:
    ensure_exp10_dirs()
    summary_rows = [_row_from_predictions(name) for name in ABLATION_ORDER]
    baseline = _baseline_row()
    effect_rows = _module_effect_rows(summary_rows)
    baseline_rows = _baseline_delta_rows(summary_rows, baseline)
    write_csv(EXP10_TABLES_DIR / "exp10_ablation_metrics_summary.csv", summary_rows, fieldnames=SUMMARY_FIELDS)
    write_csv(EXP10_TABLES_DIR / "exp10_ablation_module_effects.csv", effect_rows, fieldnames=EFFECT_FIELDS)
    write_csv(
        EXP10_TABLES_DIR / "exp10_ablation_low_score_distribution.csv",
        _low_score_distribution_rows(summary_rows),
        fieldnames=LOW_DIST_FIELDS,
    )
    write_csv(
        EXP10_TABLES_DIR / "exp10_ablation_low_to_high_by_label.csv",
        _low_to_high_by_label_rows(summary_rows),
        fieldnames=LOW_BY_LABEL_FIELDS,
    )
    write_csv(
        EXP10_TABLES_DIR / "exp10_ablation_monotonic_by_threshold.csv",
        _monotonic_by_threshold_rows(summary_rows),
        fieldnames=MONO_FIELDS,
    )
    write_csv(
        EXP10_TABLES_DIR / "exp10_ablation_vs_existing_baselines.csv",
        baseline_rows,
        fieldnames=BASELINE_FIELDS,
    )
    if baseline is not None:
        write_csv(EXP10_TABLES_DIR / "exp10_qdb1_reference_metrics.csv", [baseline])
    _write_report(summary_rows, effect_rows)
    _write_review_package(summary_rows)
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
