"""Run Exp15 post-hoc ordinal calibration diagnosis."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration import (
    DEFAULT_ECE_BINS,
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    EXP15_REPORTS_DIR,
    EXP15_TABLES_DIR,
    ensure_exp15_dirs,
)
from thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration.calibration import (
    calibrated_probs,
    candidate_specs,
    discover_array_sources,
    fit_isotonic_thresholds,
    load_ordinal_arrays,
    metric_row,
    prediction_distribution_rows,
    raw_probs_from_spec,
    select_dev_row,
    source_inventory_rows,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


DEV_FIELDS = [
    "source_id",
    "source_family",
    "run_name",
    "seed",
    "calibrator_id",
    "calibrator_family",
    "description",
    "decode_thresholds",
    "selected",
    "eligible",
    "selection_rule",
    "selection_delta",
    "uses_test_for_selection",
    "n",
    "MAE_label",
    "MAE_expected",
    "QWK",
    "Accuracy",
    "Acc@5",
    "low_to_high",
    "low_to_high_count",
    "true_low_score_count",
    "label1_low_to_high",
    "label1_low_to_high_count",
    "label2_low_to_high",
    "label2_low_to_high_count",
    "high_to_low",
    "high_to_low_count",
    "label4_recall",
    "label5_recall",
    "monotonic_violation",
    "threshold_brier",
    "threshold_ece",
    "p_gt_3_low_mean",
    "p_gt_3_label1_mean",
    "p_gt_3_label2_mean",
    "p_gt_3_low_q90",
    "p_gt_3_low_q95",
    "low_count_with_p_gt_3_over_0p5",
    "baseline_low_to_high_count",
    "delta_low_to_high_count_vs_pava",
    "baseline_MAE_label",
    "delta_MAE_label_vs_pava",
]
TEST_FIELDS = [
    "source_id",
    "source_family",
    "run_name",
    "seed",
    "calibrator_id",
    "calibrator_family",
    "description",
    "decode_thresholds",
    "selected_by_dev",
    "selection_rule",
    "selection_delta",
    "uses_test_for_selection",
    "n",
    "test_MAE_label",
    "test_MAE_expected",
    "test_QWK",
    "test_Accuracy",
    "test_Acc@5",
    "test_low_to_high",
    "test_low_to_high_count",
    "test_true_low_score_count",
    "test_label1_low_to_high",
    "test_label1_low_to_high_count",
    "test_label1_count",
    "test_label2_low_to_high",
    "test_label2_low_to_high_count",
    "test_label2_count",
    "test_high_to_low",
    "test_high_to_low_count",
    "test_label4_recall",
    "test_label5_recall",
    "test_monotonic_violation",
    "test_threshold_brier",
    "test_threshold_ece",
    "test_p_gt_3_low_mean",
    "test_p_gt_3_label1_mean",
    "test_p_gt_3_label2_mean",
    "test_p_gt_3_low_q90",
    "test_p_gt_3_low_q95",
    "test_low_count_with_p_gt_3_over_0p5",
]


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _test_row(row: dict[str, Any], selected_by_dev: bool, delta: float) -> dict[str, Any]:
    out = {
        "source_id": row.get("source_id"),
        "source_family": row.get("source_family"),
        "run_name": row.get("run_name"),
        "seed": row.get("seed"),
        "calibrator_id": row.get("calibrator_id"),
        "calibrator_family": row.get("calibrator_family"),
        "description": row.get("description"),
        "decode_thresholds": row.get("decode_thresholds"),
        "selected_by_dev": selected_by_dev,
        "selection_rule": DEFAULT_SELECTION_RULE,
        "selection_delta": delta,
        "uses_test_for_selection": False,
        "n": row.get("n"),
    }
    for key in [
        "MAE_label",
        "MAE_expected",
        "QWK",
        "Accuracy",
        "Acc@5",
        "low_to_high",
        "low_to_high_count",
        "true_low_score_count",
        "label1_low_to_high",
        "label1_low_to_high_count",
        "label1_count",
        "label2_low_to_high",
        "label2_low_to_high_count",
        "label2_count",
        "high_to_low",
        "high_to_low_count",
        "label4_recall",
        "label5_recall",
        "monotonic_violation",
        "threshold_brier",
        "threshold_ece",
        "p_gt_3_low_mean",
        "p_gt_3_label1_mean",
        "p_gt_3_label2_mean",
        "p_gt_3_low_q90",
        "p_gt_3_low_q95",
        "low_count_with_p_gt_3_over_0p5",
    ]:
        out[f"test_{key}"] = row.get(key)
    return out


def run(delta: float, ece_bins: int) -> dict[str, Any]:
    ensure_exp15_dirs()
    sources = discover_array_sources()
    inventory = source_inventory_rows(sources)
    write_csv(EXP15_TABLES_DIR / "exp15_source_inventory.csv", inventory)

    dev_ranking: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    test_diagnostic: list[dict[str, Any]] = []
    selected_test: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for source in sources:
        try:
            arrays = load_ordinal_arrays(source.arrays_dir)
        except Exception as exc:
            errors.append(
                {
                    "source_id": source.source_id,
                    "arrays_dir": relpath(source.arrays_dir),
                    "status": "SKIPPED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        specs = candidate_specs()
        isotonic_models = fit_isotonic_thresholds(raw_probs_from_spec(arrays.logits_dev, specs[1]), arrays.labels_dev)
        dev_rows_by_source: list[dict[str, Any]] = []
        test_rows_by_calibrator: dict[str, dict[str, Any]] = {}

        for spec in specs:
            models = isotonic_models if spec.use_isotonic else None
            dev_probs = calibrated_probs(arrays.logits_dev, spec, models)
            test_probs = calibrated_probs(arrays.logits_test, spec, models)
            dev_row = metric_row(source, spec, "dev", dev_probs, arrays.labels_dev, arrays.targets_dev, ece_bins)
            test_row = metric_row(source, spec, "test", test_probs, arrays.labels_test, arrays.targets_test, ece_bins)
            dev_row["selection_rule"] = DEFAULT_SELECTION_RULE
            dev_row["selection_delta"] = delta
            dev_rows_by_source.append(dev_row)
            test_rows_by_calibrator[spec.calibrator_id] = test_row
            distribution_rows.extend(prediction_distribution_rows(source, spec, "dev", dev_probs, arrays.labels_dev))
            distribution_rows.extend(prediction_distribution_rows(source, spec, "test", test_probs, arrays.labels_test))

        selected = select_dev_row(dev_rows_by_source, delta)
        for row in dev_rows_by_source:
            row["selection_rule"] = DEFAULT_SELECTION_RULE
            row["selection_delta"] = delta
        dev_ranking.extend(sorted(dev_rows_by_source, key=lambda row: (row["source_id"], not row["selected"], row["low_to_high_count"], row["MAE_label"])))
        if selected is not None:
            selected_rows.append(selected)
            for calibrator_id, row in test_rows_by_calibrator.items():
                is_selected = calibrator_id == selected["calibrator_id"]
                test_diagnostic.append(_test_row(row, is_selected, delta))
                if is_selected:
                    selected_test.append(_test_row(row, True, delta))

    write_csv(EXP15_TABLES_DIR / "exp15_calibration_dev_ranking.csv", dev_ranking, fieldnames=DEV_FIELDS)
    write_csv(EXP15_TABLES_DIR / "exp15_selected_calibrators.csv", selected_rows, fieldnames=DEV_FIELDS)
    write_csv(EXP15_TABLES_DIR / "exp15_test_diagnostic_metrics.csv", test_diagnostic, fieldnames=TEST_FIELDS)
    write_csv(EXP15_TABLES_DIR / "exp15_selected_test_metrics.csv", selected_test, fieldnames=TEST_FIELDS)
    write_csv(EXP15_TABLES_DIR / "exp15_prediction_distribution.csv", distribution_rows)
    write_csv(EXP15_TABLES_DIR / "exp15_source_errors.csv", errors)
    _write_report(selected_rows, selected_test, dev_ranking, errors)
    return {
        "status": "COMPLETED" if selected_rows else "NO_READY_SOURCES",
        "sources": len(sources),
        "selected_rows": len(selected_rows),
        "errors": len(errors),
    }


def _write_report(
    selected_rows: list[dict[str, Any]],
    selected_test: list[dict[str, Any]],
    dev_ranking: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    recommendation = "NO_FORMAL_RECOMMENDED"
    improved = [
        row
        for row in selected_rows
        if int(row.get("delta_low_to_high_count_vs_pava", 0)) < 0 and bool(row.get("eligible"))
    ]
    if improved:
        recommendation = "FORMAL_CALIBRATION_CANDIDATE"
    lines = [
        "# Exp15 Post-hoc Ordinal Calibration",
        "",
        f"Status: `{'COMPLETED' if selected_rows else 'NO_READY_SOURCES'}`",
        f"Recommendation: `{recommendation}`",
        "",
        "Exp15 is a dev-only post-hoc calibration diagnosis. It does not train a model and does not use",
        "test metrics for calibrator selection, checkpoint selection, or tuning.",
        "",
        "The selection rule is `pava_mae_guard_low_to_high_then_label2_then_calibration`: keep",
        "calibrators inside the PAVA baseline dev MAE guard, then prefer lower dev low-to-high",
        "count, lower label-2 low-to-high count, lower high-to-low count, and better calibration.",
        "",
        "## Selected Dev Calibrators",
        "",
    ]
    if not selected_rows:
        lines.append("No ready source arrays were found.")
    else:
        lines.extend(
            [
                "| source | calibrator | dev MAE | dev low-to-high | delta vs pava | dev label2 low-to-high | dev brier |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_id", "")),
                        str(row.get("calibrator_id", "")),
                        _fmt(row.get("MAE_label")),
                        f"{row.get('low_to_high_count')}/{row.get('true_low_score_count')}",
                        str(row.get("delta_low_to_high_count_vs_pava", "")),
                        f"{row.get('label2_low_to_high_count')}/{row.get('label2_count')}",
                        _fmt(row.get("threshold_brier")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Selected Test Diagnostics", ""])
    if selected_test:
        lines.extend(
            [
                "| source | calibrator | test MAE | test low-to-high | test label2 low-to-high | test QWK |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in selected_test:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_id", "")),
                        str(row.get("calibrator_id", "")),
                        _fmt(row.get("test_MAE_label")),
                        f"{row.get('test_low_to_high_count')}/{row.get('test_true_low_score_count')}",
                        f"{row.get('test_label2_low_to_high_count')}/{row.get('test_label2_count', '')}",
                        _fmt(row.get("test_QWK")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No selected test diagnostics were written.")
    tradeoff_rows = [
        row
        for row in sorted(dev_ranking, key=lambda item: (int(item.get("low_to_high_count", 9999)), float(item.get("MAE_label", 9999.0))))
        if not bool(row.get("eligible")) and int(row.get("delta_low_to_high_count_vs_pava", 0)) < 0
    ][:5]
    lines.extend(["", "## Dev Risk-Accuracy Tradeoff", ""])
    if tradeoff_rows:
        lines.extend(
            [
                "Some calibrators lower dev low-to-high but fall outside the PAVA baseline MAE guard.",
                "",
                "| source | calibrator | dev MAE | dev low-to-high | delta vs pava | MAE delta vs pava |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in tradeoff_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("source_id", "")),
                        str(row.get("calibrator_id", "")),
                        _fmt(row.get("MAE_label")),
                        f"{row.get('low_to_high_count')}/{row.get('true_low_score_count')}",
                        str(row.get("delta_low_to_high_count_vs_pava", "")),
                        _fmt(row.get("delta_MAE_label_vs_pava")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No ineligible dev calibrator lowered low-to-high relative to PAVA.")
    if errors:
        lines.extend(["", "## Skipped Sources", ""])
        for row in errors:
            lines.append(f"- `{row['source_id']}`: {row['error']}")
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Raw, PAVA, temperature, threshold, q3-bias, and threshold-wise isotonic calibrators are compared.",
            "- Isotonic models are fitted on dev threshold labels only and then applied unchanged to test.",
            "- Prediction arrays, logits, checkpoints, and raw JSONL predictions are not written to Exp15 outputs.",
        ]
    )
    write_text(EXP15_REPORTS_DIR / "exp15_posthoc_ordinal_calibration_report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp15 post-hoc ordinal calibration diagnosis.")
    parser.add_argument("--selection_delta", type=float, default=DEFAULT_SELECTION_DELTA)
    parser.add_argument("--ece_bins", type=int, default=DEFAULT_ECE_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(delta=args.selection_delta, ece_bins=args.ece_bins)
    print(result)


if __name__ == "__main__":
    main()
