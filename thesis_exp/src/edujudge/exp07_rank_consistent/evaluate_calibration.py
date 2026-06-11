"""Run Exp7-C risk-aware ordinal calibration without training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_PREDICTIONS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    EXP07_OUTPUT_DIR,
    EXP07_RUN_ID,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    ensure_exp07_calibration_dirs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_temperature import (
    evaluate_temperature_grid,
    select_temperature,
    temperature_probs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_thresholds import (
    compute_metrics,
    confusion_rows,
    distribution_rows,
    per_label_accuracy_rows,
    predictions_from_thresholds,
    quick_metrics,
    row_with_metrics,
    select_global_thresholds,
    select_risk_aware_thresholds,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import (
    ETA_HIGH_GRID,
    LAMBDA_LOW_GRID,
    BaseCalibrationData,
    expected_score_from_probs,
    load_available_bases,
    raw_prediction_from_probs,
    unified_prediction_rows,
    write_base_inventory,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_inventory_exp07 import build_inventory
from thesis_exp.src.edujudge.exp07_rank_consistent.write_exp07_calibration_report import write_report
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv


DELTA_METRICS = [
    "Accuracy",
    "MAE_label",
    "QWK",
    "Kendall tau",
    "low_to_high_rate",
    "Acc@5",
    "high_to_mid_or_low_rate",
    "severe_error_rate",
]


def _threshold_prefix(base_model: str, thresholds: tuple[float, float, float, float], method: str) -> dict[str, Any]:
    return {
        "base_model": base_model,
        "method": method,
        "theta_1": thresholds[0],
        "theta_2": thresholds[1],
        "theta_3": thresholds[2],
        "theta_4": thresholds[3],
    }


def _raw_metrics_rows(base: BaseCalibrationData) -> list[dict[str, Any]]:
    rows = []
    for split_data in [base.dev, base.test]:
        pred = raw_prediction_from_probs(split_data.probs)
        metrics = compute_metrics(split_data.labels, split_data.human_mean, pred, split_data.probs)
        rows.append(row_with_metrics({"base_model": base.base_model, "method": "raw", "split": split_data.split}, metrics))
    return rows


def _temperature_rows(base: BaseCalibrationData) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if base.dev.logits is None or base.test.logits is None:
        return [], [], None
    selected = select_temperature(base.base_model, base.dev)
    if selected is None:
        return [], [], None
    selected_t = float(selected["temperature"])
    dev_rows = evaluate_temperature_grid(base.base_model, base.dev, selected_t=selected_t)
    test_rows = evaluate_temperature_grid(base.base_model, base.test, selected_t=selected_t)
    return dev_rows, test_rows, selected


def _global_rows(base: BaseCalibrationData) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float, float, float]]:
    selected = select_global_thresholds(base.dev.labels, base.dev.human_mean, base.dev.probs)
    thresholds = selected["thresholds"]
    dev_pred = predictions_from_thresholds(base.dev.probs, thresholds)
    test_pred = predictions_from_thresholds(base.test.probs, thresholds)
    dev_metrics = compute_metrics(base.dev.labels, base.dev.human_mean, dev_pred, base.dev.probs)
    test_metrics = compute_metrics(base.test.labels, base.test.human_mean, test_pred, base.test.probs)
    prefix = {
        **_threshold_prefix(base.base_model, thresholds, "global_threshold"),
        "selection_objective": "dev_MAE_label",
        "selection_split": "dev",
        "dev_MAE_label": dev_metrics["MAE_label"],
        "dev_low_to_high": dev_metrics["low_to_high_rate"],
        "dev_Acc5": dev_metrics["Acc@5"],
        "test_MAE_label": test_metrics["MAE_label"],
        "test_low_to_high": test_metrics["low_to_high_rate"],
        "test_Acc5": test_metrics["Acc@5"],
    }
    return row_with_metrics({**prefix, "split": "dev"}, dev_metrics), row_with_metrics({**prefix, "split": "test"}, test_metrics), thresholds


def _risk_rows(
    base: BaseCalibrationData,
    raw_dev_metrics: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], tuple[float, float, float, float]]:
    dev_rows: list[dict[str, Any]] = []
    selected_by_pair = []
    for lambda_low in LAMBDA_LOW_GRID:
        for eta_high in ETA_HIGH_GRID:
            selected = select_risk_aware_thresholds(
                base.dev.labels,
                base.dev.human_mean,
                base.dev.probs,
                lambda_low=lambda_low,
                eta_high=eta_high,
                raw_dev_metrics=raw_dev_metrics,
            )
            thresholds = selected["thresholds"]
            dev_pred = predictions_from_thresholds(base.dev.probs, thresholds)
            dev_metrics = compute_metrics(base.dev.labels, base.dev.human_mean, dev_pred, base.dev.probs)
            row = row_with_metrics(
                {
                    **_threshold_prefix(base.base_model, thresholds, "risk_aware_threshold"),
                    "split": "dev",
                    "lambda_low": lambda_low,
                    "eta_high": eta_high,
                    "dev_objective": selected["objective"],
                    "satisfies_constraints": selected["satisfies_constraints"],
                    "constraint_available": selected["constraint_available"],
                    "selection_type": selected["selection_type"],
                    "selection_objective": "dev_MAE_plus_low_high_risk",
                    "selection_split": "dev",
                    "dev_MAE_label": dev_metrics["MAE_label"],
                    "dev_low_to_high": dev_metrics["low_to_high_rate"],
                    "dev_Acc5": dev_metrics["Acc@5"],
                },
                dev_metrics,
            )
            dev_rows.append(row)
            selected_by_pair.append(row)
    constrained = [row for row in selected_by_pair if str(row["satisfies_constraints"]) == "True"]
    pool = constrained or selected_by_pair
    best = min(
        pool,
        key=lambda row: (
            float(row["dev_objective"]),
            float(row["low_to_high_rate"]),
            float(row["MAE_label"]),
            -float(row["Acc@5"]),
        ),
    )
    thresholds = (float(best["theta_1"]), float(best["theta_2"]), float(best["theta_3"]), float(best["theta_4"]))
    test_pred = predictions_from_thresholds(base.test.probs, thresholds)
    test_metrics = compute_metrics(base.test.labels, base.test.human_mean, test_pred, base.test.probs)
    best_config = {
        "base_model": base.base_model,
        "method": "risk_aware_threshold",
        "lambda_low": best["lambda_low"],
        "eta_high": best["eta_high"],
        "theta_1": best["theta_1"],
        "theta_2": best["theta_2"],
        "theta_3": best["theta_3"],
        "theta_4": best["theta_4"],
        "dev_objective": best["dev_objective"],
        "dev_MAE_label": best["MAE_label"],
        "dev_low_to_high": best["low_to_high_rate"],
        "dev_Acc5": best["Acc@5"],
        "selection_type": "best_constrained" if constrained else "best_unconstrained",
        "selection_split": "dev",
    }
    test_row = row_with_metrics(
        {
            **_threshold_prefix(base.base_model, thresholds, "risk_aware_threshold"),
            "split": "test",
            "lambda_low": best["lambda_low"],
            "eta_high": best["eta_high"],
            "dev_objective": best["dev_objective"],
            "selection_type": best_config["selection_type"],
            "selection_objective": "dev_MAE_plus_low_high_risk",
            "selection_split": "dev",
            "test_MAE_label": test_metrics["MAE_label"],
            "test_low_to_high": test_metrics["low_to_high_rate"],
            "test_Acc5": test_metrics["Acc@5"],
        },
        test_metrics,
    )
    return dev_rows, best_config, test_row, thresholds


def _reference_rows() -> list[dict[str, Any]]:
    path = EXP07_OUTPUT_DIR / "tables" / "exp07_r1_comparison.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        if row.get("split") != "test" or row.get("run_id") not in {QD_B0_RUN_ID, QD_B1_RUN_ID}:
            continue
        rows.append(
            {
                "base_model": row["run_id"],
                "method": "raw_reference_existing_summary",
                "split": "test",
                "source": "exp07_r1_comparison.csv",
                "Accuracy": row.get("Accuracy", ""),
                "MAE_label": row.get("MAE_label", ""),
                "QWK": row.get("Quadratic Weighted Kappa", ""),
                "Kendall tau": row.get("Kendall tau", ""),
                "low_to_high_rate": row.get("low_to_high_rate", ""),
                "Acc@5": row.get("Acc@5", ""),
                "high_to_mid_or_low_rate": row.get("high_to_mid_or_low_rate", ""),
                "severe_error_rate": row.get("severe_error_rate", ""),
            }
        )
    return rows


def _delta_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_base = {
        row["base_model"]: row
        for row in summary_rows
        if row.get("method") == "raw" and row.get("split") == "test"
    }
    out = []
    for row in summary_rows:
        if row.get("split") != "test" or row.get("method") in {"raw", "raw_reference_existing_summary"}:
            continue
        raw = raw_by_base.get(row["base_model"])
        if not raw:
            continue
        for metric in DELTA_METRICS:
            try:
                delta = float(row[metric]) - float(raw[metric])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                {
                    "base_model": row["base_model"],
                    "method": row["method"],
                    "metric": metric,
                    "raw_value": raw.get(metric, ""),
                    "calibrated_value": row.get(metric, ""),
                    "delta": delta,
                }
            )
    return out


def _low_high_comparisons(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low_fields = ["base_model", "method", "split", "Acc@1", "Acc@2", "low_exact_match", "low_MAE", "low_signed_bias", "low_to_high_rate", "mean_pred_low"]
    high_fields = ["base_model", "method", "split", "Acc@5", "high_to_mid_or_low_rate", "high_signed_bias", "mean_pred_high"]
    low_rows = [{field: row.get(field, "") for field in low_fields} for row in summary_rows if row.get("split") == "test"]
    high_rows = [{field: row.get(field, "") for field in high_fields} for row in summary_rows if row.get("split") == "test"]
    return low_rows, high_rows


def _rejection_row(base: BaseCalibrationData, thresholds: tuple[float, float, float, float]) -> dict[str, Any]:
    pred = predictions_from_thresholds(base.test.probs, thresholds)
    p3 = base.test.probs[:, 2]
    p4 = base.test.probs[:, 3]
    review = (pred >= 4) & ((np.abs(p3 - thresholds[2]) <= 0.05) | (np.abs(p4 - thresholds[3]) <= 0.05))
    keep = ~review
    all_metrics = quick_metrics(base.test.labels, base.test.human_mean, pred)
    kept_metrics = quick_metrics(base.test.labels[keep], base.test.human_mean[keep], pred[keep]) if np.any(keep) else {}
    return {
        "base_model": base.base_model,
        "method": "risk_aware_threshold",
        "split": "test",
        "review_rate": float(np.mean(review)) if review.size else float("nan"),
        "coverage": float(np.mean(keep)) if keep.size else float("nan"),
        "low_to_high_all": all_metrics.get("low_to_high_rate", ""),
        "low_to_high_non_rejected": kept_metrics.get("low_to_high_rate", ""),
        "Acc@5_all": all_metrics.get("Acc@5", ""),
        "Acc@5_non_rejected": kept_metrics.get("Acc@5", ""),
    }


def _write_selected_predictions(base: BaseCalibrationData, thresholds: tuple[float, float, float, float]) -> None:
    pred = predictions_from_thresholds(base.test.probs, thresholds)
    rows = unified_prediction_rows(base.test, pred, "risk_aware_threshold")
    write_csv(EXP07_CALIBRATION_PREDICTIONS_DIR / f"{base.base_model}_test_risk_aware_predictions.csv", rows)


def _source_rows_from_server(args: argparse.Namespace) -> dict[str, dict[str, str]] | None:
    if not args.server_host:
        return None
    rows, _server_status = build_inventory(
        server_host=args.server_host,
        server_port=args.server_port,
        server_repo=args.server_repo,
    )
    return {row["run_id"]: row for row in rows}


def run(args: argparse.Namespace) -> None:
    ensure_exp07_calibration_dirs()
    inventory_rows = write_base_inventory(_source_rows_from_server(args))
    bases = load_available_bases(inventory_rows)

    raw_rows: list[dict[str, Any]] = []
    temp_dev_rows: list[dict[str, Any]] = []
    temp_test_rows: list[dict[str, Any]] = []
    global_dev_rows: list[dict[str, Any]] = []
    global_test_rows: list[dict[str, Any]] = []
    risk_dev_rows: list[dict[str, Any]] = []
    risk_test_rows: list[dict[str, Any]] = []
    risk_best_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    distribution_all: list[dict[str, Any]] = []
    confusion_all: list[dict[str, Any]] = []
    per_label_all: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    for base in bases:
        base_raw_rows = _raw_metrics_rows(base)
        raw_rows.extend(base_raw_rows)
        raw_test = next(row for row in base_raw_rows if row["split"] == "test")
        raw_dev_pred = raw_prediction_from_probs(base.dev.probs)
        raw_dev_metrics = quick_metrics(base.dev.labels, base.dev.human_mean, raw_dev_pred)
        raw_test_pred = raw_prediction_from_probs(base.test.probs)
        distribution_all.extend(distribution_rows(base.base_model, "raw", "test", raw_test_pred))
        confusion_all.extend(confusion_rows(base.base_model, "raw", "test", base.test.labels, raw_test_pred))
        per_label_all.extend(per_label_accuracy_rows(base.base_model, "raw", "test", base.test.labels, raw_test_pred))
        summary_rows.append({**raw_test, "source": "local_arrays"})

        dev_temp, test_temp, selected_temp = _temperature_rows(base)
        temp_dev_rows.extend(dev_temp)
        temp_test_rows.extend(test_temp)
        if selected_temp is not None:
            selected_t = float(selected_temp["temperature"])
            temp_probs_test = temperature_probs(base.test.logits, selected_t)
            temp_pred_test = raw_prediction_from_probs(temp_probs_test)
            temp_metrics = compute_metrics(base.test.labels, base.test.human_mean, temp_pred_test, temp_probs_test)
            summary_rows.append(
                row_with_metrics(
                    {"base_model": base.base_model, "method": "temperature_scaling", "split": "test", "temperature": selected_t, "source": "dev_selected"},
                    temp_metrics,
                )
            )

        global_dev, global_test, global_thresholds = _global_rows(base)
        global_dev_rows.append(global_dev)
        global_test_rows.append(global_test)
        summary_rows.append({**global_test, "source": "dev_selected"})
        global_pred = predictions_from_thresholds(base.test.probs, global_thresholds)
        distribution_all.extend(distribution_rows(base.base_model, "global_threshold", "test", global_pred))
        confusion_all.extend(confusion_rows(base.base_model, "global_threshold", "test", base.test.labels, global_pred))
        per_label_all.extend(per_label_accuracy_rows(base.base_model, "global_threshold", "test", base.test.labels, global_pred))

        base_risk_dev, risk_best, risk_test, risk_thresholds = _risk_rows(base, raw_dev_metrics)
        risk_dev_rows.extend(base_risk_dev)
        risk_best_rows.append(risk_best)
        risk_test_rows.append(risk_test)
        summary_rows.append({**risk_test, "source": "dev_selected"})
        risk_pred = predictions_from_thresholds(base.test.probs, risk_thresholds)
        distribution_all.extend(distribution_rows(base.base_model, "risk_aware_threshold", "test", risk_pred))
        confusion_all.extend(confusion_rows(base.base_model, "risk_aware_threshold", "test", base.test.labels, risk_pred))
        per_label_all.extend(per_label_accuracy_rows(base.base_model, "risk_aware_threshold", "test", base.test.labels, risk_pred))
        rejection_rows.append(_rejection_row(base, risk_thresholds))
        _write_selected_predictions(base, risk_thresholds)

    summary_rows.extend(_reference_rows())
    delta_rows = _delta_rows(summary_rows)
    low_rows, high_rows = _low_high_comparisons(summary_rows)

    write_csv(EXP07_CALIBRATION_TABLES_DIR / "raw_baseline_metrics.csv", raw_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "temperature_scaling_dev.csv", temp_dev_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "temperature_scaling_test.csv", temp_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "global_threshold_calibration_dev.csv", global_dev_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "global_threshold_calibration_test.csv", global_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "risk_aware_threshold_calibration_dev.csv", risk_dev_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "risk_aware_threshold_calibration_test.csv", risk_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "risk_aware_threshold_best_configs.csv", risk_best_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "calibration_summary.csv", summary_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "calibration_delta_vs_raw.csv", delta_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "low_score_calibration_comparison.csv", low_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "high_score_calibration_comparison.csv", high_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "rejection_analysis.csv", rejection_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "pred_label_distribution.csv", distribution_all)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "true_pred_confusion.csv", confusion_all)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "per_label_accuracy.csv", per_label_all)
    write_report()
    print(f"Wrote Exp7-C calibration outputs: {EXP07_CALIBRATION_TABLES_DIR.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp7-C risk-aware ordinal calibration.")
    parser.add_argument("--server_host", default=None, help="Optional SSH host for inventory status only.")
    parser.add_argument("--server_port", type=int, default=22)
    parser.add_argument("--server_repo", default="/home/jpang/edubench-eval-exp2")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
