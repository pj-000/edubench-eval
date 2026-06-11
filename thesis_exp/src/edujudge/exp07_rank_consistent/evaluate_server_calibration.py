"""Unified Exp7-C2 calibration after syncing server artifacts locally."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import EXP07_CALIBRATION_TABLES_DIR, ensure_exp07_calibration_dirs
from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_thresholds import (
    predictions_from_thresholds,
    quick_metrics,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import (
    BASE_CONFIGS,
    BaseCalibrationData,
    CalibrationBaseConfig,
    load_available_bases,
    raw_prediction_from_probs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.evaluate_calibration import (
    _delta_rows,
    _global_rows,
    _low_high_comparisons,
    _raw_metrics_rows,
    _rejection_row,
    _risk_rows,
    _temperature_rows,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.write_exp07_c2_report import write_c2_report
from thesis_exp.src.edujudge.utils.io import write_csv


def _array_path(config: CalibrationBaseConfig, name: str, split: str) -> Path:
    return config.run_dir / "arrays" / f"{name}_{split}.npy"


def _prediction_path(config: CalibrationBaseConfig, split: str) -> Path:
    return config.run_dir / "predictions" / f"predictions_{split}.jsonl"


def _yes(path: Path) -> str:
    return "yes" if path.exists() else "no"


def _blocking_reason(row: dict[str, str]) -> str:
    if row["calibration_ready"] == "yes":
        return "ready"
    missing = [
        key.replace("_available", "")
        for key in [
            "dev_probs_available",
            "test_probs_available",
            "dev_labels_available",
            "test_labels_available",
            "dev_predictions_available",
            "test_predictions_available",
        ]
        if row[key] != "yes"
    ]
    return "BLOCKED_MISSING_LOCAL_ARTIFACTS: " + ", ".join(missing)


def server_artifact_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for config in BASE_CONFIGS:
        row = {
            "base_model": config.base_model,
            "dev_logits_available": _yes(_array_path(config, "logits", "dev")),
            "test_logits_available": _yes(_array_path(config, "logits", "test")),
            "dev_probs_available": _yes(_array_path(config, "probs", "dev")),
            "test_probs_available": _yes(_array_path(config, "probs", "test")),
            "dev_labels_available": _yes(_array_path(config, "labels", "dev")),
            "test_labels_available": _yes(_array_path(config, "labels", "test")),
            "dev_predictions_available": _yes(_prediction_path(config, "dev")),
            "test_predictions_available": _yes(_prediction_path(config, "test")),
        }
        ready_keys = [
            "dev_probs_available",
            "test_probs_available",
            "dev_labels_available",
            "test_labels_available",
            "dev_predictions_available",
            "test_predictions_available",
        ]
        row["calibration_ready"] = "yes" if all(row[key] == "yes" for key in ready_keys) else "no"
        row["blocking_reason"] = _blocking_reason(row)
        rows.append(row)
    return rows


def _inventory_for_loader(server_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    loader_rows: list[dict[str, Any]] = []
    for row in server_rows:
        can_threshold = row["calibration_ready"] == "yes"
        can_temperature = can_threshold and row["dev_logits_available"] == "yes" and row["test_logits_available"] == "yes"
        loader_rows.append(
            {
                "base_model": row["base_model"],
                "dev_probs_available": "yes_local" if row["dev_probs_available"] == "yes" else "no",
                "test_probs_available": "yes_local" if row["test_probs_available"] == "yes" else "no",
                "dev_logits_available": "yes_local" if row["dev_logits_available"] == "yes" else "no",
                "test_logits_available": "yes_local" if row["test_logits_available"] == "yes" else "no",
                "can_threshold_calibrate": "yes" if can_threshold else "no",
                "can_temperature_calibrate": "yes" if can_temperature else "no",
                "blocking_reason": row["blocking_reason"],
            }
        )
    return loader_rows


def _best_selection_rows(summary_rows: list[dict[str, Any]], risk_best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_models = sorted({row["base_model"] for row in summary_rows if row.get("method") == "raw"})
    risk_by_base = {row["base_model"]: row for row in risk_best_rows}
    for base_model in base_models:
        base_rows = [
            row
            for row in summary_rows
            if row.get("base_model") == base_model and row.get("split") == "test" and row.get("method") != "raw"
        ]
        if not base_rows:
            continue
        best = min(
            base_rows,
            key=lambda row: (
                float(row["low_to_high_rate"]),
                float(row["MAE_label"]),
                -float(row["Acc@5"]),
            ),
        )
        raw = next(
            row
            for row in summary_rows
            if row.get("base_model") == base_model and row.get("split") == "test" and row.get("method") == "raw"
        )
        risk = risk_by_base.get(base_model, {})
        is_threshold_method = "threshold" in str(best.get("method", ""))
        out.append(
            {
                "base_model": base_model,
                "best_method": best["method"],
                "temperature": best.get("temperature", ""),
                "theta_1": best.get("theta_1", "") if is_threshold_method else "",
                "theta_2": best.get("theta_2", "") if is_threshold_method else "",
                "theta_3": best.get("theta_3", "") if is_threshold_method else "",
                "theta_4": best.get("theta_4", "") if is_threshold_method else "",
                "lambda_low": best.get("lambda_low", "") if best.get("method") == "risk_aware_threshold" else "",
                "eta_high": best.get("eta_high", "") if best.get("method") == "risk_aware_threshold" else "",
                "risk_aware_theta_1": risk.get("theta_1", ""),
                "risk_aware_theta_2": risk.get("theta_2", ""),
                "risk_aware_theta_3": risk.get("theta_3", ""),
                "risk_aware_theta_4": risk.get("theta_4", ""),
                "risk_aware_lambda_low": risk.get("lambda_low", ""),
                "risk_aware_eta_high": risk.get("eta_high", ""),
                "raw_low_to_high_rate": raw.get("low_to_high_rate", ""),
                "low_to_high_rate": best.get("low_to_high_rate", ""),
                "raw_MAE_label": raw.get("MAE_label", ""),
                "MAE_label": best.get("MAE_label", ""),
                "raw_QWK": raw.get("QWK", ""),
                "QWK": best.get("QWK", ""),
                "raw_Acc@5": raw.get("Acc@5", ""),
                "Acc@5": best.get("Acc@5", ""),
                "selection_split": best.get("selection_split", ""),
            }
        )
    return out


def _run_base(base: BaseCalibrationData) -> dict[str, Any]:
    raw_rows = _raw_metrics_rows(base)
    raw_dev_pred = raw_prediction_from_probs(base.dev.probs)
    raw_dev_metrics = quick_metrics(base.dev.labels, base.dev.human_mean, raw_dev_pred)
    _temp_dev, temp_test, _selected_temp = _temperature_rows(base)
    global_dev, global_test, _global_thresholds = _global_rows(base)
    risk_dev, risk_best, risk_test, risk_thresholds = _risk_rows(base, raw_dev_metrics)
    summary = [
        {**next(row for row in raw_rows if row["split"] == "test"), "source": "server_local_arrays"},
    ]
    if temp_test:
        selected_temp_test = next((row for row in temp_test if str(row.get("selected")) == "True"), temp_test[0])
        summary.append({**selected_temp_test, "method": "temperature_scaling", "source": "dev_selected"})
    summary.append({**global_test, "source": "dev_selected"})
    summary.append({**risk_test, "source": "dev_selected"})
    return {
        "raw_rows": raw_rows,
        "temp_test": temp_test,
        "global_dev": global_dev,
        "global_test": global_test,
        "risk_dev": risk_dev,
        "risk_test": risk_test,
        "risk_best": risk_best,
        "summary": summary,
        "rejection": _rejection_row(base, risk_thresholds),
    }


def run() -> None:
    ensure_exp07_calibration_dirs()
    server_rows = server_artifact_inventory()
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "server_calibration_artifact_inventory.csv", server_rows)
    bases = load_available_bases(_inventory_for_loader(server_rows))

    raw_rows: list[dict[str, Any]] = []
    temp_test_rows: list[dict[str, Any]] = []
    global_test_rows: list[dict[str, Any]] = []
    risk_test_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    risk_best_rows: list[dict[str, Any]] = []

    for base in bases:
        result = _run_base(base)
        raw_rows.extend(result["raw_rows"])
        temp_test_rows.extend(result["temp_test"])
        global_test_rows.append(result["global_test"])
        risk_test_rows.append(result["risk_test"])
        summary_rows.extend(result["summary"])
        rejection_rows.append(result["rejection"])
        risk_best_rows.append(result["risk_best"])

    best_rows = _best_selection_rows(summary_rows, risk_best_rows)
    low_rows, high_rows = _low_high_comparisons(summary_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_raw_baseline_metrics.csv", raw_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_temperature_scaling_test.csv", temp_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_global_threshold_test.csv", global_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_risk_aware_threshold_test.csv", risk_test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_calibration_summary.csv", summary_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_calibration_delta_vs_raw.csv", _delta_rows(summary_rows))
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_rejection_analysis.csv", rejection_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "best_calibrated_model_selection.csv", best_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_low_score_calibration_comparison.csv", low_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "unified_high_score_calibration_comparison.csv", high_rows)
    write_c2_report()
    print(f"Wrote Exp7-C2 unified calibration outputs: {EXP07_CALIBRATION_TABLES_DIR.parent}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
