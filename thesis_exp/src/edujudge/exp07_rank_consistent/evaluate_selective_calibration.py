"""Evaluate Exp7-C3 selective risk-aware calibration without training."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_REPORTS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    ensure_exp07_calibration_dirs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_thresholds import (
    compute_metrics,
    predictions_from_thresholds,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import (
    BaseCalibrationData,
    load_available_bases,
    raw_prediction_from_probs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.evaluate_server_calibration import (
    _inventory_for_loader,
    server_artifact_inventory,
)
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_text


REVIEW_BUDGETS = [0.05, 0.10, 0.15, 0.20]
POLICIES = [
    "high_threshold_margin_risk",
    "low_score_overestimation_risk",
    "entropy_confidence_risk",
    "combined_risk",
]
RAW_THRESHOLDS = (0.5, 0.5, 0.5, 0.5)
LOW_TO_HIGH_TARGET = 0.35
MIN_COVERAGE = 0.80
MAX_REVIEW_RATE = 0.20
ACC5_TOLERANCE = 0.05


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _threshold_tuple(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["theta_1"]),
        float(row["theta_2"]),
        float(row["theta_3"]),
        float(row["theta_4"]),
    )


def _load_risk_thresholds() -> dict[str, dict[str, Any]]:
    path = EXP07_CALIBRATION_TABLES_DIR / "unified_risk_aware_threshold_test.csv"
    rows = read_csv(path) if path.exists() else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        base_model = row.get("base_model", "")
        if not base_model:
            continue
        out[base_model] = {
            "thresholds": _threshold_tuple(row),
            "lambda_low": row.get("lambda_low", ""),
            "eta_high": row.get("eta_high", ""),
            "selection_split": row.get("selection_split", "dev"),
            "selection_type": row.get("selection_type", ""),
        }
    return out


def _label_distribution(values: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(values == label)) for label in range(1, 6)}


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.min(values)) if values.size else 0.0
    high = float(np.max(values)) if values.size else 0.0
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _binary_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-8, 1.0 - 1e-8)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)) / math.log(2.0)
    return entropy


def _risk_scores(
    probs: np.ndarray,
    pred: np.ndarray,
    thresholds: tuple[float, float, float, float],
    policy: str,
) -> np.ndarray:
    thresholds_arr = np.asarray(thresholds, dtype=float).reshape(1, 4)
    p3 = probs[:, 2]
    p4 = probs[:, 3]
    theta3 = float(thresholds[2])
    theta4 = float(thresholds[3])
    high_pred = (pred >= 4).astype(float)
    boundary_distance = np.minimum(np.abs(p3 - theta3), np.abs(p4 - theta4))
    margin_risk = high_pred * np.maximum(0.0, 0.10 - boundary_distance)
    over_risk = high_pred * (p3 + p4) * (1.0 + np.maximum(0.0, 0.50 - boundary_distance))
    entropy = np.mean(_binary_entropy(probs), axis=1)
    boundary_closeness = np.mean(np.maximum(0.0, 0.15 - np.abs(probs - thresholds_arr)) / 0.15, axis=1)
    entropy_risk = (0.65 * entropy + 0.35 * boundary_closeness) * (1.0 + 0.35 * high_pred)
    if policy == "high_threshold_margin_risk":
        return margin_risk
    if policy == "low_score_overestimation_risk":
        return over_risk
    if policy == "entropy_confidence_risk":
        return entropy_risk
    if policy == "combined_risk":
        high_label_boost = high_pred * (pred.astype(float) / 5.0)
        return (
            0.40 * _normalize(margin_risk)
            + 0.35 * _normalize(over_risk)
            + 0.15 * _normalize(entropy_risk)
            + 0.10 * high_label_boost
        )
    raise ValueError(f"Unknown selective calibration policy: {policy}")


def _top_k_review_mask(risk: np.ndarray, budget: float) -> np.ndarray:
    n = risk.size
    if n == 0 or budget <= 0:
        return np.zeros(n, dtype=bool)
    k = min(n, max(1, int(math.floor(budget * n))))
    order = np.argsort(-risk, kind="mergesort")
    review = np.zeros(n, dtype=bool)
    review[order[:k]] = True
    return review


def _method_predictions(
    base: BaseCalibrationData,
    method: str,
    thresholds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return raw_prediction_from_probs(base.dev.probs), raw_prediction_from_probs(base.test.probs)
    if method == "risk_aware_threshold":
        return (
            predictions_from_thresholds(base.dev.probs, thresholds),
            predictions_from_thresholds(base.test.probs, thresholds),
        )
    raise ValueError(f"Unknown method: {method}")


def _full_metrics(split_data: Any, pred: np.ndarray) -> dict[str, float]:
    return compute_metrics(split_data.labels, split_data.human_mean, pred, split_data.probs)


def _row_for_mask(
    *,
    base_model: str,
    method: str,
    split_data: Any,
    pred: np.ndarray,
    thresholds: tuple[float, float, float, float],
    full_metrics: dict[str, float],
    policy: str,
    review_budget: float,
    review_mask: np.ndarray,
    lambda_low: Any = "",
    eta_high: Any = "",
    selected_scope: str = "",
    overall_selected: str = "no",
) -> dict[str, Any]:
    keep = ~review_mask
    labels = split_data.labels
    potential_low_to_high = (labels <= 2) & (pred >= 4)
    captured_low_to_high = review_mask & potential_low_to_high
    low_mask = labels <= 2
    kept_metrics = compute_metrics(labels[keep], split_data.human_mean[keep], pred[keep], split_data.probs[keep])
    review_count = int(np.sum(review_mask))
    non_rejected_count = int(np.sum(keep))
    row = {
        "base_model": base_model,
        "method": method,
        "split": split_data.split,
        "risk_policy": policy,
        "review_budget": review_budget,
        "selection_split": "dev",
        "selection_objective": "dev_min_low_to_high_non_rejected_with_coverage_and_acc5_constraints",
        "selected_scope": selected_scope,
        "overall_selected": overall_selected,
        "theta_1": thresholds[0],
        "theta_2": thresholds[1],
        "theta_3": thresholds[2],
        "theta_4": thresholds[3],
        "lambda_low": lambda_low,
        "eta_high": eta_high,
        "n": int(labels.size),
        "reviewed_count": review_count,
        "non_rejected_count": non_rejected_count,
        "review_rate": _safe_rate(review_count, labels.size),
        "coverage": _safe_rate(non_rejected_count, labels.size),
        "full_low_to_high_rate": full_metrics["low_to_high_rate"],
        "low_to_high_non_rejected": kept_metrics["low_to_high_rate"],
        "low_to_high_delta_non_rejected_vs_full": kept_metrics["low_to_high_rate"] - full_metrics["low_to_high_rate"],
        "full_MAE_label": full_metrics["MAE_label"],
        "MAE_non_rejected": kept_metrics["MAE_label"],
        "MAE_delta_non_rejected_vs_full": kept_metrics["MAE_label"] - full_metrics["MAE_label"],
        "full_QWK": full_metrics["QWK"],
        "QWK_non_rejected": kept_metrics["QWK"],
        "QWK_delta_non_rejected_vs_full": kept_metrics["QWK"] - full_metrics["QWK"],
        "full_Acc@5": full_metrics["Acc@5"],
        "Acc@5_non_rejected": kept_metrics["Acc@5"],
        "Acc@5_delta_non_rejected_vs_full": kept_metrics["Acc@5"] - full_metrics["Acc@5"],
        "high_to_mid_or_low_non_rejected": kept_metrics["high_to_mid_or_low_rate"],
        "low_exact_non_rejected": kept_metrics["low_exact_match"],
        "severe_error_non_rejected": kept_metrics["severe_error_rate"],
        "potential_low_to_high_count": int(np.sum(potential_low_to_high)),
        "rejected_low_to_high_count": int(np.sum(captured_low_to_high)),
        "rejected_low_to_high_capture_rate": _safe_rate(int(np.sum(captured_low_to_high)), int(np.sum(potential_low_to_high))),
        "true_low_count": int(np.sum(low_mask)),
        "rejected_true_low_count": int(np.sum(review_mask & low_mask)),
        "fraction_true_low_sent_to_review": _safe_rate(int(np.sum(review_mask & low_mask)), int(np.sum(low_mask))),
        "fraction_potential_low_to_high_sent_to_review": _safe_rate(
            int(np.sum(captured_low_to_high)),
            int(np.sum(potential_low_to_high)),
        ),
        "rejected_count": review_count,
        "rejected_label_distribution": _label_distribution(labels[review_mask]),
        "rejected_pred_distribution": _label_distribution(pred[review_mask]),
    }
    row["meets_success_criteria"] = (
        row["coverage"] >= MIN_COVERAGE
        and row["review_rate"] <= MAX_REVIEW_RATE + 1e-12
        and row["low_to_high_non_rejected"] <= LOW_TO_HIGH_TARGET
    )
    row["acc5_within_tolerance"] = row["Acc@5_non_rejected"] >= row["full_Acc@5"] - ACC5_TOLERANCE
    return row


def _evaluate_candidate_rows(
    *,
    base: BaseCalibrationData,
    method: str,
    thresholds: tuple[float, float, float, float],
    pred: np.ndarray,
    split_data: Any,
    lambda_low: Any,
    eta_high: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full = _full_metrics(split_data, pred)
    for policy in POLICIES:
        risk = _risk_scores(split_data.probs, pred, thresholds, policy)
        for budget in REVIEW_BUDGETS:
            review = _top_k_review_mask(risk, budget)
            rows.append(
                _row_for_mask(
                    base_model=base.base_model,
                    method=method,
                    split_data=split_data,
                    pred=pred,
                    thresholds=thresholds,
                    full_metrics=full,
                    policy=policy,
                    review_budget=budget,
                    review_mask=review,
                    lambda_low=lambda_low,
                    eta_high=eta_high,
                )
            )
    return rows


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    success = bool(row["meets_success_criteria"])
    acc_ok = bool(row["acc5_within_tolerance"])
    improved = float(row["low_to_high_non_rejected"]) <= float(row["full_low_to_high_rate"]) + 1e-12
    return (
        0 if success else 1,
        0 if acc_ok else 1,
        0 if improved else 1,
        float(row["low_to_high_non_rejected"]),
        float(row["MAE_non_rejected"]),
        -float(row["Acc@5_non_rejected"]),
        float(row["review_rate"]),
        row["risk_policy"],
    )


def _select_best_rows(dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for key in sorted({(row["base_model"], row["method"]) for row in dev_rows}):
        rows = [row for row in dev_rows if (row["base_model"], row["method"]) == key]
        best = dict(min(rows, key=_selection_key))
        best["selected_scope"] = "base_method"
        best["dev_selection_rank"] = 1
        best_rows.append(best)
    overall = dict(min(best_rows, key=_selection_key))
    for row in best_rows:
        row["overall_selected"] = (
            "yes"
            if row["base_model"] == overall["base_model"]
            and row["method"] == overall["method"]
            and row["risk_policy"] == overall["risk_policy"]
            and float(row["review_budget"]) == float(overall["review_budget"])
            else "no"
        )
    return best_rows


def _evaluate_selected_test_rows(
    *,
    bases: list[BaseCalibrationData],
    best_rows: list[dict[str, Any]],
    risk_thresholds: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base_by_name = {base.base_model: base for base in bases}
    out: list[dict[str, Any]] = []
    for selected in best_rows:
        base = base_by_name[selected["base_model"]]
        method = selected["method"]
        thresholds = (
            RAW_THRESHOLDS
            if method == "raw"
            else tuple(float(selected[f"theta_{idx}"]) for idx in range(1, 5))
        )
        _dev_pred, test_pred = _method_predictions(base, method, thresholds)
        full = _full_metrics(base.test, test_pred)
        policy = selected["risk_policy"]
        budget = float(selected["review_budget"])
        risk = _risk_scores(base.test.probs, test_pred, thresholds, policy)
        review = _top_k_review_mask(risk, budget)
        risk_config = risk_thresholds.get(base.base_model, {})
        out.append(
            _row_for_mask(
                base_model=base.base_model,
                method=method,
                split_data=base.test,
                pred=test_pred,
                thresholds=thresholds,
                full_metrics=full,
                policy=policy,
                review_budget=budget,
                review_mask=review,
                lambda_low=risk_config.get("lambda_low", "") if method == "risk_aware_threshold" else "",
                eta_high=risk_config.get("eta_high", "") if method == "risk_aware_threshold" else "",
                selected_scope="base_method",
                overall_selected=selected["overall_selected"],
            )
        )
    return out


def _rejected_distribution_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "base_model": row["base_model"],
                "method": row["method"],
                "split": row["split"],
                "risk_policy": row["risk_policy"],
                "review_budget": row["review_budget"],
                "selected_scope": row.get("selected_scope", ""),
                "overall_selected": row.get("overall_selected", "no"),
                "rejected_count": row["rejected_count"],
                "rejected_label_distribution": row["rejected_label_distribution"],
                "rejected_pred_distribution": row["rejected_pred_distribution"],
                "potential_low_to_high_count": row["potential_low_to_high_count"],
                "rejected_low_to_high_count": row["rejected_low_to_high_count"],
                "rejected_low_to_high_capture_rate": row["rejected_low_to_high_capture_rate"],
                "fraction_true_low_sent_to_review": row["fraction_true_low_sent_to_review"],
                "fraction_potential_low_to_high_sent_to_review": row["fraction_potential_low_to_high_sent_to_review"],
            }
        )
    return out


def _compact_test_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "base_model",
        "method",
        "risk_policy",
        "review_budget",
        "coverage",
        "review_rate",
        "full_low_to_high_rate",
        "low_to_high_non_rejected",
        "rejected_low_to_high_capture_rate",
        "full_MAE_label",
        "MAE_non_rejected",
        "full_QWK",
        "QWK_non_rejected",
        "full_Acc@5",
        "Acc@5_non_rejected",
        "meets_success_criteria",
        "overall_selected",
    ]
    return [{field: row.get(field, "") for field in fields} for row in test_rows]


def _write_report(
    *,
    dev_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    overall = next(row for row in test_rows if row.get("overall_selected") == "yes")
    full_lth = float(overall["full_low_to_high_rate"])
    kept_lth = float(overall["low_to_high_non_rejected"])
    c2_full_best = min(test_rows, key=lambda row: float(row["full_low_to_high_rate"]))
    beats_full = kept_lth < float(c2_full_best["full_low_to_high_rate"])
    success = bool(overall["meets_success_criteria"]) and bool(overall["acc5_within_tolerance"])
    low_capture = int(overall["rejected_low_to_high_count"])
    potential_low = int(overall["potential_low_to_high_count"])
    lines = [
        "# Exp7-C3 Selective Risk-aware Calibration Report",
        "",
        "Formal status: completed.",
        "",
        "Training run: NO.",
        "",
        "API calls: NO.",
        "",
        "Synthetic data generation: NO.",
        "",
        "Selection protocol: rejection policy and review budget are selected on dev only; test is used once for final evaluation of the dev-selected policies.",
        "",
        "## Answer Summary",
        "",
        f"- Selective calibration outperforms full-coverage calibration on the target risk metric: {'YES' if beats_full else 'NO'} for low-to-high risk among non-rejected samples.",
        f"- Best coverage>=80% policy: {overall['base_model']} / {overall['method']} / {overall['risk_policy']} at review budget {_fmt(overall['review_budget'], 2)}.",
        f"- Test review rate: {_fmt(overall['review_rate'])}; coverage: {_fmt(overall['coverage'])}.",
        f"- Test low_to_high: {_fmt(full_lth)} full-coverage -> {_fmt(kept_lth)} non-rejected.",
        f"- Rejection captures {low_capture}/{potential_low} potential low_to_high errors ({_fmt(overall['rejected_low_to_high_capture_rate'])}).",
        f"- Thesis suitability: {'YES, as a human-in-the-loop selective scoring method' if success else 'NOT YET, criteria not fully met'}.",
        "- Framing: this is not a pure automatic scorer; it is automatic scoring with a targeted human-review option for high-risk cases.",
        "",
        "## Test Results For Dev-selected Policies",
        "",
        "| base | method | policy | review | coverage | low_to_high full -> kept | MAE full -> kept | QWK full -> kept | Acc@5 full -> kept | capture |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(test_rows, key=lambda item: (item["base_model"], item["method"])):
        marker = " **selected**" if row.get("overall_selected") == "yes" else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{row['base_model']}{marker}",
                    row["method"],
                    row["risk_policy"],
                    _fmt(row["review_rate"]),
                    _fmt(row["coverage"]),
                    f"{_fmt(row['full_low_to_high_rate'])} -> {_fmt(row['low_to_high_non_rejected'])}",
                    f"{_fmt(row['full_MAE_label'])} -> {_fmt(row['MAE_non_rejected'])}",
                    f"{_fmt(row['full_QWK'])} -> {_fmt(row['QWK_non_rejected'])}",
                    f"{_fmt(row['full_Acc@5'])} -> {_fmt(row['Acc@5_non_rejected'])}",
                    _fmt(row["rejected_low_to_high_capture_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selection Notes",
            "",
            f"- Dev candidates evaluated: {len(dev_rows)}.",
            f"- Dev-selected base/method policies: {len(best_rows)}.",
            "- Success criteria: coverage >= 0.80, low_to_high_non_rejected <= 0.35, review_rate <= 0.20, and Acc@5 within 0.05 of the corresponding full-coverage model.",
            "- Exp7-B gate: remains blocked; C3 is a post-hoc human-review option, not evidence for starting more training.",
        ]
    )
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c3_selective_calibration_report.md", "\n".join(lines))

    review_lines = [
        "# Exp7-C3 Review Package",
        "",
        "Status: completed.",
        "",
        f"Best selective policy: {overall['base_model']} / {overall['method']} / {overall['risk_policy']}.",
        f"Review rate: {_fmt(overall['review_rate'])}; coverage: {_fmt(overall['coverage'])}.",
        f"Low_to_high full -> non-rejected: {_fmt(overall['full_low_to_high_rate'])} -> {_fmt(overall['low_to_high_non_rejected'])}.",
        f"Acc@5 full -> non-rejected: {_fmt(overall['full_Acc@5'])} -> {_fmt(overall['Acc@5_non_rejected'])}.",
        f"Captured low_to_high errors: {low_capture}/{potential_low}.",
        "",
        "Conclusion: usable as human-in-the-loop selective scoring if the thesis accepts review budget as part of the method; not a full-coverage automatic scoring improvement.",
        "",
        "No training, API calls, synthetic generation, raw prediction edits, or checkpoint outputs were used.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c3_review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp7-C3 Selective Calibration Summary",
        "",
        f"Best: {overall['base_model']} / {overall['method']} / {overall['risk_policy']} at budget {_fmt(overall['review_budget'], 2)}.",
        f"Test review_rate={_fmt(overall['review_rate'])}, coverage={_fmt(overall['coverage'])}.",
        f"Low_to_high full -> non-rejected: {_fmt(overall['full_low_to_high_rate'])} -> {_fmt(overall['low_to_high_non_rejected'])}.",
        f"Acc@5 non-rejected: {_fmt(overall['Acc@5_non_rejected'])}.",
        f"Rejected low_to_high capture: {low_capture}/{potential_low} ({_fmt(overall['rejected_low_to_high_capture_rate'])}).",
        f"Final framing: {'human-in-the-loop thesis method candidate' if success else 'diagnostic only for now'}.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "notion_exp07_c3_selective_calibration_summary.md", "\n".join(notion_lines))


def run() -> None:
    ensure_exp07_calibration_dirs()
    server_rows = server_artifact_inventory()
    bases = load_available_bases(_inventory_for_loader(server_rows))
    risk_thresholds = _load_risk_thresholds()
    dev_rows: list[dict[str, Any]] = []

    for base in bases:
        for method in ["raw", "risk_aware_threshold"]:
            risk_config = risk_thresholds.get(base.base_model, {})
            if method == "risk_aware_threshold" and not risk_config:
                continue
            thresholds = RAW_THRESHOLDS if method == "raw" else risk_config["thresholds"]
            dev_pred, _test_pred = _method_predictions(base, method, thresholds)
            dev_rows.extend(
                _evaluate_candidate_rows(
                    base=base,
                    method=method,
                    thresholds=thresholds,
                    pred=dev_pred,
                    split_data=base.dev,
                    lambda_low=risk_config.get("lambda_low", "") if method == "risk_aware_threshold" else "",
                    eta_high=risk_config.get("eta_high", "") if method == "risk_aware_threshold" else "",
                )
            )

    best_rows = _select_best_rows(dev_rows)
    test_rows = _evaluate_selected_test_rows(bases=bases, best_rows=best_rows, risk_thresholds=risk_thresholds)
    rejected_rows = _rejected_distribution_rows(best_rows + test_rows)

    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_calibration_dev.csv", dev_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_best_policies.csv", best_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_calibration_test.csv", test_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_rejected_distribution.csv", rejected_rows)
    _write_report(dev_rows=dev_rows, best_rows=best_rows, test_rows=test_rows)
    print(f"Wrote Exp7-C3 selective calibration outputs: {EXP07_CALIBRATION_TABLES_DIR.parent}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
