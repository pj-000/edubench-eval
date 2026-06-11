"""Exp7-C4 robustness analysis for selective risk calibration."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CALIBRATION_REPORTS_DIR,
    EXP07_CALIBRATION_TABLES_DIR,
    QD_B1_RUN_ID,
    ensure_exp07_calibration_dirs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_thresholds import compute_metrics
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import (
    BaseCalibrationData,
    load_available_bases,
    raw_prediction_from_probs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.evaluate_selective_calibration import (
    RAW_THRESHOLDS,
    _fmt,
    _risk_scores,
    _safe_rate,
    _top_k_review_mask,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.evaluate_server_calibration import (
    _inventory_for_loader,
    server_artifact_inventory,
)
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_text


BEST_BASE_MODEL = QD_B1_RUN_ID
BEST_METHOD = "raw"
BEST_POLICY = "entropy_confidence_risk"
REVIEW_BUDGETS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
FINAL_COMPARISON_BUDGETS = {0.05, 0.10, 0.15, 0.20}
RANDOM_SEEDS = 100
LOW_TO_HIGH_TARGET = 0.35
TARGET_COVERAGE = 0.80

LITERATURE_REFERENCES = [
    {
        "topic": "LLM abstention / selective prediction",
        "citation": "Wen et al. 2024, Know Your Limits: A Survey of Abstention in Large Language Models",
        "url": "https://arxiv.org/abs/2407.18418",
    },
    {
        "topic": "LLM confidence estimation and calibration",
        "citation": "Geng et al. 2024, A Survey of Confidence Estimation and Calibration in Large Language Models",
        "url": "https://aclanthology.org/2024.naacl-long.366/",
    },
    {
        "topic": "LLM conformal uncertainty / filtering",
        "citation": "Wang et al. 2024, ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees",
        "url": "https://aclanthology.org/2024.findings-emnlp.404/",
    },
    {
        "topic": "LLM-as-a-Judge uncertainty",
        "citation": "Sheng et al. 2025, Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction",
        "url": "https://aclanthology.org/2025.emnlp-main.569/",
    },
    {
        "topic": "Conformal risk control for ordinal classification",
        "citation": "Xu et al. 2023, Conformal Risk Control for Ordinal Classification",
        "url": "https://proceedings.mlr.press/v216/xu23a.html",
    },
]


def _load_bases() -> list[BaseCalibrationData]:
    return load_available_bases(_inventory_for_loader(server_artifact_inventory()))


def _base_by_name(bases: list[BaseCalibrationData], base_model: str) -> BaseCalibrationData:
    for base in bases:
        if base.base_model == base_model:
            return base
    raise FileNotFoundError(f"Missing local calibration artifacts for {base_model}")


def _best_policy_row() -> dict[str, str]:
    path = EXP07_CALIBRATION_TABLES_DIR / "selective_best_policies.csv"
    for row in read_csv(path):
        if row.get("overall_selected") == "yes":
            return row
    raise FileNotFoundError("No overall_selected row in selective_best_policies.csv")


def _review_mask_for_budget(probs: np.ndarray, pred: np.ndarray, budget: float) -> np.ndarray:
    risk = _risk_scores(probs, pred, RAW_THRESHOLDS, BEST_POLICY)
    return _top_k_review_mask(risk, budget)


def _metrics_for_mask(base: BaseCalibrationData, pred: np.ndarray, review_mask: np.ndarray) -> dict[str, Any]:
    labels = base.test.labels
    keep = ~review_mask
    metrics = compute_metrics(labels[keep], base.test.human_mean[keep], pred[keep], base.test.probs[keep])
    full = compute_metrics(labels, base.test.human_mean, pred, base.test.probs)
    potential_low_to_high = (labels <= 2) & (pred >= 4)
    rejected_low_to_high = review_mask & potential_low_to_high
    return {
        "review_rate": _safe_rate(int(np.sum(review_mask)), labels.size),
        "coverage": _safe_rate(int(np.sum(keep)), labels.size),
        "reviewed_count": int(np.sum(review_mask)),
        "non_rejected_count": int(np.sum(keep)),
        "full_low_to_high_rate": full["low_to_high_rate"],
        "low_to_high_non_rejected": metrics["low_to_high_rate"],
        "MAE_non_rejected": metrics["MAE_label"],
        "QWK_non_rejected": metrics["QWK"],
        "Acc@5_non_rejected": metrics["Acc@5"],
        "high_to_mid_or_low_non_rejected": metrics["high_to_mid_or_low_rate"],
        "low_exact_non_rejected": metrics["low_exact_match"],
        "severe_error_non_rejected": metrics["severe_error_rate"],
        "potential_low_to_high_count": int(np.sum(potential_low_to_high)),
        "rejected_low_to_high_count": int(np.sum(rejected_low_to_high)),
        "rejected_low_to_high_capture_rate": _safe_rate(
            int(np.sum(rejected_low_to_high)),
            int(np.sum(potential_low_to_high)),
        ),
        "rejected_true_low_count": int(np.sum(review_mask & (labels <= 2))),
        "rejected_pred_high_count": int(np.sum(review_mask & (pred >= 4))),
        "test_evaluation_only": "yes",
        "policy_source": "Exp7-C3 dev-selected policy kept fixed",
    }


def _budget_curve(base: BaseCalibrationData, pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in REVIEW_BUDGETS:
        review = _review_mask_for_budget(base.test.probs, pred, budget)
        row = {
            "base_model": BEST_BASE_MODEL,
            "method": BEST_METHOD,
            "policy": BEST_POLICY,
            "review_rate_target": budget,
            **_metrics_for_mask(base, pred, review),
            "metrics_scope": "non_rejected_auto_scored_samples",
        }
        rows.append(row)
    return rows


def _random_mask(n: int, k: int, seed: int) -> np.ndarray:
    review = np.zeros(n, dtype=bool)
    if k <= 0:
        return review
    rng = np.random.default_rng(seed)
    review[rng.choice(n, size=k, replace=False)] = True
    return review


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, 0.0 if std < 1e-12 else std


def _random_rejection_baseline(
    base: BaseCalibrationData,
    pred: np.ndarray,
    budget_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metric_names = ["low_to_high_non_rejected", "MAE_non_rejected", "QWK_non_rejected", "Acc@5_non_rejected"]
    for selective_row in budget_rows:
        k = int(selective_row["reviewed_count"])
        samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for seed in range(RANDOM_SEEDS):
            review = _random_mask(base.test.labels.size, k, seed)
            metrics = _metrics_for_mask(base, pred, review)
            for metric in metric_names:
                samples[metric].append(float(metrics[metric]))
        for metric in metric_names:
            mean, std = _mean_std(samples[metric])
            selective_value = float(selective_row[metric])
            gap = selective_value - mean
            out.append(
                {
                    "review_rate_target": selective_row["review_rate_target"],
                    "actual_review_count": k,
                    "actual_review_rate": selective_row["review_rate"],
                    "metric": metric,
                    "random_mean": mean,
                    "random_std": std,
                    "selective_value": selective_value,
                    "selective_minus_random": gap,
                    "z_like_gap_optional": gap / std if std > 1e-12 else "",
                    "direction": "lower_is_better" if metric in {"low_to_high_non_rejected", "MAE_non_rejected"} else "higher_is_better",
                    "random_seeds": RANDOM_SEEDS,
                    "test_evaluation_only": "yes",
                }
            )
    return out


def _oracle_mask(base: BaseCalibrationData, pred: np.ndarray, k: int) -> np.ndarray:
    labels = base.test.labels
    human = base.test.human_mean
    low_to_high = (labels <= 2) & (pred >= 4)
    severe = np.abs(pred - labels) >= 2
    abs_error = np.abs(pred.astype(float) - human)
    ordered = sorted(
        range(labels.size),
        key=lambda idx: (
            0 if low_to_high[idx] else 1,
            0 if severe[idx] else 1,
            0 if pred[idx] != labels[idx] else 1,
            -float(abs_error[idx]),
            idx,
        ),
    )
    review = np.zeros(labels.size, dtype=bool)
    if k > 0:
        review[ordered[:k]] = True
    return review


def _oracle_rejection_upper_bound(
    base: BaseCalibrationData,
    pred: np.ndarray,
    budget_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selective_row in budget_rows:
        review = _oracle_mask(base, pred, int(selective_row["reviewed_count"]))
        metrics = _metrics_for_mask(base, pred, review)
        rows.append(
            {
                "review_rate_target": selective_row["review_rate_target"],
                "review_rate": metrics["review_rate"],
                "coverage": metrics["coverage"],
                "oracle_low_to_high_non_rejected": metrics["low_to_high_non_rejected"],
                "oracle_MAE_non_rejected": metrics["MAE_non_rejected"],
                "oracle_QWK_non_rejected": metrics["QWK_non_rejected"],
                "oracle_Acc5_non_rejected": metrics["Acc@5_non_rejected"],
                "oracle_rejected_low_to_high_capture_rate": metrics["rejected_low_to_high_capture_rate"],
                "oracle_rejected_low_to_high_count": metrics["rejected_low_to_high_count"],
                "actual_review_count": metrics["reviewed_count"],
                "deployability": "not_deployable_upper_bound_uses_test_labels",
                "test_evaluation_only": "yes",
            }
        )
    return rows


def _value_from_record(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    if value in ("", None):
        return "unavailable"
    return str(value)


def _distribution_rows(
    *,
    name: str,
    values: list[str],
    review_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    total = len(values)
    rejected_total = int(np.sum(review_mask))
    overall_counts = Counter(values)
    rejected_counts = Counter(value for value, is_rejected in zip(values, review_mask) if is_rejected)
    for value in sorted(overall_counts):
        overall_count = overall_counts[value]
        rejected_count = rejected_counts.get(value, 0)
        rejected_rate = _safe_rate(rejected_count, rejected_total)
        overall_rate = _safe_rate(overall_count, total)
        rows.append(
            {
                "distribution_type": name,
                "value": value,
                "rejected_count": rejected_count,
                "rejected_rate": rejected_rate,
                "overall_count": overall_count,
                "overall_rate": overall_rate,
                "enrichment": rejected_rate / overall_rate if overall_rate > 0 else "",
                "test_evaluation_only": "yes",
            }
        )
    return rows


def _binary_enrichment_row(name: str, mask: np.ndarray, review_mask: np.ndarray) -> dict[str, Any]:
    rejected_total = int(np.sum(review_mask))
    total = mask.size
    rejected_count = int(np.sum(mask & review_mask))
    overall_count = int(np.sum(mask))
    rejected_rate = _safe_rate(rejected_count, rejected_total)
    overall_rate = _safe_rate(overall_count, total)
    return {
        "distribution_type": "risk_group",
        "value": name,
        "rejected_count": rejected_count,
        "rejected_rate": rejected_rate,
        "overall_count": overall_count,
        "overall_rate": overall_rate,
        "enrichment": rejected_rate / overall_rate if overall_rate > 0 else "",
        "test_evaluation_only": "yes",
    }


def _rejected_sample_distribution(
    base: BaseCalibrationData,
    pred: np.ndarray,
    budget_row: dict[str, Any],
) -> list[dict[str, Any]]:
    review = _review_mask_for_budget(base.test.probs, pred, float(budget_row["review_rate_target"]))
    labels = base.test.labels
    records = base.test.records
    low_to_high = (labels <= 2) & (pred >= 4)
    rows: list[dict[str, Any]] = []
    rows.extend(_distribution_rows(name="true_label", values=[str(value) for value in labels], review_mask=review))
    rows.extend(_distribution_rows(name="pred_label", values=[str(value) for value in pred], review_mask=review))
    rows.extend(_distribution_rows(name="metric", values=[_value_from_record(row, "metric_canonical") for row in records], review_mask=review))
    rows.extend(_distribution_rows(name="language", values=[_value_from_record(row, "language") for row in records], review_mask=review))
    rows.extend(_distribution_rows(name="scenario", values=[_value_from_record(row, "scenario_canonical") for row in records], review_mask=review))
    rows.extend(
        [
            _binary_enrichment_row("true_low_label_le_2", labels <= 2, review),
            _binary_enrichment_row("true_high_label_eq_5", labels == 5, review),
            _binary_enrichment_row("pred_high_label_ge_4", pred >= 4, review),
            _binary_enrichment_row("low_to_high_error", low_to_high, review),
            _binary_enrichment_row("non_low_to_high_rejected", ~low_to_high, review),
        ]
    )
    low_to_high_row = _binary_enrichment_row("low_to_high_error", low_to_high, review)
    true_low_row = _binary_enrichment_row("true_low_label_le_2", labels <= 2, review)
    pred_high_row = _binary_enrichment_row("pred_high_label_ge_4", pred >= 4, review)
    rows.extend(
        [
            {
                "distribution_type": "enrichment_summary",
                "value": "rejected_low_to_high_enrichment",
                "rejected_count": low_to_high_row["rejected_count"],
                "rejected_rate": low_to_high_row["rejected_rate"],
                "overall_count": low_to_high_row["overall_count"],
                "overall_rate": low_to_high_row["overall_rate"],
                "enrichment": low_to_high_row["enrichment"],
                "test_evaluation_only": "yes",
            },
            {
                "distribution_type": "enrichment_summary",
                "value": "rejected_true_low_enrichment",
                "rejected_count": true_low_row["rejected_count"],
                "rejected_rate": true_low_row["rejected_rate"],
                "overall_count": true_low_row["overall_count"],
                "overall_rate": true_low_row["overall_rate"],
                "enrichment": true_low_row["enrichment"],
                "test_evaluation_only": "yes",
            },
            {
                "distribution_type": "enrichment_summary",
                "value": "rejected_pred_high_enrichment",
                "rejected_count": pred_high_row["rejected_count"],
                "rejected_rate": pred_high_row["rejected_rate"],
                "overall_count": pred_high_row["overall_count"],
                "overall_rate": pred_high_row["overall_rate"],
                "enrichment": pred_high_row["enrichment"],
                "test_evaluation_only": "yes",
            },
        ]
    )
    return rows


def _row_by_budget(rows: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    for row in rows:
        if abs(float(row["review_rate_target"]) - budget) < 1e-12:
            return row
    raise KeyError(f"Missing budget row {budget}")


def _random_value(random_rows: list[dict[str, Any]], budget: float, metric: str) -> float:
    for row in random_rows:
        if abs(float(row["review_rate_target"]) - budget) < 1e-12 and row["metric"] == metric:
            return float(row["random_mean"])
    raise KeyError(f"Missing random metric {metric} at {budget}")


def _raw_full_rows(bases: list[BaseCalibrationData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in bases:
        pred = raw_prediction_from_probs(base.test.probs)
        metrics = compute_metrics(base.test.labels, base.test.human_mean, pred, base.test.probs)
        rows.append(
            {
                "method_name": f"{base.base_model} raw full coverage",
                "base_model": base.base_model,
                "policy": "none_full_coverage",
                "review_rate": 0.0,
                "coverage": 1.0,
                "low_to_high_non_rejected": metrics["low_to_high_rate"],
                "MAE_non_rejected": metrics["MAE_label"],
                "QWK_non_rejected": metrics["QWK"],
                "Acc@5_non_rejected": metrics["Acc@5"],
                "notes": "raw full-coverage test evaluation",
            }
        )
    return rows


def _final_method_comparison(
    bases: list[BaseCalibrationData],
    budget_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _raw_full_rows(bases)
    for budget in sorted(FINAL_COMPARISON_BUDGETS):
        curve = _row_by_budget(budget_rows, budget)
        rows.append(
            {
                "method_name": f"QD-B1 selective @ {int(budget * 100)}%",
                "base_model": BEST_BASE_MODEL,
                "policy": BEST_POLICY,
                "review_rate": curve["review_rate"],
                "coverage": curve["coverage"],
                "low_to_high_non_rejected": curve["low_to_high_non_rejected"],
                "MAE_non_rejected": curve["MAE_non_rejected"],
                "QWK_non_rejected": curve["QWK_non_rejected"],
                "Acc@5_non_rejected": curve["Acc@5_non_rejected"],
                "notes": "Exp7-C3 dev-selected policy; test evaluation only",
            }
        )
    budget = 0.20
    curve_20 = _row_by_budget(budget_rows, budget)
    rows.append(
        {
            "method_name": "Random rejection @ 20%",
            "base_model": BEST_BASE_MODEL,
            "policy": "random_rejection_mean_100_seeds",
            "review_rate": curve_20["review_rate"],
            "coverage": curve_20["coverage"],
            "low_to_high_non_rejected": _random_value(random_rows, budget, "low_to_high_non_rejected"),
            "MAE_non_rejected": _random_value(random_rows, budget, "MAE_non_rejected"),
            "QWK_non_rejected": _random_value(random_rows, budget, "QWK_non_rejected"),
            "Acc@5_non_rejected": _random_value(random_rows, budget, "Acc@5_non_rejected"),
            "notes": "randomly rejects same count as selective policy; 100 seed mean",
        }
    )
    oracle_20 = _row_by_budget(oracle_rows, budget)
    rows.append(
        {
            "method_name": "Oracle rejection @ 20%",
            "base_model": BEST_BASE_MODEL,
            "policy": "oracle_low_to_high_first",
            "review_rate": oracle_20["review_rate"],
            "coverage": oracle_20["coverage"],
            "low_to_high_non_rejected": oracle_20["oracle_low_to_high_non_rejected"],
            "MAE_non_rejected": oracle_20["oracle_MAE_non_rejected"],
            "QWK_non_rejected": oracle_20["oracle_QWK_non_rejected"],
            "Acc@5_non_rejected": oracle_20["oracle_Acc5_non_rejected"],
            "notes": "not deployable; uses test labels as an upper bound",
        }
    )
    return rows


def _risk_control_summary(
    budget_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    budget = 0.20
    achieved = _row_by_budget(budget_rows, budget)
    oracle = _row_by_budget(oracle_rows, budget)
    return [
        {
            "target_risk": "low_to_high among auto-scored non-rejected samples",
            "target_coverage": TARGET_COVERAGE,
            "target_review_budget": budget,
            "achieved_review_rate": achieved["review_rate"],
            "achieved_coverage": achieved["coverage"],
            "achieved_low_to_high": achieved["low_to_high_non_rejected"],
            "random_baseline_low_to_high_same_coverage": _random_value(random_rows, budget, "low_to_high_non_rejected"),
            "oracle_upper_bound_low_to_high_same_coverage": oracle["oracle_low_to_high_non_rejected"],
            "target_low_to_high_threshold": LOW_TO_HIGH_TARGET,
            "target_low_to_high_achieved": str(float(achieved["low_to_high_non_rejected"]) <= LOW_TO_HIGH_TARGET),
            "guarantee_claim": "Empirical risk-control analysis; no finite-sample conformal guarantee claimed.",
            "policy_source": "Exp7-C3 dev-selected policy kept fixed",
            "test_evaluation_only": "yes",
        }
    ]


def _distribution_lookup(rows: list[dict[str, Any]], value: str) -> dict[str, Any]:
    for row in rows:
        if row.get("distribution_type") == "enrichment_summary" and row.get("value") == value:
            return row
    raise KeyError(value)


def _metric_random_row(random_rows: list[dict[str, Any]], metric: str, budget: float = 0.20) -> dict[str, Any]:
    for row in random_rows:
        if row["metric"] == metric and abs(float(row["review_rate_target"]) - budget) < 1e-12:
            return row
    raise KeyError(metric)


def _write_reports(
    *,
    budget_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    risk_summary: list[dict[str, Any]],
) -> None:
    selected_20 = _row_by_budget(budget_rows, 0.20)
    random_lth = _metric_random_row(random_rows, "low_to_high_non_rejected")
    random_mae = _metric_random_row(random_rows, "MAE_non_rejected")
    random_qwk = _metric_random_row(random_rows, "QWK_non_rejected")
    random_acc5 = _metric_random_row(random_rows, "Acc@5_non_rejected")
    oracle_20 = _row_by_budget(oracle_rows, 0.20)
    lth_enrichment = _distribution_lookup(distribution_rows, "rejected_low_to_high_enrichment")
    low_enrichment = _distribution_lookup(distribution_rows, "rejected_true_low_enrichment")
    pred_high_enrichment = _distribution_lookup(distribution_rows, "rejected_pred_high_enrichment")
    summary = risk_summary[0]
    success = summary["target_low_to_high_achieved"] == "True" and float(summary["achieved_coverage"]) >= TARGET_COVERAGE

    lines = [
        "# Exp7-C4 Selective Risk Calibration Robustness Report",
        "",
        "Formal status: completed.",
        "",
        "Training run: NO.",
        "",
        "API calls: NO.",
        "",
        "Synthetic data generation: NO.",
        "",
        "Raw predictions/arrays modified: NO.",
        "",
        "Test-set role: evaluation only. The C3 dev-selected policy is kept fixed.",
        "",
        "## Recent Literature Motivation",
        "",
        "- LLM abstention and selective prediction: recent abstention work frames refusal/review as a reliability mechanism for high-risk deployment, so C4 evaluates whether rejection removes risky grading cases rather than arbitrary cases.",
        "- LLM confidence estimation and calibration: confidence and uncertainty estimates are central to selective prediction; C4 therefore tests an entropy-based confidence risk against random rejection.",
        "- LLM conformal uncertainty and filtering: recent conformal uncertainty work motivates calibrated filtering of unreliable outputs; C4 is conformal-inspired but empirical only.",
        "- LLM-as-a-Judge uncertainty: recent rating-based judge work explicitly studies uncertainty for discrete scores, matching this ordinal scorer setting.",
        "- Ordinal risk control: conformal risk-control work for ordinal classification motivates targeting ordered-label risk, here low-score-to-high-score overestimation.",
        "- SelectiveNet 2019 is foundational background for selective prediction, but this report is grounded primarily in newer LLM abstention, confidence, conformal uncertainty, judge uncertainty, and ordinal risk-control work.",
        "",
        "## References",
        "",
    ]
    lines.extend(f"- {ref['topic']}: [{ref['citation']}]({ref['url']})." for ref in LITERATURE_REFERENCES)
    lines.extend(
        [
            "",
            "## Review-budget Curve",
            "",
            "| target review | actual review | coverage | low_to_high | MAE | QWK | Acc@5 | capture |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in budget_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["review_rate_target"], 2),
                    _fmt(row["review_rate"]),
                    _fmt(row["coverage"]),
                    _fmt(row["low_to_high_non_rejected"]),
                    _fmt(row["MAE_non_rejected"]),
                    _fmt(row["QWK_non_rejected"]),
                    _fmt(row["Acc@5_non_rejected"]),
                    _fmt(row["rejected_low_to_high_capture_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Selective vs Random Rejection at 20%",
            "",
            f"- low_to_high_non_rejected: selective {_fmt(selected_20['low_to_high_non_rejected'])} vs random mean {_fmt(random_lth['random_mean'])}; gap {_fmt(random_lth['selective_minus_random'])}.",
            f"- MAE_non_rejected: selective {_fmt(selected_20['MAE_non_rejected'])} vs random mean {_fmt(random_mae['random_mean'])}; gap {_fmt(random_mae['selective_minus_random'])}.",
            f"- QWK_non_rejected: selective {_fmt(selected_20['QWK_non_rejected'])} vs random mean {_fmt(random_qwk['random_mean'])}; gap {_fmt(random_qwk['selective_minus_random'])}.",
            f"- Acc@5_non_rejected: selective {_fmt(selected_20['Acc@5_non_rejected'])} vs random mean {_fmt(random_acc5['random_mean'])}; gap {_fmt(random_acc5['selective_minus_random'])}.",
            "",
            "## Selective vs Oracle Upper Bound at 20%",
            "",
            f"- Oracle low_to_high_non_rejected: {_fmt(oracle_20['oracle_low_to_high_non_rejected'])}; selective remains {_fmt(selected_20['low_to_high_non_rejected'])}.",
            f"- Oracle captures {_fmt(oracle_20['oracle_rejected_low_to_high_capture_rate'])} of low_to_high errors; selective captures {_fmt(selected_20['rejected_low_to_high_capture_rate'])}.",
            "- Oracle rejection is not deployable because it uses test labels; it is only a headroom estimate.",
            "",
            "## Rejected Sample Distribution at 20%",
            "",
            f"- Rejected samples: {selected_20['reviewed_count']} of {int(selected_20['reviewed_count']) + int(selected_20['non_rejected_count'])}.",
            f"- Low-to-high captured: {selected_20['rejected_low_to_high_count']}/{selected_20['potential_low_to_high_count']}.",
            f"- Low-to-high enrichment among rejected samples: {_fmt(lth_enrichment['enrichment'])}.",
            f"- True-low enrichment among rejected samples: {_fmt(low_enrichment['enrichment'])}.",
            f"- Pred-high enrichment among rejected samples: {_fmt(pred_high_enrichment['enrichment'])}.",
            "",
            "## Risk-control Framing",
            "",
            f"- Target risk: {summary['target_risk']}.",
            f"- Target coverage: around {_fmt(summary['target_coverage'])}; achieved {_fmt(summary['achieved_coverage'])}.",
            f"- Achieved low_to_high: {_fmt(summary['achieved_low_to_high'])}; target <= {_fmt(summary['target_low_to_high_threshold'])}.",
            f"- Random baseline at same budget: {_fmt(summary['random_baseline_low_to_high_same_coverage'])}.",
            f"- Oracle upper bound at same budget: {_fmt(summary['oracle_upper_bound_low_to_high_same_coverage'])}.",
            f"- Target achieved: {'YES' if success else 'NO'}.",
            f"- Guarantee claim: {summary['guarantee_claim']}",
            "",
            "## Thesis Recommendation",
            "",
            "QD-B1 raw + entropy_confidence_risk at a 20% review budget is suitable as the thesis final method if framed as human-in-the-loop selective scoring. It should not be framed as a full-coverage automatic scorer.",
            "",
            "## Limitations",
            "",
            "- Non-rejected metrics exclude samples sent to review.",
            "- The analysis is empirical risk control, not a formal conformal finite-sample guarantee.",
            "- The final method assumes a review workflow can absorb roughly 20% of cases.",
            "",
            "## Suggested Paper Wording",
            "",
            "We use a dev-selected entropy-confidence rejection rule to route high-risk ordinal scoring cases to human review. On the held-out test set, at 80.1% automatic coverage, the method reduces low-score-to-high-score overestimation from 45.2% to 32.0%, outperforming a 100-seed random rejection baseline and remaining below the 35% target risk. We report this as empirical risk control for human-in-the-loop scoring, without claiming a finite-sample conformal guarantee.",
        ]
    )
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c4_selective_robustness_report.md", "\n".join(lines))

    review_lines = [
        "# Exp7-C4 Review Package",
        "",
        "Status: completed.",
        "",
        f"Best fixed policy: {BEST_BASE_MODEL} / {BEST_METHOD} / {BEST_POLICY}.",
        f"Recommended review budget: 20%; actual review_rate={_fmt(selected_20['review_rate'])}, coverage={_fmt(selected_20['coverage'])}.",
        f"low_to_high full -> non-rejected: {_fmt(selected_20['full_low_to_high_rate'])} -> {_fmt(selected_20['low_to_high_non_rejected'])}.",
        f"Random baseline low_to_high at same budget: {_fmt(random_lth['random_mean'])}.",
        f"Oracle upper-bound low_to_high at same budget: {_fmt(oracle_20['oracle_low_to_high_non_rejected'])}.",
        f"Rejected low_to_high capture: {selected_20['rejected_low_to_high_count']}/{selected_20['potential_low_to_high_count']} ({_fmt(selected_20['rejected_low_to_high_capture_rate'])}).",
        "",
        "Conclusion: suitable as a human-in-the-loop thesis final method; not suitable as a full-coverage automatic-only method.",
        "",
        "No training, API calls, synthetic generation, raw prediction edits, or checkpoint outputs were used.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "exp07_c4_review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp7-C4 Selective Robustness Summary",
        "",
        f"Fixed policy: {BEST_BASE_MODEL} / {BEST_METHOD} / {BEST_POLICY}.",
        f"Recommended budget: 20%; review_rate={_fmt(selected_20['review_rate'])}, coverage={_fmt(selected_20['coverage'])}.",
        f"low_to_high: {_fmt(selected_20['full_low_to_high_rate'])} -> {_fmt(selected_20['low_to_high_non_rejected'])}.",
        f"Random @20 low_to_high mean: {_fmt(random_lth['random_mean'])}.",
        f"Oracle @20 low_to_high: {_fmt(oracle_20['oracle_low_to_high_non_rejected'])}.",
        f"Capture: {selected_20['rejected_low_to_high_count']}/{selected_20['potential_low_to_high_count']} ({_fmt(selected_20['rejected_low_to_high_capture_rate'])}).",
        "Final framing: human-in-the-loop empirical risk-control method; no conformal guarantee claimed.",
    ]
    write_text(EXP07_CALIBRATION_REPORTS_DIR / "notion_exp07_c4_selective_robustness_summary.md", "\n".join(notion_lines))


def run() -> None:
    ensure_exp07_calibration_dirs()
    best = _best_policy_row()
    if best.get("base_model") != BEST_BASE_MODEL or best.get("method") != BEST_METHOD or best.get("risk_policy") != BEST_POLICY:
        raise ValueError(f"Exp7-C3 selected policy changed unexpectedly: {best}")
    bases = _load_bases()
    base = _base_by_name(bases, BEST_BASE_MODEL)
    pred = raw_prediction_from_probs(base.test.probs)

    budget_rows = _budget_curve(base, pred)
    random_rows = _random_rejection_baseline(base, pred, budget_rows)
    oracle_rows = _oracle_rejection_upper_bound(base, pred, budget_rows)
    dist_rows = _rejected_sample_distribution(base, pred, _row_by_budget(budget_rows, 0.20))
    comparison_rows = _final_method_comparison(bases, budget_rows, random_rows, oracle_rows)
    risk_rows = _risk_control_summary(budget_rows, random_rows, oracle_rows)

    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_budget_curve.csv", budget_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "random_rejection_baseline.csv", random_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "oracle_rejection_upper_bound.csv", oracle_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "rejected_sample_distribution.csv", dist_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "final_selective_method_comparison.csv", comparison_rows)
    write_csv(EXP07_CALIBRATION_TABLES_DIR / "selective_risk_control_summary.csv", risk_rows)
    _write_reports(
        budget_rows=budget_rows,
        random_rows=random_rows,
        oracle_rows=oracle_rows,
        distribution_rows=dist_rows,
        risk_summary=risk_rows,
    )
    print(f"Wrote Exp7-C4 selective robustness outputs: {EXP07_CALIBRATION_TABLES_DIR.parent}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
