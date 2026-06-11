"""Metrics for Exp8 EduRisk ordinal scorer outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    high_score_metrics,
    low_score_metrics,
    per_bin_metrics,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import compute_metrics as base_compute_metrics
from thesis_exp.src.edujudge.exp04.train_objective import add_extra_metrics, grouped_metrics
from thesis_exp.src.edujudge.exp08_edurisk import DEFAULT_LAMBDA_HL, DEFAULT_LAMBDA_LH
from thesis_exp.src.edujudge.exp08_edurisk.coral_distribution import (
    expected_score_from_q,
    monotonicity_metrics,
    pred_label_argmax,
    pred_label_cumulative,
)
from thesis_exp.src.edujudge.exp08_edurisk.losses import make_edurisk_cost_matrix
from thesis_exp.src.edujudge.utils.io import write_csv


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = base_compute_metrics(predictions)
    return add_extra_metrics(metrics, predictions, "ordinal")


def expected_risk_values(
    labels: Any,
    q_raw: Any,
    lambda_lh: float = DEFAULT_LAMBDA_LH,
    lambda_hl: float = DEFAULT_LAMBDA_HL,
    normalized_cost: bool = True,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    q_arr = np.asarray(q_raw, dtype=np.float64)
    costs = np.asarray(
        make_edurisk_cost_matrix(
            labels_arr,
            lambda_lh=lambda_lh,
            lambda_hl=lambda_hl,
            normalized_cost=normalized_cost,
        ),
        dtype=np.float64,
    )
    return np.sum(q_arr * costs, axis=1)


def expected_risk_metric_row(split: str, labels: Any, q_raw: Any) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    risks = expected_risk_values(labels_arr, q_raw)
    if risks.size == 0:
        return {
            "split": split,
            "n": 0,
            "expected_edurisk": float("nan"),
            "expected_edurisk_low": float("nan"),
            "expected_edurisk_mid": float("nan"),
            "expected_edurisk_high": float("nan"),
        }
    low_mask = labels_arr <= 2
    high_mask = labels_arr == 5
    mid_mask = ~(low_mask | high_mask)
    return {
        "split": split,
        "n": int(risks.size),
        "expected_edurisk": float(np.mean(risks)),
        "expected_edurisk_low": float(np.mean(risks[low_mask])) if low_mask.any() else float("nan"),
        "expected_edurisk_mid": float(np.mean(risks[mid_mask])) if mid_mask.any() else float("nan"),
        "expected_edurisk_high": float(np.mean(risks[high_mask])) if high_mask.any() else float("nan"),
    }


def metrics_with_edurisk(
    predictions: list[dict[str, Any]],
    probs: Any,
    q_raw: Any,
    labels: Any,
    split: str,
) -> dict[str, Any]:
    metrics = compute_metrics(predictions)
    metrics.update(monotonicity_metrics(probs))
    metrics.update(expected_risk_metric_row(split, labels, q_raw))
    metrics["split"] = split
    return metrics


def prediction_distribution_rows(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    y_pred = [int(row["pred_label_5"]) for row in predictions]
    y_true = [int(row["label_5"]) for row in predictions]
    rows = []
    n = len(y_pred)
    for label in [1, 2, 3, 4, 5]:
        rows.append(
            {
                "split": split,
                "label_5": label,
                "true_count": sum(1 for value in y_true if value == label),
                "pred_count": sum(1 for value in y_pred if value == label),
                "pred_rate": sum(1 for value in y_pred if value == label) / n if n else float("nan"),
            }
        )
    return rows


def cumulative_vs_argmax_rows(
    base_predictions: list[dict[str, Any]],
    probs: Any,
    q_raw: Any,
    split: str,
) -> list[dict[str, Any]]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    q_arr = np.asarray(q_raw, dtype=np.float64)
    cumulative = pred_label_cumulative(probs_arr)
    argmax = pred_label_argmax(q_arr)
    expected = expected_score_from_q(q_arr)
    rows: list[dict[str, Any]] = []
    for decode_name, preds in [("cumulative", cumulative), ("argmax_q", argmax)]:
        adjusted = []
        for idx, row in enumerate(base_predictions):
            clone = dict(row)
            pred = int(preds[idx])
            human_mean = float(clone["human_mean_5"])
            clone["pred_label_5"] = pred
            clone["pred_label"] = pred
            clone["pred_score_5"] = float(pred)
            clone["pred_score_expected"] = float(expected[idx])
            clone["abs_error_label"] = abs(float(pred) - human_mean)
            clone["signed_error_label"] = float(pred) - human_mean
            clone["abs_error_expected"] = abs(float(expected[idx]) - human_mean)
            clone["signed_error_expected"] = float(expected[idx]) - human_mean
            adjusted.append(clone)
        metrics = compute_metrics(adjusted)
        metrics.update(monotonicity_metrics(probs_arr))
        metrics["split"] = split
        metrics["decode_method"] = decode_name
        metrics["label_changed_rate_vs_cumulative"] = (
            float(np.mean(argmax != cumulative)) if decode_name == "argmax_q" and cumulative.size else 0.0
        )
        rows.append(metrics)
    return rows


def save_metric_tables(output_dir: Any, results: list[Any]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [result.metrics for result in results]
    per_bin_rows = []
    low_rows = []
    high_rows = []
    metric_rows = []
    scenario_rows = []
    language_rows = []
    distribution_rows = []
    monotonic_rows = []
    risk_rows = []
    decoding_rows = []
    for result in results:
        split = result.metrics["split"]
        per_bin_rows.extend(per_bin_metrics(result.predictions, split))
        low_rows.append(low_score_metrics(result.metrics, split))
        high_rows.append(high_score_metrics(result.metrics, split))
        metric_rows.extend(grouped_metrics(result.predictions, split, "metric_canonical", "ordinal"))
        scenario_rows.extend(grouped_metrics(result.predictions, split, "scenario_canonical", "ordinal"))
        language_rows.extend(grouped_metrics(result.predictions, split, "language", "ordinal"))
        distribution_rows.extend(prediction_distribution_rows(result.predictions, split))
        monotonic_rows.append(monotonicity_metrics(result.probs, split))
        risk_rows.append(expected_risk_metric_row(split, result.labels, result.q_raw))
        decoding_rows.extend(cumulative_vs_argmax_rows(result.predictions, result.probs, result.q_raw, split))
    write_csv(tables_dir / "metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "per_bin_metrics.csv", per_bin_rows)
    write_csv(tables_dir / "low_score_metrics.csv", low_rows)
    write_csv(tables_dir / "high_score_metrics.csv", high_rows)
    write_csv(tables_dir / "metric_level_metrics.csv", metric_rows)
    write_csv(tables_dir / "scenario_level_metrics.csv", scenario_rows)
    write_csv(tables_dir / "language_level_metrics.csv", language_rows)
    write_csv(tables_dir / "score_distribution.csv", distribution_rows)
    write_csv(tables_dir / "monotonicity_metrics.csv", monotonic_rows)
    write_csv(tables_dir / "expected_risk_metrics.csv", risk_rows)
    write_csv(tables_dir / "cumulative_decoding_vs_argmax_decoding.csv", decoding_rows)
