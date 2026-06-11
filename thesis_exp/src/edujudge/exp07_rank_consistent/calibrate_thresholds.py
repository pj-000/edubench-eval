"""Threshold search and metrics for Exp7-C calibration."""

from __future__ import annotations

import math
from itertools import combinations_with_replacement
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import THRESHOLD_GRID, expected_score_from_probs


METRIC_FIELDS = [
    "n",
    "Accuracy",
    "MAE_label",
    "QWK",
    "Kendall tau",
    "Spearman rho",
    "Macro-F1",
    "severe_error_rate",
    "Acc@1",
    "Acc@2",
    "low_exact_match",
    "low_MAE",
    "low_signed_bias",
    "low_to_high_rate",
    "mean_pred_low",
    "Acc@5",
    "high_to_mid_or_low_rate",
    "high_signed_bias",
    "mean_pred_high",
]


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    observed = np.zeros((5, 5), dtype=float)
    for true, pred in zip(y_true, y_pred):
        observed[int(true) - 1, int(pred) - 1] += 1
    if observed.sum() == 0:
        return float("nan")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            weights[i, j] = ((i - j) ** 2) / 16
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return float("nan")
    return 1 - float((weights * observed).sum()) / denominator


def _rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(np.array(x_values, dtype=float), np.array(y_values, dtype=float))[0, 1])


def spearman(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return float("nan")
    return _pearson(_rank_average(x_values), _rank_average(y_values))


def kendall_tau_b(x_values: list[float], y_values: list[float]) -> float:
    n = len(x_values)
    if n < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = (x_values[j] > x_values[i]) - (x_values[j] < x_values[i])
            dy = (y_values[j] > y_values[i]) - (y_values[j] < y_values[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float("nan") if denom == 0 else (concordant - discordant) / denom


def macro_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    f1_values = []
    weighted = []
    for label in range(1, 6):
        true_label = y_true == label
        pred_label = y_pred == label
        tp = int(np.sum(true_label & pred_label))
        fp = int(np.sum(~true_label & pred_label))
        fn = int(np.sum(true_label & ~pred_label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(np.sum(true_label))
        f1_values.append(f1)
        weighted.append(f1 * support)
    total = int(y_true.size)
    return float(np.mean(f1_values)), float(sum(weighted) / total) if total else float("nan")


def predictions_from_thresholds(probs: np.ndarray, thresholds: tuple[float, float, float, float]) -> np.ndarray:
    threshold_arr = np.asarray(thresholds, dtype=float).reshape(1, 4)
    return np.clip(1 + (probs > threshold_arr).sum(axis=1), 1, 5).astype(int)


def quick_metrics(labels: np.ndarray, human_mean: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    low_mask = labels <= 2
    high_mask = labels == 5
    mae = _safe_mean(np.abs(pred.astype(float) - human_mean))
    return {
        "MAE_label": mae,
        "low_to_high_rate": _safe_mean((pred[low_mask] >= 4).astype(float)) if low_mask.any() else float("nan"),
        "Acc@5": _safe_mean((pred[high_mask] == 5).astype(float)) if high_mask.any() else float("nan"),
        "high_to_mid_or_low_rate": _safe_mean((pred[high_mask] <= 3).astype(float)) if high_mask.any() else float("nan"),
    }


def compute_metrics(labels: np.ndarray, human_mean: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    pred_float = pred.astype(float)
    expected = expected_score_from_probs(probs) if probs is not None else pred_float
    label_diffs = pred_float - human_mean
    low_mask = labels <= 2
    high_mask = labels == 5
    macro_f1, weighted_f1 = macro_weighted_f1(labels, pred)
    metrics = {
        "n": int(labels.size),
        "Accuracy": _safe_mean((labels == pred).astype(float)),
        "MAE_label": _safe_mean(np.abs(label_diffs)),
        "MAE_expected": _safe_mean(np.abs(expected - human_mean)),
        "QWK": qwk(labels, pred),
        "Kendall tau": kendall_tau_b(human_mean.tolist(), pred_float.tolist()),
        "Spearman rho": spearman(human_mean.tolist(), pred_float.tolist()),
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "severe_error_rate": _safe_mean((np.abs(pred - labels) >= 2).astype(float)),
        "Acc@1": _safe_mean((pred[labels == 1] == 1).astype(float)) if np.any(labels == 1) else float("nan"),
        "Acc@2": _safe_mean((pred[labels == 2] == 2).astype(float)) if np.any(labels == 2) else float("nan"),
        "low_exact_match": _safe_mean((pred[low_mask] == labels[low_mask]).astype(float)) if low_mask.any() else float("nan"),
        "low_MAE": _safe_mean(np.abs(label_diffs[low_mask])) if low_mask.any() else float("nan"),
        "low_signed_bias": _safe_mean(label_diffs[low_mask]) if low_mask.any() else float("nan"),
        "low_to_high_rate": _safe_mean((pred[low_mask] >= 4).astype(float)) if low_mask.any() else float("nan"),
        "mean_pred_low": _safe_mean(pred_float[low_mask]) if low_mask.any() else float("nan"),
        "Acc@5": _safe_mean((pred[high_mask] == 5).astype(float)) if high_mask.any() else float("nan"),
        "high_to_mid_or_low_rate": _safe_mean((pred[high_mask] <= 3).astype(float)) if high_mask.any() else float("nan"),
        "high_signed_bias": _safe_mean(label_diffs[high_mask]) if high_mask.any() else float("nan"),
        "mean_pred_high": _safe_mean(pred_float[high_mask]) if high_mask.any() else float("nan"),
    }
    return metrics


def threshold_candidates(grid: list[float] | None = None) -> list[tuple[float, float, float, float]]:
    values = grid or THRESHOLD_GRID
    return [tuple(candidate) for candidate in combinations_with_replacement(values, 4)]


def row_with_metrics(prefix: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = dict(prefix)
    for key in METRIC_FIELDS:
        row[key] = metrics.get(key, "")
    if "MAE_expected" in metrics:
        row["MAE_expected"] = metrics["MAE_expected"]
    if "Weighted-F1" in metrics:
        row["Weighted-F1"] = metrics["Weighted-F1"]
    return row


def select_global_thresholds(labels: np.ndarray, human_mean: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for thresholds in threshold_candidates():
        pred = predictions_from_thresholds(probs, thresholds)
        metrics = quick_metrics(labels, human_mean, pred)
        score = (
            metrics["MAE_label"],
            metrics["low_to_high_rate"],
            -metrics["Acc@5"],
            sum(thresholds),
        )
        if best is None or score < best["selection_tuple"]:
            best = {
                "thresholds": thresholds,
                "selection_tuple": score,
                "dev_MAE_label": metrics["MAE_label"],
                "dev_low_to_high": metrics["low_to_high_rate"],
                "dev_Acc5": metrics["Acc@5"],
            }
    if best is None:
        raise ValueError("No threshold candidates available")
    return best


def _risk_objective(metrics: dict[str, float], lambda_low: float, eta_high: float) -> float:
    return (
        metrics["MAE_label"]
        + lambda_low * metrics["low_to_high_rate"]
        + eta_high * metrics["high_to_mid_or_low_rate"]
    )


def select_risk_aware_thresholds(
    labels: np.ndarray,
    human_mean: np.ndarray,
    probs: np.ndarray,
    *,
    lambda_low: float,
    eta_high: float,
    raw_dev_metrics: dict[str, float],
) -> dict[str, Any]:
    best_any: dict[str, Any] | None = None
    best_constrained: dict[str, Any] | None = None
    for thresholds in threshold_candidates():
        pred = predictions_from_thresholds(probs, thresholds)
        metrics = quick_metrics(labels, human_mean, pred)
        objective = _risk_objective(metrics, lambda_low, eta_high)
        satisfies = (
            metrics["Acc@5"] >= raw_dev_metrics["Acc@5"] - 0.05
            and metrics["MAE_label"] <= raw_dev_metrics["MAE_label"] + 0.03
        )
        score = (objective, metrics["low_to_high_rate"], metrics["MAE_label"], -metrics["Acc@5"], sum(thresholds))
        candidate = {
            "thresholds": thresholds,
            "objective": objective,
            "selection_tuple": score,
            "satisfies_constraints": satisfies,
            "dev_MAE_label": metrics["MAE_label"],
            "dev_low_to_high": metrics["low_to_high_rate"],
            "dev_Acc5": metrics["Acc@5"],
            "dev_high_to_mid_or_low": metrics["high_to_mid_or_low_rate"],
        }
        if best_any is None or score < best_any["selection_tuple"]:
            best_any = candidate
        if satisfies and (best_constrained is None or score < best_constrained["selection_tuple"]):
            best_constrained = candidate
    if best_any is None:
        raise ValueError("No threshold candidates available")
    selected = best_constrained or best_any
    selected = dict(selected)
    selected["selection_type"] = "best_constrained" if best_constrained else "best_unconstrained"
    selected["constraint_available"] = best_constrained is not None
    return selected


def distribution_rows(base_model: str, method: str, split: str, pred: np.ndarray) -> list[dict[str, Any]]:
    total = pred.size
    rows = []
    for label in range(1, 6):
        count = int(np.sum(pred == label))
        rows.append(
            {
                "base_model": base_model,
                "method": method,
                "split": split,
                "pred_label": label,
                "count": count,
                "rate": count / total if total else float("nan"),
            }
        )
    return rows


def confusion_rows(base_model: str, method: str, split: str, labels: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    total = labels.size
    for true_label in range(1, 6):
        mask = labels == true_label
        denom = int(np.sum(mask))
        for pred_label in range(1, 6):
            count = int(np.sum(mask & (pred == pred_label)))
            rows.append(
                {
                    "base_model": base_model,
                    "method": method,
                    "split": split,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "rate_within_true": count / denom if denom else float("nan"),
                    "rate_overall": count / total if total else float("nan"),
                }
            )
    return rows


def per_label_accuracy_rows(base_model: str, method: str, split: str, labels: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for label in range(1, 6):
        mask = labels == label
        rows.append(
            {
                "base_model": base_model,
                "method": method,
                "split": split,
                "label_5": label,
                "n": int(np.sum(mask)),
                "accuracy": _safe_mean((pred[mask] == label).astype(float)) if mask.any() else float("nan"),
                "mean_pred_label": _safe_mean(pred[mask].astype(float)) if mask.any() else float("nan"),
            }
        )
    return rows
