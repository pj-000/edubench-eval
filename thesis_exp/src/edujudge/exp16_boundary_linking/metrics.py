"""Metrics and threshold diagnostics for Exp16A."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

import numpy as np


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def quadratic_weighted_kappa(gold: list[int], pred: list[int]) -> float:
    if not gold:
        return float("nan")
    labels = [1, 2, 3, 4, 5]
    index = {label: idx for idx, label in enumerate(labels)}
    observed = np.zeros((5, 5), dtype=float)
    for g, p in zip(gold, pred):
        if g in index and p in index:
            observed[index[g], index[p]] += 1.0
    hist_g = observed.sum(axis=1)
    hist_p = observed.sum(axis=0)
    expected = np.outer(hist_g, hist_p) / max(1.0, observed.sum())
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            weights[i, j] = ((i - j) ** 2) / ((5 - 1) ** 2)
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else float("nan")
    return 1.0 - numerator / denominator


def monotonic_violation_rate(probs: np.ndarray, tolerance: float = 1e-7) -> float:
    if probs.size == 0:
        return float("nan")
    violations = np.any(probs[:, 1:] > probs[:, :-1] + tolerance, axis=1)
    return float(np.mean(violations))


def compute_metrics(predictions: list[dict[str, Any]], split: str) -> dict[str, Any]:
    gold = [int(row["gold_label"]) for row in predictions]
    pred = [int(row["pred_label"]) for row in predictions]
    probs = np.asarray([row["probs"] for row in predictions], dtype=float) if predictions else np.empty((0, 4))
    abs_errors = [abs(g - p) for g, p in zip(gold, pred)]
    low_idx = [idx for idx, value in enumerate(gold) if value <= 2]
    high_idx = [idx for idx, value in enumerate(gold) if value >= 4]
    low_to_high = sum(1 for idx in low_idx if pred[idx] >= 4)
    high_to_low = sum(1 for idx in high_idx if pred[idx] <= 2)
    metrics: dict[str, Any] = {
        "split": split,
        "n": len(predictions),
        "MAE": _safe_mean([float(value) for value in abs_errors]),
        "QWK": quadratic_weighted_kappa(gold, pred),
        "Accuracy": _safe_rate(sum(1 for g, p in zip(gold, pred) if g == p), len(pred)),
        "low_to_high_rate": _safe_rate(low_to_high, len(low_idx)),
        "low_to_high_count": low_to_high,
        "true_low_n": len(low_idx),
        "high_to_low_rate": _safe_rate(high_to_low, len(high_idx)),
        "high_to_low_count": high_to_low,
        "true_high_n": len(high_idx),
        "monotonic_violation_rate": monotonic_violation_rate(probs),
    }
    for label in [1, 2, 3, 4, 5]:
        denom = sum(1 for value in gold if value == label)
        hit = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        label_recall = _safe_rate(hit, denom)
        metrics[f"label{label}_recall"] = label_recall
        metrics[f"recall_label_{label}"] = label_recall
        metrics[f"n_label_{label}"] = denom
    metrics["Label5_Recall"] = metrics["label5_recall"]
    metrics["Acc@5"] = metrics["Accuracy"]
    return metrics


def threshold_stats(predictions: list[dict[str, Any]], split: str) -> dict[str, Any]:
    stats: dict[str, Any] = {"split": split, "n": len(predictions)}
    for idx in range(1, 5):
        values = [float(row[f"tau{idx}"]) for row in predictions]
        stats[f"tau{idx}_mean"] = _safe_mean(values)
        stats[f"tau{idx}_std"] = _safe_std(values)
        stats[f"tau{idx}_min"] = float(np.min(values)) if values else float("nan")
        stats[f"tau{idx}_max"] = float(np.max(values)) if values else float("nan")
    for label in [1, 2, 3, 4, 5]:
        group = [row for row in predictions if int(row["gold_label"]) == label]
        stats[f"s_mean_gold_{label}"] = _safe_mean([float(row["quality_score_s"]) for row in group])
    return stats


def threshold_stats_by_metric(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[str(row.get("metric") or "")].append(row)
    rows: list[dict[str, Any]] = []
    for metric, items in sorted(groups.items()):
        gold = [int(row["gold_label"]) for row in items]
        pred = [int(row["pred_label"]) for row in items]
        low_idx = [idx for idx, value in enumerate(gold) if value <= 2]
        low_to_high = sum(1 for idx in low_idx if pred[idx] >= 4)
        out = {
            "split": split,
            "metric": metric,
            "n": len(items),
            "low_to_high_count": low_to_high,
            "true_low_n": len(low_idx),
            "low_to_high_rate": _safe_rate(low_to_high, len(low_idx)),
        }
        for idx in range(1, 5):
            out[f"tau{idx}_mean"] = _safe_mean([float(row[f"tau{idx}"]) for row in items])
        out["quality_score_s_mean"] = _safe_mean([float(row["quality_score_s"]) for row in items])
        rows.append(out)
    return rows


def margins_by_gold_label(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in [1, 2, 3, 4, 5]:
        items = [row for row in predictions if int(row["gold_label"]) == label]
        row = {"split": split, "gold_label": label, "n": len(items)}
        for field in ["quality_score_s", "tau2", "tau3", "margin_tau2", "margin_tau3"]:
            values = [float(item[field]) for item in items]
            row[f"{field}_mean"] = _safe_mean(values)
            row[f"{field}_median"] = float(median(values)) if values else float("nan")
        rows.append(row)
    return rows
