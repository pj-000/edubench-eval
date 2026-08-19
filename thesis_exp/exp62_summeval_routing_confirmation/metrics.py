"""Frozen Exp62 hard-head metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    result = spearmanr(left, right)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def _qwk(true: np.ndarray, predicted: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(true, predicted, labels=[1, 2, 3, 4, 5], weights="quadratic"))


def _block(probabilities: np.ndarray, means: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    expected = probabilities @ np.arange(1, 6, dtype=np.float64)
    argmax = probabilities.argmax(axis=1) + 1
    error = expected - means
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(math.sqrt(np.square(error).mean())),
        "exact_match": float((argmax == labels).mean()),
        "spearman": _spearman(means, expected),
        "qwk": _qwk(labels, argmax),
    }


def evaluate_hard_head(
    probabilities: np.ndarray,
    human_means: np.ndarray,
    hard_labels: np.ndarray,
    dimensions: list[str],
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    means = np.asarray(human_means, dtype=np.float64)
    labels = np.asarray(hard_labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("Exp62 probabilities must have shape [n, 5]")
    if not (len(probabilities) == len(means) == len(labels) == len(dimensions)):
        raise ValueError("Exp62 metric inputs differ in length")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Exp62 probabilities do not sum to one")
    by_dimension: dict[str, dict[str, float]] = {}
    for dimension in sorted(set(dimensions)):
        selected = np.asarray([value == dimension for value in dimensions])
        by_dimension[dimension] = _block(
            probabilities[selected], means[selected], labels[selected]
        )
    metric_names = tuple(next(iter(by_dimension.values())))
    macro = {
        name: float(np.mean([values[name] for values in by_dimension.values()]))
        for name in metric_names
    }
    return {
        "primary_macro_mae": macro["mae"],
        "macro": macro,
        "overall": _block(probabilities, means, labels),
        "by_dimension": by_dimension,
    }

