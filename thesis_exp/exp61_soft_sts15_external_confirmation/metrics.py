"""Pre-registered Exp61 metrics and component-cluster bootstrap."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    result = spearmanr(left, right)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def quadratic_weighted_kappa(true: np.ndarray, predicted: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(true, predicted, labels=list(range(6)), weights="quadratic"))


def evaluate_hard_head(
    probabilities: np.ndarray,
    human_means: np.ndarray,
    hard_labels: np.ndarray,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    human_means = np.asarray(human_means, dtype=np.float64)
    hard_labels = np.asarray(hard_labels, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 6:
        raise ValueError("Exp61 hard-head probabilities must be [n, 6]")
    if len(probabilities) != len(human_means) or len(human_means) != len(hard_labels):
        raise ValueError("Exp61 metric arrays differ in length")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("probabilities do not sum to one")
    expectation = probabilities @ np.arange(6, dtype=np.float64)
    argmax = probabilities.argmax(axis=1)
    absolute_error = np.abs(expectation - human_means)
    squared_error = np.square(expectation - human_means)
    argmax_error = np.abs(argmax - human_means)
    return {
        "nmae": float(absolute_error.mean() / 5.0),
        "mae": float(absolute_error.mean()),
        "nrmse": float(math.sqrt(squared_error.mean()) / 5.0),
        "spearman": _spearman(human_means, expectation),
        "argmax_nmae": float(argmax_error.mean() / 5.0),
        "exact_match": float((argmax == hard_labels).mean()),
        "qwk": quadratic_weighted_kappa(hard_labels, argmax),
    }


def component_cluster_bootstrap(
    aligned_predictions_by_seed: dict[int, np.ndarray],
    shuffled_predictions_by_seed: dict[int, np.ndarray],
    human_means: np.ndarray,
    components: list[str],
    *,
    n_resamples: int = 10_000,
    rng_seed: int = 610061,
) -> dict[str, Any]:
    """Item-weighted paired NMAE bootstrap over sentence components.

    Per-record error differences are averaged over the three seeds first.
    Components are then resampled with replacement and all rows belonging to
    sampled components are retained, so the statistic remains item weighted.
    """

    if set(aligned_predictions_by_seed) != set(shuffled_predictions_by_seed):
        raise ValueError("paired arms must contain identical seed sets")
    if len(aligned_predictions_by_seed) != 3:
        raise ValueError("confirmatory bootstrap is conditional on exactly three seeds")
    human = np.asarray(human_means, dtype=np.float64)
    n = len(human)
    if len(components) != n:
        raise ValueError("component vector length mismatch")
    per_seed: list[np.ndarray] = []
    for seed in sorted(aligned_predictions_by_seed):
        aligned = np.asarray(aligned_predictions_by_seed[seed], dtype=np.float64)
        shuffled = np.asarray(shuffled_predictions_by_seed[seed], dtype=np.float64)
        if aligned.shape != (n,) or shuffled.shape != (n,):
            raise ValueError("prediction vector length mismatch")
        per_seed.append((np.abs(aligned - human) - np.abs(shuffled - human)) / 5.0)
    record_delta = np.stack(per_seed).mean(axis=0)
    component_indices: dict[str, list[int]] = defaultdict(list)
    for index, component in enumerate(components):
        component_indices[str(component)].append(index)
    component_names = sorted(component_indices)
    if not component_names:
        raise ValueError("no components available for bootstrap")
    rng = np.random.default_rng(rng_seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        sampled = rng.integers(0, len(component_names), size=len(component_names))
        indices = [
            index
            for component_index in sampled
            for index in component_indices[component_names[int(component_index)]]
        ]
        replicates[draw] = float(record_delta[indices].mean())
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "statistic": "aligned_nmae_minus_shuffled_nmae",
        "aggregation": "average paired record errors across seeds, then item-weighted component bootstrap",
        "seeds": sorted(aligned_predictions_by_seed),
        "components": len(component_names),
        "records": n,
        "n_resamples": n_resamples,
        "rng_seed": rng_seed,
        "mean_delta_nmae": float(record_delta.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def confirmation_gates(
    aligned: dict[int, dict[str, float]],
    shuffled: dict[int, dict[str, float]],
    quantized_mean_only: dict[int, dict[str, float]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    seeds = sorted(aligned)
    if seeds != sorted(shuffled) or seeds != sorted(quantized_mean_only) or len(seeds) != 3:
        raise ValueError("confirmation gates require the same three seeds in all arms")
    aligned_minus_shuffled = [aligned[s]["nmae"] - shuffled[s]["nmae"] for s in seeds]
    aligned_minus_main = [aligned[s]["nmae"] - quantized_mean_only[s]["nmae"] for s in seeds]
    spearman_delta = float(np.mean([aligned[s]["spearman"] - shuffled[s]["spearman"] for s in seeds]))
    gates = {
        "aligned_beats_shuffled_nmae_all_3_seeds": all(value < 0 for value in aligned_minus_shuffled),
        "aligned_minus_shuffled_mean_nmae_le_minus_0_00125": float(np.mean(aligned_minus_shuffled)) <= -0.00125,
        "component_bootstrap_ci_upper_below_zero": float(bootstrap["ci95_upper"]) < 0.0,
        "aligned_minus_shuffled_spearman_ge_minus_0_005": spearman_delta >= -0.005,
        "aligned_beats_quantized_mean_only_at_least_2_of_3": sum(value < 0 for value in aligned_minus_main) >= 2,
        "aligned_minus_quantized_mean_only_mean_nmae_le_minus_0_00125": float(np.mean(aligned_minus_main)) <= -0.00125,
    }
    return {
        "decision": "PASS" if all(gates.values()) else "NO_GO",
        "gates": gates,
        "aligned_minus_shuffled_nmae_by_seed": dict(zip(map(str, seeds), aligned_minus_shuffled)),
        "aligned_minus_quantized_mean_only_nmae_by_seed": dict(zip(map(str, seeds), aligned_minus_main)),
        "aligned_minus_shuffled_mean_spearman": spearman_delta,
    }
