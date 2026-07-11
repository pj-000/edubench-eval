"""Crossed seed by common-question-key paired bootstrap."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp27p.bootstrap_exp27p_dev_differences import align
from thesis_exp.src.edujudge.exp27p.common import finite, prediction_metrics


METRICS = (
    "MAE_argmax", "QWK", "low_to_high_rate", "low_to_high_count",
    "label2_recall", "label5_recall", "Signed_Bias_argmax",
)


def crossed_bootstrap(
    left_by_seed: dict[int, list[dict[str, Any]]],
    right_by_seed: dict[int, list[dict[str, Any]]],
    left_name: str,
    right_name: str,
    effect: str,
    resamples: int = 5000,
    random_seed: int = 27017,
) -> list[dict[str, Any]]:
    seeds = sorted(left_by_seed)
    if seeds != sorted(right_by_seed) or len(seeds) != 3:
        raise ValueError("Crossed bootstrap requires the same three seeds")
    aligned = {}
    common_keys: set[str] | None = None
    for seed in seeds:
        left, right = align(left_by_seed[seed], right_by_seed[seed])
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(left):
            groups[str(row["question_key"])].append(index)
        keys = set(groups)
        common_keys = keys if common_keys is None else common_keys & keys
        aligned[seed] = (left, right, groups)
    keys = sorted(common_keys or set())
    if not keys:
        raise ValueError("No common question keys across seeds")

    rng = random.Random(random_seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        seed_draw = [rng.choice(seeds) for _ in seeds]
        question_draw = [rng.choice(keys) for _ in keys]
        differences = {metric: [] for metric in METRICS}
        for seed in seed_draw:
            left, right, groups = aligned[seed]
            indices = [index for key in question_draw for index in groups[key]]
            left_metrics = prediction_metrics([left[index] for index in indices])
            right_metrics = prediction_metrics([right[index] for index in indices])
            for metric in METRICS:
                value = float(right_metrics[metric]) - float(left_metrics[metric])
                if finite(value):
                    differences[metric].append(value)
        for metric, values in differences.items():
            if len(values) == len(seeds):
                samples[metric].append(float(np.mean(values)))

    point = {}
    for metric in METRICS:
        point[metric] = float(np.mean([
            float(prediction_metrics(aligned[seed][1])[metric])
            - float(prediction_metrics(aligned[seed][0])[metric])
            for seed in seeds
        ]))
    rows = []
    for metric, raw in samples.items():
        values = np.asarray(raw, dtype=float)
        rows.append({
            "left_variant": left_name, "right_variant": right_name, "effect": effect,
            "difference_definition": "right_minus_left", "metric": metric,
            "point_difference": point[metric], "ci_low_95": float(np.quantile(values, 0.025)),
            "ci_high_95": float(np.quantile(values, 0.975)), "bootstrap_resamples": len(values),
            "seed_resampling": "with_replacement_3", "question_resampling": "one_common_draw",
            "question_key_count": len(keys),
        })
    return rows

