"""Two-level seed then question-key bootstrap for Safe16 minus V3."""

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


def two_level_bootstrap(
    safe_by_seed: dict[int, list[dict[str, Any]]],
    v3_by_seed: dict[int, list[dict[str, Any]]],
    resamples: int = 5000,
    random_seed: int = 27016,
) -> list[dict[str, Any]]:
    seeds = sorted(safe_by_seed)
    if seeds != sorted(v3_by_seed) or len(seeds) != 3:
        raise ValueError("Two-level bootstrap requires matching three-seed predictions")
    aligned: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]] = {}
    for seed in seeds:
        safe, v3 = align(safe_by_seed[seed], v3_by_seed[seed])
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(safe):
            groups[str(row["question_key"])].append(index)
        aligned[seed] = (safe, v3, groups)

    rng = random.Random(random_seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        seed_draw = [rng.choice(seeds) for _ in seeds]
        differences = {metric: [] for metric in METRICS}
        for seed in seed_draw:
            safe, v3, groups = aligned[seed]
            keys = sorted(groups)
            selected = [rng.choice(keys) for _ in keys]
            indices = [index for key in selected for index in groups[key]]
            safe_metrics = prediction_metrics([safe[index] for index in indices])
            v3_metrics = prediction_metrics([v3[index] for index in indices])
            for metric in METRICS:
                value = float(safe_metrics[metric]) - float(v3_metrics[metric])
                if finite(value):
                    differences[metric].append(value)
        for metric, values in differences.items():
            if len(values) == len(seeds):
                samples[metric].append(float(np.mean(values)))

    point = {}
    for metric in METRICS:
        point[metric] = float(np.mean([
            float(prediction_metrics(aligned[seed][0])[metric])
            - float(prediction_metrics(aligned[seed][1])[metric])
            for seed in seeds
        ]))
    return [{
        "left_variant": "v3_selective_soft_audit",
        "right_variant": "v3_safe16_original_low_anchor",
        "difference_definition": "safe16_minus_v3",
        "metric": metric,
        "point_difference": point[metric],
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
        "bootstrap_resamples": len(values),
        "level_1": "seed_with_replacement",
        "level_2": "question_key_with_replacement",
    } for metric, raw in samples.items() if (values := np.asarray(raw, dtype=float)).size]

