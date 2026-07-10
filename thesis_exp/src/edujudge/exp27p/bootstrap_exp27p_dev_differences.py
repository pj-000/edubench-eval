"""Question-key cluster bootstrap for fixed Exp27P dev comparisons."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp27p import PAIRWISE_COMPARISONS, RUN_ROOT
from thesis_exp.src.edujudge.exp27p.common import finite, prediction_metrics, read_jsonl, write_csv


METRICS = (
    "MAE_argmax",
    "QWK",
    "low_to_high_count",
    "low_to_high_rate",
    "high_to_low_count",
    "high_to_low_rate",
    "label2_recall",
    "label5_recall",
)


def align(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("Pairwise dev predictions do not contain identical sample IDs")
    ids = sorted(left_by_id)
    for sid in ids:
        if left_by_id[sid].get("question_key") != right_by_id[sid].get("question_key"):
            raise ValueError(f"Question-key mismatch: {sid}")
    return [left_by_id[sid] for sid in ids], [right_by_id[sid] for sid in ids]


def bootstrap_pair(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    effect: str,
    resamples: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    left, right = align(left, right)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(left):
        groups[str(row["question_key"])].append(index)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        selected = [rng.choice(keys) for _ in keys]
        indices = [index for key in selected for index in groups[key]]
        left_metrics = prediction_metrics([left[index] for index in indices])
        right_metrics = prediction_metrics([right[index] for index in indices])
        for metric in METRICS:
            difference = float(right_metrics[metric]) - float(left_metrics[metric])
            if finite(difference):
                samples[metric].append(difference)
    left_point = prediction_metrics(left)
    right_point = prediction_metrics(right)
    rows = []
    for metric in METRICS:
        values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "left_variant": left_name,
                "right_variant": right_name,
                "effect": effect,
                "metric": metric,
                "difference_definition": "right_minus_left",
                "point_difference": float(right_point[metric]) - float(left_point[metric]),
                "ci_low_95": float(np.quantile(values, 0.025)) if len(values) else float("nan"),
                "ci_high_95": float(np.quantile(values, 0.975)) if len(values) else float("nan"),
                "bootstrap_resamples": len(values),
                "cluster_unit": "question_key",
            }
        )
    return rows


def bootstrap_all(run_root: Path, seed: int = 42, resamples: int = 2000) -> list[dict[str, Any]]:
    predictions = {
        variant: read_jsonl(run_root / variant / f"seed_{seed}" / "predictions_private" / "selected_dev_predictions.jsonl")
        for pair in PAIRWISE_COMPARISONS
        for variant in pair[:2]
    }
    rows = []
    for left, right, effect in PAIRWISE_COMPARISONS:
        rows.extend(bootstrap_pair(predictions[left], predictions[right], left, right, effect, resamples, seed))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = bootstrap_all(args.run_root, args.seed, args.resamples)
    write_csv(args.output, rows)
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(args.output)}, sort_keys=True))
