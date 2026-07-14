"""Run locked per-seed and crossed seed-by-question bootstraps for Exp42A."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp42_rubidist.common import (  # noqa: E402
    COMPARISONS,
    METRICS,
    ROOT,
    RUN_ROOT,
    SEEDS,
    all_metrics,
    load_oof_predictions,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--question-resamples", type=int, default=2000)
    parser.add_argument("--crossed-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42042)
    return parser.parse_args()


def index_by_qkey(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[str(row["question_key"])].append(row)
    return dict(indexed)


def summarize(values: dict[str, list[float]], observed: dict[str, float], resamples: int, unit: str) -> list[dict[str, Any]]:
    output = []
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output.append(
            {
                "metric": metric,
                "observed_delta": observed[metric],
                "ci_lower": float(np.nanpercentile(array, 2.5)),
                "ci_upper": float(np.nanpercentile(array, 97.5)),
                "bootstrap_mean": float(np.nanmean(array)),
                "resamples": resamples,
                "unit": unit,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    required_variants = sorted({variant for left, right, _ in COMPARISONS for variant in (left, right)})
    predictions = {
        (variant, seed): load_oof_predictions(args.run_root, variant, seed)
        for variant in required_variants
        for seed in args.seeds
    }
    indexed = {key: index_by_qkey(rows) for key, rows in predictions.items()}
    common_qkeys = sorted(set.intersection(*(set(value) for value in indexed.values())))
    if any(set(value) != set(common_qkeys) for value in indexed.values()):
        raise RuntimeError("Variants/seeds do not share the exact question-key set")

    per_seed_rows = []
    for comparison_index, (left, right, comparison) in enumerate(COMPARISONS):
        for seed in args.seeds:
            left_rows, right_rows = predictions[(left, seed)], predictions[(right, seed)]
            left_metrics, right_metrics = all_metrics(left_rows), all_metrics(right_rows)
            observed = {metric: float(left_metrics[metric]) - float(right_metrics[metric]) for metric in METRICS}
            rng = random.Random(args.bootstrap_seed + comparison_index * 100 + seed)
            values = {metric: [] for metric in METRICS}
            for _ in range(args.question_resamples):
                draw = [common_qkeys[rng.randrange(len(common_qkeys))] for _ in common_qkeys]
                sampled_left = [row for qkey in draw for row in indexed[(left, seed)][qkey]]
                sampled_right = [row for qkey in draw for row in indexed[(right, seed)][qkey]]
                lm, rm = all_metrics(sampled_left), all_metrics(sampled_right)
                for metric in METRICS:
                    values[metric].append(float(lm[metric]) - float(rm[metric]))
            for row in summarize(values, observed, args.question_resamples, "question_key"):
                per_seed_rows.append(
                    {
                        "comparison": comparison,
                        "left_variant": left,
                        "right_variant": right,
                        "seed": seed,
                        **row,
                    }
                )
    write_csv(args.out_dir / "tables/exp42a_question_key_bootstrap_ci.csv", per_seed_rows)

    crossed_rows = []
    for comparison_index, (left, right, comparison) in enumerate(COMPARISONS):
        rng = random.Random(args.bootstrap_seed + 10000 + comparison_index)
        values = {metric: [] for metric in METRICS}
        per_seed_observed = []
        for seed in args.seeds:
            lm, rm = all_metrics(predictions[(left, seed)]), all_metrics(predictions[(right, seed)])
            per_seed_observed.append({metric: float(lm[metric]) - float(rm[metric]) for metric in METRICS})
        observed = {metric: float(np.mean([row[metric] for row in per_seed_observed])) for metric in METRICS}
        for _ in range(args.crossed_resamples):
            selected_seeds = [args.seeds[rng.randrange(len(args.seeds))] for _ in args.seeds]
            qkey_draw = [common_qkeys[rng.randrange(len(common_qkeys))] for _ in common_qkeys]
            selected_deltas = {metric: [] for metric in METRICS}
            for seed in selected_seeds:
                sampled_left = [row for qkey in qkey_draw for row in indexed[(left, seed)][qkey]]
                sampled_right = [row for qkey in qkey_draw for row in indexed[(right, seed)][qkey]]
                lm, rm = all_metrics(sampled_left), all_metrics(sampled_right)
                for metric in METRICS:
                    selected_deltas[metric].append(float(lm[metric]) - float(rm[metric]))
            for metric in METRICS:
                values[metric].append(float(np.mean(selected_deltas[metric])))
        for row in summarize(values, observed, args.crossed_resamples, "seed_x_question_key"):
            crossed_rows.append(
                {
                    "comparison": comparison,
                    "left_variant": left,
                    "right_variant": right,
                    **row,
                }
            )
    write_csv(args.out_dir / "tables/exp42a_crossed_seed_question_bootstrap_ci.csv", crossed_rows)
    print(
        json.dumps(
            {
                "status": "BOOTSTRAPPED",
                "question_keys": len(common_qkeys),
                "per_seed_resamples": args.question_resamples,
                "crossed_resamples": args.crossed_resamples,
                "dev_access_count": 0,
                "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
