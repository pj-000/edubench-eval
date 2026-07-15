"""Paired 5000-replicate question-key bootstrap for Exp44A fixed comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.exp44_taco_score.collect_exp44a_seed42 import METRICS
from thesis_exp.exp44_taco_score.common import ROOT, RUN_ROOT, prediction_metrics, read_jsonl, run_dir, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--replicates", type=int, default=5000)
    return parser.parse_args()


def load(run_root: Path, variant: str) -> list[dict]:
    rows = []
    for fold in range(5):
        rows.extend(read_jsonl(run_dir(run_root, variant, fold) / "heldout_predictions.jsonl"))
    return rows


def main() -> None:
    args = parse_args()
    comparisons = (
        ("C2_TACO", "C0_E4_baseline"),
        ("C2_TACO", "C1_balanced_plain_contrastive"),
        ("C2_TACO", "C3_shuffled_margin_control"),
        ("C1_balanced_plain_contrastive", "C0_E4_baseline"),
    )
    variants = sorted({variant for pair in comparisons for variant in pair})
    cache = {variant: load(args.run_root, variant) for variant in variants}
    output = []
    for comparison_index, (left, right) in enumerate(comparisons):
        left_by = {row["sample_id"]: row for row in cache[left]}
        right_by = {row["sample_id"]: row for row in cache[right]}
        if set(left_by) != set(right_by):
            raise RuntimeError(f"Bootstrap sample IDs differ for {left} vs {right}")
        groups: dict[str, list[str]] = {}
        for sample_id, row in left_by.items():
            groups.setdefault(str(row["question_key"]), []).append(sample_id)
        keys = sorted(groups)
        rng = np.random.default_rng(44000 + comparison_index)
        values = {metric: [] for metric in METRICS}
        for _ in range(args.replicates):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            sample_ids = [sample_id for key in sampled for sample_id in groups[str(key)]]
            left_metrics = prediction_metrics([left_by[sample_id] for sample_id in sample_ids])
            right_metrics = prediction_metrics([right_by[sample_id] for sample_id in sample_ids])
            for metric in METRICS:
                values[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
        for metric, metric_values in values.items():
            data = np.asarray(metric_values, dtype=float)
            output.append({"comparison": f"{left}_vs_{right}", "left": left, "right": right, "metric": metric, "delta_mean": float(np.nanmean(data)), "ci_low": float(np.nanquantile(data, 0.025)), "ci_high": float(np.nanquantile(data, 0.975)), "replicates": args.replicates, "unit": "question_key"})
    write_csv(args.out_dir / "tables/exp44a_question_key_bootstrap_ci.csv", output)
    print(json.dumps({"status": "BOOTSTRAP_COMPLETE", "rows": len(output), "replicates": args.replicates, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()

