"""Question-key paired bootstrap for Exp46A teacher and student comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_exp.exp46_hato_kd.common import ROOT, RUN_ROOT, load_predictions, prediction_metrics, write_csv


METRICS = ("MAE", "QWK", "Exact_Match", "Kendall_tau", "Signed_Bias", "abs_Signed_Bias", "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--replicates", type=int, default=5000)
    return parser.parse_args()


def comparisons(stage: str) -> tuple[tuple[str, str], ...]:
    if stage == "teacher":
        return (("T1_4B_teacher", "K0_E4"),)
    return (("K2_hato_kd", "C1_strongest_point"), ("K2_hato_kd", "K1_standard_kd"), ("K2_hato_kd", "K3_shuffled_hato_control"))


def main() -> None:
    args = parse_args()
    output = []
    for comparison_index, (left, right) in enumerate(comparisons(args.stage)):
        left_rows = load_predictions(left, args.run_root)
        right_rows = load_predictions(right, args.run_root)
        left_by = {row["sample_id"]: row for row in left_rows}
        right_by = {row["sample_id"]: row for row in right_rows}
        if set(left_by) != set(right_by):
            raise RuntimeError(f"Exp46 bootstrap IDs differ: {left} vs {right}")
        groups: dict[str, list[str]] = {}
        for sample_id, row in left_by.items():
            groups.setdefault(str(row["question_key"]), []).append(sample_id)
        keys = sorted(groups)
        rng = np.random.default_rng(46000 + comparison_index)
        values = {metric: [] for metric in METRICS}
        for _ in range(args.replicates):
            sampled_keys = rng.choice(keys, size=len(keys), replace=True)
            sampled_ids = [sample_id for key in sampled_keys for sample_id in groups[str(key)]]
            left_metrics = prediction_metrics([left_by[sample_id] for sample_id in sampled_ids])
            right_metrics = prediction_metrics([right_by[sample_id] for sample_id in sampled_ids])
            for metric in METRICS:
                values[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
        for metric, raw in values.items():
            data = np.asarray(raw, dtype=float)
            output.append({"stage": args.stage, "comparison": f"{left}_vs_{right}", "left": left, "right": right, "metric": metric, "delta_mean": float(np.nanmean(data)), "ci_low": float(np.nanquantile(data, 0.025)), "ci_high": float(np.nanquantile(data, 0.975)), "replicates": args.replicates, "unit": "question_key"})
    path = args.out_dir / f"tables/exp46a_{args.stage}_question_key_bootstrap.csv"
    write_csv(path, output)
    print(json.dumps({"status": "BOOTSTRAP_COMPLETE", "stage": args.stage, "rows": len(output), "replicates": args.replicates, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
