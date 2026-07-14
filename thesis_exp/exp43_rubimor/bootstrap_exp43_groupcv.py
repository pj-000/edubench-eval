"""Question-key and crossed seed-by-question bootstrap for formal Exp43 GroupCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis_exp.exp43_rubimor.collect_exp43_groupcv import METRICS, paths
from thesis_exp.exp43_rubimor.common import ROOT, RUN_ROOT, SEEDS, prediction_metrics, read_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--qkey-reps", type=int, default=2000)
    parser.add_argument("--crossed-reps", type=int, default=5000)
    return parser.parse_args()


def load(run_root: Path, variant: str, seed: int) -> list[dict]:
    rows = []
    for fold in range(5):
        _, path = paths(run_root, variant, seed, fold)
        rows.extend(read_jsonl(path))
    return rows


def direction(metric: str) -> float:
    return -1.0 if metric in {"MAE", "abs_Signed_Bias", "expected_score_MAE", "human_CE", "human_Brier", "human_RPS", "low_to_high_rate", "high_to_low_rate"} else 1.0


def main() -> None:
    args = parse_args()
    comparisons = (("E6", "E0"), ("E6", "E3"), ("E6", "E6N"))
    qkey_rows, crossed_rows = [], []
    cache = {(variant, seed): load(args.run_root, variant, seed) for pair in comparisons for variant in pair for seed in SEEDS}
    for left, right in comparisons:
        for seed in SEEDS:
            left_by = {row["sample_id"]: row for row in cache[(left, seed)]}; right_by = {row["sample_id"]: row for row in cache[(right, seed)]}
            joined = [{"left": left_by[sid], "right": right_by[sid], "question_key": left_by[sid]["question_key"]} for sid in left_by]
            groups: dict[str, list[dict]] = {}
            for row in joined: groups.setdefault(str(row["question_key"]), []).append(row)
            keys = sorted(groups); rng = np.random.default_rng(43000 + seed)
            deltas = {metric: [] for metric in METRICS}
            for _ in range(args.qkey_reps):
                selected = rng.choice(keys, size=len(keys), replace=True)
                left_rows = [item["left"] for key in selected for item in groups[str(key)]]
                right_rows = [item["right"] for key in selected for item in groups[str(key)]]
                lm, rm = prediction_metrics(left_rows), prediction_metrics(right_rows)
                for metric in METRICS: deltas[metric].append(float(lm[metric]) - float(rm[metric]))
            for metric, values in deltas.items():
                qkey_rows.append({"comparison": f"{left}_vs_{right}", "left": left, "right": right, "seed": seed, "metric": metric, "delta_mean": float(np.mean(values)), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "favorable_probability": float(np.mean(np.asarray(values) * direction(metric) > 0)), "replicates": args.qkey_reps, "unit": "question_key"})
        rng = np.random.default_rng(43999)
        values_by_metric = {metric: [] for metric in METRICS}
        for _ in range(args.crossed_reps):
            sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            seed_deltas = {metric: [] for metric in METRICS}
            for seed in sampled_seeds:
                left_source, right_source = cache[(left, int(seed))], cache[(right, int(seed))]
                right_by_id = {row["sample_id"]: row for row in right_source}
                groups: dict[str, list[dict]] = {}
                for row in left_source: groups.setdefault(str(row["question_key"]), []).append(row)
                keys = sorted(groups); selected = rng.choice(keys, size=len(keys), replace=True)
                left_rows = [row for key in selected for row in groups[str(key)]]
                right_rows = [right_by_id[row["sample_id"]] for row in left_rows]
                lm, rm = prediction_metrics(left_rows), prediction_metrics(right_rows)
                for metric in METRICS: seed_deltas[metric].append(float(lm[metric]) - float(rm[metric]))
            for metric in METRICS: values_by_metric[metric].append(float(np.mean(seed_deltas[metric])))
        for metric, values in values_by_metric.items():
            crossed_rows.append({"comparison": f"{left}_vs_{right}", "left": left, "right": right, "metric": metric, "delta_mean": float(np.mean(values)), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)), "favorable_probability": float(np.mean(np.asarray(values) * direction(metric) > 0)), "replicates": args.crossed_reps, "unit": "seed_x_question_key"})
    write_csv(args.out_dir / "tables/exp43_groupcv_qkey_bootstrap_ci.csv", qkey_rows)
    write_csv(args.out_dir / "tables/exp43_groupcv_crossed_bootstrap_ci.csv", crossed_rows)
    print({"status": "BOOTSTRAP_COMPLETE", "qkey_rows": len(qkey_rows), "crossed_rows": len(crossed_rows), "test_access_count": 0})


if __name__ == "__main__":
    main()
