"""Paired seed/triple bootstrap for the frozen Exp28 final test comparison."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.bootstrap_exp28f_dev_differences import (
    METRICS,
    metrics,
    percentile,
    read_jsonl,
    write_csv,
    write_json,
)


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test"
)
BASELINE = "b0_original_human"
MAIN = "b2_selective_dual_teacher"
SEEDS = (42, 43, 44)


def load(out_dir: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    output = {}
    expected: set[str] | None = None
    for variant in (BASELINE, MAIN):
        for seed in SEEDS:
            path = out_dir / "runs" / variant / f"seed_{seed}" / "predictions" / "predictions_test.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            by_id = {str(row["record_id"]): row for row in rows}
            if len(by_id) != 2218:
                raise ValueError(f"Expected 2218 unique test predictions: {path}")
            if expected is None:
                expected = set(by_id)
            elif set(by_id) != expected:
                raise ValueError(f"Test prediction IDs differ: {path}")
            output[(variant, seed)] = list(by_id.values())
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load(args.out_dir)
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    clusters: set[str] | None = None
    for key, rows in predictions.items():
        value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value[str(row["triple_key"])].append(row)
        grouped[key] = value
        if clusters is None:
            clusters = set(value)
        elif set(value) != clusters:
            raise ValueError("Test triple clusters differ across runs")
    cluster_list = sorted(clusters or [])
    observed_main = metrics([row for seed in SEEDS for row in predictions[(MAIN, seed)]])
    observed_baseline = metrics([row for seed in SEEDS for row in predictions[(BASELINE, seed)]])
    distributions = {metric: [] for metric in METRICS}
    rng = random.Random(args.seed)
    for _ in range(args.resamples):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_clusters = [rng.choice(cluster_list) for _ in cluster_list]
        main_rows = []
        baseline_rows = []
        for seed in sampled_seeds:
            for cluster in sampled_clusters:
                main_rows.extend(grouped[(MAIN, seed)][cluster])
                baseline_rows.extend(grouped[(BASELINE, seed)][cluster])
        left = metrics(main_rows)
        right = metrics(baseline_rows)
        for metric in METRICS:
            distributions[metric].append(left[metric] - right[metric])
    rows = []
    lookup = {}
    for metric in METRICS:
        row = {
            "metric": metric,
            "baseline": observed_baseline[metric],
            "main": observed_main[metric],
            "delta": observed_main[metric] - observed_baseline[metric],
            "ci_low": percentile(distributions[metric], 0.025),
            "ci_high": percentile(distributions[metric], 0.975),
            "resamples": args.resamples,
        }
        rows.append(row)
        lookup[metric] = row
    write_csv(
        args.out_dir / "tables" / "exp28h_final_test_bootstrap_ci.csv",
        rows,
        ["metric", "baseline", "main", "delta", "ci_low", "ci_high", "resamples"],
    )
    checks = {
        "mae_significant_improvement": lookup["MAE"]["ci_high"] < 0.0,
        "exact_significant_improvement": lookup["Exact_Match"]["ci_low"] > 0.0,
        "kendall_significant_improvement": lookup["Kendall_tau"]["ci_low"] > 0.0,
        "bin_agreement_noninferior": lookup["Bin_Agreement"]["ci_low"] >= -0.005,
        "low_to_high_directional_improvement": lookup["low_to_high_rate"]["delta"] < 0.0,
        "label5_guard": lookup["label5_recall"]["ci_low"] >= -0.03,
    }
    positive = all(checks.values())
    decision = {
        "status": "POSITIVE_SMALL_PAPER_RESULT" if positive else "MIXED_OR_NEGATIVE_SMALL_PAPER_RESULT",
        "checks": checks,
        "baseline": BASELINE,
        "main": MAIN,
        "test_rows": 2218,
        "resamples": args.resamples,
        "test_used_for_selection": False,
        "method_frozen_after_test": True,
    }
    write_json(args.out_dir / "decision" / "exp28h_final_statistical_decision.json", decision)
    report = f"""# Exp28H Final Test Statistical Report

- decision: **{decision['status']}**
- comparison: B2 selective dual-teacher audit vs B0 original labels
- test rows: 2,218
- seeds: 42, 43, 44
- bootstrap: paired two-level seed and paper triple clusters
- resamples: {args.resamples}
- test used for method selection: no

All conclusions must follow the confidence intervals in the accompanying table. The method,
training variants, hyperparameters, and checkpoints remain frozen after this test.
"""
    report_path = args.out_dir / "reports" / "exp28h_final_test_statistical_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=28084)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
