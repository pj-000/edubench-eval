"""Run paired two-level seed/triple bootstrap for Exp28 dev predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp27p.common import kendall_tau_b, qwk


DEFAULT_RUN_ROOT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/runs"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28f_paper_dev_statistical_lock"
)
VARIANTS = (
    "b0_original_human",
    "b1_primary_teacher_all",
    "b2_selective_dual_teacher",
    "b3_filter_unresolved",
    "b4_random_transition_control",
)
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("b2_selective_dual_teacher", "b0_original_human", "main_vs_original"),
    ("b2_selective_dual_teacher", "b4_random_transition_control", "main_vs_random_control"),
    ("b1_primary_teacher_all", "b0_original_human", "all_teacher_vs_original"),
    ("b3_filter_unresolved", "b0_original_human", "filter_vs_original"),
)
METRICS = (
    "MAE",
    "Signed_Bias",
    "Exact_Match",
    "QWK",
    "Kendall_tau",
    "Bin_Agreement",
    "low_to_high_rate",
    "high_to_low_rate",
    "label1_recall",
    "label2_recall",
    "label5_recall",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {metric: float("nan") for metric in METRICS}
    label = [int(row["label_5"]) for row in rows]
    human_mean = [float(row["human_mean_5"]) for row in rows]
    pred = [int(row["pred_label_5"]) for row in rows]
    low = [index for index, value in enumerate(label) if value <= 2]
    high = [index for index, value in enumerate(label) if value >= 4]
    output = {
        "MAE": sum(abs(right - left) for left, right in zip(human_mean, pred)) / len(rows),
        "Signed_Bias": sum(right - left for left, right in zip(human_mean, pred)) / len(rows),
        "Exact_Match": sum(left == right for left, right in zip(label, pred)) / len(rows),
        "QWK": qwk(label, pred),
        "Kendall_tau": kendall_tau_b(human_mean, pred),
        "Bin_Agreement": sum(
            (0 if left <= 2 else 1 if left == 3 else 2) == (0 if right <= 2 else 1 if right == 3 else 2)
            for left, right in zip(label, pred)
        ) / len(rows),
        "low_to_high_rate": sum(pred[index] >= 4 for index in low) / len(low) if low else float("nan"),
        "high_to_low_rate": sum(pred[index] <= 2 for index in high) / len(high) if high else float("nan"),
    }
    for score in (1, 2, 5):
        indices = [index for index, value in enumerate(label) if value == score]
        output[f"label{score}_recall"] = (
            sum(pred[index] == score for index in indices) / len(indices) if indices else float("nan")
        )
    return output


def load_predictions(run_root: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    output = {}
    expected_ids: set[str] | None = None
    for variant in VARIANTS:
        for seed in SEEDS:
            path = run_root / variant / f"seed_{seed}" / "predictions" / "predictions_dev.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            by_id = {str(row["record_id"]): row for row in rows}
            if len(by_id) != 664:
                raise ValueError(f"Expected 664 unique dev rows: {path}")
            if expected_ids is None:
                expected_ids = set(by_id)
            elif set(by_id) != expected_ids:
                raise ValueError(f"Dev prediction identity mismatch: {path}")
            output[(variant, seed)] = list(by_id.values())
    return output


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_predictions(args.run_root)
    by_key: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    triple_keys: set[str] | None = None
    for key, rows in predictions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["triple_key"])].append(row)
        by_key[key] = grouped
        if triple_keys is None:
            triple_keys = set(grouped)
        elif set(grouped) != triple_keys:
            raise ValueError("Triple-key clusters differ across runs")
    clusters = sorted(triple_keys or [])
    rng = random.Random(args.seed)
    result_rows = []
    decision_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for left, right, comparison in COMPARISONS:
        observed_left = [row for seed in SEEDS for row in predictions[(left, seed)]]
        observed_right = [row for seed in SEEDS for row in predictions[(right, seed)]]
        observed_metrics_left = metrics(observed_left)
        observed_metrics_right = metrics(observed_right)
        distributions = {metric: [] for metric in METRICS}
        for _ in range(args.resamples):
            sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
            sampled_clusters = [rng.choice(clusters) for _ in clusters]
            left_rows = []
            right_rows = []
            for seed in sampled_seeds:
                for cluster in sampled_clusters:
                    left_rows.extend(by_key[(left, seed)][cluster])
                    right_rows.extend(by_key[(right, seed)][cluster])
            left_metrics = metrics(left_rows)
            right_metrics = metrics(right_rows)
            for metric in METRICS:
                distributions[metric].append(left_metrics[metric] - right_metrics[metric])
        for metric in METRICS:
            values = distributions[metric]
            observed_delta = observed_metrics_left[metric] - observed_metrics_right[metric]
            row = {
                "comparison": comparison,
                "left_variant": left,
                "right_variant": right,
                "metric": metric,
                "observed_left": observed_metrics_left[metric],
                "observed_right": observed_metrics_right[metric],
                "observed_delta": observed_delta,
                "ci_low": percentile(values, 0.025),
                "ci_high": percentile(values, 0.975),
                "resamples": args.resamples,
                "seed": args.seed,
            }
            result_rows.append(row)
            decision_lookup[(comparison, metric)] = row

    write_csv(
        args.out_dir / "tables" / "exp28f_two_level_bootstrap_ci.csv",
        result_rows,
        [
            "comparison", "left_variant", "right_variant", "metric", "observed_left", "observed_right",
            "observed_delta", "ci_low", "ci_high", "resamples", "seed",
        ],
    )
    main = {metric: decision_lookup[("main_vs_original", metric)] for metric in METRICS}
    random_control = {metric: decision_lookup[("main_vs_random_control", metric)] for metric in METRICS}
    checks = {
        "mae_guard": main["MAE"]["ci_high"] <= 0.01,
        "exact_guard": main["Exact_Match"]["ci_low"] >= -0.01,
        "kendall_guard": main["Kendall_tau"]["ci_low"] >= -0.01,
        "label5_guard": main["label5_recall"]["ci_low"] >= -0.03,
        "low_to_high_directional_improvement": main["low_to_high_rate"]["observed_delta"] < 0.0,
        "targeting_beats_random_mae": random_control["MAE"]["ci_high"] < 0.0,
    }
    passed = all(checks.values())
    decision = {
        "status": "READY_FOR_FINAL_DEV_LOCK" if passed else "BOOTSTRAP_SUCCESS_CRITERIA_NOT_MET",
        "checks": checks,
        "main_variant": "b2_selective_dual_teacher",
        "baseline_variant": "b0_original_human",
        "random_control_variant": "b4_random_transition_control",
        "resamples": args.resamples,
        "test_read": False,
        "test_open_authorized": False,
    }
    write_json(args.out_dir / "decision" / "exp28f_bootstrap_decision.json", decision)
    report = f"""# Exp28F Dev Statistical Lock

- two-level bootstrap: seed and paper triple cluster
- resamples: {args.resamples}
- main comparison: B2 selective dual teacher vs B0 original labels
- targeting control: B2 vs B4 random transition control
- decision: **{decision['status']}**
- test read: no

The held-out test remains closed. A separate lock step verifies this decision, the multiseed dev
decision, dataset hashes, and checkpoint availability before authorizing final evaluation.
"""
    report_path = args.out_dir / "reports" / "exp28f_bootstrap_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=28042)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(bootstrap(parse_args()), ensure_ascii=False, sort_keys=True))
