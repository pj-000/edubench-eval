"""Run fixed paired question-key bootstrap and finalize the Exp38A decision."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from thesis_exp.exp38_hails_score.common import ROOT, prediction_metrics, read_jsonl, write_csv, write_json

COMPARISONS = (
    ("v4_hails", "v0h_human_empirical"),
    ("v4_hails", "v2_qwen_interval_only"),
    ("v4_hails", "v3_naive_interval"),
    ("v4_hails", "v5_shuffled_interval"),
)
METRICS = ("MAE", "Exact_Match", "Kendall_tau", "QWK", "Bin_Agreement", "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(out_dir: Path, variant: str) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        rows.extend(read_jsonl(out_dir / f"private/groupcv_predictions/{variant}/fold_{fold}/heldout_predictions.jsonl"))
    if len(rows) != 2654:
        raise ValueError(f"Expected 2654 OOF predictions for {variant}, found {len(rows)}")
    return rows


def bootstrap_pair(left: list[dict[str, Any]], right: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("Paired variants have different sample IDs")
    by_qkey: dict[str, list[str]] = defaultdict(list)
    for sid, row in left_by_id.items():
        by_qkey[str(row["question_key"])].append(sid)
    qkeys = sorted(by_qkey)
    rng = random.Random(seed)
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        sampled = [qkeys[rng.randrange(len(qkeys))] for _ in qkeys]
        left_sample: list[dict[str, Any]] = []
        right_sample: list[dict[str, Any]] = []
        for qkey in sampled:
            for sid in by_qkey[qkey]:
                left_sample.append(left_by_id[sid])
                right_sample.append(right_by_id[sid])
        left_metrics, right_metrics = prediction_metrics(left_sample), prediction_metrics(right_sample)
        for metric in METRICS:
            values[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
    output = []
    observed_left, observed_right = prediction_metrics(left), prediction_metrics(right)
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output.append({
            "metric": metric, "observed_delta": float(observed_left[metric]) - float(observed_right[metric]),
            "ci_lower": float(np.nanpercentile(array, 2.5)), "ci_upper": float(np.nanpercentile(array, 97.5)),
            "resamples": resamples, "unit": "question_key",
        })
    return output


def main() -> None:
    args = parse_args()
    predictions = {variant: load_predictions(args.out_dir, variant) for variant in {name for pair in COMPARISONS for name in pair}}
    ci_rows = []
    for index, (left, right) in enumerate(COMPARISONS):
        for row in bootstrap_pair(predictions[left], predictions[right], args.resamples, args.seed + index):
            ci_rows.append({"left_variant": left, "right_variant": right, **row})
    write_csv(args.out_dir / "tables/exp38a_groupcv_question_key_bootstrap_ci.csv", ci_rows)
    metrics_path = args.out_dir / "tables/exp38a_groupcv_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        metrics = {row["variant"]: {key: (float(value) if key != "variant" and value not in {"", "nan"} else value) for key, value in row.items()} for row in csv.DictReader(handle)}
    baseline, hails = metrics["v0h_human_empirical"], metrics["v4_hails"]
    meaningful_gain = (
        (baseline["low_to_high_rate"] > 0 and (baseline["low_to_high_rate"] - hails["low_to_high_rate"]) / baseline["low_to_high_rate"] >= 0.10)
        or hails["label2_recall"] - baseline["label2_recall"] >= 0.05
        or baseline["MAE"] - hails["MAE"] >= 0.005
    )
    attribution = any(
        hails["MAE"] < metrics[name]["MAE"] or hails["low_to_high_rate"] < metrics[name]["low_to_high_rate"]
        for name in ("v3_naive_interval", "v5_shuffled_interval")
    )
    v5_not_match = hails["MAE"] < metrics["v5_shuffled_interval"]["MAE"] or hails["low_to_high_rate"] < metrics["v5_shuffled_interval"]["low_to_high_rate"]
    v0h_ci = {(row["metric"]): row for row in ci_rows if row["left_variant"] == "v4_hails" and row["right_variant"] == "v0h_human_empirical"}
    no_significant_harm = (
        float(v0h_ci["MAE"]["ci_lower"]) <= 0
        and float(v0h_ci["QWK"]["ci_upper"]) >= 0
        and float(v0h_ci["Exact_Match"]["ci_upper"]) >= 0
    )
    gates = {
        "mae_guard": hails["MAE"] <= baseline["MAE"] + 0.005,
        "exact_guard": hails["Exact_Match"] >= baseline["Exact_Match"] - 0.005,
        "qwk_guard": hails["QWK"] >= baseline["QWK"] - 0.01,
        "kendall_guard": hails["Kendall_tau"] >= baseline["Kendall_tau"] - 0.01,
        "label5_guard": hails["label5_recall"] >= baseline["label5_recall"] - 0.02,
        "high_to_low_guard": hails["high_to_low_rate"] <= baseline["high_to_low_rate"] + 0.01,
        "meaningful_gain": meaningful_gain,
        "method_attribution": attribution,
        "not_matched_by_shuffled": v5_not_match,
        "no_significant_bootstrap_harm": no_significant_harm,
    }
    go = all(gates.values())
    decision = {
        "status": "GROUPCV_GO" if go else "GROUPCV_STOP",
        "gates": gates,
        "recommend_full_train_dev_multiseed": go,
        "stop_interval_supervision": not go,
        "bootstrap_resamples": args.resamples,
        "bootstrap_unit": "question_key",
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp38a_groupcv_decision.json", decision)
    report = [
        "# Exp38A HAILS GroupCV report", "",
        f"- Status: **{decision['status']}**",
        f"- V0H metrics: `{json.dumps(baseline, sort_keys=True)}`",
        f"- V4 metrics: `{json.dumps(hails, sort_keys=True)}`",
        f"- Gate checks: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend full-train/dev multiseed: `{str(go).lower()}`",
        f"- Stop interval supervision: `{str(not go).lower()}`",
        "- Training used only standard hard/soft cross-entropy; no reason/failure supervision or custom loss.",
        "- No paper-like dev/test data were accessed.",
    ]
    report_path = args.out_dir / "reports/exp38a_hails_groupcv_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
