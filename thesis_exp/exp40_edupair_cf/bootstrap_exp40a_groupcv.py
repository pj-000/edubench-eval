"""Run locked question-key bootstrap and finalize the Exp40A GroupCV gate."""

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

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    ROOT,
    VARIANTS,
    prediction_metrics,
    read_jsonl,
    write_csv,
    write_json,
)

COMPARISONS = (
    ("v3_edupair_cf", "v0h_human_soft"),
    ("v3_edupair_cf", "v1_human_real_pairs"),
    ("v3_edupair_cf", "v2_unverified_counterfactual_pairs"),
    ("v3_edupair_cf", "v4_shuffled_pair_alignment"),
)
METRICS = (
    "MAE", "Exact_Match", "QWK", "Kendall_tau", "low_to_high_rate",
    "high_to_low_rate", "label2_recall", "label5_recall",
)


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
    if len(rows) != 2654 or len({row["sample_id"] for row in rows}) != 2654:
        raise ValueError(f"Expected 2654 unique OOF predictions for {variant}")
    return rows


def bootstrap_pair(
    left: list[dict[str, Any]], right: list[dict[str, Any]], resamples: int, seed: int,
) -> list[dict[str, Any]]:
    left_by_id = {row["sample_id"]: row for row in left}
    right_by_id = {row["sample_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("Paired variants have different OOF sample IDs")
    by_qkey: dict[str, list[str]] = defaultdict(list)
    for sid, row in left_by_id.items():
        by_qkey[str(row["question_key"])].append(sid)
    qkeys = sorted(by_qkey)
    rng = random.Random(seed)
    values = {metric: [] for metric in METRICS}
    for _ in range(resamples):
        sampled = [qkeys[rng.randrange(len(qkeys))] for _ in qkeys]
        left_sample, right_sample = [], []
        for qkey in sampled:
            for sid in by_qkey[qkey]:
                left_sample.append(left_by_id[sid])
                right_sample.append(right_by_id[sid])
        left_metrics, right_metrics = prediction_metrics(left_sample), prediction_metrics(right_sample)
        for metric in METRICS:
            values[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
    observed_left, observed_right = prediction_metrics(left), prediction_metrics(right)
    output = []
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output.append(
            {
                "metric": metric,
                "observed_delta": float(observed_left[metric]) - float(observed_right[metric]),
                "ci_lower": float(np.nanpercentile(array, 2.5)),
                "ci_upper": float(np.nanpercentile(array, 97.5)),
                "resamples": resamples,
                "unit": "question_key",
            }
        )
    return output


def noninferior(main: dict[str, float], other: dict[str, float]) -> bool:
    return (
        main["MAE"] <= other["MAE"] + 0.005
        and main["Exact_Match"] >= other["Exact_Match"] - 0.005
        and main["QWK"] >= other["QWK"] - 0.01
        and main["Kendall_tau"] >= other["Kendall_tau"] - 0.01
        and main["label5_recall"] >= other["label5_recall"] - 0.02
        and main["high_to_low_rate"] <= other["high_to_low_rate"] + 0.01
    )


def meaningful_gain(main: dict[str, float], other: dict[str, float]) -> bool:
    relative_l2h = (
        (other["low_to_high_rate"] - main["low_to_high_rate"]) / other["low_to_high_rate"]
        if other["low_to_high_rate"] > 0 else 0.0
    )
    return (
        relative_l2h >= 0.10
        or main["label2_recall"] - other["label2_recall"] >= 0.05
        or other["MAE"] - main["MAE"] >= 0.005
    )


def main() -> None:
    args = parse_args()
    predictions = {variant: load_predictions(args.out_dir, variant) for variant in VARIANTS}
    ci_rows = []
    for index, (left, right) in enumerate(COMPARISONS):
        for row in bootstrap_pair(predictions[left], predictions[right], args.resamples, args.seed + index):
            ci_rows.append({"left_variant": left, "right_variant": right, **row})
    write_csv(args.out_dir / "tables/exp40a_groupcv_question_key_bootstrap_ci.csv", ci_rows)

    with (args.out_dir / "tables/exp40a_groupcv_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        metrics = {
            row["variant"]: {
                key: (float(value) if key != "variant" and value not in {"", "nan"} else value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        }
    baseline, method = metrics["v0h_human_soft"], metrics["v3_edupair_cf"]
    relative_reduction = (
        (baseline["low_to_high_rate"] - method["low_to_high_rate"]) / baseline["low_to_high_rate"]
        if baseline["low_to_high_rate"] > 0 else 0.0
    )
    ci_by_comparison = defaultdict(dict)
    for row in ci_rows:
        ci_by_comparison[row["right_variant"]][row["metric"]] = row
    no_significant_harm = True
    for right, by_metric in ci_by_comparison.items():
        no_significant_harm &= (
            float(by_metric["MAE"]["ci_lower"]) <= 0
            and float(by_metric["QWK"]["ci_upper"]) >= 0
            and float(by_metric["Exact_Match"]["ci_upper"]) >= 0
        )
    gates = {
        "mae_guard": method["MAE"] <= baseline["MAE"] + 0.005,
        "exact_guard": method["Exact_Match"] >= baseline["Exact_Match"] - 0.005,
        "qwk_guard": method["QWK"] >= baseline["QWK"] - 0.01,
        "kendall_guard": method["Kendall_tau"] >= baseline["Kendall_tau"] - 0.01,
        "label5_guard": method["label5_recall"] >= baseline["label5_recall"] - 0.02,
        "high_to_low_guard": method["high_to_low_rate"] <= baseline["high_to_low_rate"] + 0.01,
        "meaningful_gain_vs_v0h": meaningful_gain(method, baseline),
        "outperforms_v2_unverified": noninferior(method, metrics["v2_unverified_counterfactual_pairs"])
        and meaningful_gain(method, metrics["v2_unverified_counterfactual_pairs"]),
        "outperforms_v4_shuffled": noninferior(method, metrics["v4_shuffled_pair_alignment"])
        and meaningful_gain(method, metrics["v4_shuffled_pair_alignment"]),
        "noninferior_to_v1_human_pair": noninferior(method, metrics["v1_human_real_pairs"]),
        "no_significant_bootstrap_harm_mae_qwk_exact": bool(no_significant_harm),
    }
    go = all(gates.values())
    decision = {
        "status": "GROUPCV_GO" if go else "GROUPCV_STOP",
        "gates": gates,
        "recommend_run_dev_multiseed": go,
        "stop_model_assisted_positive_route": not go,
        "bootstrap_resamples": args.resamples,
        "bootstrap_unit": "question_key",
        "low_to_high_relative_reduction": relative_reduction,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp40a_groupcv_decision.json", decision)
    report = [
        "# Exp40A EduPair-CF GroupCV report", "",
        f"- Status: **{decision['status']}**",
        f"- V0H metrics: `{json.dumps(baseline, sort_keys=True)}`",
        f"- V3 EduPair-CF metrics: `{json.dumps(method, sort_keys=True)}`",
        f"- Gate checks: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend dev multiseed: `{str(go).lower()}`",
        f"- Stop model-assisted positive route: `{str(not go).lower()}`",
        "- Held-out metrics contain original human rows only; no counterfactual row enters evaluation.",
        "- The only added objective is the locked standard RankNet loss with lambda_pair=0.25.",
        "- No failure/reason auxiliary head and no custom risk/ordinal loss were used.",
        "- No paper-like dev/test data were accessed.",
    ]
    report_path = args.out_dir / "reports/exp40a_groupcv_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
