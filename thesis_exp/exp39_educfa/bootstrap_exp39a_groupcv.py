"""Run locked question-key bootstrap and finalize the Exp39A GroupCV gate."""

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

from thesis_exp.exp39_educfa.common import ROOT, VARIANTS, prediction_metrics, read_jsonl, write_csv, write_json  # noqa: E402

COMPARISONS = tuple(("v4_educfa", variant) for variant in VARIANTS if variant != "v4_educfa")
METRICS = (
    "MAE", "Exact_Match", "Kendall_tau", "QWK", "Bin_Agreement",
    "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall",
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


def bootstrap_pair(left: list[dict[str, Any]], right: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
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
        output.append({
            "metric": metric,
            "observed_delta": float(observed_left[metric]) - float(observed_right[metric]),
            "ci_lower": float(np.nanpercentile(array, 2.5)),
            "ci_upper": float(np.nanpercentile(array, 97.5)),
            "resamples": resamples,
            "unit": "question_key",
        })
    return output


def main() -> None:
    args = parse_args()
    predictions = {variant: load_predictions(args.out_dir, variant) for variant in VARIANTS}
    ci_rows = []
    for index, (left, right) in enumerate(COMPARISONS):
        for row in bootstrap_pair(predictions[left], predictions[right], args.resamples, args.seed + index):
            ci_rows.append({"left_variant": left, "right_variant": right, **row})
    write_csv(args.out_dir / "tables/exp39a_groupcv_question_key_bootstrap_ci.csv", ci_rows)

    with (args.out_dir / "tables/exp39a_groupcv_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        metrics = {
            row["variant"]: {key: (float(value) if key != "variant" and value not in {"", "nan"} else value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        }
    baseline, educfa = metrics["v0h_human_soft"], metrics["v4_educfa"]
    relative_reduction = (
        (baseline["low_to_high_rate"] - educfa["low_to_high_rate"]) / baseline["low_to_high_rate"]
        if baseline["low_to_high_rate"] > 0 else 0.0
    )
    meaningful_gain = (
        relative_reduction >= 0.15
        or educfa["label2_recall"] - baseline["label2_recall"] >= 0.05
        or baseline["MAE"] - educfa["MAE"] >= 0.005
    )

    def outperforms(name: str) -> bool:
        other = metrics[name]
        other_relative_l2h = (
            (other["low_to_high_rate"] - educfa["low_to_high_rate"]) / other["low_to_high_rate"]
            if other["low_to_high_rate"] > 0 else 0.0
        )
        no_material_harm = (
            educfa["MAE"] <= other["MAE"] + 0.005
            and educfa["Exact_Match"] >= other["Exact_Match"] - 0.005
            and educfa["QWK"] >= other["QWK"] - 0.01
            and educfa["label5_recall"] >= other["label5_recall"] - 0.02
            and educfa["high_to_low_rate"] <= other["high_to_low_rate"] + 0.01
        )
        material_gain = (
            other["MAE"] - educfa["MAE"] >= 0.005
            or other_relative_l2h >= 0.10
            or educfa["label2_recall"] - other["label2_recall"] >= 0.05
        )
        return no_material_harm and material_gain

    v0_ci = {row["metric"]: row for row in ci_rows if row["right_variant"] == "v0h_human_soft"}
    no_significant_harm = (
        float(v0_ci["MAE"]["ci_lower"]) <= 0
        and float(v0_ci["QWK"]["ci_upper"]) >= 0
        and float(v0_ci["Exact_Match"]["ci_upper"]) >= 0
    )
    gates = {
        "mae_guard": educfa["MAE"] <= baseline["MAE"] + 0.005,
        "exact_guard": educfa["Exact_Match"] >= baseline["Exact_Match"] - 0.005,
        "qwk_guard": educfa["QWK"] >= baseline["QWK"] - 0.01,
        "kendall_guard": educfa["Kendall_tau"] >= baseline["Kendall_tau"] - 0.01,
        "label5_guard": educfa["label5_recall"] >= baseline["label5_recall"] - 0.02,
        "high_to_low_guard": educfa["high_to_low_rate"] <= baseline["high_to_low_rate"] + 0.01,
        "meaningful_gain": meaningful_gain,
        "outperforms_real_low_oversampling": outperforms("v1_matched_real_low_oversampling"),
        "outperforms_unverified_or_generic": outperforms("v2_unverified_counterfactual") or outperforms("v3_generic_corruption"),
        "not_fully_matched_by_shuffled": outperforms("v5_shuffled_counterfactual"),
        "no_significant_bootstrap_harm": no_significant_harm,
    }
    go = all(gates.values())
    decision = {
        "status": "GROUPCV_GO" if go else "GROUPCV_STOP",
        "gates": gates,
        "recommend_run_seeds_43_44": go,
        "stop_positive_small_paper_method_search": not go,
        "bootstrap_resamples": args.resamples,
        "bootstrap_unit": "question_key",
        "low_to_high_relative_reduction": relative_reduction,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39a_groupcv_decision.json", decision)
    report = [
        "# Exp39A EduCFA GroupCV report", "",
        f"- Status: **{decision['status']}**",
        f"- V0H metrics: `{json.dumps(baseline, sort_keys=True)}`",
        f"- V4 metrics: `{json.dumps(educfa, sort_keys=True)}`",
        f"- Gate checks: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend seeds 43/44: `{str(go).lower()}`",
        f"- Stop positive small-paper method search: `{str(not go).lower()}`",
        "- Evaluation contains only original held-out human rows; synthetic rows never enter metrics.",
        "- Standard soft cross-entropy only; no custom loss or architecture change.",
        "- No paper-like dev/test data were accessed.",
    ]
    report_path = args.out_dir / "reports/exp39a_groupcv_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
