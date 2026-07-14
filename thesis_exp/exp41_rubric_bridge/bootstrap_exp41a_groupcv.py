"""Run locked question-key bootstrap and finalize the Exp41A seed42 gate."""

from __future__ import annotations

import argparse
import csv
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

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    FORMAL_VARIANTS, ROOT, prediction_metrics, read_jsonl, write_csv, write_json,
)

COMPARISONS = (
    ("v3_rubric_bridge", "v0h_human_soft"),
    ("v3_rubric_bridge", "v1_raw_rubric"),
    ("v3_rubric_bridge", "v2_deterministic_checklist"),
    ("v3_rubric_bridge", "v4_shuffled_compiled_rubric"),
    ("v3_rubric_bridge", "v5_human_soft_logit_adjustment"),
    ("v1_raw_rubric", "v0h_human_soft"),
)
METRICS = (
    "MAE", "QWK", "Exact_Match", "Kendall_tau", "abs_Signed_Bias",
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
        sampled_qkeys = [qkeys[rng.randrange(len(qkeys))] for _ in qkeys]
        left_sample, right_sample = [], []
        for qkey in sampled_qkeys:
            for sid in by_qkey[qkey]:
                left_sample.append(left_by_id[sid]); right_sample.append(right_by_id[sid])
        left_metrics, right_metrics = prediction_metrics(left_sample), prediction_metrics(right_sample)
        for metric in METRICS:
            values[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
    observed_left, observed_right = prediction_metrics(left), prediction_metrics(right)
    output = []
    for metric in METRICS:
        array = np.asarray(values[metric], dtype=float)
        output.append({"metric": metric, "observed_delta": float(observed_left[metric]) - float(observed_right[metric]),
                       "ci_lower": float(np.nanpercentile(array, 2.5)), "ci_upper": float(np.nanpercentile(array, 97.5)),
                       "resamples": resamples, "unit": "question_key"})
    return output


def main() -> None:
    args = parse_args()
    predictions = {variant: load_predictions(args.out_dir, variant) for variant in FORMAL_VARIANTS}
    ci_rows = []
    for index, (left, right) in enumerate(COMPARISONS):
        for row in bootstrap_pair(predictions[left], predictions[right], args.resamples, args.seed + index):
            ci_rows.append({"left_variant": left, "right_variant": right, **row})
    write_csv(args.out_dir / "tables/exp41a_groupcv_question_key_bootstrap_ci.csv", ci_rows)
    with (args.out_dir / "tables/exp41a_groupcv_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        metrics = {row["variant"]: {key: (float(value) if key != "variant" and value not in {"", "nan"} else value)
                                    for key, value in row.items()} for row in csv.DictReader(handle)}
    baseline, method = metrics["v0h_human_soft"], metrics["v3_rubric_bridge"]
    by_comparison: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ci_rows:
        if row["left_variant"] == "v3_rubric_bridge":
            by_comparison[row["right_variant"]][row["metric"]] = row
    no_significant_harm = all(
        float(rows["MAE"]["ci_lower"]) <= 0
        and float(rows["QWK"]["ci_upper"]) >= 0
        and float(rows["Exact_Match"]["ci_upper"]) >= 0
        for rows in by_comparison.values()
    )
    primary_gain = method["MAE"] <= baseline["MAE"] - 0.005 or method["QWK"] >= baseline["QWK"] + 0.01
    outperform_v1 = method["MAE"] <= metrics["v1_raw_rubric"]["MAE"] - 0.003 or method["QWK"] >= metrics["v1_raw_rubric"]["QWK"] + 0.005
    outperform_v2 = method["MAE"] <= metrics["v2_deterministic_checklist"]["MAE"] - 0.003 or method["QWK"] >= metrics["v2_deterministic_checklist"]["QWK"] + 0.005
    outperform_v4 = method["MAE"] < metrics["v4_shuffled_compiled_rubric"]["MAE"] or method["QWK"] > metrics["v4_shuffled_compiled_rubric"]["QWK"]
    gates = {
        "primary_gain_mae_or_qwk": primary_gain,
        "exact_guard": method["Exact_Match"] >= baseline["Exact_Match"] - 0.005,
        "kendall_guard": method["Kendall_tau"] >= baseline["Kendall_tau"] - 0.01,
        "absolute_bias_guard": abs(method["Signed_Bias"]) <= abs(baseline["Signed_Bias"]) + 0.01,
        "label5_guard": method["label5_recall"] >= baseline["label5_recall"] - 0.02,
        "high_to_low_guard": method["high_to_low_rate"] <= baseline["high_to_low_rate"] + 0.01,
        "low_to_high_guard": method["low_to_high_rate"] <= baseline["low_to_high_rate"],
        "label2_guard": method["label2_recall"] >= baseline["label2_recall"],
        "outperforms_v1_raw_rubric": outperform_v1,
        "outperforms_v2_deterministic_checklist": outperform_v2,
        "outperforms_v4_shuffled": outperform_v4,
        "no_significant_bootstrap_harm_mae_qwk_exact": bool(no_significant_harm),
    }
    go = all(gates.values())
    decision = {
        "status": "GROUPCV_GO" if go else "GROUPCV_STOP", "gates": gates,
        "recommend_run_multiseed": go, "stop_llm_rubric_compiler_route": not go,
        "bootstrap_resamples": args.resamples, "bootstrap_unit": "question_key",
        "label_free_inference_preprocessing": True, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp41a_groupcv_decision.json", decision)
    report = [
        "# Exp41A RUBRIC-Bridge GroupCV report", "", f"- Status: **{decision['status']}**",
        f"- V0H metrics: `{json.dumps(baseline, sort_keys=True)}`",
        f"- V3 RUBRIC-Bridge metrics: `{json.dumps(method, sort_keys=True)}`",
        f"- Gate checks: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend run multiseed: `{str(go).lower()}`",
        f"- Stop LLM rubric compiler route: `{str(not go).lower()}`",
        "- The Qwen teacher compiled rubrics without seeing answers or labels.",
        "- Human labels were not replaced; training used standard hard/soft cross-entropy only.",
        "- Held-out evaluation contains original human rows only with question-key-disjoint folds.",
        "- No paper-like dev/test data were accessed.",
    ]
    path = args.out_dir / "reports/exp41a_rubric_bridge_groupcv_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
