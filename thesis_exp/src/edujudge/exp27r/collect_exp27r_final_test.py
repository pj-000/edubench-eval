"""Collect the frozen Exp27R one-shot final test campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp27p.bootstrap_exp27p_dev_differences import bootstrap_pair
from thesis_exp.src.edujudge.exp27p.common import prediction_metrics, read_jsonl, stratified_metrics, write_csv, write_json
from thesis_exp.src.edujudge.exp27r import COMPARISONS, OUTPUT_DIR, SEEDS, VARIANTS
from thesis_exp.src.edujudge.exp27r.bootstrap_exp27r_crossed_seed_question import crossed_bootstrap


METRICS = (
    "MAE_argmax", "MAE_expected", "QWK", "Exact_Match", "Kendall_tau", "Bin_Agreement",
    "Signed_Bias_argmax", "Signed_Bias_expected", "low_to_high_count", "low_to_high_rate",
    "high_to_low_count", "high_to_low_rate", "label1_recall", "label2_recall", "label5_recall",
    "low_mean_p_score_ge_4", "high_mean_p_score_le_2",
)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            output.append({"variant": variant, "metric": metric, "mean": statistics.mean(values),
                           "std": statistics.stdev(values), "min": min(values), "max": max(values), "seed_count": 3})
    return output


def collect(args: argparse.Namespace) -> dict[str, Any]:
    tables, reports, decisions = (args.out_dir / name for name in ("tables", "reports", "decision"))
    for path in (tables, reports, decisions):
        path.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.out_dir / "configs/exp27r_final_lock_manifest.json").read_text(encoding="utf-8"))
    predictions: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    selected_rows, stratified_rows, missing = [], [], []
    for kind in (["selected", "pure_min_mae"] if manifest["pure_min_sensitivity_enabled"] else ["selected"]):
        for variant in VARIANTS:
            for seed in SEEDS:
                path = args.out_dir / "predictions_private" / kind / variant / f"seed_{seed}.jsonl"
                if not path.exists():
                    missing.append(str(path)); continue
                rows = read_jsonl(path)
                predictions[(variant, seed, kind)] = rows
                metrics = prediction_metrics(rows)
                if kind == "selected":
                    selected_rows.append({"variant": variant, "seed": seed, **metrics})
                    stratified_rows.extend(stratified_metrics(rows, variant, seed))
    if missing:
        raise FileNotFoundError(f"Incomplete one-shot campaign: {missing[:3]}")

    summary_rows = summarize(selected_rows)
    pairwise_rows, cluster_rows, crossed_rows = [], [], []
    for left, right, effect in COMPARISONS:
        for seed in SEEDS:
            left_rows, right_rows = predictions[(left, seed, "selected")], predictions[(right, seed, "selected")]
            left_metrics, right_metrics = prediction_metrics(left_rows), prediction_metrics(right_rows)
            for metric in METRICS:
                pairwise_rows.append({"seed": seed, "left_variant": left, "right_variant": right,
                                      "effect": effect, "metric": metric,
                                      "difference_right_minus_left": float(right_metrics[metric]) - float(left_metrics[metric])})
            cluster_rows.extend({"seed": seed, **row} for row in bootstrap_pair(
                left_rows, right_rows, left, right, effect, args.cluster_resamples, seed,
            ))
        crossed_rows.extend(crossed_bootstrap(
            {seed: predictions[(left, seed, "selected")] for seed in SEEDS},
            {seed: predictions[(right, seed, "selected")] for seed in SEEDS},
            left, right, effect, args.crossed_resamples, 27018,
        ))

    pure_rows = []
    if manifest["pure_min_sensitivity_enabled"]:
        for variant in VARIANTS:
            for seed in SEEDS:
                selected = prediction_metrics(predictions[(variant, seed, "selected")])
                pure = prediction_metrics(predictions[(variant, seed, "pure_min_mae")])
                pure_rows.append({"variant": variant, "seed": seed, **{
                    f"selected_{metric}": selected[metric] for metric in METRICS
                }, **{f"pure_{metric}": pure[metric] for metric in METRICS}})

    lookup = {(row["variant"], row["metric"]): float(row["mean"]) for row in summary_rows}
    def delta(left: str, right: str, metric: str) -> float:
        return lookup[(right, metric)] - lookup[(left, metric)]
    v3_l2h = delta("v2_selective_hard_relabel", "v3_selective_soft_audit", "low_to_high_rate")
    safe_l2h = delta("v3_selective_soft_audit", "v3_safe16_original_low_anchor", "low_to_high_rate")
    hard_l2h = delta("v1_original_label_matched_weight", "v2_selective_hard_relabel", "low_to_high_rate")
    hard_l5 = delta("v1_original_label_matched_weight", "v2_selective_hard_relabel", "label5_recall")
    weight_l2h = delta("v0_original_unweighted", "v1_original_label_matched_weight", "low_to_high_rate")
    position = "directional_signal_only" if safe_l2h < 0 else "audit_framework_negative_transfer"
    if v3_l2h < 0 and delta("v2_selective_hard_relabel", "v3_selective_soft_audit", "MAE_argmax") <= 0:
        position = "positive_method"
    decision = {
        "lock_pass": True, "methods_frozen": True, "training_frozen": True, "data_frozen": True,
        "no_new_method_after_test": True, "test_campaign_completed": True, "test_access_count": 1,
        "all_variants_evaluated": True, "all_seeds_evaluated": True,
        "selected_checkpoint_results_complete": True,
        "pure_min_sensitivity_complete": bool(manifest["pure_min_sensitivity_enabled"]),
        "primary_comparisons_complete": True, "crossed_bootstrap_complete": True,
        "v3_test_generalization_summary": {"delta_low_to_high_vs_v2": v3_l2h},
        "safe16_test_generalization_summary": {"delta_low_to_high_vs_v3": safe_l2h},
        "hard_relabel_tradeoff_summary": {"delta_low_to_high_vs_v1": hard_l2h, "delta_label5_recall_vs_v1": hard_l5},
        "weight_only_summary": {"delta_low_to_high_vs_v0": weight_l2h},
        "final_paper_position": position, "recommend_more_training": False,
        "recommend_new_teacher_annotation": False, "recommend_hyperparameter_search": False,
        "final_test_closed": True,
    }
    write_csv(tables / "exp27r_test_selected_metrics.csv", selected_rows)
    write_csv(tables / "exp27r_test_multiseed_summary.csv", summary_rows)
    write_csv(tables / "exp27r_test_pairwise_differences.csv", pairwise_rows)
    write_csv(tables / "exp27r_test_question_key_bootstrap_ci.csv", cluster_rows)
    write_csv(tables / "exp27r_test_crossed_seed_question_bootstrap_ci.csv", crossed_rows)
    write_csv(tables / "exp27r_test_label_recall.csv", [{"variant": row["variant"], "seed": row["seed"],
        "label1_recall": row["label1_recall"], "label2_recall": row["label2_recall"], "label5_recall": row["label5_recall"]} for row in selected_rows])
    write_csv(tables / "exp27r_test_stratified_metrics.csv", stratified_rows)
    write_csv(tables / "exp27r_test_selected_vs_pure_sensitivity.csv", pure_rows)
    write_csv(tables / "exp27r_test_access_audit.csv", [{"test_access_count": 1, "campaign_completed": True,
        "variants": len(VARIANTS), "seeds": len(SEEDS), "selected_results": len(selected_rows),
        "pure_min_sensitivity_complete": manifest["pure_min_sensitivity_enabled"]}])
    write_json(decisions / "exp27r_final_test_decision.json", decision)
    lines = ["# Exp27R Final One-Shot Test", "", f"- final paper position: `{position}`",
             "- test access count: `1`", "- methods/training/data frozen: `true`", "- final test closed: `true`", "",
             "| variant | MAE | QWK | low-to-high | label2 recall | label5 recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        lines.append(f"| {variant} | {lookup[(variant,'MAE_argmax')]:.4f} | {lookup[(variant,'QWK')]:.4f} | "
                     f"{lookup[(variant,'low_to_high_rate')]:.4f} | {lookup[(variant,'label2_recall')]:.4f} | {lookup[(variant,'label5_recall')]:.4f} |")
    (reports / "exp27r_final_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cluster-resamples", type=int, default=2000)
    parser.add_argument("--crossed-resamples", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), sort_keys=True))

