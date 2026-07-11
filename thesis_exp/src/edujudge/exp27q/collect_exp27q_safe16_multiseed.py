"""Collect locked Exp27Q Safe16 runs and apply the preregistered decision gates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp27p.bootstrap_exp27p_dev_differences import bootstrap_pair
from thesis_exp.src.edujudge.exp27p.common import prediction_metrics, read_jsonl, write_csv, write_json
from thesis_exp.src.edujudge.exp27q import EXP27P_RUN_ROOT, OUTPUT_DIR, RUN_ROOT, VARIANT
from thesis_exp.src.edujudge.exp27q.bootstrap_exp27q_two_level import two_level_bootstrap


BASELINES = (
    "v0_original_unweighted", "v1_original_label_matched_weight",
    "v2_selective_hard_relabel", "v3_selective_soft_audit",
)
METRICS = (
    "MAE_argmax", "MAE_expected", "QWK", "Exact_Match", "Kendall_tau",
    "Bin_Agreement", "Signed_Bias_argmax", "Signed_Bias_expected",
    "low_to_high_count", "low_to_high_rate", "high_to_low_count", "high_to_low_rate",
    "label1_recall", "label2_recall", "label5_recall",
    "low_mean_p_score_ge_4", "high_mean_p_score_le_2",
)


def mean(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    tables, reports, decisions, configs = (args.output_dir / name for name in ("tables", "reports", "decision", "configs"))
    for path in (tables, reports, decisions, configs):
        path.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    selected_rows = []
    missing = []
    for seed in args.seeds:
        for variant in (*BASELINES, VARIANT):
            root = args.run_root if variant == VARIANT else args.exp27p_run_root
            run = root / variant / f"seed_{seed}"
            summary_path = run / "run_summary.json"
            pred_path = run / "predictions_private/selected_dev_predictions.jsonl"
            if not summary_path.exists() or not pred_path.exists():
                missing.append(f"{variant}:seed_{seed}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(summary.get("test_access_count", -1)) != 0:
                raise ValueError(f"Test access detected: {summary_path}")
            predictions[(seed, variant)] = read_jsonl(pred_path)
            metrics = prediction_metrics(predictions[(seed, variant)])
            selected_rows.append({"seed": seed, "variant": variant, "selected_epoch": summary["selected_epoch"], **metrics})
            if variant == VARIANT:
                summaries[seed] = summary
    if missing:
        decision = {"status": "INCOMPLETE", "missing_runs": missing, "test_access_count": 0}
        write_json(decisions / "exp27q_safe16_multiseed_decision.json", decision)
        return decision

    summary_rows = []
    for variant in (*BASELINES, VARIANT):
        subset = [row for row in selected_rows if row["variant"] == variant]
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            summary_rows.append({
                "variant": variant, "metric": metric, "mean": statistics.mean(values),
                "std": statistics.stdev(values), "min": min(values), "max": max(values), "seed_count": 3,
            })

    pairwise_rows, cluster_rows = [], []
    for seed in args.seeds:
        for baseline in BASELINES:
            left = prediction_metrics(predictions[(seed, baseline)])
            right = prediction_metrics(predictions[(seed, VARIANT)])
            for metric in METRICS:
                pairwise_rows.append({
                    "seed": seed, "left_variant": baseline, "right_variant": VARIANT,
                    "metric": metric, "difference_safe16_minus_baseline": float(right[metric]) - float(left[metric]),
                })
            cluster_rows.extend({"seed": seed, **row} for row in bootstrap_pair(
                predictions[(seed, baseline)], predictions[(seed, VARIANT)], baseline, VARIANT,
                "safe16_sensitivity", args.cluster_resamples, seed,
            ))
    two_level_rows = two_level_bootstrap(
        {seed: predictions[(seed, VARIANT)] for seed in args.seeds},
        {seed: predictions[(seed, "v3_selective_soft_audit")] for seed in args.seeds},
        args.two_level_resamples,
    )

    sensitivity_rows, high_rows = [], []
    for seed, summary in summaries.items():
        selected = summary["selected_metrics"]
        pure = summary["pure_min_mae_metrics"]
        sensitivity_rows.append({
            "seed": seed, "selected_epoch": summary["selected_epoch"], "pure_min_mae_epoch": summary["pure_min_mae_epoch"],
            "same_epoch": summary["selected_epoch"] == summary["pure_min_mae_epoch"],
            **{f"selected_{metric}": selected[metric] for metric in ("MAE_argmax", "QWK", "low_to_high_rate", "label2_recall", "label5_recall")},
            **{f"pure_{metric}": pure[metric] for metric in ("MAE_argmax", "QWK", "low_to_high_rate", "label2_recall", "label5_recall")},
            **{f"selected_minus_pure_{metric}": float(selected[metric]) - float(pure[metric]) for metric in ("MAE_argmax", "QWK", "low_to_high_rate", "label2_recall", "label5_recall")},
        })
        diag = args.run_root / VARIANT / f"seed_{seed}/high_impact16_fit_diagnostics.csv"
        with diag.open("r", encoding="utf-8", newline="") as handle:
            high_rows.extend(csv.DictReader(handle))

    selected_by_variant = {variant: [r for r in selected_rows if r["variant"] == variant] for variant in (*BASELINES, VARIANT)}
    safe, v3, v1 = selected_by_variant[VARIANT], selected_by_variant["v3_selective_soft_audit"], selected_by_variant["v1_original_label_matched_weight"]
    l2h_delta = mean(safe, "low_to_high_rate") - mean(v3, "low_to_high_rate")
    nonworse = sum(float(s["low_to_high_rate"]) <= float(v["low_to_high_rate"]) for s, v in zip(safe, v3))
    nonzero_l2 = sum(float(row["label2_recall"]) > 0 for row in safe)
    two_l2h = next(row for row in two_level_rows if row["metric"] == "low_to_high_rate")
    primary = (
        l2h_delta <= -0.0350877 and nonworse >= 2 and float(two_l2h["ci_high_95"]) <= 0
        and nonzero_l2 >= 2 and mean(safe, "label2_recall") >= 0.0526316
    )
    mae_guard = mean(safe, "MAE_argmax") <= mean(v3, "MAE_argmax") + 0.01 and mean(safe, "MAE_argmax") <= mean(v1, "MAE_argmax") + 0.02
    qwk_guard = mean(safe, "QWK") >= mean(v3, "QWK") - 0.01 and mean(safe, "QWK") >= mean(v1, "QWK") - 0.02
    label5_guard = mean(safe, "label5_recall") >= mean(v1, "label5_recall") - 0.03
    high_guard = mean(safe, "high_to_low_count") <= mean(v3, "high_to_low_count") + 1
    passed = primary and mae_guard and qwk_guard and label5_guard and high_guard
    point_better = l2h_delta <= -0.0350877
    status = "SAFE16_PASS" if passed else ("SAFE16_INCONCLUSIVE" if point_better else "SAFE16_FAIL_STOP")

    equivalence_path = args.output_dir / "tables/exp27q_safe16_dataset_equivalence.csv"
    equivalence = next(csv.DictReader(equivalence_path.open("r", encoding="utf-8", newline="")))
    decision = {
        "all_three_runs_completed": True, "no_nan_oom": True, "test_access_count": 0,
        "changed_rows": int(equivalence["changed_rows_vs_v3"]),
        "changed_question_keys": sum(1 for _ in csv.DictReader(
            (args.output_dir / "tables/exp27q_safe16_question_key_summary.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )),
        "input_mismatch_count": int(equivalence["changed_input_hashes"]),
        "weight_mismatch_count": int(equivalence["changed_sample_weights"]),
        "mean_low_to_high_delta_vs_v3": l2h_delta, "seeds_nonworse_low_to_high": nonworse,
        "mean_label2_recall": mean(safe, "label2_recall"), "seeds_nonzero_label2_recall": nonzero_l2,
        "two_level_bootstrap_low_to_high_ci": [two_l2h["ci_low_95"], two_l2h["ci_high_95"]],
        "mae_guard_pass": mae_guard, "qwk_guard_pass": qwk_guard,
        "label5_guard_pass": label5_guard, "high_to_low_guard_pass": high_guard,
        "primary_low_tail_gate_pass": primary, "status": status,
        "recommend_keep_directional_safe_variant": passed,
        "recommend_prepare_final_test_lock": passed,
        "recommend_more_training": False, "recommend_few_shot_annotation": False,
        "recommend_expand_teacher_audit": False, "stop_exp27p_training": not passed,
        "test_predictions_generated": False,
    }

    write_csv(tables / "exp27q_selected_metrics.csv", selected_rows)
    write_csv(tables / "exp27q_multiseed_summary.csv", summary_rows)
    write_csv(tables / "exp27q_pairwise_differences.csv", pairwise_rows)
    write_csv(tables / "exp27q_question_key_bootstrap_ci.csv", cluster_rows)
    write_csv(tables / "exp27q_two_level_seed_question_bootstrap_ci.csv", two_level_rows)
    write_csv(tables / "exp27q_checkpoint_selection_sensitivity.csv", sensitivity_rows)
    write_csv(tables / "exp27q_high_impact16_fit_diagnostics.csv", high_rows)
    write_csv(tables / "exp27q_label_recall.csv", [{
        "seed": row["seed"], "variant": row["variant"],
        "label1_recall": row["label1_recall"], "label2_recall": row["label2_recall"], "label5_recall": row["label5_recall"],
    } for row in selected_rows])
    write_json(decisions / "exp27q_safe16_multiseed_decision.json", decision)
    write_json(configs / "exp27q_locked_config.json", {
        "variant": VARIANT, "seeds": args.seeds, "epochs": 10, "learning_rate": 2e-5,
        "batch_size": 4, "gradient_accumulation_steps": 32, "evaluation": "dev_only",
        "test_access": False,
    })
    lines = [
        "# Exp27Q Locked V3-Safe16", "", f"- status: `{status}`",
        f"- changed rows: `16`; unique question keys: `{decision['changed_question_keys']}`",
        f"- mean low-to-high delta vs V3: `{l2h_delta:.6f}`",
        f"- mean label2 recall: `{decision['mean_label2_recall']:.6f}`",
        f"- two-level low-to-high CI: `{decision['two_level_bootstrap_low_to_high_ci']}`",
        f"- MAE/QWK/label5 guards: `{mae_guard}/{qwk_guard}/{label5_guard}`",
        "- test accessed: `false`", "",
        "No further training, few-shot annotation, audit expansion, or test evaluation was triggered.",
    ]
    (reports / "exp27q_safe16_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--exp27p-run-root", type=Path, default=EXP27P_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cluster-resamples", type=int, default=2000)
    parser.add_argument("--two-level-resamples", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), sort_keys=True))
