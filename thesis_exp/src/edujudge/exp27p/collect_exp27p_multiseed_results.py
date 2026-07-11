"""Collect Exp27P seeds 42/43/44 into dev-only stability tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp27p import PAIRWISE_COMPARISONS, RUN_ROOT, VARIANTS
from thesis_exp.src.edujudge.exp27p.bootstrap_exp27p_dev_differences import bootstrap_pair
from thesis_exp.src.edujudge.exp27p.common import prediction_metrics, read_jsonl, write_csv, write_json


DEFAULT_OUTPUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp27p_soft_target_reranker_multiseed_seed42_44"
)
SUMMARY_METRICS = (
    "MAE_argmax",
    "MAE_expected",
    "QWK",
    "Accuracy",
    "Exact_Match",
    "Kendall_tau",
    "Bin_Agreement",
    "Signed_Bias_argmax",
    "low_to_high_count",
    "low_to_high_rate",
    "high_to_low_count",
    "high_to_low_rate",
    "label1_recall",
    "label2_recall",
    "label3_recall",
    "label4_recall",
    "label5_recall",
    "low_mean_p_score_ge_4",
    "high_mean_p_score_le_2",
)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def collect(args: argparse.Namespace) -> dict[str, Any]:
    tables = args.output_dir / "tables"
    reports = args.output_dir / "reports"
    decisions = args.output_dir / "decision"
    configs = args.output_dir / "configs"
    for path in (tables, reports, decisions, configs):
        path.mkdir(parents=True, exist_ok=True)

    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    predictions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    missing: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            run_dir = args.run_root / variant / f"seed_{seed}"
            summary_path = run_dir / "run_summary.json"
            prediction_path = run_dir / "predictions_private" / "selected_dev_predictions.jsonl"
            if not summary_path.exists() or not prediction_path.exists():
                missing.append(f"{variant}:seed_{seed}")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "COMPLETED" or int(summary.get("test_access_count", -1)) != 0:
                raise ValueError(f"Invalid run summary: {summary_path}")
            summaries[(seed, variant)] = summary
            rows = read_jsonl(prediction_path)
            predictions[(seed, variant)] = rows
            metrics = prediction_metrics(rows)
            selected_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "selected_epoch": summary["selected_epoch"],
                    "pure_min_mae_epoch": summary["pure_min_mae_epoch"],
                    "checkpoint_reload_pass": summary["checkpoint_reload_pass"],
                    "zero_weight_window_count": summary["zero_weight_window_count"],
                    **metrics,
                }
            )

    if missing:
        decision = {
            "experiment": "exp27p_multiseed_seed42_44",
            "status": "INCOMPLETE",
            "missing_runs": missing,
            "test_access_count": 0,
        }
        write_json(decisions / "exp27p_multiseed_decision.json", decision)
        return decision

    aggregate_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = [row for row in selected_rows if row["variant"] == variant]
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in variant_rows]
            mean, std = mean_std(values)
            aggregate_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "seed_count": len(values),
                }
            )

    pairwise_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for left, right, effect in PAIRWISE_COMPARISONS:
            left_metrics = prediction_metrics(predictions[(seed, left)])
            right_metrics = prediction_metrics(predictions[(seed, right)])
            for metric in SUMMARY_METRICS:
                pairwise_rows.append(
                    {
                        "seed": seed,
                        "left_variant": left,
                        "right_variant": right,
                        "effect": effect,
                        "metric": metric,
                        "difference_right_minus_left": float(right_metrics[metric])
                        - float(left_metrics[metric]),
                    }
                )
            bootstrap_rows.extend(
                {
                    "seed": seed,
                    **row,
                }
                for row in bootstrap_pair(
                    predictions[(seed, left)],
                    predictions[(seed, right)],
                    left,
                    right,
                    effect,
                    args.bootstrap_resamples,
                    seed,
                )
            )

    metric_lookup = {
        (row["variant"], row["metric"]): float(row["mean"]) for row in aggregate_rows
    }
    v1 = "v1_original_label_matched_weight"
    v2 = "v2_selective_hard_relabel"
    v3 = "v3_selective_soft_audit"
    mean_gate = (
        metric_lookup[(v3, "MAE_argmax")] <= metric_lookup[(v1, "MAE_argmax")] + 0.02
        and metric_lookup[(v3, "QWK")] >= metric_lookup[(v1, "QWK")] - 0.02
        and metric_lookup[(v3, "label5_recall")] >= metric_lookup[(v1, "label5_recall")] - 0.03
    )
    seedwise_gate_count = 0
    for seed in args.seeds:
        by_variant = {
            row["variant"]: row for row in selected_rows if int(row["seed"]) == seed
        }
        if (
            float(by_variant[v3]["MAE_argmax"]) <= float(by_variant[v1]["MAE_argmax"]) + 0.02
            and float(by_variant[v3]["QWK"]) >= float(by_variant[v1]["QWK"]) - 0.02
            and float(by_variant[v3]["label5_recall"]) >= float(by_variant[v1]["label5_recall"]) - 0.03
        ):
            seedwise_gate_count += 1
    recommend_safe16 = (
        metric_lookup[(v3, "low_to_high_rate")] > metric_lookup[(v2, "low_to_high_rate")]
        or metric_lookup[(v3, "Signed_Bias_argmax")]
        > metric_lookup[(v1, "Signed_Bias_argmax")] + 0.05
    )
    decision = {
        "experiment": "exp27p_multiseed_seed42_44",
        "status": "PASS" if mean_gate else "SOFT_TARGET_NOT_STABLE",
        "seeds": args.seeds,
        "all_twelve_runs_completed": True,
        "no_nan_oom": all(summary.get("status") == "COMPLETED" for summary in summaries.values()),
        "no_test_access": all(int(summary.get("test_access_count", -1)) == 0 for summary in summaries.values()),
        "v3_mean_performance_gate_pass": mean_gate,
        "v3_seedwise_gate_pass_count": seedwise_gate_count,
        "recommend_keep_soft_target_method": mean_gate and seedwise_gate_count >= 2,
        "recommend_run_v3_safe16": recommend_safe16,
        "test_access_count": 0,
        "test_predictions_generated": False,
    }

    write_csv(tables / "exp27p_multiseed_selected_metrics.csv", selected_rows)
    write_csv(tables / "exp27p_multiseed_variant_summary.csv", aggregate_rows)
    write_csv(tables / "exp27p_multiseed_pairwise_differences.csv", pairwise_rows)
    write_csv(tables / "exp27p_multiseed_question_key_bootstrap_ci.csv", bootstrap_rows)
    write_json(decisions / "exp27p_multiseed_decision.json", decision)
    write_json(
        configs / "exp27p_multiseed_locked_config.json",
        {
            "seeds": args.seeds,
            "variants": list(VARIANTS),
            "base_model": "Qwen3-Reranker-0.6B",
            "epochs": 10,
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 32,
            "effective_batch_size": 128,
            "checkpoint_selection": "min_MAE_plus_0.005_then_low_to_high_QWK_expected_MAE_earliest",
            "evaluation_split": "dev_only",
        },
    )

    lines = [
        "# Exp27P Multiseed Stability",
        "",
        f"- status: `{decision['status']}`",
        f"- seeds: `{args.seeds}`",
        f"- v3 mean performance gate: `{str(mean_gate).lower()}`",
        f"- v3 seedwise gate pass: `{seedwise_gate_count}/{len(args.seeds)}`",
        f"- recommend v3_safe16: `{str(recommend_safe16).lower()}`",
        "- test accessed: `false`",
        "",
        "| variant | MAE | Bias | Exact | Kendall tau | Bin agree | QWK | low-to-high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        values = {
            row["metric"]: row for row in aggregate_rows if row["variant"] == variant
        }
        lines.append(
            f"| {variant} | {float(values['MAE_argmax']['mean']):.4f}±{float(values['MAE_argmax']['std']):.4f} | "
            f"{float(values['Signed_Bias_argmax']['mean']):.4f}±{float(values['Signed_Bias_argmax']['std']):.4f} | "
            f"{float(values['Exact_Match']['mean']):.4f}±{float(values['Exact_Match']['std']):.4f} | "
            f"{float(values['Kendall_tau']['mean']):.4f}±{float(values['Kendall_tau']['std']):.4f} | "
            f"{float(values['Bin_Agreement']['mean']):.4f}±{float(values['Bin_Agreement']['std']):.4f} | "
            f"{float(values['QWK']['mean']):.4f}±{float(values['QWK']['std']):.4f} | "
            f"{float(values['low_to_high_rate']['mean']):.4f}±{float(values['low_to_high_rate']['std']):.4f} |"
        )
    (reports / "exp27p_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), ensure_ascii=False, sort_keys=True))
