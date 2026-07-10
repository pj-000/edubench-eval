"""Collect Exp27P seed42 runs into lightweight dev-only tables and decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp27p import EXP27O_DIR, OUTPUT_DIR, PAIRWISE_COMPARISONS, RUN_ROOT, VARIANTS
from thesis_exp.src.edujudge.exp27p.bootstrap_exp27p_dev_differences import bootstrap_all
from thesis_exp.src.edujudge.exp27p.common import (
    prediction_metrics,
    read_jsonl,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pairwise_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "MAE_argmax",
        "MAE_expected",
        "QWK",
        "Accuracy",
        "Signed_Bias_argmax",
        "low_to_high_count",
        "low_to_high_rate",
        "high_to_low_count",
        "high_to_low_rate",
        "label1_recall",
        "label2_recall",
        "label5_recall",
    )
    output = []
    for left, right, effect in PAIRWISE_COMPARISONS:
        for metric in keys:
            output.append(
                {
                    "left_variant": left,
                    "right_variant": right,
                    "effect": effect,
                    "metric": metric,
                    "left_value": metrics[left][metric],
                    "right_value": metrics[right][metric],
                    "difference_right_minus_left": float(metrics[right][metric]) - float(metrics[left][metric]),
                }
            )
    return output


def pivot_high_impact(rows_by_variant: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    ids = sorted({row["sample_id"] for rows in rows_by_variant.values() for row in rows})
    output = []
    for sid in ids:
        row: dict[str, Any] = {"sample_id": sid}
        for variant, rows in rows_by_variant.items():
            current = next((item for item in rows if item["sample_id"] == sid), None)
            if current is None:
                continue
            for key in (
                "sample_id_hash",
                "original_score",
                "silver_hard_score",
                "silver_score_range",
                "soft_target_5",
                "sample_weight",
            ):
                row.setdefault(key, current[key])
            for key in ("pred_label_5", "pred_score_expected", "agrees_original", "agrees_silver"):
                row[f"{variant}_{key}"] = current[key]
            for label in range(1, 6):
                row[f"{variant}_prob_{label}"] = current[f"prob_{label}"]
        output.append(row)
    return output


def matched_question_key_audit(
    predictions: list[dict[str, Any]], manifest: list[dict[str, str]]
) -> dict[str, int]:
    strata = {
        (str(row.get("language")), str(row.get("metric_group")), str(row.get("subject")))
        for row in manifest
    }
    matched = [
        row
        for row in predictions
        if int(row["gold_label_5"]) <= 2
        and (str(row.get("language")), str(row.get("metric_group")), str(row.get("subject_canonical")))
        in strata
    ]
    train_hashes = {str(row["question_key_hash"]) for row in manifest}
    dev_hashes = {
        hashlib.sha1(str(row["question_key"]).encode("utf-8")).hexdigest()
        for row in matched
    }
    return {
        "train_high_impact_question_key_count": len(train_hashes),
        "dev_matched_question_key_count": len(dev_hashes),
        "question_key_overlap_count": len(train_hashes & dev_hashes),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    tables = args.output_dir / "tables"
    reports = args.output_dir / "reports"
    decisions = args.output_dir / "decision"
    for path in (tables, reports, decisions):
        path.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    missing = []
    for variant in VARIANTS:
        path = args.run_root / variant / f"seed_{args.seed}" / "run_summary.json"
        if not path.exists():
            missing.append(variant)
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETED" or summary.get("test_access_count") != 0:
            raise ValueError(f"Invalid Exp27P run summary: {path}")
        summaries[variant] = summary
    if missing:
        decision = {
            "experiment": "exp27p_seed42_scout",
            "status": "INCOMPLETE",
            "missing_variants": missing,
            "recommend_run_seeds_43_44": False,
            "recommend_run_v3_safe16": False,
            "test_access_count": 0,
        }
        write_json(decisions / "exp27p_seed42_scout_decision.json", decision)
        return decision

    selected_rows = []
    epoch_rows = []
    predictions: dict[str, list[dict[str, Any]]] = {}
    train_tier_rows = []
    high_by_variant = {}
    matched_rows = []
    strata_rows = []
    with (EXP27O_DIR / "tables" / "exp27o_high_impact16_manifest_light.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        high_impact_manifest = list(csv.DictReader(handle))
    for variant, summary in summaries.items():
        selected = dict(summary["selected_metrics"])
        selected.update(
            {
                "variant": variant,
                "seed": args.seed,
                "selected_epoch": summary["selected_epoch"],
                "pure_min_mae_epoch": summary["pure_min_mae_epoch"],
                "checkpoint_reload_pass": summary["checkpoint_reload_pass"],
                "zero_weight_window_count": summary["zero_weight_window_count"],
            }
        )
        selected_rows.append(selected)
        run_dir = args.run_root / variant / f"seed_{args.seed}"
        epoch_rows.extend(json.loads((run_dir / "epoch_metrics.json").read_text(encoding="utf-8")))
        predictions[variant] = read_jsonl(run_dir / "predictions_private" / "selected_dev_predictions.jsonl")
        train_tier_rows.extend(read_csv(run_dir / "train_tier_fit_diagnostics.csv"))
        high_by_variant[variant] = read_csv(run_dir / "high_impact16_fit_diagnostics.csv")
        variant_matched_rows = read_csv(run_dir / "dev_high_impact_matched_strata.csv")
        for row in variant_matched_rows:
            row.update(matched_question_key_audit(predictions[variant], high_impact_manifest))
        matched_rows.extend(variant_matched_rows)
        strata_rows.extend(read_csv(run_dir / "dev_stratified_metrics.csv"))
    metric_map = {variant: prediction_metrics(rows) for variant, rows in predictions.items()}
    pair_rows = pairwise_rows(metric_map)
    bootstrap_rows = bootstrap_all(args.run_root, args.seed, args.bootstrap_resamples)
    label_rows = []
    for variant, metrics in metric_map.items():
        for label in (1, 2, 5):
            label_rows.append(
                {
                    "variant": variant,
                    "seed": args.seed,
                    "label": label,
                    "recall": metrics[f"label{label}_recall"],
                }
            )
    high_pivot = pivot_high_impact(high_by_variant)

    write_csv(tables / "exp27p_selected_checkpoint_metrics.csv", selected_rows)
    write_csv(tables / "exp27p_epoch_metrics.csv", epoch_rows)
    write_csv(tables / "exp27p_pairwise_dev_differences.csv", pair_rows)
    write_csv(tables / "exp27p_question_key_bootstrap_ci.csv", bootstrap_rows)
    write_csv(tables / "exp27p_label_recall.csv", label_rows)
    write_csv(tables / "exp27p_language_metric_subject_metrics.csv", strata_rows)
    write_csv(tables / "exp27p_train_tier_fit_diagnostics.csv", train_tier_rows)
    write_csv(tables / "exp27p_high_impact16_fit_diagnostics.csv", high_pivot)
    write_csv(tables / "exp27p_dev_high_impact_matched_strata.csv", matched_rows)

    v1 = metric_map["v1_original_label_matched_weight"]
    v2 = metric_map["v2_selective_hard_relabel"]
    v3 = metric_map["v3_selective_soft_audit"]
    no_nan_oom = all(summary.get("status") == "COMPLETED" for summary in summaries.values())
    performance_gate = (
        float(v3["MAE_argmax"]) <= float(v1["MAE_argmax"]) + 0.02
        and float(v3["QWK"]) >= float(v1["QWK"]) - 0.02
        and float(v3["label5_recall"]) >= float(v1["label5_recall"]) - 0.03
    )
    simultaneously_worse_risk = (
        float(v3["low_to_high_rate"]) > max(float(v1["low_to_high_rate"]), float(v2["low_to_high_rate"]))
        and float(v3["label2_recall"]) < min(float(v1["label2_recall"]), float(v2["label2_recall"]))
        and float(v3["high_to_low_rate"]) > max(float(v1["high_to_low_rate"]), float(v2["high_to_low_rate"]))
    )
    recommend_multiseed = no_nan_oom and performance_gate and not simultaneously_worse_risk
    signed_bias_rise = float(v3["Signed_Bias_argmax"]) - float(v1["Signed_Bias_argmax"])
    recommend_safe16 = (
        float(v3["low_to_high_rate"]) > float(v1["low_to_high_rate"])
        or signed_bias_rise > 0.05
        or abs(float(v3["MAE_argmax"]) - (float(v1["MAE_argmax"]) + 0.02)) <= 0.005
    )
    decision = {
        "experiment": "exp27p_seed42_scout",
        "status": "PASS" if recommend_multiseed else "STOP_AFTER_SEED42",
        "all_four_runs_completed": True,
        "no_nan_oom": no_nan_oom,
        "no_test_access": all(summary.get("test_access_count") == 0 for summary in summaries.values()),
        "performance_gate_pass": performance_gate,
        "v3_not_simultaneously_worse_all_risk": not simultaneously_worse_risk,
        "recommend_run_seeds_43_44": recommend_multiseed,
        "recommend_run_v3_safe16": recommend_safe16,
        "test_access_count": 0,
        "test_predictions_generated": False,
    }
    write_json(decisions / "exp27p_seed42_scout_decision.json", decision)
    lines = [
        "# Exp27P Seed42 Scout",
        "",
        f"- status: `{decision['status']}`",
        f"- recommend seeds 43/44: `{str(recommend_multiseed).lower()}`",
        f"- recommend v3_safe16: `{str(recommend_safe16).lower()}`",
        "- test accessed: `false`",
        "",
        "| variant | epoch | MAE | QWK | low-to-high | label2 recall | label5 recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(selected_rows, key=lambda item: item["variant"]):
        lines.append(
            f"| {row['variant']} | {row['selected_epoch']} | {float(row['MAE_argmax']):.4f} | "
            f"{float(row['QWK']):.4f} | {float(row['low_to_high_rate']):.4f} | "
            f"{float(row['label2_recall']):.4f} | {float(row['label5_recall']):.4f} |"
        )
    (reports / "exp27p_seed42_scout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), ensure_ascii=False, sort_keys=True))
