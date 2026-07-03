"""Collect Exp19-R5 next-scout dev predictions.

This collector handles the follow-up scout runs requested after the first R5
DPO scout:

- R5E hard-synthetic control from R1b.
- R5C no-mid score-risk from R1b.
- R5C no-mid score-risk from R2c.

It reuses the same dev metric implementation as the earlier Exp19 SFT/R5
collectors so that MAE/QWK/low-to-high comparisons stay on one evaluation
surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.exp17_low_score_evidence.collect_exp19_r5_dpo_scout_dev_results import (  # noqa: E402
    D1_FIELDS,
    FAILURE_FIELDS,
    METRIC_FIELDS,
    copy_d1_fields,
    copy_failure_fields,
    copy_metric_fields,
    d1_score_cap_rate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_next_scout")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_BASELINE_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round")
DEFAULT_R5_SCOUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_dpo_scout")

RUNS = [
    {
        "run_name": "r5e_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5e_hard_synthetic_control",
    },
    {
        "run_name": "r5c_no_mid_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5c_score_risk_no_mid",
    },
    {
        "run_name": "r5c_no_mid_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5c_score_risk_no_mid",
    },
]

BASELINE_META = {
    "r1b_score_only_balanced": {
        "run_name": "r1b_score_only_balanced",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "none_init_baseline",
    },
    "r2c_clean_reason_score_balanced": {
        "run_name": "r2c_clean_reason_score_balanced",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "none_init_baseline",
    },
    "r4b_shuffled_reason_balanced": {
        "run_name": "r4b_shuffled_reason_balanced",
        "init_adapter": "r4b_shuffled_reason_balanced",
        "dpo_dataset": "none_init_baseline",
    },
}

PREVIOUS_R5_META = {
    "r5c_from_r1b": {
        "run_name": "r5c_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5c_score_risk",
    },
    "r5c_from_r2c": {
        "run_name": "r5c_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5c_score_risk",
    },
    "r5e_from_r2c": {
        "run_name": "r5e_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5e_hard_synthetic_control",
    },
}


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    val = to_float(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows}


def collect_new_runs(
    prediction_root: Path,
    reference: list[dict[str, str]],
    d1_dir: Path,
    allow_missing_predictions: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    d1_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for meta in RUNS:
        run_name = meta["run_name"]
        run_dir = prediction_root / run_name
        try:
            prediction_file = sft_collect.find_prediction_file(run_dir)
        except FileNotFoundError:
            missing.append(run_name)
            if allow_missing_predictions:
                continue
            raise
        predictions = sft_collect.load_prediction_records(prediction_file)
        parsed_rows = sft_collect.align_predictions(reference, predictions, run_name, run_name)
        metric_rows.append(copy_metric_fields(sft_collect.metric_summary(parsed_rows, run_name, run_name, "dev"), meta))
        d1_rows.append(copy_d1_fields(sft_collect.d1_eval_row(parsed_rows, d1_dir, run_name, run_name), run_name))
        failure_rows.append(
            copy_failure_fields(
                sft_collect.failure_type_eval_row(parsed_rows, d1_dir, run_name, run_name),
                run_name,
                d1_score_cap_rate(parsed_rows, d1_dir),
            )
        )
    return metric_rows, d1_rows, failure_rows, missing


def append_prior_rows(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    included: list[str] = []
    missing: list[str] = []

    baseline_metrics = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.baseline_dir / "tables" / "exp19_sft_second_round_dev_metrics.csv")
    }
    baseline_d1 = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.baseline_dir / "tables" / "exp19_sft_second_round_dev_d1_hidden_eval.csv")
    }
    baseline_failure = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.baseline_dir / "tables" / "exp19_sft_second_round_dev_failure_type_eval.csv")
    }
    previous_metrics = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.r5_scout_dir / "tables" / "exp19_r5_dpo_scout_dev_metrics.csv")
    }
    previous_d1 = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.r5_scout_dir / "tables" / "exp19_r5_dpo_scout_d1_hidden_eval.csv")
    }
    previous_failure = {
        row.get("run_name", ""): row
        for row in read_csv_if_exists(args.r5_scout_dir / "tables" / "exp19_r5_dpo_scout_failure_type_eval.csv")
    }

    for run_name, meta in {**BASELINE_META, **PREVIOUS_R5_META}.items():
        source_metrics = baseline_metrics if run_name in BASELINE_META else previous_metrics
        source_d1 = baseline_d1 if run_name in BASELINE_META else previous_d1
        source_failure = baseline_failure if run_name in BASELINE_META else previous_failure
        if run_name not in source_metrics:
            missing.append(run_name)
            continue
        metric_rows.append(copy_metric_fields(source_metrics[run_name], meta))
        if run_name in source_d1:
            d1_rows.append(copy_d1_fields(source_d1[run_name], run_name))
        if run_name in source_failure:
            failure_rows.append(copy_failure_fields(source_failure[run_name], run_name))
        included.append(run_name)
    return {"included": included, "missing": missing}


def delta(new: dict[str, Any] | None, old: dict[str, Any] | None, key: str) -> float:
    if not new or not old:
        return float("nan")
    return to_float(new.get(key)) - to_float(old.get(key))


def make_decision(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    d1 = by_run(d1_rows)
    r5e_r1b_gap = abs(delta(metrics.get("r5e_from_r1b"), metrics.get("r5c_from_r1b"), "low_to_high_rate"))
    r5e_r1b_similar = (
        r5e_r1b_gap <= 0.02
        and abs(delta(metrics.get("r5e_from_r1b"), metrics.get("r5c_from_r1b"), "MAE")) <= 0.05
        and abs(delta(metrics.get("r5e_from_r1b"), metrics.get("r5c_from_r1b"), "label2_recall")) <= 0.05
    )
    no_mid_r1b_l2h_gain = -delta(metrics.get("r5c_no_mid_from_r1b"), metrics.get("r5c_from_r1b"), "low_to_high_rate")
    no_mid_r2c_l2h_gain = -delta(metrics.get("r5c_no_mid_from_r2c"), metrics.get("r5c_from_r2c"), "low_to_high_rate")
    no_mid_r1b_d1_gain = -delta(d1.get("r5c_no_mid_from_r1b"), d1.get("r5c_from_r1b"), "pred_ge4_rate_d1_hidden")
    no_mid_r2c_d1_gain = -delta(d1.get("r5c_no_mid_from_r2c"), d1.get("r5c_from_r2c"), "pred_ge4_rate_d1_hidden")

    if r5e_r1b_similar:
        recommendation = "prioritize_r5f_rejection_mining"
        reason = "R5E from R1b is close to R5C from R1b; synthetic-control effects remain a concern."
    elif max(no_mid_r1b_l2h_gain, no_mid_r2c_l2h_gain) >= 0.03:
        recommendation = "consider_no_mid_plus_r5f"
        reason = "No-mid score-risk improves low-to-high enough to keep as a candidate, but R5F mining is still needed."
    else:
        recommendation = "prioritize_r5f_rejection_mining"
        reason = "No-mid or control scouts do not provide enough evidence for full DPO."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "r5e_from_r1b_vs_r5c_from_r1b": {
            "low_to_high_abs_gap": r5e_r1b_gap,
            "similar": r5e_r1b_similar,
        },
        "r5c_no_mid_from_r1b_vs_r5c_main": {
            "low_to_high_improvement": no_mid_r1b_l2h_gain,
            "d1_pred_ge4_improvement": no_mid_r1b_d1_gain,
            "MAE_delta": delta(metrics.get("r5c_no_mid_from_r1b"), metrics.get("r5c_from_r1b"), "MAE"),
            "QWK_delta": delta(metrics.get("r5c_no_mid_from_r1b"), metrics.get("r5c_from_r1b"), "QWK"),
        },
        "r5c_no_mid_from_r2c_vs_r5c_main": {
            "low_to_high_improvement": no_mid_r2c_l2h_gain,
            "d1_pred_ge4_improvement": no_mid_r2c_d1_gain,
            "MAE_delta": delta(metrics.get("r5c_no_mid_from_r2c"), metrics.get("r5c_from_r2c"), "MAE"),
            "QWK_delta": delta(metrics.get("r5c_no_mid_from_r2c"), metrics.get("r5c_from_r2c"), "QWK"),
        },
        "guardrails": {
            "no_test_read": True,
            "d1_used_for_eval_only": True,
            "human_rationale_not_in_prompt": True,
        },
    }


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    prior_summary: dict[str, list[str]],
    missing_predictions: list[str],
) -> None:
    lines = [
        "# Exp19-R5 Next Scout Dev Evaluation",
        "",
        "This report evaluates the R5E-from-R1b control and R5C-no-mid score-risk scouts on the original dev split.",
        "Raw predictions/logs/checkpoints remain gitignored. No test split is read.",
        "",
        "## Dev Metrics",
        "",
        "| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['run_name']}` | `{row['init_adapter']}` | `{row['dpo_dataset']}` | "
            f"{fmt(row['MAE'])} | {fmt(row['QWK'])} | {fmt(row['low_to_high_rate'])} | "
            f"{fmt(row['high_to_low_rate'])} | {fmt(row['label2_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## D1 Hidden And Failure-Type Tables",
            "",
            "| run | D1 pred>=4 | D1 label2 recall | failure micro-F1 | D1 nonempty failure | score_cap nonnull |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    d1_by_name = by_run(d1_rows)
    for row in failure_rows:
        name = str(row["run_name"])
        d1 = d1_by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(d1.get('pred_ge4_rate_d1_hidden'))} | {fmt(d1.get('label2_recall_d1'))} | "
            f"{fmt(row.get('failure_type_micro_f1'))} | {fmt(row.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(row.get('score_cap_nonnull_rate_on_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"- R5E from R1b similar to R5C from R1b: `{decision['r5e_from_r1b_vs_r5c_from_r1b']['similar']}` "
            f"(low-to-high gap={fmt(decision['r5e_from_r1b_vs_r5c_from_r1b']['low_to_high_abs_gap'])}).",
            "- R5C no-mid from R1b low-to-high improvement over R5C main: "
            f"{fmt(decision['r5c_no_mid_from_r1b_vs_r5c_main']['low_to_high_improvement'])}.",
            "- R5C no-mid from R2c low-to-high improvement over R5C main: "
            f"{fmt(decision['r5c_no_mid_from_r2c_vs_r5c_main']['low_to_high_improvement'])}.",
            f"- recommendation: `{decision['recommendation']}`.",
            f"- reason: {decision['reason']}",
            "",
            "## Sources",
            "",
            f"- included prior rows: {', '.join(prior_summary.get('included') or []) or 'none'}",
            f"- missing prior rows: {', '.join(prior_summary.get('missing') or []) or 'none'}",
            f"- missing new prediction runs: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- Evaluation uses the original dev split, not balanced train distribution.",
            "- Test split is not read.",
            "- D1 annotations are evaluation references only.",
            "- Human rationale is not included in the prediction prompt.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r5_next_scout_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = sft_collect.read_csv_rows(args.reference_csv)
    metric_rows, d1_rows, failure_rows, missing_predictions = collect_new_runs(
        args.prediction_root,
        reference,
        args.d1_dir,
        args.allow_missing_predictions,
    )
    prior_summary = append_prior_rows(args, metric_rows, d1_rows, failure_rows)
    decision = make_decision(metric_rows, d1_rows)
    write_csv(args.out_dir / "tables" / "exp19_r5_next_scout_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5_next_scout_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5_next_scout_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_json(args.out_dir / "decision" / "exp19_r5_next_scout_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, decision, prior_summary, missing_predictions)
    return {
        "new_runs_collected": len([row for row in metric_rows if row["run_name"] in {item["run_name"] for item in RUNS}]),
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19-R5 next-scout dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--r5-scout-dir", type=Path, default=DEFAULT_R5_SCOUT_DIR)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
