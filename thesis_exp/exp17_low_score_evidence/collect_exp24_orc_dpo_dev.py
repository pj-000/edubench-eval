"""Collect Exp24 ORC-DPO dev predictions and compare to Exp23/baselines."""

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
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_EXP23_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp23_r7_dpo_scout")
DEFAULT_TRAINING_SUMMARY_DIR = DEFAULT_OUT_DIR / "training_summaries"


RUNS = [
    {
        "run_name": "exp24_orc_a_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r7g_orc_score_channel_reason_aux",
    },
    {
        "run_name": "exp24_orc_b_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r7g_orc_score_channel_reason_aux",
    },
    {
        "run_name": "exp24_orc_c_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r7g_orc_score_channel_reason_aux",
    },
]

BASELINE_NAMES = [
    "r2c_clean_reason_score_balanced",
    "r4b_shuffled_reason_balanced",
    "r7d_reason_real_s100_b0p03_lr5em6",
    "r7e_matched_score_only_s100_b0p03_lr5em6",
    "r7f_score_reason_consistency_s100_b0p03_lr5em6",
]

MINIMUM_SUCCESS = {
    "low_to_high_delta_vs_r7e_max": -0.10,
    "mae_delta_vs_r7e_max": 0.03,
    "qwk_delta_vs_r7e_min": -0.03,
    "label5_recall_delta_vs_r7e_min": 0.0,
    "d1_hidden_pred_ge4_max": 0.90,
}

STRONG_SUCCESS = {
    "MAE_max": 0.43,
    "QWK_min": 0.50,
    "low_to_high_rate_max": 0.5965,
    "label2_recall_min_exclusive": 0.10,
    "label5_recall_min": 0.80,
    "d1_hidden_pred_ge4_max": 0.8462,
}


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


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


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def by_run(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_name", "")): row for row in rows}


def collect_orc_rows(
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


def append_exp23_rows(
    exp23_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    exp23_metrics = by_run(read_csv_if_exists(exp23_dir / "tables" / "exp23_r7_dpo_scout_dev_metrics.csv"))
    exp23_d1 = by_run(read_csv_if_exists(exp23_dir / "tables" / "exp23_r7_dpo_scout_d1_hidden_eval.csv"))
    exp23_failure = by_run(read_csv_if_exists(exp23_dir / "tables" / "exp23_r7_dpo_scout_failure_type_eval.csv"))
    included: list[str] = []
    missing: list[str] = []
    for run_name in BASELINE_NAMES:
        if run_name not in exp23_metrics:
            missing.append(run_name)
            continue
        row = dict(exp23_metrics[run_name])
        # Exp23 rows already have init_adapter and dpo_dataset columns.
        metric_rows.append(row)
        if run_name in exp23_d1:
            d1_rows.append(dict(exp23_d1[run_name]))
        if run_name in exp23_failure:
            failure_rows.append(dict(exp23_failure[run_name]))
        included.append(run_name)
    return {"included": included, "missing": missing}


def load_training_summaries(summary_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        path = summary_dir / f"{run['run_name']}.json"
        if not path.exists():
            rows.append({"run_name": run["run_name"], "completed": False})
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        last = data.get("last_metrics") or {}
        rows.append(
            {
                "run_name": data.get("run_name", run["run_name"]),
                "completed": int(data.get("completed_steps", 0)) >= int(data.get("max_steps", 0)),
                "output_dir": data.get("output_dir", ""),
                "rows": data.get("rows", ""),
                "max_steps": data.get("max_steps", ""),
                "completed_steps": data.get("completed_steps", ""),
                "learning_rate": data.get("learning_rate", ""),
                "beta": data.get("beta", ""),
                "pref_ftx": data.get("pref_ftx", ""),
                "lambda_reason": data.get("lambda_reason", ""),
                "alpha_lh": data.get("alpha_lh", ""),
                "alpha_hl": data.get("alpha_hl", ""),
                "alpha_lm": data.get("alpha_lm", ""),
                "alpha_hm": data.get("alpha_hm", ""),
                "alpha_d": data.get("alpha_d", ""),
                "margin_lh": data.get("margin_lh", ""),
                "margin_hl": data.get("margin_hl", ""),
                "margin_d": data.get("margin_d", ""),
                "last_loss": last.get("loss", ""),
                "last_orc_loss": last.get("orc_loss", ""),
                "last_reason_loss": last.get("reason_loss", ""),
                "elapsed_seconds": data.get("elapsed_seconds", ""),
            }
        )
    return rows


def run_success(metric: dict[str, Any], d1: dict[str, Any], r7e: dict[str, Any]) -> dict[str, Any]:
    l2h_delta = to_float(metric.get("low_to_high_rate")) - to_float(r7e.get("low_to_high_rate"))
    mae_delta = to_float(metric.get("MAE")) - to_float(r7e.get("MAE"))
    qwk_delta = to_float(metric.get("QWK")) - to_float(r7e.get("QWK"))
    label5_delta = to_float(metric.get("label5_recall")) - to_float(r7e.get("label5_recall"))
    d1_pred_ge4 = to_float(d1.get("pred_ge4_rate_d1_hidden"))
    minimum_checks = {
        "low_to_high_delta_vs_r7e": l2h_delta <= MINIMUM_SUCCESS["low_to_high_delta_vs_r7e_max"],
        "mae_delta_vs_r7e": mae_delta <= MINIMUM_SUCCESS["mae_delta_vs_r7e_max"],
        "qwk_delta_vs_r7e": qwk_delta >= MINIMUM_SUCCESS["qwk_delta_vs_r7e_min"],
        "label5_delta_vs_r7e": label5_delta >= MINIMUM_SUCCESS["label5_recall_delta_vs_r7e_min"],
        "d1_hidden_pred_ge4": d1_pred_ge4 < MINIMUM_SUCCESS["d1_hidden_pred_ge4_max"],
    }
    strong_checks = {
        "MAE": to_float(metric.get("MAE")) <= STRONG_SUCCESS["MAE_max"],
        "QWK": to_float(metric.get("QWK")) >= STRONG_SUCCESS["QWK_min"],
        "low_to_high": to_float(metric.get("low_to_high_rate")) <= STRONG_SUCCESS["low_to_high_rate_max"],
        "label2_recall": to_float(metric.get("label2_recall")) > STRONG_SUCCESS["label2_recall_min_exclusive"],
        "label5_recall": to_float(metric.get("label5_recall")) >= STRONG_SUCCESS["label5_recall_min"],
        "d1_hidden_pred_ge4": d1_pred_ge4 <= STRONG_SUCCESS["d1_hidden_pred_ge4_max"],
    }
    return {
        "minimum_success": all(minimum_checks.values()),
        "strong_success": all(strong_checks.values()),
        "minimum_checks": minimum_checks,
        "strong_checks": strong_checks,
        "deltas_vs_r7e": {
            "low_to_high_rate": l2h_delta,
            "MAE": mae_delta,
            "QWK": qwk_delta,
            "label5_recall": label5_delta,
            "D1_hidden_pred_ge4": d1_pred_ge4 - to_float(d1.get("r7e_pred_ge4_rate_d1_hidden", float("nan"))),
        },
        "score": sum(1 for ok in minimum_checks.values() if ok) + sum(1 for ok in strong_checks.values() if ok),
    }


def make_decision(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    d1_by_name = by_run(d1_rows)
    if "r7e_matched_score_only_s100_b0p03_lr5em6" not in metrics:
        return {
            "recommendation": "wait_for_r7e_baseline",
            "reason": "Exp23 R7E baseline is required for Exp24 ORC comparison.",
        }
    r7e = metrics["r7e_matched_score_only_s100_b0p03_lr5em6"]
    r7e_d1_pred = to_float(d1_by_name.get("r7e_matched_score_only_s100_b0p03_lr5em6", {}).get("pred_ge4_rate_d1_hidden"))
    evaluations: dict[str, Any] = {}
    for run in RUNS:
        name = run["run_name"]
        if name not in metrics:
            continue
        d1 = dict(d1_by_name.get(name, {}))
        d1["r7e_pred_ge4_rate_d1_hidden"] = r7e_d1_pred
        evaluations[name] = run_success(metrics[name], d1, r7e)
    minimum = [name for name, row in evaluations.items() if row["minimum_success"]]
    strong = [name for name, row in evaluations.items() if row["strong_success"]]
    best_name = ""
    if evaluations:
        best_name = sorted(
            evaluations,
            key=lambda name: (
                to_float(metrics[name].get("low_to_high_rate")),
                to_float(metrics[name].get("MAE")),
                -to_float(metrics[name].get("QWK")),
            ),
        )[0]
    if strong:
        recommendation = "proceed_to_exp25_orc_src_auxiliary"
        reason = f"{strong[0]} satisfies the strong ORC-DPO success rule."
    elif minimum:
        recommendation = "orc_minimum_success_consider_r4b_secondary_or_exp25"
        reason = f"{minimum[0]} satisfies the minimum success rule vs R7E."
    elif evaluations:
        recommendation = "orc_not_yet_successful"
        reason = "No Exp24 ORC run satisfies the minimum success rule vs R7E."
    else:
        recommendation = "wait_for_exp24_predictions"
        reason = "No Exp24 ORC predictions were collected."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "best_by_low_to_high_mae_qwk": best_name,
        "minimum_success_runs": minimum,
        "strong_success_runs": strong,
        "evaluations": json_safe(evaluations),
        "minimum_success_rule": MINIMUM_SUCCESS,
        "strong_success_rule": STRONG_SUCCESS,
        "guardrails": {
            "no_test_read": True,
            "dev_used_for_eval_only": True,
            "human_reason_not_in_prompt": True,
        },
    }


def write_report(
    out_dir: Path,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    included: dict[str, list[str]],
    missing_predictions: list[str],
) -> None:
    d1_by_name = by_run(d1_rows)
    failure_by_name = by_run(failure_rows)
    lines = [
        "# Exp24 Score-Channel ORC-DPO Dev Evaluation",
        "",
        "Exp24 uses score-only DPO responses plus ordinal/risk weights and margins.",
        "Human rationales are auxiliary targets, not contrasted directly against rejected score responses.",
        "",
        "## Dev Metrics",
        "",
        "| run | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall | D1 pred>=4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        d1 = d1_by_name.get(str(row["run_name"]), {})
        lines.append(
            f"| `{row['run_name']}` | `{row.get('dpo_dataset', '')}` | {fmt(row.get('MAE'))} | "
            f"{fmt(row.get('QWK'))} | {fmt(row.get('low_to_high_rate'))} | "
            f"{fmt(row.get('high_to_low_rate'))} | {fmt(row.get('label2_recall'))} | "
            f"{fmt(row.get('label5_recall'))} | {fmt(d1.get('pred_ge4_rate_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Training Summary",
            "",
            "| run | completed | steps | alpha_lh | alpha_hl | alpha_d | margin_lh | margin_hl | margin_d | lambda_reason | last loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training_rows:
        lines.append(
            f"| `{row.get('run_name', '')}` | {row.get('completed', '')} | {row.get('completed_steps', '')}/{row.get('max_steps', '')} | "
            f"{row.get('alpha_lh', '')} | {row.get('alpha_hl', '')} | {row.get('alpha_d', '')} | "
            f"{row.get('margin_lh', '')} | {row.get('margin_hl', '')} | {row.get('margin_d', '')} | "
            f"{row.get('lambda_reason', '')} | {fmt(row.get('last_loss'))} |"
        )
    lines.extend(
        [
            "",
            "## Structured Failure Diagnostics",
            "",
            "| run | failure micro-F1 | D1 nonempty failure | score_cap nonnull |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        failure = failure_by_name.get(str(row["run_name"]), {})
        lines.append(
            f"| `{row['run_name']}` | {fmt(failure.get('failure_type_micro_f1'))} | "
            f"{fmt(failure.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(failure.get('score_cap_nonnull_rate_on_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            f"- best_by_low_to_high_mae_qwk: `{decision.get('best_by_low_to_high_mae_qwk', '')}`",
            f"- minimum_success_runs: {', '.join(decision.get('minimum_success_runs') or []) or 'none'}",
            f"- strong_success_runs: {', '.join(decision.get('strong_success_runs') or []) or 'none'}",
            "",
            "## Sources",
            "",
            f"- included Exp23/baseline rows: {', '.join(included.get('included') or []) or 'none'}",
            f"- missing Exp23/baseline rows: {', '.join(included.get('missing') or []) or 'none'}",
            f"- missing Exp24 predictions: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- Test split is not read.",
            "- Dev labels are used only for evaluation.",
            "- Human rationale is not included in the prediction prompt.",
            "- Raw predictions, logs, checkpoints, and adapter weights must not be committed.",
        ]
    )
    write_text(out_dir / "reports" / "exp24_orc_dpo_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = sft_collect.read_csv_rows(args.reference_csv)
    metric_rows, d1_rows, failure_rows, missing_predictions = collect_orc_rows(
        args.prediction_root,
        reference,
        args.d1_dir,
        args.allow_missing_predictions,
    )
    included = append_exp23_rows(args.exp23_dir, metric_rows, d1_rows, failure_rows)
    training_rows = load_training_summaries(args.training_summary_dir)
    decision = json_safe(make_decision(metric_rows, d1_rows))
    write_csv(args.out_dir / "tables" / "exp24_orc_dpo_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp24_orc_dpo_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp24_orc_dpo_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_csv(args.out_dir / "tables" / "exp24_orc_dpo_training_summary.csv", training_rows)
    write_json(args.out_dir / "decision" / "exp24_orc_dpo_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, training_rows, decision, included, missing_predictions)
    return {
        "exp24_runs_collected": len([row for row in metric_rows if str(row.get("run_name", "")).startswith("exp24_")]),
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp24 ORC-DPO dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--exp23-dir", type=Path, default=DEFAULT_EXP23_DIR)
    parser.add_argument("--training-summary-dir", type=Path, default=DEFAULT_TRAINING_SUMMARY_DIR)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
