"""Collect Exp19-R5H two-stage DPO scout dev predictions."""

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
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5h_two_stage_dpo import RUN_CONFIGS  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_dpo_scout")
DEFAULT_PREDICTION_ROOT = DEFAULT_OUT_DIR / "dev_predictions"
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)
DEFAULT_BASELINE_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round")
DEFAULT_R5F2_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_dpo_scout")
DEFAULT_R5G_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_dpo_scout")

RUNS = [
    {
        "run_name": str(run["run_name"]),
        "init_adapter": str(run["init_family"]),
        "dpo_dataset": "r5h_high_protection_only",
    }
    for run in RUN_CONFIGS
]

BASELINE_META = {
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
R5F2_META = {
    "r5f2_real_only_from_r2c": {
        "run_name": "r5f2_real_only_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5f2_real_only_small",
    },
}
R5G_META = {
    "r5g_a3_real_only_s50_b0p05_lr5em6": {
        "run_name": "r5g_a3_real_only_s50_b0p05_lr5em6",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5g_real_only",
    },
}

SUCCESS_THRESHOLDS = {
    "low_to_high_rate_max": 0.35,
    "label2_recall_min": 0.20,
    "d1_pred_ge4_rate_max": 0.65,
    "high_to_low_rate_max": 0.04,
    "label5_recall_min": 0.75,
    "MAE_max": 0.50,
    "QWK_min": 0.50,
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
    value_float = to_float(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


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


def append_rows_from_dir(
    rows_dir: Path,
    table_prefix: str,
    metas: dict[str, dict[str, str]],
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    metrics = {row.get("run_name", ""): row for row in read_csv_if_exists(rows_dir / "tables" / f"{table_prefix}_dev_metrics.csv")}
    d1 = {row.get("run_name", ""): row for row in read_csv_if_exists(rows_dir / "tables" / f"{table_prefix}_d1_hidden_eval.csv")}
    failure = {row.get("run_name", ""): row for row in read_csv_if_exists(rows_dir / "tables" / f"{table_prefix}_failure_type_eval.csv")}
    included: list[str] = []
    missing: list[str] = []
    for run_name, meta in metas.items():
        if run_name not in metrics:
            missing.append(run_name)
            continue
        metric_rows.append(copy_metric_fields(metrics[run_name], meta))
        if run_name in d1:
            d1_rows.append(copy_d1_fields(d1[run_name], run_name))
        if run_name in failure:
            failure_rows.append(copy_failure_fields(failure[run_name], run_name))
        included.append(run_name)
    return {"included": included, "missing": missing}


def append_prior_rows(
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    d1_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    included: list[str] = []
    missing: list[str] = []
    for summary in [
        append_rows_from_dir(args.baseline_dir, "exp19_sft_second_round", BASELINE_META, metric_rows, d1_rows, failure_rows),
        append_rows_from_dir(args.r5f2_dir, "exp19_r5f2_dpo_scout", R5F2_META, metric_rows, d1_rows, failure_rows),
        append_rows_from_dir(args.r5g_dir, "exp19_r5g_dpo_scout", R5G_META, metric_rows, d1_rows, failure_rows),
    ]:
        included.extend(summary["included"])
        missing.extend(summary["missing"])
    return {"included": included, "missing": missing}


def run_passes_thresholds(metric: dict[str, Any], d1: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "low_to_high": to_float(metric.get("low_to_high_rate")) <= SUCCESS_THRESHOLDS["low_to_high_rate_max"],
        "label2_recall": to_float(metric.get("label2_recall")) >= SUCCESS_THRESHOLDS["label2_recall_min"],
        "d1_pred_ge4": to_float(d1.get("pred_ge4_rate_d1_hidden")) <= SUCCESS_THRESHOLDS["d1_pred_ge4_rate_max"],
        "high_to_low": to_float(metric.get("high_to_low_rate")) <= SUCCESS_THRESHOLDS["high_to_low_rate_max"],
        "label5_recall": to_float(metric.get("label5_recall")) >= SUCCESS_THRESHOLDS["label5_recall_min"],
        "MAE": to_float(metric.get("MAE")) <= SUCCESS_THRESHOLDS["MAE_max"],
        "QWK": to_float(metric.get("QWK")) >= SUCCESS_THRESHOLDS["QWK_min"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "score": sum(1 for ok in checks.values() if ok),
    }


def metric_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        to_float(row.get("low_to_high_rate")),
        to_float(row.get("high_to_low_rate")),
        to_float(row.get("MAE")),
        -to_float(row.get("QWK")),
    )


def failure_reason(best_metric: dict[str, Any], best_d1: dict[str, Any]) -> str:
    low_to_high = to_float(best_metric.get("low_to_high_rate"))
    high_to_low = to_float(best_metric.get("high_to_low_rate"))
    d1_ge4 = to_float(best_d1.get("pred_ge4_rate_d1_hidden"))
    label5 = to_float(best_metric.get("label5_recall"))
    if low_to_high <= SUCCESS_THRESHOLDS["low_to_high_rate_max"] and high_to_low > SUCCESS_THRESHOLDS["high_to_low_rate_max"]:
        return "low risk remains controlled but high-protection is too weak."
    if high_to_low <= SUCCESS_THRESHOLDS["high_to_low_rate_max"] and label5 >= SUCCESS_THRESHOLDS["label5_recall_min"] and low_to_high > SUCCESS_THRESHOLDS["low_to_high_rate_max"]:
        return "high protection recovers, but low-to-high rebounds."
    if d1_ge4 > SUCCESS_THRESHOLDS["d1_pred_ge4_rate_max"]:
        return "D1 hidden pred>=4 remains high; more D1-like data or annotation may be needed."
    return "no R5H run satisfies all guardrails; continue calibration only if needed."


def make_decision(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    d1_by_name = by_run(d1_rows)
    new_run_names = [run["run_name"] for run in RUNS if run["run_name"] in metrics]
    evaluations = {
        name: run_passes_thresholds(metrics[name], d1_by_name.get(name, {}))
        for name in new_run_names
    }
    passed = [name for name, result in evaluations.items() if result["passed"]]
    best_name = ""
    if new_run_names:
        best_name = sorted(new_run_names, key=lambda name: metric_key(metrics[name]))[0]
    if passed:
        recommendation = "consider_full_dpo_candidate"
        reason = f"{passed[0]} passes the R5H two-stage success rule."
    elif best_name:
        recommendation = "continue_or_stop_by_tradeoff"
        reason = failure_reason(metrics[best_name], d1_by_name.get(best_name, {}))
    else:
        recommendation = "missing_predictions"
        reason = "No R5H prediction rows were collected."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "best_by_low_high_mae_qwk": best_name,
        "passed_runs": passed,
        "thresholds": SUCCESS_THRESHOLDS,
        "run_evaluations": evaluations,
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
    d1_by_name = by_run(d1_rows)
    lines = [
        "# Exp19-R5H Two-Stage DPO Scout Dev Evaluation",
        "",
        "R5H starts from low-risk DPO adapters and applies lightweight high-protection-only DPO.",
        "",
        "## Dev Metrics",
        "",
        "| run | init | dataset | MAE | QWK | low-to-high | high-to-low | label2 recall | label5 recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['run_name']}` | `{row['init_adapter']}` | `{row['dpo_dataset']}` | "
            f"{fmt(row['MAE'])} | {fmt(row['QWK'])} | {fmt(row['low_to_high_rate'])} | "
            f"{fmt(row['high_to_low_rate'])} | {fmt(row['label2_recall'])} | {fmt(row['label5_recall'])} |"
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
            "## R5H Success Rule",
            "",
            f"- low-to-high <= {SUCCESS_THRESHOLDS['low_to_high_rate_max']}",
            f"- label2 recall >= {SUCCESS_THRESHOLDS['label2_recall_min']}",
            f"- D1 hidden pred>=4 <= {SUCCESS_THRESHOLDS['d1_pred_ge4_rate_max']}",
            f"- high-to-low <= {SUCCESS_THRESHOLDS['high_to_low_rate_max']}",
            f"- label5 recall >= {SUCCESS_THRESHOLDS['label5_recall_min']}",
            f"- MAE <= {SUCCESS_THRESHOLDS['MAE_max']}",
            f"- QWK >= {SUCCESS_THRESHOLDS['QWK_min']}",
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            f"- best_by_low_high_mae_qwk: `{decision.get('best_by_low_high_mae_qwk', '')}`",
            f"- passed_runs: {', '.join(decision.get('passed_runs') or []) or 'none'}",
            "",
            "## Sources",
            "",
            f"- included prior rows: {', '.join(prior_summary.get('included') or []) or 'none'}",
            f"- missing prior rows: {', '.join(prior_summary.get('missing') or []) or 'none'}",
            f"- missing R5H prediction runs: {', '.join(missing_predictions) or 'none'}",
            "",
            "## Guardrails",
            "",
            "- Evaluation uses the original dev split.",
            "- Test split is not read.",
            "- D1 annotations are evaluation references only.",
            "- Human rationale is not included in the prediction prompt.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r5h_dpo_scout_report.md", "\n".join(lines))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    reference = sft_collect.read_csv_rows(args.reference_csv)
    metric_rows, d1_rows, failure_rows, missing_predictions = collect_new_runs(
        args.prediction_root,
        reference,
        args.d1_dir,
        args.allow_missing_predictions,
    )
    prior_summary = append_prior_rows(args, metric_rows, d1_rows, failure_rows)
    decision = json_safe(make_decision(metric_rows, d1_rows))
    write_csv(args.out_dir / "tables" / "exp19_r5h_dpo_scout_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5h_dpo_scout_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5h_dpo_scout_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_json(args.out_dir / "decision" / "exp19_r5h_dpo_scout_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, decision, prior_summary, missing_predictions)
    return {
        "new_runs_collected": len([row for row in metric_rows if row["run_name"] in {item["run_name"] for item in RUNS}]),
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19-R5H DPO scout dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--r5f2-dir", type=Path, default=DEFAULT_R5F2_DIR)
    parser.add_argument("--r5g-dir", type=Path, default=DEFAULT_R5G_DIR)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
