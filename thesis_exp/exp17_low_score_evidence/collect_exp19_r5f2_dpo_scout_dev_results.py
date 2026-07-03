"""Collect Exp19-R5F2 DPO scout dev predictions.

R5F2 is a small-step DPO scout built from expanded rejection mining. The
collector keeps the same dev metric surface as the earlier Exp19 SFT/R5
collectors and appends prior baseline/control rows for direct comparison.
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


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_dpo_scout")
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
DEFAULT_R5_NEXT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_next_scout")

RUNS = [
    {
        "run_name": "r5f2_main_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5f2_score_risk_main",
    },
    {
        "run_name": "r5f2_main_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5f2_score_risk_main",
    },
    {
        "run_name": "r5f2_real_only_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5f2_real_only_small",
    },
    {
        "run_name": "r5f2_real_only_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5f2_real_only_small",
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

R5_SCOUT_META = {
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

R5_NEXT_META = {
    "r5e_from_r1b": {
        "run_name": "r5e_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5e_hard_synthetic_control",
    },
    "r5c_no_mid_from_r1b": {
        "run_name": "r5c_no_mid_from_r1b",
        "init_adapter": "r1b_score_only_balanced",
        "dpo_dataset": "r5c_score_risk_no_mid",
    },
    "r5c_no_mid_from_r2c": {
        "run_name": "r5c_no_mid_from_r2c",
        "init_adapter": "r2c_clean_reason_score_balanced",
        "dpo_dataset": "r5c_score_risk_no_mid",
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
    baseline = append_rows_from_dir(
        args.baseline_dir,
        "exp19_sft_second_round",
        BASELINE_META,
        metric_rows,
        d1_rows,
        failure_rows,
    )
    r5_scout = append_rows_from_dir(
        args.r5_scout_dir,
        "exp19_r5_dpo_scout",
        R5_SCOUT_META,
        metric_rows,
        d1_rows,
        failure_rows,
    )
    r5_next = append_rows_from_dir(
        args.r5_next_dir,
        "exp19_r5_next_scout",
        R5_NEXT_META,
        metric_rows,
        d1_rows,
        failure_rows,
    )
    return {
        "included": baseline["included"] + r5_scout["included"] + r5_next["included"],
        "missing": baseline["missing"] + r5_scout["missing"] + r5_next["missing"],
    }


def delta(new: dict[str, Any] | None, old: dict[str, Any] | None, key: str) -> float:
    if not new or not old:
        return float("nan")
    return to_float(new.get(key)) - to_float(old.get(key))


def improvement_lower(new: dict[str, Any] | None, old: dict[str, Any] | None, key: str) -> float:
    val = delta(new, old, key)
    return -val if not math.isnan(val) else val


def dpo_vs_init(
    run: dict[str, Any] | None,
    base: dict[str, Any] | None,
    run_d1: dict[str, Any] | None,
    base_d1: dict[str, Any] | None,
) -> dict[str, Any]:
    low_to_high_improvement = improvement_lower(run, base, "low_to_high_rate")
    label2_delta = delta(run, base, "label2_recall")
    high_to_low_delta = delta(run, base, "high_to_low_rate")
    mae_delta = delta(run, base, "MAE")
    qwk_delta = delta(run, base, "QWK")
    d1_pred_ge4_improvement = improvement_lower(run_d1, base_d1, "pred_ge4_rate_d1_hidden")
    success = (
        low_to_high_improvement >= 0.05
        and label2_delta > 0
        and high_to_low_delta <= 0.05
        and mae_delta <= 0.05
        and qwk_delta >= -0.05
        and d1_pred_ge4_improvement > 0
    )
    return {
        "success": bool(success),
        "low_to_high_improvement": low_to_high_improvement,
        "label2_recall_delta": label2_delta,
        "high_to_low_delta": high_to_low_delta,
        "MAE_delta": mae_delta,
        "QWK_delta": qwk_delta,
        "d1_pred_ge4_improvement": d1_pred_ge4_improvement,
    }


def beats_control(run: dict[str, Any] | None, control: dict[str, Any] | None, run_d1: dict[str, Any] | None, control_d1: dict[str, Any] | None) -> dict[str, Any]:
    low_to_high_advantage = improvement_lower(run, control, "low_to_high_rate")
    d1_advantage = improvement_lower(run_d1, control_d1, "pred_ge4_rate_d1_hidden")
    mae_delta = delta(run, control, "MAE")
    qwk_delta = delta(run, control, "QWK")
    beats = low_to_high_advantage >= 0.02 and d1_advantage >= 0 and mae_delta <= 0.05 and qwk_delta >= -0.05
    return {
        "beats": bool(beats),
        "low_to_high_advantage": low_to_high_advantage,
        "d1_pred_ge4_advantage": d1_advantage,
        "MAE_delta": mae_delta,
        "QWK_delta": qwk_delta,
    }


def make_decision(metric_rows: list[dict[str, Any]], d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = by_run(metric_rows)
    d1 = by_run(d1_rows)
    main_r1b = dpo_vs_init(
        metrics.get("r5f2_main_from_r1b"),
        metrics.get("r1b_score_only_balanced"),
        d1.get("r5f2_main_from_r1b"),
        d1.get("r1b_score_only_balanced"),
    )
    main_r2c = dpo_vs_init(
        metrics.get("r5f2_main_from_r2c"),
        metrics.get("r2c_clean_reason_score_balanced"),
        d1.get("r5f2_main_from_r2c"),
        d1.get("r2c_clean_reason_score_balanced"),
    )
    real_only_r1b = dpo_vs_init(
        metrics.get("r5f2_real_only_from_r1b"),
        metrics.get("r1b_score_only_balanced"),
        d1.get("r5f2_real_only_from_r1b"),
        d1.get("r1b_score_only_balanced"),
    )
    real_only_r2c = dpo_vs_init(
        metrics.get("r5f2_real_only_from_r2c"),
        metrics.get("r2c_clean_reason_score_balanced"),
        d1.get("r5f2_real_only_from_r2c"),
        d1.get("r2c_clean_reason_score_balanced"),
    )
    main_r1b_vs_r5e = beats_control(
        metrics.get("r5f2_main_from_r1b"),
        metrics.get("r5e_from_r1b"),
        d1.get("r5f2_main_from_r1b"),
        d1.get("r5e_from_r1b"),
    )
    main_r2c_vs_r5e = beats_control(
        metrics.get("r5f2_main_from_r2c"),
        metrics.get("r5e_from_r2c"),
        d1.get("r5f2_main_from_r2c"),
        d1.get("r5e_from_r2c"),
    )
    main_r1b_vs_real = beats_control(
        metrics.get("r5f2_main_from_r1b"),
        metrics.get("r5f2_real_only_from_r1b"),
        d1.get("r5f2_main_from_r1b"),
        d1.get("r5f2_real_only_from_r1b"),
    )
    main_r2c_vs_real = beats_control(
        metrics.get("r5f2_main_from_r2c"),
        metrics.get("r5f2_real_only_from_r2c"),
        d1.get("r5f2_main_from_r2c"),
        d1.get("r5f2_real_only_from_r2c"),
    )

    if main_r1b["success"] and main_r1b_vs_r5e["beats"]:
        recommendation = "consider_full_r5f2_main_from_r1b"
        reason = "R5F2 main from R1b passes the small-step success rule and beats the hard-synthetic control."
    elif main_r2c["success"] and main_r2c_vs_r5e["beats"]:
        recommendation = "consider_full_r5f2_main_from_r2c"
        reason = "R5F2 main from R2c passes the small-step success rule and beats the hard-synthetic control."
    elif main_r1b["low_to_high_improvement"] > 0 or main_r2c["low_to_high_improvement"] > 0:
        recommendation = "continue_rejection_mining_or_adjust_dpo"
        reason = "R5F2 gives some risk-side movement, but it does not pass the guarded full-DPO rule."
    else:
        recommendation = "do_not_scale_r5f2"
        reason = "R5F2 does not improve the required score-risk metrics enough on dev."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "r5f2_main_from_r1b_vs_init": main_r1b,
        "r5f2_main_from_r2c_vs_init": main_r2c,
        "r5f2_real_only_from_r1b_vs_init": real_only_r1b,
        "r5f2_real_only_from_r2c_vs_init": real_only_r2c,
        "r5f2_main_from_r1b_vs_r5e_control": main_r1b_vs_r5e,
        "r5f2_main_from_r2c_vs_r5e_control": main_r2c_vs_r5e,
        "r5f2_main_from_r1b_vs_real_only": main_r1b_vs_real,
        "r5f2_main_from_r2c_vs_real_only": main_r2c_vs_real,
        "guardrails": {
            "no_test_read": True,
            "d1_used_for_eval_only": True,
            "human_rationale_not_in_prompt": True,
            "plus_evidence_dataset_not_trained": True,
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
        "# Exp19-R5F2 DPO Scout Dev Evaluation",
        "",
        "This report evaluates the R5F2 expanded rejection-mined DPO scout on the original question-disjoint dev split.",
        "Raw predictions/logs/checkpoints stay gitignored. No test split is read.",
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
    for row in failure_rows:
        name = str(row["run_name"])
        d1_row = d1_by_name.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(d1_row.get('pred_ge4_rate_d1_hidden'))} | {fmt(d1_row.get('label2_recall_d1'))} | "
            f"{fmt(row.get('failure_type_micro_f1'))} | {fmt(row.get('major_failure_nonempty_rate_on_d1_hidden'))} | "
            f"{fmt(row.get('score_cap_nonnull_rate_on_d1_hidden'))} |"
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            "- Does R5F2 score-risk main from R1b reduce low-to-high vs R1b: "
            f"{fmt(decision['r5f2_main_from_r1b_vs_init']['low_to_high_improvement'])}.",
            "- Does R5F2 score-risk main from R2c reduce low-to-high vs R2c: "
            f"{fmt(decision['r5f2_main_from_r2c_vs_init']['low_to_high_improvement'])}.",
            "- Does R5F2 score-risk main beat hard-synthetic controls: "
            f"R1b={decision['r5f2_main_from_r1b_vs_r5e_control']['beats']}, "
            f"R2c={decision['r5f2_main_from_r2c_vs_r5e_control']['beats']}.",
            "- Does real-only diagnostic behave differently from score-risk main: compare the real-only rows above; "
            "real-only is diagnostic and not a full-DPO candidate by itself.",
            "- Does any run damage high-score protection: inspect high-to-low rates; the success rule allows at most +0.05.",
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
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
            "- Evaluation uses the original dev split, not a balanced train distribution.",
            "- Test split is not read.",
            "- D1 annotations are evaluation references only.",
            "- Human rationale is not included in the prediction prompt.",
            "- R5F2 plus-evidence data is not trained in this scout.",
        ]
    )
    write_text(out_dir / "reports" / "exp19_r5f2_dpo_scout_report.md", "\n".join(lines))


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
    write_csv(args.out_dir / "tables" / "exp19_r5f2_dpo_scout_dev_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5f2_dpo_scout_d1_hidden_eval.csv", d1_rows, D1_FIELDS)
    write_csv(args.out_dir / "tables" / "exp19_r5f2_dpo_scout_failure_type_eval.csv", failure_rows, FAILURE_FIELDS)
    write_json(args.out_dir / "decision" / "exp19_r5f2_dpo_scout_decision.json", decision)
    write_report(args.out_dir, metric_rows, d1_rows, failure_rows, decision, prior_summary, missing_predictions)
    return {
        "new_runs_collected": len([row for row in metric_rows if row["run_name"] in {item["run_name"] for item in RUNS}]),
        "missing_predictions": missing_predictions,
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp19-R5F2 DPO scout dev predictions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--r5-scout-dir", type=Path, default=DEFAULT_R5_SCOUT_DIR)
    parser.add_argument("--r5-next-dir", type=Path, default=DEFAULT_R5_NEXT_DIR)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
