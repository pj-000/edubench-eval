"""Audit Exp20B best-rule case transitions on dev predictions.

Exp20C is evaluation-only. It audits which cases are changed by the Exp20B
best automatic downgrade rule and which hard cases remain unresolved. It does
not train, does not read test, and does not use human rationale as a decision
input.
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

from thesis_exp.exp17_low_score_evidence.evaluate_exp20_dual_model_risk_gate import (  # noqa: E402
    DEFAULT_D1_DIR,
    DEFAULT_REFERENCE,
    DEFAULT_RISK_RUNS,
    DEFAULT_SCORE_RUNS,
    load_d1_hidden_ids,
    load_run,
    metric_values,
    read_csv_rows,
    safe_int,
    to_float,
)
from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import safe_rate  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp20c_best_rule_case_audit_seed42")
DEFAULT_DOWNGRADE_SCORE_NAME = "r5g_a3_real_only_s50_b0p05_lr5em6"
DEFAULT_DOWNGRADE_RISK_NAME = "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6"
DEFAULT_SELECTIVE_SCORE_NAME = "r4b_shuffled_reason_balanced"
DEFAULT_SELECTIVE_RISK_NAME = "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6"


def read_single_run_arg(value: str | None, defaults: dict[str, Path], default_name: str) -> tuple[str, Path]:
    if not value:
        return default_name, defaults[default_name]
    if ":" not in value:
        raise ValueError(f"Expected run override as name:path, got: {value}")
    name, path = value.split(":", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected run override as name:path, got: {value}")
    return name.strip(), Path(path.strip())


def fmt(value: Any, digits: int = 4) -> str:
    val = to_float(value)
    if math.isnan(val):
        return "nan"
    return f"{val:.{digits}f}"


def case_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_id", ""))


def align_risk(score_rows: list[dict[str, Any]], risk_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    risk_by_id = {case_key(row): row for row in risk_rows}
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for score_row in score_rows:
        risk_row = risk_by_id.get(case_key(score_row))
        if risk_row is not None:
            out.append((score_row, risk_row))
    return out


def gold_group(row: dict[str, Any]) -> str:
    gold = int(row["gold_label"])
    if gold <= 2:
        return "gold_low_1_2"
    if gold == 3:
        return "gold_mid_3"
    return "gold_high_4_5"


def flag_downgrade(
    score_row: dict[str, Any],
    risk_row: dict[str, Any],
    score_threshold: int,
    risk_threshold: int,
) -> bool:
    score_pred = safe_int(score_row.get("pred_label"))
    risk_pred = safe_int(risk_row.get("pred_label"))
    return score_pred is not None and risk_pred is not None and score_pred >= score_threshold and risk_pred <= risk_threshold


def flag_selective_gap(
    score_row: dict[str, Any],
    risk_row: dict[str, Any],
    score_threshold: int,
    gap_threshold: int,
) -> bool:
    score_pred = safe_int(score_row.get("pred_label"))
    risk_pred = safe_int(risk_row.get("pred_label"))
    return (
        score_pred is not None
        and risk_pred is not None
        and score_pred >= score_threshold
        and (score_pred - risk_pred) >= gap_threshold
    )


def build_downgrade_rows(
    score_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    d1_ids: set[str],
    score_threshold: int,
    risk_threshold: int,
    downgrade_to: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_row, risk_row in align_risk(score_rows, risk_rows):
        score_pred = safe_int(score_row.get("pred_label"))
        risk_pred = safe_int(risk_row.get("pred_label"))
        flagged = flag_downgrade(score_row, risk_row, score_threshold, risk_threshold)
        final_pred = downgrade_to if flagged else score_pred
        gold = int(score_row["gold_label"])
        baseline_l2h = gold <= 2 and score_pred is not None and score_pred >= 4
        final_l2h = gold <= 2 and final_pred is not None and final_pred >= 4
        high_to_mid = gold >= 4 and flagged and final_pred == 3
        rescued_l2h = baseline_l2h and not final_l2h
        d1_hidden = case_key(score_row) in d1_ids
        if rescued_l2h and d1_hidden:
            bucket = "rescued_d1_low_to_high"
        elif rescued_l2h:
            bucket = "rescued_low_to_high"
        elif high_to_mid and d1_hidden:
            bucket = "downgraded_d1_gold_high_to_3"
        elif high_to_mid:
            bucket = "downgraded_gold_high_to_3"
        elif flagged and gold == 3:
            bucket = "downgraded_gold_mid_to_3"
        elif flagged and gold <= 2:
            bucket = "downgraded_gold_low_to_3"
        elif d1_hidden and final_pred is not None and final_pred >= 4:
            bucket = "residual_d1_pred_ge4"
        elif final_l2h:
            bucket = "residual_low_to_high"
        else:
            bucket = "unchanged_or_safe"
        row = {
            "sample_id": score_row.get("sample_id", ""),
            "question_key": score_row.get("question_key", ""),
            "question_group_id": score_row.get("question_group_id", ""),
            "metric": score_row.get("metric", ""),
            "language": score_row.get("language", ""),
            "subject": score_row.get("subject", ""),
            "gold_label": gold,
            "gold_group": gold_group(score_row),
            "score_pred": score_pred,
            "risk_pred": risk_pred,
            "risk_score_cap": risk_row.get("score_cap"),
            "final_pred": final_pred,
            "flagged": flagged,
            "is_d1_hidden": d1_hidden,
            "baseline_low_to_high": baseline_l2h,
            "final_low_to_high": final_l2h,
            "rescued_low_to_high": rescued_l2h,
            "gold_high_downgraded_to_3": high_to_mid,
            "audit_bucket": bucket,
        }
        rows.append(row)
    return rows


def build_selective_rows(
    score_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    d1_ids: set[str],
    score_threshold: int,
    gap_threshold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_row, risk_row in align_risk(score_rows, risk_rows):
        score_pred = safe_int(score_row.get("pred_label"))
        risk_pred = safe_int(risk_row.get("pred_label"))
        flagged = flag_selective_gap(score_row, risk_row, score_threshold, gap_threshold)
        gold = int(score_row["gold_label"])
        baseline_l2h = gold <= 2 and score_pred is not None and score_pred >= 4
        rows.append(
            {
                "sample_id": score_row.get("sample_id", ""),
                "question_key": score_row.get("question_key", ""),
                "question_group_id": score_row.get("question_group_id", ""),
                "metric": score_row.get("metric", ""),
                "language": score_row.get("language", ""),
                "subject": score_row.get("subject", ""),
                "gold_label": gold,
                "gold_group": gold_group(score_row),
                "score_pred": score_pred,
                "risk_pred": risk_pred,
                "flagged_for_review": flagged,
                "covered": not flagged,
                "is_d1_hidden": case_key(score_row) in d1_ids,
                "baseline_low_to_high": baseline_l2h,
            }
        )
    return rows


def metric_summary_row(
    name: str,
    rows: list[dict[str, Any]],
    pred_key: str,
    d1_ids: set[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metric_values(rows, pred_key)
    d1_cases = [row for row in rows if case_key(row) in d1_ids or row.get("is_d1_hidden")]
    d1_pred_ge4 = safe_rate(
        sum(1 for row in d1_cases if safe_int(row.get(pred_key)) is not None and int(row[pred_key]) >= 4),
        len(d1_cases),
    )
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    high_to_mid = safe_rate(
        sum(1 for row in high if safe_int(row.get(pred_key)) == 3),
        len(high),
    )
    out = {
        "name": name,
        "n": len(rows),
        "MAE": metrics["MAE"],
        "QWK": metrics["QWK"],
        "Signed_Bias": metrics["Signed_Bias"],
        "low_to_high": metrics["low_to_high"],
        "label2_recall": metrics["label2_recall"],
        "high_to_low": metrics["high_to_low"],
        "gold_high_to_mid_rate": high_to_mid,
        "label5_recall": metrics["label5_recall"],
        "D1_hidden_pred_ge4": d1_pred_ge4,
    }
    if extra:
        out.update(extra)
    return out


def flagged_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [row for row in rows if row["flagged"]]
    baseline_l2h = [row for row in rows if row["baseline_low_to_high"]]
    rescued = [row for row in rows if row["rescued_low_to_high"]]
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    high_downgraded = [row for row in rows if row["gold_high_downgraded_to_3"]]
    d1 = [row for row in rows if row["is_d1_hidden"]]
    d1_residual = [row for row in rows if row["is_d1_hidden"] and safe_int(row.get("final_pred")) is not None and int(row["final_pred"]) >= 4]
    return {
        "n": len(rows),
        "flagged_count": len(flagged),
        "flagged_rate": safe_rate(len(flagged), len(rows)),
        "baseline_low_to_high_count": len(baseline_l2h),
        "rescued_low_to_high_count": len(rescued),
        "rescued_low_to_high_rate": safe_rate(len(rescued), len(baseline_l2h)),
        "gold_high_downgraded_to_3_count": len(high_downgraded),
        "gold_high_downgraded_to_3_rate": safe_rate(len(high_downgraded), len(high)),
        "flagged_gold_high_rate": safe_rate(sum(1 for row in flagged if int(row["gold_label"]) >= 4), len(flagged)),
        "d1_hidden_count": len(d1),
        "d1_hidden_residual_pred_ge4_count": len(d1_residual),
        "d1_hidden_residual_pred_ge4_rate": safe_rate(len(d1_residual), len(d1)),
    }


def group_breakdown(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_field) or "")
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        summary = flagged_summary(group)
        out.append({"group_field": group_field, "group_value": key, **summary})
    return out


def bucket_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["audit_bucket"]), []).append(row)
    total = len(rows)
    flagged_total = sum(1 for row in rows if row["flagged"])
    out: list[dict[str, Any]] = []
    for bucket, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        out.append(
            {
                "audit_bucket": bucket,
                "count": len(group),
                "rate_all": safe_rate(len(group), total),
                "rate_among_flagged": safe_rate(sum(1 for row in group if row["flagged"]), flagged_total),
                "d1_hidden_count": sum(1 for row in group if row["is_d1_hidden"]),
                "gold_low_count": sum(1 for row in group if int(row["gold_label"]) <= 2),
                "gold_high_count": sum(1 for row in group if int(row["gold_label"]) >= 4),
            }
        )
    return out


def residual_cases(rows: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    interesting = [
        row
        for row in rows
        if row["final_low_to_high"]
        or (row["is_d1_hidden"] and safe_int(row.get("final_pred")) is not None and int(row["final_pred"]) >= 4)
        or row["gold_high_downgraded_to_3"]
        or row["rescued_low_to_high"]
    ]
    order = {
        "residual_d1_pred_ge4": 0,
        "residual_low_to_high": 1,
        "downgraded_d1_gold_high_to_3": 2,
        "downgraded_gold_high_to_3": 3,
        "rescued_d1_low_to_high": 4,
        "rescued_low_to_high": 5,
    }
    return sorted(interesting, key=lambda row: (order.get(row["audit_bucket"], 99), str(row["sample_id"])))[:limit]


def make_decision(summary: dict[str, Any], baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    rescued_rate = to_float(summary["rescued_low_to_high_rate"])
    d1_residual = to_float(summary["d1_hidden_residual_pred_ge4_rate"])
    high_to_mid = to_float(summary["gold_high_downgraded_to_3_rate"])
    mae_delta = to_float(final["MAE"]) - to_float(baseline["MAE"])
    qwk_delta = to_float(baseline["QWK"]) - to_float(final["QWK"])
    auto_ready = (
        rescued_rate >= 0.30
        and d1_residual <= 0.50
        and high_to_mid <= 0.10
        and mae_delta <= 0.05
        and qwk_delta <= 0.05
    )
    review_ready = rescued_rate >= 0.30 and high_to_mid > 0.10
    if auto_ready:
        recommendation = "automatic_downgrade_candidate"
        reason = "The rule rescues low-to-high cases while keeping D1 residuals and high-score demotion low."
    elif review_ready:
        recommendation = "prefer_selective_review_over_automatic_downgrade"
        reason = "The rule catches risk but demotes too many true high-score cases for automatic score changes."
    else:
        recommendation = "d1_like_data_expansion_before_formal_rq3"
        reason = "The rule is informative but leaves too much D1 hidden risk or insufficiently reliable interventions."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "automatic_downgrade_ready": auto_ready,
        "selective_review_preferred": review_ready,
        "recommend_automatic_downgrade_formal": auto_ready,
        "recommend_selective_review_formal": review_ready,
        "recommend_data_expansion": recommendation == "d1_like_data_expansion_before_formal_rq3",
        "rescued_low_to_high_rate": rescued_rate,
        "d1_hidden_residual_pred_ge4_rate": d1_residual,
        "gold_high_downgraded_to_3_rate": high_to_mid,
        "MAE_delta_vs_score_baseline": mae_delta,
        "QWK_delta_vs_score_baseline": qwk_delta,
        "guardrails": {
            "no_test_read": True,
            "no_training": True,
            "d1_used_for_eval_only": True,
            "human_rationale_not_used_as_decision_input": True,
        },
    }


def write_report(
    out_dir: Path,
    downgrade_score_name: str,
    downgrade_risk_name: str,
    selective_score_name: str,
    selective_risk_name: str,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    final: dict[str, Any],
    selective_summary: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Exp20C Best-Rule Case Audit",
        "",
        "Exp20C audits case-level effects of the Exp20B best automatic downgrade rule and",
        "the best selective review rule. It uses existing dev predictions only, does not train,",
        "and does not read test.",
        "",
        "## Audited Rules",
        "",
        f"- automatic downgrade score model: `{downgrade_score_name}`",
        f"- automatic downgrade risk model: `{downgrade_risk_name}`",
        "- automatic downgrade rule: `score_pred >= 5 and risk_pred <= 3 -> final_pred = 3`",
        f"- selective review score model: `{selective_score_name}`",
        f"- selective review risk model: `{selective_risk_name}`",
        "- selective review rule: `score_pred >= 4 and score_pred - risk_pred >= 1 -> review`",
        "",
        "## Automatic Downgrade Effects",
        "",
        f"- flagged_count: {summary['flagged_count']}",
        f"- flagged_rate: {fmt(summary['flagged_rate'])}",
        f"- rescued_low_to_high_count: {summary['rescued_low_to_high_count']}",
        f"- rescued_low_to_high_rate: {fmt(summary['rescued_low_to_high_rate'])}",
        f"- gold_high_downgraded_to_3_count: {summary['gold_high_downgraded_to_3_count']}",
        f"- gold_high_downgraded_to_3_rate: {fmt(summary['gold_high_downgraded_to_3_rate'])}",
        f"- d1_hidden_residual_pred_ge4_rate: {fmt(summary['d1_hidden_residual_pred_ge4_rate'])}",
        "",
        "## Score Metrics",
        "",
        "| row | MAE | QWK | low-to-high | label2 recall | high-to-low | high-to-mid | label5 recall | D1 pred>=4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| score baseline | {fmt(baseline['MAE'])} | {fmt(baseline['QWK'])} | "
            f"{fmt(baseline['low_to_high'])} | {fmt(baseline['label2_recall'])} | "
            f"{fmt(baseline['high_to_low'])} | {fmt(baseline['gold_high_to_mid_rate'])} | "
            f"{fmt(baseline['label5_recall'])} | {fmt(baseline['D1_hidden_pred_ge4'])} |"
        ),
        (
            f"| after downgrade | {fmt(final['MAE'])} | {fmt(final['QWK'])} | "
            f"{fmt(final['low_to_high'])} | {fmt(final['label2_recall'])} | "
            f"{fmt(final['high_to_low'])} | {fmt(final['gold_high_to_mid_rate'])} | "
            f"{fmt(final['label5_recall'])} | {fmt(final['D1_hidden_pred_ge4'])} |"
        ),
        "",
        "## Selective Review Rule",
        "",
        f"- flagged_for_review_count: {selective_summary['flagged_count']}",
        f"- flagged_for_review_rate: {fmt(selective_summary['flagged_rate'])}",
        f"- low-to-high recall among baseline errors: {fmt(selective_summary['rescued_low_to_high_rate'])}",
        f"- gold-high flag rate: {fmt(selective_summary['gold_high_downgraded_to_3_rate'])}",
        f"- D1 hidden residual if covered-only: {fmt(selective_summary['d1_hidden_residual_pred_ge4_rate'])}",
        "",
        "## Decision",
        "",
        f"- recommendation: `{decision['recommendation']}`",
        f"- reason: {decision['reason']}",
        f"- recommend_automatic_downgrade_formal: {decision['recommend_automatic_downgrade_formal']}",
        f"- recommend_selective_review_formal: {decision['recommend_selective_review_formal']}",
        f"- recommend_data_expansion: {decision['recommend_data_expansion']}",
        "",
        "## Guardrails",
        "",
        "- Test split is not read.",
        "- No model training is performed.",
        "- D1 annotations are used only for evaluation.",
        "- Human rationale is not used as decision input.",
        "- Raw predictions are not written by this script.",
    ]
    write_text(out_dir / "reports" / "exp20c_best_rule_case_audit_report.md", "\n".join(lines))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    d1_ids = load_d1_hidden_ids(args.d1_dir)
    downgrade_score_name, downgrade_score_path = read_single_run_arg(
        args.downgrade_score_run,
        DEFAULT_SCORE_RUNS,
        DEFAULT_DOWNGRADE_SCORE_NAME,
    )
    downgrade_risk_name, downgrade_risk_path = read_single_run_arg(
        args.downgrade_risk_run,
        DEFAULT_RISK_RUNS,
        DEFAULT_DOWNGRADE_RISK_NAME,
    )
    selective_score_name, selective_score_path = read_single_run_arg(
        args.selective_score_run,
        DEFAULT_SCORE_RUNS,
        DEFAULT_SELECTIVE_SCORE_NAME,
    )
    selective_risk_name, selective_risk_path = read_single_run_arg(
        args.selective_risk_run,
        DEFAULT_RISK_RUNS,
        DEFAULT_SELECTIVE_RISK_NAME,
    )
    downgrade_score_rows = load_run(
        downgrade_score_name,
        downgrade_score_path,
        reference,
        args.allow_missing_predictions,
    )
    downgrade_risk_rows = load_run(
        downgrade_risk_name,
        downgrade_risk_path,
        reference,
        args.allow_missing_predictions,
    )
    selective_score_rows = load_run(
        selective_score_name,
        selective_score_path,
        reference,
        args.allow_missing_predictions,
    )
    selective_risk_rows = load_run(
        selective_risk_name,
        selective_risk_path,
        reference,
        args.allow_missing_predictions,
    )
    missing = []
    for name, rows in [
        (downgrade_score_name, downgrade_score_rows),
        (downgrade_risk_name, downgrade_risk_rows),
        (selective_score_name, selective_score_rows),
        (selective_risk_name, selective_risk_rows),
    ]:
        if rows is None:
            missing.append(name)
    if missing:
        decision = {
            "recommendation": "missing_predictions",
            "missing_prediction_runs": missing,
            "guardrails": {"no_test_read": True, "no_training": True},
        }
        write_json(args.out_dir / "decision" / "exp20c_best_rule_case_audit_decision.json", decision)
        return {"missing_prediction_runs": missing}

    assert downgrade_score_rows is not None
    assert downgrade_risk_rows is not None
    assert selective_score_rows is not None
    assert selective_risk_rows is not None

    downgrade_rows = build_downgrade_rows(
        downgrade_score_rows,
        downgrade_risk_rows,
        d1_ids,
        args.downgrade_score_threshold,
        args.downgrade_risk_threshold,
        args.downgrade_to,
    )
    selective_rows = build_selective_rows(
        selective_score_rows,
        selective_risk_rows,
        d1_ids,
        args.selective_score_threshold,
        args.selective_gap_threshold,
    )
    baseline_rows = [dict(row, final_pred=row.get("score_pred")) for row in downgrade_rows]
    baseline = metric_summary_row("score_baseline", baseline_rows, "score_pred", d1_ids)
    final = metric_summary_row("after_downgrade", downgrade_rows, "final_pred", d1_ids)
    summary = flagged_summary(downgrade_rows)

    selective_flagged = [row for row in selective_rows if row["flagged_for_review"]]
    selective_covered = [dict(row, final_pred=row.get("score_pred"), flagged=False) for row in selective_rows if not row["flagged_for_review"]]
    baseline_errors = [row for row in selective_rows if row["baseline_low_to_high"]]
    high_rows = [row for row in selective_rows if int(row["gold_label"]) >= 4]
    d1_rows = [row for row in selective_rows if row["is_d1_hidden"]]
    d1_covered_pred_ge4 = [
        row
        for row in selective_covered
        if row["is_d1_hidden"] and safe_int(row.get("score_pred")) is not None and int(row["score_pred"]) >= 4
    ]
    selective_summary = {
        "flagged_count": len(selective_flagged),
        "flagged_rate": safe_rate(len(selective_flagged), len(selective_rows)),
        "rescued_low_to_high_rate": safe_rate(
            sum(1 for row in selective_flagged if row["baseline_low_to_high"]),
            len(baseline_errors),
        ),
        "gold_high_downgraded_to_3_rate": safe_rate(
            sum(1 for row in selective_flagged if int(row["gold_label"]) >= 4),
            len(high_rows),
        ),
        "d1_hidden_residual_pred_ge4_rate": safe_rate(len(d1_covered_pred_ge4), len(d1_rows)),
    }

    decision = make_decision(summary, baseline, final)
    decision["missing_prediction_runs"] = []
    decision["audited_rules"] = {
        "automatic_downgrade": {
            "score_model": downgrade_score_name,
            "risk_model": downgrade_risk_name,
            "score_threshold": args.downgrade_score_threshold,
            "risk_threshold": args.downgrade_risk_threshold,
            "downgrade_to": args.downgrade_to,
        },
        "selective_review": {
            "score_model": selective_score_name,
            "risk_model": selective_risk_name,
            "score_threshold": args.selective_score_threshold,
            "gap_threshold": args.selective_gap_threshold,
        },
    }

    write_csv(
        args.out_dir / "tables" / "exp20c_metric_summary.csv",
        [
            baseline,
            final,
            {
                "name": "selective_review_covered_only",
                **metric_summary_row("selective_review_covered_only", selective_covered, "score_pred", d1_ids),
                "coverage": safe_rate(len(selective_covered), len(selective_rows)),
            },
        ],
    )
    write_csv(
        args.out_dir / "tables" / "exp20c_downgrade_case_audit.csv",
        [row for row in downgrade_rows if row["flagged"]],
        [
            "sample_id",
            "question_key",
            "question_group_id",
            "metric",
            "language",
            "subject",
            "gold_label",
            "gold_group",
            "score_pred",
            "risk_pred",
            "risk_score_cap",
            "final_pred",
            "flagged",
            "is_d1_hidden",
            "baseline_low_to_high",
            "final_low_to_high",
            "rescued_low_to_high",
            "gold_high_downgraded_to_3",
            "audit_bucket",
        ],
    )
    write_csv(args.out_dir / "tables" / "exp20c_bucket_summary.csv", bucket_summary(downgrade_rows))
    write_csv(args.out_dir / "tables" / "exp20c_residual_and_changed_cases.csv", residual_cases(downgrade_rows))
    breakdown: list[dict[str, Any]] = []
    for field in ["metric", "language", "subject", "question_group_id"]:
        breakdown.extend(group_breakdown(downgrade_rows, field))
    write_csv(args.out_dir / "tables" / "exp20c_group_breakdown.csv", breakdown)
    write_json(args.out_dir / "decision" / "exp20c_best_rule_case_audit_decision.json", decision)
    write_report(
        args.out_dir,
        downgrade_score_name,
        downgrade_risk_name,
        selective_score_name,
        selective_risk_name,
        summary,
        baseline,
        final,
        selective_summary,
        decision,
    )
    return {
        "downgrade_score_model": downgrade_score_name,
        "downgrade_risk_model": downgrade_risk_name,
        "flagged_count": summary["flagged_count"],
        "rescued_low_to_high_count": summary["rescued_low_to_high_count"],
        "gold_high_downgraded_to_3_count": summary["gold_high_downgraded_to_3_count"],
        "d1_hidden_residual_pred_ge4_rate": summary["d1_hidden_residual_pred_ge4_rate"],
        "recommendation": decision["recommendation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Exp20C best-rule case transitions.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--downgrade-score-run", default=None, help="Override as name:path")
    parser.add_argument("--downgrade-risk-run", default=None, help="Override as name:path")
    parser.add_argument("--selective-score-run", default=None, help="Override as name:path")
    parser.add_argument("--selective-risk-run", default=None, help="Override as name:path")
    parser.add_argument("--downgrade-score-threshold", type=int, default=5)
    parser.add_argument("--downgrade-risk-threshold", type=int, default=3)
    parser.add_argument("--downgrade-to", type=int, default=3)
    parser.add_argument("--selective-score-threshold", type=int, default=4)
    parser.add_argument("--selective-gap-threshold", type=int, default=1)
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
