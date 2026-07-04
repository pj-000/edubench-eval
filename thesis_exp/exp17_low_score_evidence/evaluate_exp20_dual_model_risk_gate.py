"""Evaluate Exp20 dual-model risk-aware high-score gates on dev predictions.

Exp20 does not train a model. It combines an existing balanced scoring model
with an existing risk-sensitive model and evaluates whether simple gates can
flag or downgrade dangerous high-score predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import (  # noqa: E402
    mean,
    quadratic_weighted_kappa,
    safe_rate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp20_dual_model_risk_gate_seed42")
DEFAULT_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_D1_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev/"
    "summary_human_rationale_recovered"
)

DEFAULT_SCORE_RUNS = {
    "r4b_shuffled_reason_balanced": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_second_round/dev_predictions/"
        "r4b_shuffled_reason_balanced"
    ),
    "r5g_a3_real_only_s50_b0p05_lr5em6": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_dpo_scout/dev_predictions/"
        "r5g_a3_real_only_s50_b0p05_lr5em6"
    ),
    "r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_dpo_scout/dev_predictions/"
        "r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6"
    ),
}

DEFAULT_RISK_RUNS = {
    "r5f2_real_only_from_r2c": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_dpo_scout/dev_predictions/"
        "r5f2_real_only_from_r2c"
    ),
    "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_dpo_scout/dev_predictions/"
        "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6"
    ),
    "r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6": Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_dpo_scout/dev_predictions/"
        "r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6"
    ),
}

ABSTAIN_RULES = [
    "disagreement_abstain",
    "gap_abstain",
    "score_cap_abstain",
    "hybrid_abstain",
]
DOWNGRADE_RULES = [
    "disagreement_downgrade_to_3",
    "gap_downgrade_to_3",
]

ABSTAIN_SUCCESS = {
    "coverage_min": 0.85,
    "low_to_high_covered_max": 0.30,
    "d1_hidden_pred_ge4_covered_max": 0.50,
}
DOWNGRADE_SUCCESS = {
    "low_to_high_max": 0.35,
    "label2_recall_min": 0.20,
    "high_to_low_max": 0.05,
    "label5_recall_min": 0.70,
    "mae_qwk_delta_max": 0.05,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def parse_run_arg(values: list[str] | None, defaults: dict[str, Path]) -> dict[str, Path]:
    if not values:
        return dict(defaults)
    out: dict[str, Path] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected --run name:path, got: {value}")
        name, path = value.split(":", 1)
        if not name.strip() or not path.strip():
            raise ValueError(f"Expected --run name:path, got: {value}")
        out[name.strip()] = Path(path.strip())
    return out


def load_run(
    name: str,
    run_dir: Path,
    reference: list[dict[str, str]],
    allow_missing: bool,
) -> list[dict[str, Any]] | None:
    try:
        prediction_file = sft_collect.find_prediction_file(run_dir)
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    records = sft_collect.load_prediction_records(prediction_file)
    return sft_collect.align_predictions(reference, records, name, name)


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        out = int(float(value))
        return out if 1 <= out <= 5 else None
    except Exception:
        return None


def has_substantive_failure(row: dict[str, Any]) -> bool:
    return sft_collect.has_substantive_failure(row)


def trigger(rule: str, score_row: dict[str, Any], risk_row: dict[str, Any]) -> bool:
    score_pred = safe_int(score_row.get("pred_label"))
    risk_pred = safe_int(risk_row.get("pred_label"))
    if score_pred is None or score_pred < 4:
        return False
    risk_cap = safe_int(risk_row.get("score_cap"))
    risk_has_failure = has_substantive_failure(risk_row)
    if rule.startswith("disagreement_"):
        return risk_pred is not None and risk_pred <= 2
    if rule.startswith("gap_"):
        return risk_pred is not None and (score_pred - risk_pred) >= 2
    if rule == "score_cap_abstain":
        return risk_cap is not None and risk_cap <= 2
    if rule == "hybrid_abstain":
        return (risk_pred is not None and risk_pred <= 2) or (risk_cap is not None and risk_cap <= 2) or risk_has_failure
    raise ValueError(f"Unknown rule: {rule}")


def load_d1_hidden_ids(d1_dir: Path) -> set[str]:
    annotations, _pairs, _controls = sft_collect.load_d1_annotations(d1_dir)
    return set(annotations)


def metric_values(rows: list[dict[str, Any]], pred_key: str = "pred_label") -> dict[str, Any]:
    valid = [row for row in rows if safe_int(row.get(pred_key)) is not None]
    gold = [int(row["gold_label"]) for row in valid]
    pred = [int(row[pred_key]) for row in valid]
    low = [row for row in rows if int(row["gold_label"]) <= 2]
    high = [row for row in rows if int(row["gold_label"]) >= 4]
    label2 = [row for row in rows if int(row["gold_label"]) == 2]
    label5 = [row for row in rows if int(row["gold_label"]) == 5]
    low_to_high = sum(1 for row in low if safe_int(row.get(pred_key)) is not None and int(row[pred_key]) >= 4)
    high_to_low = sum(1 for row in high if safe_int(row.get(pred_key)) is not None and int(row[pred_key]) <= 2)
    label2_hits = sum(1 for row in label2 if safe_int(row.get(pred_key)) == 2)
    label5_hits = sum(1 for row in label5 if safe_int(row.get(pred_key)) == 5)
    return {
        "n": len(rows),
        "valid_n": len(valid),
        "MAE": mean([abs(g - p) for g, p in zip(gold, pred)]),
        "QWK": quadratic_weighted_kappa(gold, pred),
        "Signed_Bias": mean([p - g for g, p in zip(gold, pred)]),
        "low_to_high_count": low_to_high,
        "low_to_high": safe_rate(low_to_high, len(low)),
        "high_to_low_count": high_to_low,
        "high_to_low": safe_rate(high_to_low, len(high)),
        "label2_recall": safe_rate(label2_hits, len(label2)),
        "label5_recall": safe_rate(label5_hits, len(label5)),
    }


def d1_pred_ge4(rows: list[dict[str, Any]], d1_ids: set[str], pred_key: str = "pred_label") -> float:
    cases = [row for row in rows if str(row.get("sample_id", "")) in d1_ids]
    return safe_rate(sum(1 for row in cases if safe_int(row.get(pred_key)) is not None and int(row[pred_key]) >= 4), len(cases))


def annotate_pairs(
    score_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    rule: str,
    d1_ids: set[str],
) -> list[dict[str, Any]]:
    risk_by_id = {str(row["sample_id"]): row for row in risk_rows}
    out: list[dict[str, Any]] = []
    for score_row in score_rows:
        sid = str(score_row["sample_id"])
        risk_row = risk_by_id.get(sid)
        if risk_row is None:
            continue
        score_pred = safe_int(score_row.get("pred_label"))
        is_flagged = trigger(rule, score_row, risk_row)
        item = dict(score_row)
        item["risk_pred_label"] = risk_row.get("pred_label")
        item["risk_score_cap"] = risk_row.get("score_cap")
        item["risk_major_failure_nonempty"] = has_substantive_failure(risk_row)
        item["flagged"] = is_flagged
        item["is_d1_hidden"] = sid in d1_ids
        item["baseline_low_to_high"] = int(score_row["gold_label"]) <= 2 and score_pred is not None and score_pred >= 4
        item["final_pred"] = 3 if is_flagged and rule.endswith("downgrade_to_3") else score_pred
        out.append(item)
    return out


def abstention_row(
    score_model: str,
    risk_model: str,
    rule: str,
    rows: list[dict[str, Any]],
    d1_ids: set[str],
) -> dict[str, Any]:
    flagged = [row for row in rows if row["flagged"]]
    covered = [row for row in rows if not row["flagged"]]
    low_all = [row for row in rows if int(row["gold_label"]) <= 2]
    low_covered = [row for row in covered if int(row["gold_label"]) <= 2]
    label2_covered = [row for row in covered if int(row["gold_label"]) == 2]
    high_all = [row for row in rows if int(row["gold_label"]) >= 4]
    baseline_l2h = [row for row in rows if row["baseline_low_to_high"]]
    covered_l2h = [
        row
        for row in covered
        if int(row["gold_label"]) <= 2 and safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4
    ]
    d1_covered = [row for row in covered if row["is_d1_hidden"]]
    metric = metric_values(covered)
    return {
        "score_model": score_model,
        "risk_model": risk_model,
        "rule": rule,
        "coverage": safe_rate(len(covered), len(rows)),
        "abstention_rate": safe_rate(len(flagged), len(rows)),
        "low_to_high_covered": safe_rate(len(covered_l2h), len(low_covered)),
        "low_to_high_all_treat_abstain_safe": safe_rate(len(covered_l2h), len(low_all)),
        "label2_recall_covered": safe_rate(
            sum(1 for row in label2_covered if safe_int(row.get("pred_label")) == 2),
            len(label2_covered),
        ),
        "MAE_covered": metric["MAE"],
        "QWK_covered": metric["QWK"],
        "D1_hidden_pred_ge4_covered": safe_rate(
            sum(1 for row in d1_covered if safe_int(row.get("pred_label")) is not None and int(row["pred_label"]) >= 4),
            len(d1_covered),
        ),
        "flag_rate_on_gold_high": safe_rate(sum(1 for row in flagged if int(row["gold_label"]) >= 4), len(high_all)),
        "flag_precision_for_low": safe_rate(sum(1 for row in flagged if int(row["gold_label"]) <= 2), len(flagged)),
        "flag_recall_for_low_to_high": safe_rate(sum(1 for row in flagged if row["baseline_low_to_high"]), len(baseline_l2h)),
    }


def downgrade_row(
    score_model: str,
    risk_model: str,
    rule: str,
    rows: list[dict[str, Any]],
    d1_ids: set[str],
) -> dict[str, Any]:
    metric = metric_values(rows, "final_pred")
    return {
        "score_model": score_model,
        "risk_model": risk_model,
        "rule": rule,
        "MAE": metric["MAE"],
        "QWK": metric["QWK"],
        "Signed_Bias": metric["Signed_Bias"],
        "low_to_high": metric["low_to_high"],
        "label2_recall": metric["label2_recall"],
        "high_to_low": metric["high_to_low"],
        "label5_recall": metric["label5_recall"],
        "D1_hidden_pred_ge4": d1_pred_ge4(rows, d1_ids, "final_pred"),
    }


def flag_analysis_row(score_model: str, risk_model: str, rule: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [row for row in rows if row["flagged"]]
    baseline_l2h = [row for row in rows if row["baseline_low_to_high"]]
    flagged_l2h = [row for row in flagged if row["baseline_low_to_high"]]
    return {
        "score_model": score_model,
        "risk_model": risk_model,
        "rule": rule,
        "n_flagged": len(flagged),
        "flagged_gold_low_rate": safe_rate(sum(1 for row in flagged if int(row["gold_label"]) <= 2), len(flagged)),
        "flagged_gold_high_rate": safe_rate(sum(1 for row in flagged if int(row["gold_label"]) >= 4), len(flagged)),
        "flagged_D1_hidden_count": sum(1 for row in flagged if row["is_d1_hidden"]),
        "flagged_low_to_high_baseline_count": len(flagged_l2h),
        "flag_precision": safe_rate(len(flagged_l2h), len(flagged)),
        "flag_recall": safe_rate(len(flagged_l2h), len(baseline_l2h)),
    }


def abstention_success(row: dict[str, Any]) -> bool:
    return (
        to_float(row.get("coverage")) >= ABSTAIN_SUCCESS["coverage_min"]
        and to_float(row.get("low_to_high_covered")) <= ABSTAIN_SUCCESS["low_to_high_covered_max"]
        and to_float(row.get("D1_hidden_pred_ge4_covered")) <= ABSTAIN_SUCCESS["d1_hidden_pred_ge4_covered_max"]
    )


def downgrade_success(row: dict[str, Any], baseline_by_score: dict[str, dict[str, Any]]) -> bool:
    baseline = baseline_by_score.get(str(row["score_model"]), {})
    mae_delta = to_float(row.get("MAE")) - to_float(baseline.get("MAE"))
    qwk_delta = to_float(baseline.get("QWK")) - to_float(row.get("QWK"))
    return (
        to_float(row.get("low_to_high")) <= DOWNGRADE_SUCCESS["low_to_high_max"]
        and to_float(row.get("label2_recall")) >= DOWNGRADE_SUCCESS["label2_recall_min"]
        and to_float(row.get("high_to_low")) <= DOWNGRADE_SUCCESS["high_to_low_max"]
        and to_float(row.get("label5_recall")) >= DOWNGRADE_SUCCESS["label5_recall_min"]
        and mae_delta <= DOWNGRADE_SUCCESS["mae_qwk_delta_max"]
        and qwk_delta <= DOWNGRADE_SUCCESS["mae_qwk_delta_max"]
    )


def best_abstention(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            not abstention_success(row),
            to_float(row.get("low_to_high_covered")),
            to_float(row.get("D1_hidden_pred_ge4_covered")),
            -to_float(row.get("coverage")),
            to_float(row.get("flag_rate_on_gold_high")),
        ),
    )[0]


def best_downgrade(rows: list[dict[str, Any]], baseline_by_score: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            not downgrade_success(row, baseline_by_score),
            to_float(row.get("low_to_high")),
            to_float(row.get("high_to_low")),
            to_float(row.get("MAE")),
            -to_float(row.get("QWK")),
        ),
    )[0]


def make_decision(
    abstain_rows: list[dict[str, Any]],
    downgrade_rows: list[dict[str, Any]],
    baseline_by_score: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    abstain_passed = [row for row in abstain_rows if abstention_success(row)]
    downgrade_passed = [row for row in downgrade_rows if downgrade_success(row, baseline_by_score)]
    best_abs = best_abstention(abstain_rows)
    best_down = best_downgrade(downgrade_rows, baseline_by_score)
    if abstain_passed and not downgrade_passed:
        recommendation = "human_in_the_loop_selective_review"
        reason = "At least one abstention gate succeeds while automatic downgrade does not."
    elif abstain_passed and downgrade_passed:
        recommendation = "compare_selective_review_and_downgrade"
        reason = "Both abstention and downgrade have successful candidates."
    elif downgrade_passed:
        recommendation = "consider_automatic_downgrade"
        reason = "At least one downgrade gate succeeds while abstention does not."
    else:
        recommendation = "d1_like_data_expansion_not_more_dpo"
        reason = "Both abstention and downgrade fail the current success criteria."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "best_abstention_rule": {
            key: best_abs.get(key)
            for key in [
                "score_model",
                "risk_model",
                "rule",
                "coverage",
                "low_to_high_covered",
                "D1_hidden_pred_ge4_covered",
                "flag_rate_on_gold_high",
            ]
        }
        if best_abs
        else {},
        "best_downgrade_rule": {
            key: best_down.get(key)
            for key in [
                "score_model",
                "risk_model",
                "rule",
                "MAE",
                "QWK",
                "low_to_high",
                "label2_recall",
                "high_to_low",
                "label5_recall",
                "D1_hidden_pred_ge4",
            ]
        }
        if best_down
        else {},
        "abstention_success_count": len(abstain_passed),
        "downgrade_success_count": len(downgrade_passed),
        "abstention_success": ABSTAIN_SUCCESS,
        "downgrade_success": DOWNGRADE_SUCCESS,
        "guardrails": {
            "no_test_read": True,
            "no_training": True,
            "d1_used_for_eval_only": True,
            "human_rationale_not_used_as_decision_input": True,
        },
    }


def write_report(
    out_dir: Path,
    score_runs: dict[str, Path],
    risk_runs: dict[str, Path],
    abstain_rows: list[dict[str, Any]],
    downgrade_rows: list[dict[str, Any]],
    flag_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    best_abs = decision.get("best_abstention_rule") or {}
    best_down = decision.get("best_downgrade_rule") or {}
    top_flags = sorted(flag_rows, key=lambda row: (-to_float(row.get("flag_recall")), -to_float(row.get("flag_precision"))))[:5]
    lines = [
        "# Exp20 Dual-Model Risk-Aware Gate Evaluation",
        "",
        "Exp20 evaluates decision-layer gates using existing dev predictions only. It does not train",
        "and does not read test.",
        "",
        "## Models",
        "",
        f"- score models: {', '.join(score_runs)}",
        f"- risk models: {', '.join(risk_runs)}",
        "",
        "## Best Abstention Rule",
        "",
        f"- score_model: `{best_abs.get('score_model', '')}`",
        f"- risk_model: `{best_abs.get('risk_model', '')}`",
        f"- rule: `{best_abs.get('rule', '')}`",
        f"- coverage: {fmt(best_abs.get('coverage'))}",
        f"- low_to_high_covered: {fmt(best_abs.get('low_to_high_covered'))}",
        f"- D1_hidden_pred_ge4_covered: {fmt(best_abs.get('D1_hidden_pred_ge4_covered'))}",
        f"- flag_rate_on_gold_high: {fmt(best_abs.get('flag_rate_on_gold_high'))}",
        "",
        "## Best Downgrade Rule",
        "",
        f"- score_model: `{best_down.get('score_model', '')}`",
        f"- risk_model: `{best_down.get('risk_model', '')}`",
        f"- rule: `{best_down.get('rule', '')}`",
        f"- MAE: {fmt(best_down.get('MAE'))}",
        f"- QWK: {fmt(best_down.get('QWK'))}",
        f"- low_to_high: {fmt(best_down.get('low_to_high'))}",
        f"- label2_recall: {fmt(best_down.get('label2_recall'))}",
        f"- high_to_low: {fmt(best_down.get('high_to_low'))}",
        f"- label5_recall: {fmt(best_down.get('label5_recall'))}",
        f"- D1_hidden_pred_ge4: {fmt(best_down.get('D1_hidden_pred_ge4'))}",
        "",
        "## Does The Risk Model Flag Dangerous High-Score Predictions?",
        "",
        "| score | risk | rule | flagged | precision | recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top_flags:
        lines.append(
            f"| `{row['score_model']}` | `{row['risk_model']}` | `{row['rule']}` | "
            f"{row['n_flagged']} | {fmt(row['flag_precision'])} | {fmt(row['flag_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            "",
            "## Required Questions",
            "",
            "- Does the risk model flag dangerous high-score predictions? See `exp20_gate_flag_analysis.csv`.",
            "- Is abstention better than automatic downgrade? Compare the decision and the best-rule tables above.",
            "- What coverage is needed to reduce low-to-high below 0.30? See `exp20_gate_abstention_frontier.csv`.",
            "- Does gate over-flag true high scores? Check `flag_rate_on_gold_high` and `flagged_gold_high_rate`.",
            "- Which score/risk pair is best? See `best_abstention_rule` and `best_downgrade_rule` in decision JSON.",
            "- Should RQ3 use selective review rather than more DPO? The decision JSON states the recommendation.",
            "",
            "## Guardrails",
            "",
            "- Test split is not read.",
            "- No model training is performed.",
            "- D1 annotations are used only for evaluation.",
            "- Human rationale is not used as decision input.",
            "- Raw predictions remain local/server-side and are not written by this script.",
        ]
    )
    write_text(out_dir / "reports" / "exp20_dual_model_risk_gate_report.md", "\n".join(lines))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    d1_ids = load_d1_hidden_ids(args.d1_dir)
    score_runs = parse_run_arg(args.score_run, DEFAULT_SCORE_RUNS)
    risk_runs = parse_run_arg(args.risk_run, DEFAULT_RISK_RUNS)

    loaded_scores: dict[str, list[dict[str, Any]]] = {}
    loaded_risks: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for name, path in score_runs.items():
        rows = load_run(name, path, reference, args.allow_missing_predictions)
        if rows is None:
            missing.append(name)
        else:
            loaded_scores[name] = rows
    for name, path in risk_runs.items():
        rows = load_run(name, path, reference, args.allow_missing_predictions)
        if rows is None:
            missing.append(name)
        else:
            loaded_risks[name] = rows

    baseline_by_score = {name: metric_values(rows) for name, rows in loaded_scores.items()}
    abstain_rows: list[dict[str, Any]] = []
    downgrade_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    for score_name, score_rows in loaded_scores.items():
        for risk_name, risk_rows in loaded_risks.items():
            for rule in ABSTAIN_RULES:
                rows = annotate_pairs(score_rows, risk_rows, rule, d1_ids)
                abstain_rows.append(abstention_row(score_name, risk_name, rule, rows, d1_ids))
                flag_rows.append(flag_analysis_row(score_name, risk_name, rule, rows))
            for rule in DOWNGRADE_RULES:
                rows = annotate_pairs(score_rows, risk_rows, rule, d1_ids)
                downgrade_rows.append(downgrade_row(score_name, risk_name, rule, rows, d1_ids))
                flag_rows.append(flag_analysis_row(score_name, risk_name, rule, rows))

    decision = make_decision(abstain_rows, downgrade_rows, baseline_by_score)
    decision["missing_prediction_runs"] = missing
    write_csv(
        args.out_dir / "tables" / "exp20_gate_abstention_frontier.csv",
        abstain_rows,
        [
            "score_model",
            "risk_model",
            "rule",
            "coverage",
            "abstention_rate",
            "low_to_high_covered",
            "low_to_high_all_treat_abstain_safe",
            "label2_recall_covered",
            "MAE_covered",
            "QWK_covered",
            "D1_hidden_pred_ge4_covered",
            "flag_rate_on_gold_high",
            "flag_precision_for_low",
            "flag_recall_for_low_to_high",
        ],
    )
    write_csv(
        args.out_dir / "tables" / "exp20_gate_downgrade_metrics.csv",
        downgrade_rows,
        [
            "score_model",
            "risk_model",
            "rule",
            "MAE",
            "QWK",
            "Signed_Bias",
            "low_to_high",
            "label2_recall",
            "high_to_low",
            "label5_recall",
            "D1_hidden_pred_ge4",
        ],
    )
    write_csv(
        args.out_dir / "tables" / "exp20_gate_flag_analysis.csv",
        flag_rows,
        [
            "score_model",
            "risk_model",
            "rule",
            "n_flagged",
            "flagged_gold_low_rate",
            "flagged_gold_high_rate",
            "flagged_D1_hidden_count",
            "flagged_low_to_high_baseline_count",
            "flag_precision",
            "flag_recall",
        ],
    )
    write_json(args.out_dir / "decision" / "exp20_dual_model_risk_gate_decision.json", decision)
    write_report(args.out_dir, score_runs, risk_runs, abstain_rows, downgrade_rows, flag_rows, decision)
    return {
        "score_models": list(loaded_scores),
        "risk_models": list(loaded_risks),
        "abstention_rows": len(abstain_rows),
        "downgrade_rows": len(downgrade_rows),
        "recommendation": decision["recommendation"],
        "missing_prediction_runs": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Exp20 dual-model risk-aware gates.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--score-run", action="append", default=None, help="Override score model candidate as name:path")
    parser.add_argument("--risk-run", action="append", default=None, help="Override risk model candidate as name:path")
    parser.add_argument("--allow-missing-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
