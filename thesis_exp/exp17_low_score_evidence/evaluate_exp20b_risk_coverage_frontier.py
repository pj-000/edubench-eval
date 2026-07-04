"""Evaluate Exp20B risk-coverage frontiers for dual-model high-score gates.

Exp20B is an evaluation-only follow-up to Exp20. It does not train a model and
does not read test. It searches thresholded high-score gate rules over existing
dev predictions to decide whether RQ3 should proceed with selective review,
automatic downgrade, or D1-like data expansion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.evaluate_exp20_dual_model_risk_gate import (  # noqa: E402
    ABSTAIN_SUCCESS,
    DEFAULT_D1_DIR,
    DEFAULT_REFERENCE,
    DEFAULT_RISK_RUNS,
    DEFAULT_SCORE_RUNS,
    DOWNGRADE_SUCCESS,
    d1_pred_ge4,
    fmt,
    load_d1_hidden_ids,
    load_run,
    metric_values,
    parse_run_arg,
    read_csv_rows,
    safe_int,
    to_float,
)
from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import safe_rate  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp20b_risk_coverage_frontier_seed42")
SCORE_THRESHOLDS = [4, 5]
RISK_THRESHOLDS = [1, 2, 3]
GAP_THRESHOLDS = [1, 2, 3]
SCORE_CAP_THRESHOLDS = [2, 3]


@dataclass(frozen=True)
class RuleSpec:
    rule: str
    score_threshold: int
    risk_threshold: int | None = None
    gap_threshold: int | None = None
    score_cap_threshold: int | None = None

    def thresholds(self) -> dict[str, int | None]:
        return {
            "score_threshold": self.score_threshold,
            "risk_threshold": self.risk_threshold,
            "gap_threshold": self.gap_threshold,
            "score_cap_threshold": self.score_cap_threshold,
        }

    def thresholds_json(self) -> str:
        return json.dumps(self.thresholds(), ensure_ascii=False, sort_keys=True)


def generate_rule_grid() -> list[RuleSpec]:
    specs: list[RuleSpec] = []
    for score_t in SCORE_THRESHOLDS:
        for risk_t in RISK_THRESHOLDS:
            specs.append(RuleSpec("risk_pred_le", score_threshold=score_t, risk_threshold=risk_t))
        for gap_t in GAP_THRESHOLDS:
            specs.append(RuleSpec("gap_ge", score_threshold=score_t, gap_threshold=gap_t))
        for cap_t in SCORE_CAP_THRESHOLDS:
            specs.append(RuleSpec("score_cap_le", score_threshold=score_t, score_cap_threshold=cap_t))
        for risk_t in RISK_THRESHOLDS:
            for cap_t in SCORE_CAP_THRESHOLDS:
                specs.append(
                    RuleSpec(
                        "risk_or_score_cap",
                        score_threshold=score_t,
                        risk_threshold=risk_t,
                        score_cap_threshold=cap_t,
                    )
                )
        for risk_t in RISK_THRESHOLDS:
            for gap_t in GAP_THRESHOLDS:
                specs.append(
                    RuleSpec(
                        "risk_and_gap",
                        score_threshold=score_t,
                        risk_threshold=risk_t,
                        gap_threshold=gap_t,
                    )
                )
        for gap_t in GAP_THRESHOLDS:
            for cap_t in SCORE_CAP_THRESHOLDS:
                specs.append(
                    RuleSpec(
                        "gap_or_score_cap",
                        score_threshold=score_t,
                        gap_threshold=gap_t,
                        score_cap_threshold=cap_t,
                    )
                )
    return specs


def trigger(spec: RuleSpec, score_row: dict[str, Any], risk_row: dict[str, Any]) -> bool:
    score_pred = safe_int(score_row.get("pred_label"))
    risk_pred = safe_int(risk_row.get("pred_label"))
    score_cap = safe_int(risk_row.get("score_cap"))
    if score_pred is None or score_pred < spec.score_threshold:
        return False

    risk_ok = risk_pred is not None and spec.risk_threshold is not None and risk_pred <= spec.risk_threshold
    gap_ok = (
        risk_pred is not None
        and spec.gap_threshold is not None
        and (score_pred - risk_pred) >= spec.gap_threshold
    )
    cap_ok = (
        score_cap is not None
        and spec.score_cap_threshold is not None
        and score_cap <= spec.score_cap_threshold
    )
    if spec.rule == "risk_pred_le":
        return risk_ok
    if spec.rule == "gap_ge":
        return gap_ok
    if spec.rule == "score_cap_le":
        return cap_ok
    if spec.rule == "risk_or_score_cap":
        return risk_ok or cap_ok
    if spec.rule == "risk_and_gap":
        return risk_ok and gap_ok
    if spec.rule == "gap_or_score_cap":
        return gap_ok or cap_ok
    raise ValueError(f"Unknown rule: {spec.rule}")


def annotate_pairs(
    score_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    spec: RuleSpec,
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
        item = dict(score_row)
        flagged = trigger(spec, score_row, risk_row)
        item["risk_pred_label"] = risk_row.get("pred_label")
        item["risk_score_cap"] = risk_row.get("score_cap")
        item["flagged"] = flagged
        item["is_d1_hidden"] = sid in d1_ids
        item["baseline_low_to_high"] = int(score_row["gold_label"]) <= 2 and score_pred is not None and score_pred >= 4
        item["final_pred"] = 3 if flagged else score_pred
        out.append(item)
    return out


def abstention_row(
    score_model: str,
    risk_model: str,
    spec: RuleSpec,
    rows: list[dict[str, Any]],
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
    d1_all = [row for row in rows if row["is_d1_hidden"]]
    d1_covered = [row for row in covered if row["is_d1_hidden"]]
    metric = metric_values(covered)
    return {
        "score_model": score_model,
        "risk_model": risk_model,
        "rule": spec.rule,
        "score_threshold": spec.score_threshold,
        "risk_threshold": spec.risk_threshold,
        "gap_threshold": spec.gap_threshold,
        "score_cap_threshold": spec.score_cap_threshold,
        "thresholds": spec.thresholds_json(),
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
        "flag_recall_for_D1_hidden": safe_rate(sum(1 for row in flagged if row["is_d1_hidden"]), len(d1_all)),
    }


def downgrade_row(
    score_model: str,
    risk_model: str,
    spec: RuleSpec,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric = metric_values(rows, "final_pred")
    return {
        "score_model": score_model,
        "risk_model": risk_model,
        "rule": spec.rule,
        "score_threshold": spec.score_threshold,
        "risk_threshold": spec.risk_threshold,
        "gap_threshold": spec.gap_threshold,
        "score_cap_threshold": spec.score_cap_threshold,
        "thresholds": spec.thresholds_json(),
        "MAE": metric["MAE"],
        "QWK": metric["QWK"],
        "Signed_Bias": metric["Signed_Bias"],
        "low_to_high": metric["low_to_high"],
        "label2_recall": metric["label2_recall"],
        "high_to_low": metric["high_to_low"],
        "label5_recall": metric["label5_recall"],
        "D1_hidden_pred_ge4": d1_pred_ge4(rows, {str(row["sample_id"]) for row in rows if row["is_d1_hidden"]}, "final_pred"),
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
            to_float(row.get("D1_hidden_pred_ge4")),
            to_float(row.get("MAE")),
            -to_float(row.get("QWK")),
            to_float(row.get("high_to_low")),
        ),
    )[0]


def objective_value(row: dict[str, Any], key: str, maximize: bool = False) -> float:
    value = to_float(row.get(key))
    if math.isnan(value):
        value = -math.inf if maximize else math.inf
    return -value if maximize else value


def non_dominated(rows: list[dict[str, Any]], objectives: list[tuple[str, bool]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    values = [[objective_value(row, key, maximize=maximize) for key, maximize in objectives] for row in rows]
    for idx, row in enumerate(rows):
        dominated = False
        for other_idx, other in enumerate(rows):
            if idx == other_idx:
                continue
            other_values = values[other_idx]
            row_values = values[idx]
            if all(o <= r for o, r in zip(other_values, row_values)) and any(
                o < r for o, r in zip(other_values, row_values)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def pareto_rows(abstain_rows: list[dict[str, Any]], downgrade_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abstain_frontier = non_dominated(
        abstain_rows,
        [
            ("coverage", True),
            ("low_to_high_covered", False),
            ("D1_hidden_pred_ge4_covered", False),
            ("flag_rate_on_gold_high", False),
        ],
    )
    downgrade_frontier = non_dominated(
        downgrade_rows,
        [
            ("MAE", False),
            ("QWK", True),
            ("low_to_high", False),
            ("high_to_low", False),
            ("label5_recall", True),
            ("D1_hidden_pred_ge4", False),
        ],
    )
    out: list[dict[str, Any]] = []
    for row in abstain_frontier:
        out.append(
            {
                "mode": "abstention",
                "score_model": row["score_model"],
                "risk_model": row["risk_model"],
                "rule": row["rule"],
                "thresholds": row["thresholds"],
                "coverage": row["coverage"],
                "low_to_high_covered": row["low_to_high_covered"],
                "D1_hidden_pred_ge4_covered": row["D1_hidden_pred_ge4_covered"],
                "flag_rate_on_gold_high": row["flag_rate_on_gold_high"],
                "MAE": "",
                "QWK": "",
                "low_to_high": "",
                "high_to_low": "",
                "label5_recall": "",
                "D1_hidden_pred_ge4": "",
            }
        )
    for row in downgrade_frontier:
        out.append(
            {
                "mode": "downgrade",
                "score_model": row["score_model"],
                "risk_model": row["risk_model"],
                "rule": row["rule"],
                "thresholds": row["thresholds"],
                "coverage": "",
                "low_to_high_covered": "",
                "D1_hidden_pred_ge4_covered": "",
                "flag_rate_on_gold_high": "",
                "MAE": row["MAE"],
                "QWK": row["QWK"],
                "low_to_high": row["low_to_high"],
                "high_to_low": row["high_to_low"],
                "label5_recall": row["label5_recall"],
                "D1_hidden_pred_ge4": row["D1_hidden_pred_ge4"],
            }
        )
    return out


def make_decision(
    abstain_rows: list[dict[str, Any]],
    downgrade_rows: list[dict[str, Any]],
    baseline_by_score: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    abstain_passed = [row for row in abstain_rows if abstention_success(row)]
    downgrade_passed = [row for row in downgrade_rows if downgrade_success(row, baseline_by_score)]
    best_abs = best_abstention(abstain_rows)
    best_down = best_downgrade(downgrade_rows, baseline_by_score)
    recommend_selective = bool(abstain_passed)
    recommend_downgrade = bool(downgrade_passed) and not recommend_selective
    recommend_expansion = not abstain_passed and not downgrade_passed
    if recommend_selective and not downgrade_passed:
        reason = "Selective review has at least one successful frontier rule while downgrade does not."
    elif recommend_selective and downgrade_passed:
        reason = "Both selective review and downgrade have successful frontier rules; compare operating costs."
    elif downgrade_passed:
        reason = "Automatic downgrade has a successful frontier rule while selective review does not."
    else:
        reason = "Neither selective review nor downgrade meets the current frontier success criteria."
    return {
        "best_abstention_rule": best_abs or {},
        "best_downgrade_rule": best_down or {},
        "abstention_success": bool(abstain_passed),
        "downgrade_success": bool(downgrade_passed),
        "abstention_success_count": len(abstain_passed),
        "downgrade_success_count": len(downgrade_passed),
        "recommend_selective_review": recommend_selective,
        "recommend_automatic_downgrade": recommend_downgrade,
        "recommend_data_expansion": recommend_expansion,
        "reason": reason,
        "success_criteria": {
            "abstention": ABSTAIN_SUCCESS,
            "downgrade": DOWNGRADE_SUCCESS,
        },
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
    rule_grid_size: int,
    abstain_rows: list[dict[str, Any]],
    downgrade_rows: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    best_abs = decision.get("best_abstention_rule") or {}
    best_down = decision.get("best_downgrade_rule") or {}
    passed_abs = [row for row in abstain_rows if abstention_success(row)]
    passed_down = [
        row
        for row in downgrade_rows
        if to_float(row.get("low_to_high")) <= DOWNGRADE_SUCCESS["low_to_high_max"]
        and to_float(row.get("high_to_low")) <= DOWNGRADE_SUCCESS["high_to_low_max"]
        and to_float(row.get("label5_recall")) >= DOWNGRADE_SUCCESS["label5_recall_min"]
    ]
    d1_good_abs = [row for row in abstain_rows if to_float(row.get("D1_hidden_pred_ge4_covered")) <= 0.50]
    selective_answer = (
        "Yes, at least one selective review rule meets the success criteria."
        if decision["abstention_success"]
        else "No. Rules that reduce low-to-high enough either miss the D1 hidden target or sacrifice too much coverage."
    )
    downgrade_answer = (
        "Downgrade is stronger under the current grid criteria, but it is an automatic score intervention."
        if decision["downgrade_success"] and not decision["abstention_success"]
        else "Downgrade does not dominate selective review under the current grid criteria."
    )
    high_flag_answer = (
        "For selective review, the best low-risk rule over-flags true high scores."
        if to_float(best_abs.get("flag_rate_on_gold_high")) > 0.20
        else "For selective review, high-score over-flagging is moderate for the best low-risk rule."
    )
    lines = [
        "# Exp20B Risk-Coverage Frontier",
        "",
        "Exp20B searches high-score gate thresholds over existing dev predictions only. It does not train",
        "and does not read test.",
        "",
        "## Grid",
        "",
        f"- score models: {', '.join(score_runs)}",
        f"- risk models: {', '.join(risk_runs)}",
        f"- rule specs per score/risk pair: {rule_grid_size}",
        f"- abstention rows: {len(abstain_rows)}",
        f"- downgrade rows: {len(downgrade_rows)}",
        f"- pareto rows: {len(pareto)}",
        "",
        "## Best Selective Review Rule",
        "",
        f"- score_model: `{best_abs.get('score_model', '')}`",
        f"- risk_model: `{best_abs.get('risk_model', '')}`",
        f"- rule: `{best_abs.get('rule', '')}`",
        f"- thresholds: `{best_abs.get('thresholds', '')}`",
        f"- coverage: {fmt(best_abs.get('coverage'))}",
        f"- low_to_high_covered: {fmt(best_abs.get('low_to_high_covered'))}",
        f"- D1_hidden_pred_ge4_covered: {fmt(best_abs.get('D1_hidden_pred_ge4_covered'))}",
        f"- flag_rate_on_gold_high: {fmt(best_abs.get('flag_rate_on_gold_high'))}",
        "",
        "## Best Automatic Downgrade Rule",
        "",
        f"- score_model: `{best_down.get('score_model', '')}`",
        f"- risk_model: `{best_down.get('risk_model', '')}`",
        f"- rule: `{best_down.get('rule', '')}`",
        f"- thresholds: `{best_down.get('thresholds', '')}`",
        f"- MAE: {fmt(best_down.get('MAE'))}",
        f"- QWK: {fmt(best_down.get('QWK'))}",
        f"- low_to_high: {fmt(best_down.get('low_to_high'))}",
        f"- high_to_low: {fmt(best_down.get('high_to_low'))}",
        f"- label5_recall: {fmt(best_down.get('label5_recall'))}",
        f"- D1_hidden_pred_ge4: {fmt(best_down.get('D1_hidden_pred_ge4'))}",
        "",
        "## Required Questions",
        "",
        f"- Selective rule with coverage>=0.85 and low_to_high_covered<=0.30: {bool(passed_abs)}.",
        f"  {selective_answer}",
        f"- Rule reducing D1_hidden_pred_ge4_covered<=0.50: {bool(d1_good_abs)}.",
        f"- Downgrade vs abstention: {downgrade_answer}",
        "- Best coverage-risk tradeoff: see the best selective review rule above and the Pareto table.",
        f"- High-score over-flagging: {high_flag_answer}",
        f"- Downgrade rules passing the low-risk/high-protection screen before MAE/QWK deltas: {len(passed_down)}.",
        f"- RQ3 selective review recommendation: {decision['recommend_selective_review']}.",
        f"- RQ3 automatic downgrade recommendation: {decision['recommend_automatic_downgrade']}.",
        f"- Data expansion recommendation: {decision['recommend_data_expansion']}.",
        "",
        "## Decision",
        "",
        f"- reason: {decision['reason']}",
        "",
        "## Guardrails",
        "",
        "- Test split is not read.",
        "- No model training is performed.",
        "- D1 annotations are used only for evaluation.",
        "- Human rationale is not used as decision input.",
        "- Raw predictions remain local/server-side and are not written by this script.",
    ]
    write_text(out_dir / "reports" / "exp20b_risk_coverage_frontier_report.md", "\n".join(lines))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_csv_rows(args.reference_csv)
    d1_ids = load_d1_hidden_ids(args.d1_dir)
    score_runs = parse_run_arg(args.score_run, DEFAULT_SCORE_RUNS)
    risk_runs = parse_run_arg(args.risk_run, DEFAULT_RISK_RUNS)
    rule_grid = generate_rule_grid()

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
    for score_name, score_rows in loaded_scores.items():
        for risk_name, risk_rows in loaded_risks.items():
            for spec in rule_grid:
                rows = annotate_pairs(score_rows, risk_rows, spec, d1_ids)
                abstain_rows.append(abstention_row(score_name, risk_name, spec, rows))
                downgrade_rows.append(downgrade_row(score_name, risk_name, spec, rows))

    pareto = pareto_rows(abstain_rows, downgrade_rows)
    decision = make_decision(abstain_rows, downgrade_rows, baseline_by_score)
    decision["missing_prediction_runs"] = missing
    decision["grid"] = {
        "score_model_count": len(loaded_scores),
        "risk_model_count": len(loaded_risks),
        "rule_specs_per_pair": len(rule_grid),
        "abstention_rows": len(abstain_rows),
        "downgrade_rows": len(downgrade_rows),
        "pareto_rows": len(pareto),
    }

    write_csv(
        args.out_dir / "tables" / "exp20b_abstention_frontier.csv",
        abstain_rows,
        [
            "score_model",
            "risk_model",
            "rule",
            "score_threshold",
            "risk_threshold",
            "gap_threshold",
            "score_cap_threshold",
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
            "flag_recall_for_D1_hidden",
        ],
    )
    write_csv(
        args.out_dir / "tables" / "exp20b_downgrade_frontier.csv",
        downgrade_rows,
        [
            "score_model",
            "risk_model",
            "rule",
            "thresholds",
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
        args.out_dir / "tables" / "exp20b_pareto_frontier.csv",
        pareto,
        [
            "mode",
            "score_model",
            "risk_model",
            "rule",
            "thresholds",
            "coverage",
            "low_to_high_covered",
            "D1_hidden_pred_ge4_covered",
            "flag_rate_on_gold_high",
            "MAE",
            "QWK",
            "low_to_high",
            "high_to_low",
            "label5_recall",
            "D1_hidden_pred_ge4",
        ],
    )
    write_json(args.out_dir / "decision" / "exp20b_risk_coverage_decision.json", decision)
    write_report(args.out_dir, score_runs, risk_runs, len(rule_grid), abstain_rows, downgrade_rows, pareto, decision)
    return {
        "score_models": list(loaded_scores),
        "risk_models": list(loaded_risks),
        "rule_grid_size": len(rule_grid),
        "abstention_rows": len(abstain_rows),
        "downgrade_rows": len(downgrade_rows),
        "pareto_rows": len(pareto),
        "missing_prediction_runs": missing,
        "abstention_success": decision["abstention_success"],
        "downgrade_success": decision["downgrade_success"],
        "recommend_selective_review": decision["recommend_selective_review"],
        "recommend_data_expansion": decision["recommend_data_expansion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Exp20B risk-coverage frontiers.")
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
