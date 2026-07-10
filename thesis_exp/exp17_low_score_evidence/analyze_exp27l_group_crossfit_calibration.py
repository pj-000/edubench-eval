"""Analyze locked Exp27L OOF calibration and risk-coverage results."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27l_common import (  # noqa: E402
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    SEED,
    REPRESENTATIVE_VIEW,
    RISK_ENRICHED_VIEW,
    brier,
    canonical_rows,
    cluster_bootstrap_ci,
    ece,
    read_csv,
    top_fraction_indices,
    weighted_average_precision,
    weighted_mae,
    weighted_mean,
    weighted_qwk,
    write_csv,
    write_json,
    write_text,
)


def as_float(value: Any) -> float:
    return float(value)


def as_int(value: Any) -> int:
    return int(float(value))


def scope_rows(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["view"] == scope]


def score_metric_row(rows: list[dict[str, Any]], name: str, prediction: Callable[[dict[str, Any]], int]) -> dict[str, Any]:
    gold = [as_int(row["silver_final_score"]) for row in rows]
    pred = [prediction(row) for row in rows]
    weights = [as_float(row["analysis_weight"]) for row in rows]
    low_weight = sum(weight for value, weight in zip(gold, weights) if value <= 2)
    high_weight = sum(weight for value, weight in zip(gold, weights) if value >= 4)
    return {
        "scope": rows[0]["view"] if rows else "empty",
        "model": name,
        "n": len(rows),
        "weighted_MAE": weighted_mae(gold, pred, weights),
        "weighted_QWK": weighted_qwk(gold, pred, weights),
        "weighted_exact_accuracy": weighted_mean([int(a == b) for a, b in zip(gold, pred)], weights),
        "weighted_severe_error_rate": weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, pred)], weights),
        "weighted_low_to_high": (
            sum(weight for a, b, weight in zip(gold, pred, weights) if a <= 2 and b >= 4) / low_weight if low_weight else None
        ),
        "weighted_high_to_low": (
            sum(weight for a, b, weight in zip(gold, pred, weights) if a >= 4 and b <= 2) / high_weight if high_weight else None
        ),
        "weighted_signed_bias": weighted_mean([b - a for a, b in zip(gold, pred)], weights),
    }


def tier_proxy(row: dict[str, Any]) -> float:
    # Exp27I's old tier is kept strictly as a heuristic baseline, not a learned calibration feature.
    return {"high_weight": 0.10, "low_weight": 0.55, "review_only": 0.90}.get(str(row["exp27i_v1_tier"]), 0.50)


def risk_metric_row(rows: list[dict[str, Any]], name: str, signal: Callable[[dict[str, Any]], float], formal: bool) -> dict[str, Any]:
    labels = [as_int(row["severe_human_silver_conflict"]) for row in rows]
    scores = [signal(row) for row in rows]
    weights = [as_float(row["analysis_weight"]) for row in rows]
    return {
        "scope": rows[0]["view"] if rows else "empty",
        "signal": name,
        "n": len(rows),
        "formal_calibrated_metric": formal,
        "weighted_AUPRC": weighted_average_precision(labels, scores, weights),
        "weighted_Brier": brier(labels, scores, weights) if formal else None,
        "weighted_ECE": ece(labels, scores, weights) if formal else None,
        "proxy_weighted_Brier": None if formal else brier(labels, scores, weights),
        "proxy_weighted_ECE": None if formal else ece(labels, scores, weights),
    }


def review_frontier(rows: list[dict[str, Any]], signal_name: str, signal: Callable[[dict[str, Any]], float]) -> list[dict[str, Any]]:
    labels = [as_int(row["severe_human_silver_conflict"]) for row in rows]
    scores = [signal(row) for row in rows]
    weights = [as_float(row["analysis_weight"]) for row in rows]
    positives = sum(weight for label, weight in zip(labels, weights) if label)
    output = []
    for population_weight in (False, True):
        selected = top_fraction_indices(scores, weights, 0.20, population_weight)
        selected_weights = [weights[idx] for idx in selected]
        flagged_positive_weight = sum(weights[idx] for idx in selected if labels[idx])
        output.append(
            {
                "scope": rows[0]["view"] if rows else "empty",
                "signal": signal_name,
                "selection": "top_20pct_population_weight" if population_weight else "top_20pct_rows",
                "selected_rows": len(selected),
                "selected_weight": sum(selected_weights),
                "weighted_review_precision": flagged_positive_weight / sum(selected_weights) if selected_weights else None,
                "weighted_review_recall": flagged_positive_weight / positives if positives else None,
            }
        )
    return output


def policy_summary(rows: list[dict[str, Any]], policy_key: str) -> list[dict[str, Any]]:
    total = sum(as_float(row["analysis_weight"]) for row in rows)
    output = []
    for tier in ("high", "low", "review"):
        selected = [row for row in rows if row[policy_key] == tier]
        weights = [as_float(row["analysis_weight"]) for row in selected]
        severe_rate = weighted_mean([as_int(row["severe_human_silver_conflict"]) for row in selected], weights)
        output.append(
            {
                "scope": rows[0]["view"] if rows else "empty",
                "policy": policy_key,
                "tier": tier,
                "n": len(selected),
                "weighted_coverage": sum(weights) / total if total else None,
                "weighted_severe_conflict_rate": severe_rate,
                "weighted_nonsevere_rate": None if severe_rate is None else 1.0 - severe_rate,
            }
        )
    return output


def pattern_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["agreement_pattern"])].append(row)
    output = []
    for pattern, grouped in sorted(groups.items()):
        weights = [as_float(row["analysis_weight"]) for row in grouped]
        severe_rate = weighted_mean([as_int(row["severe_human_silver_conflict"]) for row in grouped], weights)
        low_review_rate = weighted_mean([int(row["policy_b_tier"] == "review") for row in grouped], weights)
        output.append(
            {
                "scope": grouped[0]["view"],
                "agreement_pattern": pattern,
                "n": len(grouped),
                "weighted_population_mass": sum(weights),
                "weighted_severe_conflict_rate": severe_rate,
                "policy_b_weighted_review_rate": low_review_rate,
            }
        )
    return output


def bootstrap_table(rows: list[dict[str, Any]], resamples: int) -> list[dict[str, Any]]:
    def metric_delta(pred_field: str, baseline_field: str, metric: str) -> Callable[[list[dict[str, Any]]], float | None]:
        def statistic(sample: list[dict[str, Any]]) -> float | None:
            gold = [as_int(row["silver_final_score"]) for row in sample]
            weights = [as_float(row["analysis_weight"]) for row in sample]
            proposed = [as_int(row[pred_field]) for row in sample]
            baseline = [as_int(row[baseline_field]) for row in sample]
            if metric == "mae":
                first, second = weighted_mae(gold, proposed, weights), weighted_mae(gold, baseline, weights)
            elif metric == "qwk":
                first, second = weighted_qwk(gold, proposed, weights), weighted_qwk(gold, baseline, weights)
            else:
                first = weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, proposed)], weights)
                second = weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, baseline)], weights)
            return None if first is None or second is None else first - second
        return statistic

    def risk_ap_delta(sample: list[dict[str, Any]]) -> float | None:
        labels = [as_int(row["severe_human_silver_conflict"]) for row in sample]
        weights = [as_float(row["analysis_weight"]) for row in sample]
        model = weighted_average_precision(labels, [as_float(row["risk_probability"]) for row in sample], weights)
        heuristic = weighted_average_precision(labels, [abs(as_int(row["qwen_score"]) - as_int(row["original_score"])) for row in sample], weights)
        return None if model is None or heuristic is None else model - heuristic

    specs = [
        ("soft_minus_qwen_weighted_mae", metric_delta("soft_rounded_score", "qwen_score", "mae")),
        ("soft_minus_teacher_mean_weighted_qwk", metric_delta("soft_rounded_score", "teacher_mean_score", "qwk")),
        ("soft_minus_teacher_mean_severe_error", metric_delta("soft_rounded_score", "teacher_mean_score", "severe")),
        ("risk_model_minus_qwen_h_gap_AUPRC", risk_ap_delta),
    ]
    output = []
    for index, (name, statistic) in enumerate(specs):
        point = statistic(rows)
        low, high = cluster_bootstrap_ci(rows, statistic, SEED + index, resamples)
        output.append(
            {
                "comparison": name,
                "point_estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_unit": "question_key",
                "resamples": resamples,
            }
        )
    return output


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    base = {row["sample_id"]: row for row in canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)}
    scores = {row["sample_id"]: row for row in read_csv(args.out_dir / "data" / "exp27l_oof_soft_score_predictions.csv")}
    risks = {row["sample_id"]: row for row in read_csv(args.out_dir / "data" / "exp27l_oof_risk_predictions.csv")}
    if set(base) != set(scores) or set(base) != set(risks):
        raise ValueError("OOF score/risk artifacts must align to all 180 Exp27J rows")
    rows = [{**base[sid], **scores[sid], **risks[sid]} for sid in sorted(base)]
    representative = scope_rows(rows, REPRESENTATIVE_VIEW)
    risk_stress = scope_rows(rows, RISK_ENRICHED_VIEW)

    predictors = {
        "B0_original_human": lambda row: as_int(row["original_score"]),
        "B1_qwen": lambda row: as_int(row["qwen_score"]),
        "B2_deepseek": lambda row: as_int(row["deepseek_score"]),
        "B3_teacher_mean": lambda row: as_int(row["teacher_mean_score"]),
        "B4_human_teacher_median": lambda row: as_int(row["human_teacher_median_score"]),
        "B5_exp27i_v1": lambda row: as_int(row["exp27i_v1_score"]),
        "Exp27L_soft_ordinal_fusion": lambda row: as_int(row["soft_rounded_score"]),
    }
    score_metrics = [score_metric_row(scope, name, predictor) for scope in (representative, risk_stress) for name, predictor in predictors.items()]
    write_csv(args.out_dir / "tables" / "exp27l_oof_score_metrics.csv", score_metrics)

    signals: dict[str, tuple[Callable[[dict[str, Any]], float], bool]] = {
        "Exp27L_l2_logistic": (lambda row: as_float(row["risk_probability"]), True),
        "qwen_human_gap": (lambda row: abs(as_int(row["qwen_score"]) - as_int(row["original_score"])) / 4.0, False),
        "deepseek_human_gap": (lambda row: abs(as_int(row["deepseek_score"]) - as_int(row["original_score"])) / 4.0, False),
        "teacher_gap": (lambda row: as_int(row["teacher_gap"]) / 4.0, False),
        "max_three_way_gap": (lambda row: as_int(row["max_three_way_gap"]) / 4.0, False),
        "exp27i_tier_proxy": (tier_proxy, False),
    }
    risk_metrics = [risk_metric_row(scope, name, signal, formal) for scope in (representative, risk_stress) for name, (signal, formal) in signals.items()]
    write_csv(args.out_dir / "tables" / "exp27l_oof_risk_metrics.csv", risk_metrics)
    frontier = [item for scope in (representative, risk_stress) for name, (signal, _formal) in signals.items() for item in review_frontier(scope, name, signal)]
    write_csv(args.out_dir / "tables" / "exp27l_risk_review_frontier.csv", frontier)
    policy = [item for scope in (representative, risk_stress) for key in ("policy_a_tier", "policy_b_tier") for item in policy_summary(scope, key)]
    write_csv(args.out_dir / "tables" / "exp27l_oof_policy_metrics.csv", policy)
    patterns = pattern_rows(representative) + pattern_rows(risk_stress)
    write_csv(args.out_dir / "tables" / "exp27l_pattern_validation.csv", patterns)
    bootstrap = bootstrap_table(representative, args.bootstrap_resamples)
    write_csv(args.out_dir / "tables" / "exp27l_cluster_bootstrap_ci.csv", bootstrap)

    lookup_score = {(row["scope"], row["model"]): row for row in score_metrics}
    lookup_risk = {(row["scope"], row["signal"]): row for row in risk_metrics}
    mae_delta_ci = next(row for row in bootstrap if row["comparison"] == "soft_minus_qwen_weighted_mae")
    ap_delta_ci = next(row for row in bootstrap if row["comparison"] == "risk_model_minus_qwen_h_gap_AUPRC")
    score_signal_stable = bool(mae_delta_ci["ci95_high"] is not None and mae_delta_ci["ci95_high"] <= 0.0)
    risk_signal_stable = bool(ap_delta_ci["ci95_low"] is not None and ap_delta_ci["ci95_low"] > 0.0)
    policy_b_rep_review = next(row for row in policy if row["scope"] == REPRESENTATIVE_VIEW and row["policy"] == "policy_b_tier" and row["tier"] == "review")
    external_review_complete = False
    decision = {
        "experiment": "exp27l_group_crossfit_calibration",
        "oof_score_rows": len(scores),
        "oof_risk_rows": len(risks),
        "representative_rows": len(representative),
        "risk_stress_rows": len(risk_stress),
        "bootstrap_resamples": args.bootstrap_resamples,
        "score_signal_ci_stable": score_signal_stable,
        "risk_signal_ci_stable": risk_signal_stable,
        "policy_b_representative_review_coverage": policy_b_rep_review["weighted_coverage"],
        "external_review_complete": external_review_complete,
        "external_review_required": True,
        "proceed_to_exp27m_train": False,
        "proceed_to_reranker_training": False,
        "reason": "Exp27L OOF results are internal silver-reference evidence only. External blinded human review is not complete, so no trainer dataset may be constructed.",
    }
    write_json(args.out_dir / "decision" / "exp27l_group_crossfit_decision.json", decision)
    write_text(
        args.out_dir / "reports" / "exp27l_group_crossfit_calibration_report.md",
        "\n".join(
            [
                "# Exp27L Group Cross-Fitted Calibration Report",
                "",
                "This report is a train-only, question-key OOF audit. Exp27J adjudication is a silver target, not external human gold.",
                "",
                "## Score Calibration",
                "",
                f"- Qwen weighted MAE on representative rows: {lookup_score[(REPRESENTATIVE_VIEW, 'B1_qwen')]['weighted_MAE']}",
                f"- Soft-fusion weighted MAE on representative rows: {lookup_score[(REPRESENTATIVE_VIEW, 'Exp27L_soft_ordinal_fusion')]['weighted_MAE']}",
                f"- Soft-minus-Qwen MAE 95% cluster-bootstrap CI: [{mae_delta_ci['ci95_low']}, {mae_delta_ci['ci95_high']}]",
                "",
                "## Severe Human-Silver Conflict Detection",
                "",
                f"- Logistic OOF AUPRC on representative rows: {lookup_risk[(REPRESENTATIVE_VIEW, 'Exp27L_l2_logistic')]['weighted_AUPRC']}",
                f"- Qwen-human-gap heuristic AUPRC on representative rows: {lookup_risk[(REPRESENTATIVE_VIEW, 'qwen_human_gap')]['weighted_AUPRC']}",
                f"- AUPRC difference 95% cluster-bootstrap CI: [{ap_delta_ci['ci95_low']}, {ap_delta_ci['ci95_high']}]",
                "",
                "## Policy Boundary",
                "",
                "Policy A is fixed. Policy B selects its review budget within each outer training fold using inner OOF predictions, then applies those thresholds unchanged to the held-out fold.",
                "",
                "## Locked Decision",
                "",
                "No Exp27M trainer data is produced here. Both training flags remain false until the separate external blind-review lockbox is filled and independently adjudicated.",
                "",
            ]
        ),
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    print(analyze(parse_args()))
