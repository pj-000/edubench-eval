"""Analyze Exp27L-R1 balanced OOF calibration and risk diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_EXP27L,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    REPRESENTATIVE_VIEW,
    RISK_VIEW,
    SEED,
    brier,
    canonical_rows,
    cluster_bootstrap_ci,
    ece,
    half_up,
    nll,
    read_csv,
    rps,
    tie_safe_average_precision,
    top_fraction_indices,
    weighted_mae,
    weighted_mean,
    weighted_qwk,
    write_csv,
    write_json,
    write_text,
)


def f(value: Any) -> float:
    return float(value)


def i(value: Any) -> int:
    return int(float(value))


def tier_proxy(row: dict[str, Any]) -> float:
    return {"high_weight": 0.10, "low_weight": 0.55, "review_only": 0.90}.get(str(row["exp27i_v1_tier"]), 0.50)


def split_scopes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return ([row for row in rows if row["view"] == REPRESENTATIVE_VIEW], [row for row in rows if row["view"] == RISK_VIEW])


def score_metrics(rows: list[dict[str, Any]], model: str, prediction: Callable[[dict[str, Any]], float], continuous: bool = False) -> dict[str, Any]:
    gold = [i(row["silver_final_score"]) for row in rows]
    pred = [prediction(row) for row in rows]
    weights = [f(row["analysis_weight"]) for row in rows]
    rounded = [half_up(value) for value in pred]
    low_weight = sum(weight for value, weight in zip(gold, weights) if value <= 2)
    high_weight = sum(weight for value, weight in zip(gold, weights) if value >= 4)
    return {
        "scope": rows[0]["view"],
        "model": model,
        "prediction_type": "continuous" if continuous else "ordinal",
        "n": len(rows),
        "weighted_MAE": weighted_mae(gold, pred, weights),
        "half_up_rounded_MAE": weighted_mae(gold, rounded, weights),
        "weighted_QWK": weighted_qwk(gold, rounded, weights),
        "weighted_exact_accuracy": weighted_mean([int(a == b) for a, b in zip(gold, rounded)], weights),
        "weighted_severe_error_rate": weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, rounded)], weights),
        "weighted_low_to_high": sum(weight for a, b, weight in zip(gold, rounded, weights) if a <= 2 and b >= 4) / low_weight if low_weight else None,
        "weighted_high_to_low": sum(weight for a, b, weight in zip(gold, rounded, weights) if a >= 4 and b <= 2) / high_weight if high_weight else None,
        "weighted_signed_bias": weighted_mean([b - a for a, b in zip(gold, pred)], weights),
    }


def distribution_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    gold = [i(row["silver_final_score"]) for row in rows]
    weights = [f(row["analysis_weight"]) for row in rows]
    probabilities = [json.loads(row[f"{variant}_probs"]) for row in rows]
    expected = [f(row[f"{variant}_expected_score"]) for row in rows]
    rounded = [i(row[f"{variant}_half_up_score"]) for row in rows]
    return {
        "scope": rows[0]["view"],
        "variant": f"global_ordinal_soft_fusion_{variant}_selected",
        "n": len(rows),
        "weighted_multiclass_NLL": weighted_mean([nll(prob, label) for prob, label in zip(probabilities, gold)], weights),
        "weighted_RPS": weighted_mean([rps(prob, label) for prob, label in zip(probabilities, gold)], weights),
        "expected_score_MAE": weighted_mae(gold, expected, weights),
        "half_up_rounded_MAE": weighted_mae(gold, rounded, weights),
        "weighted_QWK": weighted_qwk(gold, rounded, weights),
        "weighted_severe_error_rate": weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, rounded)], weights),
        "weighted_signed_bias": weighted_mean([b - a for a, b in zip(gold, expected)], weights),
    }


def risk_metrics(rows: list[dict[str, Any]], name: str, signal: Callable[[dict[str, Any]], float], formal: bool) -> dict[str, Any]:
    labels = [i(row["severe_human_silver_conflict"]) for row in rows]
    signals = [signal(row) for row in rows]
    weights = [f(row["analysis_weight"]) for row in rows]
    return {
        "scope": rows[0]["view"],
        "signal": name,
        "n": len(rows),
        "formal_probability_model": formal,
        "tie_safe_weighted_AUPRC": tie_safe_average_precision(labels, signals, weights),
        "weighted_Brier": brier(labels, signals, weights) if formal else None,
        "weighted_ECE": ece(labels, signals, weights) if formal else None,
        "proxy_weighted_Brier": None if formal else brier(labels, signals, weights),
        "proxy_weighted_ECE": None if formal else ece(labels, signals, weights),
    }


def review_frontier(rows: list[dict[str, Any]], name: str, signal: Callable[[dict[str, Any]], float]) -> list[dict[str, Any]]:
    labels = [i(row["severe_human_silver_conflict"]) for row in rows]
    values = [signal(row) for row in rows]
    weights = [f(row["analysis_weight"]) for row in rows]
    positives = sum(weight for label, weight in zip(labels, weights) if label)
    output = []
    for population_weight in (False, True):
        indices = top_fraction_indices(values, weights, 0.20, population_weight)
        selected_weight = sum(weights[index] for index in indices)
        selected_positive = sum(weights[index] for index in indices if labels[index])
        output.append(
            {
                "scope": rows[0]["view"],
                "signal": name,
                "selection": "top_20pct_population_weight" if population_weight else "top_20pct_rows",
                "selected_rows": len(indices),
                "selected_weight": selected_weight,
                "weighted_review_precision": selected_positive / selected_weight if selected_weight else None,
                "weighted_review_recall": selected_positive / positives if positives else None,
            }
        )
    return output


def ap_invariance(rows: list[dict[str, Any]], name: str, signal: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    rng = random.Random(SEED)
    values = []
    for _ in range(100):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        values.append(
            tie_safe_average_precision(
                [i(row["severe_human_silver_conflict"]) for row in shuffled],
                [signal(row) for row in shuffled],
                [f(row["analysis_weight"]) for row in shuffled],
            )
        )
    values_f = [float(value) for value in values if value is not None]
    return {
        "scope": rows[0]["view"],
        "signal": name,
        "permutations": 100,
        "min_AUPRC": min(values_f),
        "max_AUPRC": max(values_f),
        "max_minus_min": max(values_f) - min(values_f),
        "pass_lt_1e12": max(values_f) - min(values_f) < 1e-12,
    }


def bootstrap_rows(rows: list[dict[str, Any]], resamples: int) -> list[dict[str, Any]]:
    def score_delta(field: str, baseline: str, metric: str):
        def statistic(sample: list[dict[str, Any]]) -> float | None:
            gold = [i(row["silver_final_score"]) for row in sample]
            weights = [f(row["analysis_weight"]) for row in sample]
            left = [i(row[field]) for row in sample]
            right = [i(row[baseline]) for row in sample]
            if metric == "mae":
                first, second = weighted_mae(gold, left, weights), weighted_mae(gold, right, weights)
            elif metric == "qwk":
                first, second = weighted_qwk(gold, left, weights), weighted_qwk(gold, right, weights)
            else:
                first = weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, left)], weights)
                second = weighted_mean([int(abs(a - b) >= 2) for a, b in zip(gold, right)], weights)
            return None if first is None or second is None else first - second
        return statistic

    def ap_delta(field: str):
        def statistic(sample: list[dict[str, Any]]) -> float | None:
            labels = [i(row["severe_human_silver_conflict"]) for row in sample]
            weights = [f(row["analysis_weight"]) for row in sample]
            left = tie_safe_average_precision(labels, [f(row[field]) for row in sample], weights)
            right = tie_safe_average_precision(labels, [abs(i(row["qwen_score"]) - i(row["original_score"])) / 4.0 for row in sample], weights)
            return None if left is None or right is None else left - right
        return statistic

    specs = [
        ("nll_soft_fusion_minus_qwen_half_up_MAE", score_delta("nll_half_up_score", "qwen_score", "mae")),
        ("nll_soft_fusion_minus_teacher_mean_QWK", score_delta("nll_half_up_score", "half_up_teacher_mean_score", "qwk")),
        ("nll_soft_fusion_minus_teacher_mean_severe_error", score_delta("nll_half_up_score", "half_up_teacher_mean_score", "severe")),
        ("full_logistic_minus_qwen_gap_AUPRC", ap_delta("full_logistic_probability")),
        ("core_logistic_minus_qwen_gap_AUPRC", ap_delta("core_logistic_probability")),
    ]
    output = []
    for offset, (name, function) in enumerate(specs):
        point = function(rows)
        low, high = cluster_bootstrap_ci(rows, function, SEED + offset, resamples)
        output.append({"comparison": name, "point_estimate": point, "ci95_low": low, "ci95_high": high, "bootstrap_unit": "question_key", "resamples": resamples})
    return output


def rounding_audit(representative: list[dict[str, Any]], old_exp27l: Path) -> list[dict[str, Any]]:
    current = score_metrics(representative, "Exp27L-R1_half_up_teacher_mean", lambda row: float(row["half_up_teacher_mean_score"]))
    continuous = score_metrics(representative, "Exp27L-R1_continuous_teacher_mean", lambda row: f(row["continuous_teacher_mean"]), continuous=True)
    old_rows = read_csv(old_exp27l / "tables" / "exp27l_oof_score_metrics.csv")
    old = next(row for row in old_rows if row["scope"] == "representative" and row["model"] == "B3_teacher_mean")
    exp27k = score_metrics(representative, "Exp27K_teacher_mean_reference", lambda row: float(row["half_up_teacher_mean_score"]))
    return [
        {"method": "Exp27K_teacher_mean_half_up", "weighted_MAE": exp27k["weighted_MAE"], "weighted_QWK": exp27k["weighted_QWK"], "rounding": "floor(mean+0.5)"},
        {"method": "Exp27L_old_teacher_mean", "weighted_MAE": old["weighted_MAE"], "weighted_QWK": old["weighted_QWK"], "rounding": "python_bankers_round"},
        {"method": "Exp27L-R1_corrected_teacher_mean", "weighted_MAE": current["weighted_MAE"], "weighted_QWK": current["weighted_QWK"], "rounding": "floor(mean+0.5)"},
        {"method": "Exp27L-R1_continuous_teacher_mean", "weighted_MAE": continuous["weighted_MAE"], "weighted_QWK": continuous["weighted_QWK"], "rounding": "continuous_no_rounding"},
    ]


def analyze(args: argparse.Namespace) -> dict[str, object]:
    base = {row["sample_id"]: row for row in canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)}
    scores = {row["sample_id"]: row for row in read_csv(args.out_dir / "data" / "exp27lr1_oof_score_predictions.csv")}
    risks = {row["sample_id"]: row for row in read_csv(args.out_dir / "data" / "exp27lr1_oof_risk_predictions.csv")}
    if set(base) != set(scores) or set(base) != set(risks):
        raise ValueError("Exp27L-R1 OOF artifacts must align to all 180 cases")
    rows = [{**base[sid], **scores[sid], **risks[sid]} for sid in sorted(base)]
    representative, risk_enriched = split_scopes(rows)

    baselines = {
        "original_human": (lambda row: float(row["original_score"]), False),
        "qwen": (lambda row: float(row["qwen_score"]), False),
        "deepseek": (lambda row: float(row["deepseek_score"]), False),
        "continuous_dual_teacher_mean": (lambda row: f(row["continuous_teacher_mean"]), True),
        "half_up_dual_teacher_mean": (lambda row: float(row["half_up_teacher_mean_score"]), False),
        "human_qwen_deepseek_median": (lambda row: float(row["human_teacher_median_score"]), False),
        "exp27i_v1": (lambda row: float(row["exp27i_v1_score"]), False),
        "global_ordinal_soft_fusion_nll": (lambda row: f(row["nll_expected_score"]), True),
        "global_ordinal_soft_fusion_rps": (lambda row: f(row["rps_expected_score"]), True),
    }
    score_rows = [score_metrics(scope, name, predictor, continuous) for scope in (representative, risk_enriched) for name, (predictor, continuous) in baselines.items()]
    distribution_rows = [distribution_metrics(scope, variant) for scope in (representative, risk_enriched) for variant in ("nll", "rps")]
    write_csv(args.out_dir / "tables" / "exp27lr1_score_metrics.csv", score_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_score_distribution_metrics.csv", distribution_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_rounding_regression_audit.csv", rounding_audit(representative, args.exp27l_dir))

    signals: dict[str, tuple[Callable[[dict[str, Any]], float], bool]] = {
        "full_feature_l2_logistic": (lambda row: f(row["full_logistic_probability"]), True),
        "core_standardized_l2_logistic": (lambda row: f(row["core_logistic_probability"]), True),
        "qwen_human_gap": (lambda row: abs(i(row["qwen_score"]) - i(row["original_score"])) / 4.0, False),
        "deepseek_human_gap": (lambda row: abs(i(row["deepseek_score"]) - i(row["original_score"])) / 4.0, False),
        "teacher_gap": (lambda row: i(row["teacher_gap"]) / 4.0, False),
        "max_three_way_gap": (lambda row: i(row["max_three_way_gap"]) / 4.0, False),
        "exp27i_tier_proxy": (tier_proxy, False),
    }
    risk_rows = [risk_metrics(scope, name, signal, formal) for scope in (representative, risk_enriched) for name, (signal, formal) in signals.items()]
    frontier_rows = [entry for scope in (representative, risk_enriched) for name, (signal, _formal) in signals.items() for entry in review_frontier(scope, name, signal)]
    invariance_rows = [ap_invariance(representative, name, signal) for name, (signal, _formal) in signals.items()]
    bootstrap = bootstrap_rows(representative, args.bootstrap_resamples)
    write_csv(args.out_dir / "tables" / "exp27lr1_risk_metrics.csv", risk_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_review_frontier.csv", frontier_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_ap_tie_invariance_test.csv", invariance_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_cluster_bootstrap_ci.csv", bootstrap)

    score_bootstrap = next(row for row in bootstrap if row["comparison"] == "nll_soft_fusion_minus_qwen_half_up_MAE")
    full_bootstrap = next(row for row in bootstrap if row["comparison"] == "full_logistic_minus_qwen_gap_AUPRC")
    core_bootstrap = next(row for row in bootstrap if row["comparison"] == "core_logistic_minus_qwen_gap_AUPRC")
    full_ap = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "full_feature_l2_logistic")["tie_safe_weighted_AUPRC"]
    core_ap = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "core_standardized_l2_logistic")["tie_safe_weighted_AUPRC"]
    qwen_gap_ap = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "qwen_human_gap")["tie_safe_weighted_AUPRC"]
    # The protocol freezes learned risk when neither model even exceeds the
    # simple gap point estimate and neither paired CI supports an improvement.
    learned_negative = bool(
        full_ap <= qwen_gap_ap
        and core_ap <= qwen_gap_ap
        and full_bootstrap["ci95_low"] is not None
        and full_bootstrap["ci95_low"] <= 0.0
        and core_bootstrap["ci95_low"] is not None
        and core_bootstrap["ci95_low"] <= 0.0
    )
    fusion_negative = bool(score_bootstrap["ci95_low"] is not None and score_bootstrap["ci95_low"] > 0.0)
    decision = {
        "experiment": "exp27lr1_balanced_group_crossfit",
        "lock_result_soft_fusion": "global_soft_fusion_negative" if fusion_negative else "not_locked",
        "lock_result_learned_risk": "learned_risk_model_negative_use_simple_disagreement" if learned_negative else "not_locked",
        "recommend_learned_risk_model": not learned_negative,
        "recommended_risk_signal": "qwen_human_gap_or_simple_disagreement" if learned_negative else "undetermined",
        "ap_tie_invariance_pass": all(row["pass_lt_1e12"] for row in invariance_rows),
        "external_human_review_complete": False,
        "proceed_to_full_3326_calibrated_dataset": False,
        "proceed_to_qwen3_reranker_training": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
    }
    write_json(args.out_dir / "decision" / "exp27lr1_balanced_crossfit_decision.json", decision)
    nll_metrics = next(row for row in score_rows if row["scope"] == REPRESENTATIVE_VIEW and row["model"] == "global_ordinal_soft_fusion_nll")
    qwen_metrics = next(row for row in score_rows if row["scope"] == REPRESENTATIVE_VIEW and row["model"] == "qwen")
    full_metric = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "full_feature_l2_logistic")
    core_metric = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "core_standardized_l2_logistic")
    gap_metric = next(row for row in risk_rows if row["scope"] == REPRESENTATIVE_VIEW and row["signal"] == "qwen_human_gap")
    write_text(
        args.out_dir / "reports" / "exp27lr1_balanced_crossfit_report.md",
        "\n".join(
            [
                "# Exp27L-R1 Balanced Crossfit Report",
                "",
                "This is a train-only internal silver-reference diagnostic. It does not construct trainer data.",
                "",
                "## Score Fusion",
                "",
                f"- Qwen weighted MAE: {qwen_metrics['weighted_MAE']}",
                f"- NLL-selected global fusion weighted MAE: {nll_metrics['weighted_MAE']}",
                f"- Fusion-minus-Qwen MAE 95% cluster CI: [{score_bootstrap['ci95_low']}, {score_bootstrap['ci95_high']}]",
                f"- Fusion lock status: {decision['lock_result_soft_fusion']}",
                "",
                "## Severe Human-Silver Conflict Ranking",
                "",
                f"- Full Logistic tie-safe AUPRC: {full_metric['tie_safe_weighted_AUPRC']}",
                f"- Core Logistic tie-safe AUPRC: {core_metric['tie_safe_weighted_AUPRC']}",
                f"- Qwen-human gap tie-safe AUPRC: {gap_metric['tie_safe_weighted_AUPRC']}",
                f"- Learned-risk lock status: {decision['lock_result_learned_risk']}",
                "",
                "No external review is complete, so all downstream training gates remain false.",
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
    parser.add_argument("--exp27l-dir", type=Path, default=DEFAULT_EXP27L)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    print(analyze(parse_args()))
