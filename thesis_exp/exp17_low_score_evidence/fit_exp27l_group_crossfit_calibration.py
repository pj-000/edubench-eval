"""Fit Exp27L score/risk calibrators with nested question-key cross-fitting."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27l_common import (  # noqa: E402
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    INNER_FOLDS,
    REPRESENTATIVE_VIEW,
    SCORE_LABELS,
    SEED,
    SIMPLEX,
    TAUS,
    brier,
    canonical_rows,
    entropy,
    expected_score,
    grouped_folds,
    require_sklearn,
    risk_feature_rows,
    rounded_score,
    soft_score_distribution,
    weighted_mean,
    write_csv,
    write_json,
)


def weighted_nll(rows: list[dict[str, Any]], lambdas: tuple[float, float, float], tau: float) -> float:
    values = []
    weights = []
    for row in rows:
        probs = soft_score_distribution(row, lambdas, tau)
        values.append(-math.log(max(probs[row["silver_final_score"] - 1], 1e-12)))
        weights.append(float(row["analysis_weight"]))
    result = weighted_mean(values, weights)
    return float("inf") if result is None else result


def choose_score_params(rows: list[dict[str, Any]], seed: int) -> tuple[tuple[float, float, float], float, float]:
    """Nested grouped-CV selection of the fixed soft ordinal fusion family."""
    folds = grouped_folds(rows, INNER_FOLDS, seed)
    best: tuple[float, tuple[float, float, float], float] | None = None
    for lambdas in SIMPLEX:
        for tau in TAUS:
            fold_losses: list[float] = []
            for fold in range(INNER_FOLDS):
                held = [row for row in rows if folds[row["question_key"]] == fold]
                fold_losses.append(weighted_nll(held, lambdas, tau))
            value = float(np.mean(fold_losses))
            candidate = (value, lambdas, tau)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No score fusion parameter was selected")
    return best[1], best[2], best[0]


def score_oof(rows: list[dict[str, Any]], lambdas: tuple[float, float, float], tau: float) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        probs = soft_score_distribution(row, lambdas, tau)
        output.append(
            {
                "sample_id": row["sample_id"],
                "score_probs": probs,
                "soft_expected_score": expected_score(probs),
                "soft_rounded_score": rounded_score(probs),
                "score_entropy": entropy(probs),
            }
        )
    return output


def fit_logistic(train_rows: list[dict[str, Any]], c_value: float):
    """Fit the required L2 LogisticRegression, with a documented rare-event fallback."""
    from sklearn.linear_model import LogisticRegression

    labels = [int(row["severe_human_silver_conflict"]) for row in train_rows]
    prior = weighted_mean(labels, [float(row["analysis_weight"]) for row in train_rows])
    if len(set(labels)) < 2:
        return None, float(prior or 0.0), None
    x_train, categories = risk_feature_rows(train_rows)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=SEED)
    model.fit(x_train, labels, sample_weight=np.asarray([row["analysis_weight"] for row in train_rows], dtype=float))
    return model, None, categories


def predict_logistic(model, constant: float | None, categories: dict[str, list[str]] | None, rows: list[dict[str, Any]]) -> list[float]:
    if model is None:
        return [float(constant or 0.0)] * len(rows)
    x, _ = risk_feature_rows(rows, categories)
    return [float(value) for value in model.predict_proba(x)[:, 1]]


def choose_risk_c(rows: list[dict[str, Any]], seed: int) -> tuple[float, float, list[dict[str, Any]]]:
    """Choose C by group-CV weighted Brier on representative rows only."""
    folds = grouped_folds(rows, INNER_FOLDS, seed)
    best: tuple[float, float, list[dict[str, Any]]] | None = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        oof: list[dict[str, Any]] = []
        for fold in range(INNER_FOLDS):
            train = [row for row in rows if folds[row["question_key"]] != fold]
            held = [row for row in rows if folds[row["question_key"]] == fold]
            model, constant, categories = fit_logistic(train, c_value)
            for row, probability in zip(held, predict_logistic(model, constant, categories, held)):
                oof.append({**row, "risk_probability": probability})
        value = brier(
            [row["severe_human_silver_conflict"] for row in oof],
            [row["risk_probability"] for row in oof],
            [float(row["analysis_weight"]) for row in oof],
        )
        candidate = (float(value if value is not None else float("inf")), c_value, oof)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No LogisticRegression C was selected")
    return best[1], best[0], best[2]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), fraction, method="higher"))


def weighted_review_threshold(rows: list[dict[str, Any]], budget: float) -> float:
    ordered = sorted(rows, key=lambda row: row["risk_probability"], reverse=True)
    target = budget * sum(float(row["analysis_weight"]) for row in ordered)
    total = 0.0
    threshold = ordered[-1]["risk_probability"] if ordered else 1.0
    for row in ordered:
        total += float(row["analysis_weight"])
        threshold = row["risk_probability"]
        if total >= target:
            return float(threshold)
    return float(threshold)


def policy_a(probability: float, score_entropy: float, entropy_median: float, teacher_gap: int, target_issue: bool) -> str:
    if probability <= 0.10 and score_entropy <= entropy_median and teacher_gap <= 1 and not target_issue:
        return "high"
    if probability > 0.35 or teacher_gap >= 2 or target_issue:
        return "review"
    return "low"


def choose_policy_b(rows: list[dict[str, Any]]) -> tuple[float, float, dict[str, float]]:
    """Select a budget only from inner-OOF predictions in an outer training fold."""
    choices: list[tuple[float, float, float, dict[str, float]]] = []
    for budget in (0.15, 0.20, 0.25, 0.30):
        threshold = weighted_review_threshold(rows, budget)
        flagged = [row for row in rows if row["risk_probability"] >= threshold]
        remaining = [row for row in rows if row not in flagged]
        weights_flagged = [float(row["analysis_weight"]) for row in flagged]
        precision = weighted_mean([row["severe_human_silver_conflict"] for row in flagged], weights_flagged) or 0.0
        high_threshold = percentile([row["risk_probability"] for row in remaining], 0.45)
        high = [row for row in remaining if row["risk_probability"] <= high_threshold]
        high_coverage = (sum(float(row["analysis_weight"]) for row in high) / sum(float(row["analysis_weight"]) for row in rows)) if rows else 0.0
        high_safe = 1.0 - (weighted_mean([row["severe_human_silver_conflict"] for row in high], [float(row["analysis_weight"]) for row in high]) or 0.0)
        stats = {
            "review_budget": budget,
            "review_threshold": threshold,
            "high_threshold": high_threshold,
            "review_precision": precision,
            "high_coverage": high_coverage,
            "high_safe_rate": high_safe,
        }
        eligible = float(high_coverage >= 0.30)
        choices.append((eligible, precision + 0.20 * high_safe, -budget, stats))
    _, _, _, selected = max(choices)
    return selected["review_threshold"], selected["high_threshold"], selected


def fit(args: argparse.Namespace) -> dict[str, Any]:
    require_sklearn()
    rows = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    with (args.out_dir / "data" / "exp27l_question_key_fold_assignment.csv").open(encoding="utf-8", newline="") as handle:
        fold_rows = {row["sample_id"]: int(row["outer_fold"]) for row in csv.DictReader(handle)}
    if len(fold_rows) != 180:
        raise ValueError("Run prepare_exp27l_group_crossfit_calibration.py before fitting")
    for row in rows:
        row["outer_fold"] = fold_rows[row["sample_id"]]

    score_outputs: list[dict[str, Any]] = []
    risk_outputs: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        outer_train = [row for row in rows if row["outer_fold"] != outer_fold]
        outer_held = [row for row in rows if row["outer_fold"] == outer_fold]
        rep_train = [row for row in outer_train if row["view"] == REPRESENTATIVE_VIEW]
        if not rep_train:
            raise RuntimeError(f"Outer fold {outer_fold} has no representative fitting rows")
        lambdas, tau, score_nll = choose_score_params(rep_train, args.seed + outer_fold)
        selected_c, risk_brier, risk_inner_oof = choose_risk_c(rep_train, args.seed + 100 + outer_fold)
        model, constant, categories = fit_logistic(rep_train, selected_c)
        risk_probs = predict_logistic(model, constant, categories, outer_held)
        held_score = {item["sample_id"]: item for item in score_oof(outer_held, lambdas, tau)}
        entropy_median = percentile([item["score_entropy"] for item in score_oof(rep_train, lambdas, tau)], 0.5)
        review_threshold, high_threshold, policy_b_selection = choose_policy_b(risk_inner_oof)
        parameter_rows.append(
            {
                "outer_fold": outer_fold,
                "score_lambda_human": lambdas[0],
                "score_lambda_qwen": lambdas[1],
                "score_lambda_deepseek": lambdas[2],
                "score_tau": tau,
                "inner_score_weighted_nll": score_nll,
                "risk_logistic_c": selected_c,
                "inner_risk_weighted_brier": risk_brier,
                "risk_fit_rows": len(rep_train),
                "risk_stress_fit_rows": 0,
                "risk_fit_population": "representative_only",
                "risk_feature_excludes_silver": True,
            }
        )
        policy_rows.append({"outer_fold": outer_fold, "policy": "B_budget_selected_on_inner_oof", **policy_b_selection})
        for row, probability in zip(outer_held, risk_probs):
            score_item = held_score[row["sample_id"]]
            tier_a = policy_a(probability, score_item["score_entropy"], entropy_median, row["teacher_gap"], row["target_issue_flag"])
            if probability >= review_threshold:
                tier_b = "review"
            elif probability <= high_threshold:
                tier_b = "high"
            else:
                tier_b = "low"
            common = {
                "sample_id": row["sample_id"],
                "question_key_hash": row["question_key_hash"],
                "outer_fold": outer_fold,
                "view": row["view"],
                "score_stratum": row["score_stratum"],
                "analysis_weight": row["analysis_weight"],
                "original_score": row["original_score"],
                "silver_final_score": row["silver_final_score"],
                "qwen_score": row["qwen_score"],
                "deepseek_score": row["deepseek_score"],
                "teacher_gap": row["teacher_gap"],
                "target_issue_flag": row["target_issue_flag"],
                "severe_human_silver_conflict": row["severe_human_silver_conflict"],
            }
            score_outputs.append(
                {
                    **common,
                    **score_item,
                    "selected_lambda_human": lambdas[0],
                    "selected_lambda_qwen": lambdas[1],
                    "selected_lambda_deepseek": lambdas[2],
                    "selected_tau": tau,
                }
            )
            risk_outputs.append(
                {
                    **common,
                    "risk_probability": probability,
                    "risk_model": "l2_logistic_representative_only",
                    "risk_logistic_c": selected_c,
                    "policy_a_tier": tier_a,
                    "policy_b_tier": tier_b,
                    "policy_b_review_threshold": review_threshold,
                    "policy_b_high_threshold": high_threshold,
                }
            )

    score_outputs.sort(key=lambda row: row["sample_id"])
    risk_outputs.sort(key=lambda row: row["sample_id"])
    write_csv(args.out_dir / "data" / "exp27l_oof_soft_score_predictions.csv", score_outputs)
    write_csv(args.out_dir / "data" / "exp27l_oof_risk_predictions.csv", risk_outputs)
    write_csv(args.out_dir / "data" / "exp27l_oof_tier_assignments.csv", risk_outputs)
    write_csv(args.out_dir / "tables" / "exp27l_selected_calibration_parameters.csv", parameter_rows)
    write_csv(args.out_dir / "tables" / "exp27l_policy_b_inner_selection.csv", policy_rows)
    decision = {
        "experiment": "exp27l_group_crossfit_calibration",
        "oof_rows": len(score_outputs),
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "risk_model": "sklearn.linear_model.LogisticRegression(penalty=L2)",
        "risk_target": "severe_human_silver_conflict",
        "risk_stress_used_for_fit": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "proceed_to_exp27m_train": False,
        "proceed_to_reranker_training": False,
    }
    write_json(args.out_dir / "decision" / "exp27l_fit_decision.json", decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(fit(parse_args()))
