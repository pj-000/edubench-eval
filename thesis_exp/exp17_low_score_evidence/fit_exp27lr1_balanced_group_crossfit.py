"""Fit Exp27L-R1 OOF score and risk diagnostics on balanced group folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    INNER_SPLITS,
    SEED,
    SIMPLEX,
    TAUS,
    brier,
    canonical_rows,
    core_feature_matrix,
    entropy,
    expected_score,
    full_feature_matrix,
    half_up,
    nll,
    rps,
    soft_distribution,
    stratified_group_assignments,
    validate_inner,
    weighted_mean,
    write_csv,
    write_json,
)


def load_outer_assignment(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = {row["sample_id"]: int(row["outer_fold"]) for row in csv.DictReader(handle)}
    if len(values) != 180:
        raise ValueError("Expected 180 Exp27L-R1 outer assignments")
    return values


def pooled_score_objective(rows: list[dict[str, Any]], lambdas: tuple[float, float, float], tau: float, kind: str) -> float:
    values, weights = [], []
    for row in rows:
        probs = soft_distribution(row, lambdas, tau)
        values.append(nll(probs, row["silver_final_score"]) if kind == "nll" else rps(probs, row["silver_final_score"]))
        weights.append(float(row["analysis_weight"]))
    return float(weighted_mean(values, weights) or float("inf"))


def choose_fusion(rows: list[dict[str, Any]], inner_assign: dict[str, int], kind: str) -> tuple[tuple[float, float, float], float, float]:
    """Select one global fusion family member using pooled inner-held loss.

    The candidates have no fitted parameters beyond lambda/tau, so every inner
    held sample contributes once to the pooled design-weighted objective.
    """
    held_once = [row for fold in range(INNER_SPLITS) for row in rows if inner_assign[row["sample_id"]] == fold]
    if len(held_once) != len(rows) or len({row["sample_id"] for row in held_once}) != len(rows):
        raise ValueError("Inner score objective must contain every held sample exactly once")
    best: tuple[float, tuple[float, float, float], float] | None = None
    for lambdas in SIMPLEX:
        for tau in TAUS:
            candidate = (pooled_score_objective(held_once, lambdas, tau, kind), lambdas, tau)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No soft-fusion parameters selected")
    return best[1], best[2], best[0]


def fit_full_logistic(rows: list[dict[str, Any]], c_value: float):
    from sklearn.linear_model import LogisticRegression

    labels = [int(row["severe_human_silver_conflict"]) for row in rows]
    weights = [float(row["analysis_weight"]) for row in rows]
    if len(set(labels)) < 2:
        return None, float(weighted_mean(labels, weights) or 0.0), None
    matrix, categories = full_feature_matrix(rows)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=SEED)
    model.fit(matrix, labels, sample_weight=np.asarray(weights, dtype=float))
    return model, None, categories


def predict_full(model, constant: float | None, categories: dict[str, list[str]] | None, rows: list[dict[str, Any]]) -> list[float]:
    if model is None:
        return [float(constant or 0.0)] * len(rows)
    matrix, _ = full_feature_matrix(rows, categories)
    return [float(value) for value in model.predict_proba(matrix)[:, 1]]


def fit_core_logistic(rows: list[dict[str, Any]], c_value: float):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    labels = [int(row["severe_human_silver_conflict"]) for row in rows]
    weights = [float(row["analysis_weight"]) for row in rows]
    if len(set(labels)) < 2:
        return None, float(weighted_mean(labels, weights) or 0.0)
    preprocessor = ColumnTransformer(
        [
            ("continuous", StandardScaler(), [0, 1, 2, 3, 4]),
            ("score_stratum", OneHotEncoder(handle_unknown="ignore"), [5]),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("logistic", LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=SEED)),
        ]
    )
    model.fit(core_feature_matrix(rows), labels, logistic__sample_weight=np.asarray(weights, dtype=float))
    return model, None


def predict_core(model, constant: float | None, rows: list[dict[str, Any]]) -> list[float]:
    if model is None:
        return [float(constant or 0.0)] * len(rows)
    return [float(value) for value in model.predict_proba(core_feature_matrix(rows))[:, 1]]


def choose_risk_c(
    rows: list[dict[str, Any]], inner_assign: dict[str, int], kind: str
) -> tuple[float, float, list[dict[str, Any]]]:
    fitter: Callable[..., Any] = fit_full_logistic if kind == "full" else fit_core_logistic
    predictor: Callable[..., list[float]] = predict_full if kind == "full" else predict_core
    best: tuple[float, float, list[dict[str, Any]]] | None = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        oof: list[dict[str, Any]] = []
        for fold in range(INNER_SPLITS):
            train = [row for row in rows if inner_assign[row["sample_id"]] != fold]
            held = [row for row in rows if inner_assign[row["sample_id"]] == fold]
            fitted = fitter(train, c_value)
            if kind == "full":
                model, constant, categories = fitted
                probabilities = predictor(model, constant, categories, held)
            else:
                model, constant = fitted
                probabilities = predictor(model, constant, held)
            for row, probability in zip(held, probabilities):
                oof.append({**row, "risk_probability": probability})
        score_value = brier(
            [row["severe_human_silver_conflict"] for row in oof],
            [row["risk_probability"] for row in oof],
            [float(row["analysis_weight"]) for row in oof],
        )
        candidate = (float(score_value if score_value is not None else float("inf")), c_value, oof)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No {kind} Logistic C selected")
    return best[1], best[0], best[2]


def weighted_quantile_threshold(rows: list[dict[str, Any]], mass: float) -> float:
    ordered = sorted(rows, key=lambda row: row["risk_probability"], reverse=True)
    target = mass * sum(float(row["analysis_weight"]) for row in ordered)
    current, threshold = 0.0, 1.0
    for row in ordered:
        current += float(row["analysis_weight"])
        threshold = float(row["risk_probability"])
        if current >= target:
            break
    return threshold


def fit(args: argparse.Namespace) -> dict[str, object]:
    rows = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    assignment = load_outer_assignment(args.out_dir / "data" / "exp27lr1_outer_fold_assignment_seed42.csv")
    if set(assignment) != {row["sample_id"] for row in rows}:
        raise ValueError("Outer assignment does not align to canonical rows")
    for row in rows:
        row["outer_fold"] = assignment[row["sample_id"]]

    score_output: list[dict[str, Any]] = []
    risk_output: list[dict[str, Any]] = []
    tier_output: list[dict[str, Any]] = []
    inner_balance: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        outer_train = [row for row in rows if row["outer_fold"] != outer_fold]
        outer_held = [row for row in rows if row["outer_fold"] == outer_fold]
        representative_train = [row for row in outer_train if row["view"] == "representative"]
        inner_assign = stratified_group_assignments(representative_train, seed=args.seed + outer_fold, inner=True)
        inner_balance.extend(validate_inner(representative_train, inner_assign, outer_fold))
        nll_params = choose_fusion(representative_train, inner_assign, "nll")
        rps_params = choose_fusion(representative_train, inner_assign, "rps")
        full_c, full_brier, _full_inner = choose_risk_c(representative_train, inner_assign, "full")
        core_c, core_brier, core_inner = choose_risk_c(representative_train, inner_assign, "core")
        full_model, full_constant, full_categories = fit_full_logistic(representative_train, full_c)
        core_model, core_constant = fit_core_logistic(representative_train, core_c)
        full_probabilities = predict_full(full_model, full_constant, full_categories, outer_held)
        core_probabilities = predict_core(core_model, core_constant, outer_held)
        core_review_threshold = weighted_quantile_threshold(core_inner, 0.20)
        core_high_threshold = float(np.quantile([row["risk_probability"] for row in core_inner], 0.25, method="higher"))
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "nll_lambda_human": nll_params[0][0],
                "nll_lambda_qwen": nll_params[0][1],
                "nll_lambda_deepseek": nll_params[0][2],
                "nll_tau": nll_params[1],
                "pooled_inner_weighted_nll": nll_params[2],
                "rps_lambda_human": rps_params[0][0],
                "rps_lambda_qwen": rps_params[0][1],
                "rps_lambda_deepseek": rps_params[0][2],
                "rps_tau": rps_params[1],
                "pooled_inner_weighted_rps": rps_params[2],
                "full_logistic_c": full_c,
                "full_inner_weighted_brier": full_brier,
                "core_logistic_c": core_c,
                "core_inner_weighted_brier": core_brier,
                "representative_fit_rows": len(representative_train),
                "risk_enriched_fit_rows": 0,
            }
        )
        for row, full_probability, core_probability in zip(outer_held, full_probabilities, core_probabilities):
            nll_probs = soft_distribution(row, nll_params[0], nll_params[1])
            rps_probs = soft_distribution(row, rps_params[0], rps_params[1])
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
                "continuous_teacher_mean": row["continuous_teacher_mean"],
                "half_up_teacher_mean_score": row["half_up_teacher_mean_score"],
                "human_teacher_median_score": row["human_teacher_median_score"],
                "exp27i_v1_score": row["exp27i_v1_score"],
                "exp27i_v1_tier": row["exp27i_v1_tier"],
                "teacher_gap": row["teacher_gap"],
                "max_three_way_gap": row["max_three_way_gap"],
                "target_issue_flag": row["target_issue_flag"],
                "severe_human_silver_conflict": row["severe_human_silver_conflict"],
            }
            score_output.append(
                {
                    **common,
                    "nll_probs": json.dumps(nll_probs),
                    "nll_expected_score": expected_score(nll_probs),
                    "nll_half_up_score": half_up(expected_score(nll_probs)),
                    "nll_entropy": entropy(nll_probs),
                    "rps_probs": json.dumps(rps_probs),
                    "rps_expected_score": expected_score(rps_probs),
                    "rps_half_up_score": half_up(expected_score(rps_probs)),
                    "rps_entropy": entropy(rps_probs),
                }
            )
            risk_output.append(
                {
                    **common,
                    "full_logistic_probability": full_probability,
                    "core_logistic_probability": core_probability,
                    "full_logistic_c": full_c,
                    "core_logistic_c": core_c,
                    "risk_fit_population": "representative_only",
                }
            )
            qwen_gap = abs(row["qwen_score"] - row["original_score"])
            simple_tier = "review" if qwen_gap >= 2 or row["target_issue_flag"] else "high" if qwen_gap == 0 else "low"
            core_tier = "review" if core_probability >= core_review_threshold else "high" if core_probability <= core_high_threshold else "low"
            tier_output.append(
                {
                    **common,
                    "simple_disagreement_tier": simple_tier,
                    "core_logistic_tier": core_tier,
                    "core_review_threshold": core_review_threshold,
                    "core_high_threshold": core_high_threshold,
                }
            )

    write_csv(args.out_dir / "data" / "exp27lr1_oof_score_predictions.csv", sorted(score_output, key=lambda row: row["sample_id"]))
    write_csv(args.out_dir / "data" / "exp27lr1_oof_risk_predictions.csv", sorted(risk_output, key=lambda row: row["sample_id"]))
    write_csv(args.out_dir / "data" / "exp27lr1_oof_tier_assignments.csv", sorted(tier_output, key=lambda row: row["sample_id"]))
    write_csv(args.out_dir / "tables" / "exp27lr1_inner_fold_balance.csv", inner_balance)
    write_csv(args.out_dir / "tables" / "exp27lr1_selected_parameters.csv", selected_rows)
    decision = {
        "experiment": "exp27lr1_balanced_group_crossfit",
        "oof_rows": len(score_output),
        "inner_constraints_pass": True,
        "risk_enriched_rows_used_for_fit": 0,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "proceed_to_full_3326_calibrated_dataset": False,
        "proceed_to_qwen3_reranker_training": False,
    }
    write_json(args.out_dir / "decision" / "exp27lr1_fit_decision.json", decision)
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
