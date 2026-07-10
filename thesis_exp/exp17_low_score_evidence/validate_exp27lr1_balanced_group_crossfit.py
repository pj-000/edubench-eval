"""Validate Exp27L-R1 crossfit, packet isolation, and review-template rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import DEFAULT_OUT, read_csv, read_jsonl  # noqa: E402


def validate_review_payload(review: dict[str, Any], evaluator_output: str) -> list[str]:
    """Return protocol violations for a filled review without assigning semantics."""
    errors: list[str] = []
    score_range = review.get("score_range")
    score_value = review.get("most_plausible_score")
    if not isinstance(score_range, list) or len(score_range) != 2 or not all(isinstance(value, int) and 1 <= value <= 5 for value in score_range):
        errors.append("invalid_score_range")
    elif score_range[0] > score_range[1]:
        errors.append("reversed_score_range")
    elif not isinstance(score_value, int) or not score_range[0] <= score_value <= score_range[1]:
        errors.append("score_outside_range")
    if review.get("target_scope_confirmed") is not True:
        errors.append("target_scope_not_confirmed")
    evidence = review.get("evaluator_output_evidence")
    missing_reason = review.get("missing_evidence_reason")
    if evidence is not None and (not isinstance(evidence, str) or evidence not in evaluator_output):
        errors.append("evidence_not_normalized_substring")
    if review.get("failure_bucket") == "hidden_or_missing_failure" and evidence is None and not isinstance(missing_reason, str):
        errors.append("missing_failure_without_missing_reason")
    high_risk_failures = {"missing_key_point", "factual_or_rubric_mismatch", "task_constraint_violation"}
    if review.get("failure_bucket") == "no_failure" and high_risk_failures & set(review.get("major_failures") or []):
        errors.append("no_failure_with_high_risk_failure")
    if review.get("confidence") == "low" and review.get("needs_adjudication") is not True:
        errors.append("low_confidence_without_adjudication")
    return errors


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_dir
    required = [
        out / "data" / "exp27lr1_outer_fold_assignment_seed42.csv",
        out / "data" / "exp27lr1_oof_score_predictions.csv",
        out / "data" / "exp27lr1_oof_risk_predictions.csv",
        out / "data" / "exp27lr1_oof_tier_assignments.csv",
        out / "tables" / "exp27lr1_outer_fold_balance.csv",
        out / "tables" / "exp27lr1_inner_fold_balance.csv",
        out / "tables" / "exp27lr1_split_seed_sensitivity.csv",
        out / "tables" / "exp27lr1_ap_tie_invariance_test.csv",
        out / "tables" / "exp27lr1_rounding_regression_audit.csv",
        out / "tables" / "exp27lr1_review_set_distribution.csv",
        out / "tables" / "exp27lr1_review_leakage_audit.csv",
        out / "decision" / "exp27lr1_balanced_crossfit_decision.json",
        out / "decision" / "exp27lr1_blind_review_prepare_decision.json",
        out / "packets" / "exp27lr1_balanced_silver_validation_60.jsonl",
        out / "packets" / "exp27lr1_external_high_control_34.jsonl",
        out / "packets" / "exp27lr1_targeted_ambiguity_20.jsonl",
    ]
    for path in required:
        require(path)

    assignment = read_csv(out / "data" / "exp27lr1_outer_fold_assignment_seed42.csv")
    if len(assignment) != 180 or len({row["sample_id"] for row in assignment}) != 180:
        raise ValueError("Expected 180 unique Exp27L-R1 assignments")
    qkey_folds: dict[str, set[str]] = defaultdict(set)
    for row in assignment:
        qkey_folds[row["question_key_hash"]].add(row["outer_fold"])
    if any(len(folds) != 1 for folds in qkey_folds.values()):
        raise ValueError("A question key crosses Exp27L-R1 outer folds")
    outer = read_csv(out / "tables" / "exp27lr1_outer_fold_balance.csv")
    if len(outer) != 5 or not all(row["constraints_pass"] == "True" for row in outer):
        raise ValueError("Outer fold constraints did not pass")
    inner = read_csv(out / "tables" / "exp27lr1_inner_fold_balance.csv")
    if len(inner) != 15 or not all(row["constraints_pass"] == "True" for row in inner):
        raise ValueError("Inner fold constraints did not pass")
    sensitivity = read_csv(out / "tables" / "exp27lr1_split_seed_sensitivity.csv")
    if {row["seed"] for row in sensitivity} != {"42", "43", "44"} or not all(row["constraints_pass"] == "True" for row in sensitivity):
        raise ValueError("42/43/44 split sensitivity did not pass")

    ap = read_csv(out / "tables" / "exp27lr1_ap_tie_invariance_test.csv")
    if not ap or not all(row["pass_lt_1e12"] == "True" for row in ap):
        raise ValueError("Tie-safe AUPRC permutation-invariance test failed")
    rounding = {row["method"]: row for row in read_csv(out / "tables" / "exp27lr1_rounding_regression_audit.csv")}
    if abs(float(rounding["Exp27K_teacher_mean_half_up"]["weighted_MAE"]) - float(rounding["Exp27L-R1_corrected_teacher_mean"]["weighted_MAE"])) > 1e-12:
        raise ValueError("Half-up teacher mean does not match Exp27K reference")
    scores = read_csv(out / "data" / "exp27lr1_oof_score_predictions.csv")
    risks = read_csv(out / "data" / "exp27lr1_oof_risk_predictions.csv")
    if len(scores) != 180 or len(risks) != 180 or {row["sample_id"] for row in scores} != {row["sample_id"] for row in risks}:
        raise ValueError("OOF score/risk rows must cover and align to 180 samples")
    selected = read_csv(out / "tables" / "exp27lr1_selected_parameters.csv")
    if any(row["risk_enriched_fit_rows"] != "0" for row in selected):
        raise ValueError("Risk-enriched samples entered a fit")
    leakage = {row["check"]: int(row["count"]) for row in read_csv(out / "tables" / "exp27lr1_review_leakage_audit.csv")}
    for key in ("dev_sample_overlap", "dev_question_overlap", "test_sample_overlap", "test_question_overlap", "dev_label_used", "test_label_read", "teacher_api_calls", "gpu_training_runs"):
        if leakage.get(key) != 0:
            raise ValueError(f"Exp27L-R1 leakage/protocol check failed: {key}={leakage.get(key)}")

    packet_paths = [
        out / "packets" / "exp27lr1_balanced_silver_validation_60.jsonl",
        out / "packets" / "exp27lr1_external_high_control_34.jsonl",
        out / "packets" / "exp27lr1_targeted_ambiguity_20.jsonl",
    ]
    expected_counts = (60, 34, 20)
    forbidden = {"original_score", "qwen_score", "deepseek_score", "silver_final_score", "risk_probability", "exp27i_v1_tier"}
    for path, expected in zip(packet_paths, expected_counts):
        packets = read_jsonl(path)
        if len(packets) != expected or any(forbidden & set(packet) for packet in packets):
            raise ValueError(f"Invalid blind packet boundary: {path.name}")
        if any("<CONTEXT_ONLY_ORIGINAL_TASK>" not in packet["messages"][1]["content"] or "<EVALUATOR_OUTPUT_TO_SCORE>" not in packet["messages"][1]["content"] for packet in packets):
            raise ValueError(f"Missing target-aware packet isolation: {path.name}")
    distribution = read_csv(out / "tables" / "exp27lr1_review_set_distribution.csv")
    balanced = {row["original_score_stratum"]: int(row["selected_rows"]) for row in distribution if row["review_set"] == "balanced_silver_validation"}
    if balanced != {"low_1_2": 20, "mid_3": 20, "high_4_5": 20}:
        raise ValueError(f"Balanced silver-validation view is not 20/20/20: {balanced}")

    for path in (out / "decision" / "exp27lr1_balanced_crossfit_decision.json", out / "decision" / "exp27lr1_blind_review_prepare_decision.json"):
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("proceed_to_full_3326_calibrated_dataset") is not False or decision.get("proceed_to_qwen3_reranker_training") is not False:
            raise ValueError(f"Training gate incorrectly open: {path.name}")
    return {
        "status": "PASS",
        "outer_question_key_leakage": 0,
        "oof_rows": 180,
        "outer_folds": 5,
        "inner_folds": 15,
        "ap_tie_invariant": True,
        "training_gate": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
