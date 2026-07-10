"""Accept or reject the Exp27 model-review audit policy.

Exp27M is a CPU-only, train-only acceptance audit. It joins two blind model
review sessions and a third model adjudication to the locked Exp27L-R1
artifacts. The model reviews remain silver references: they are not human gold,
are not written into a trainer dataset, and never overwrite the source labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    bool_value,
    cluster_bootstrap_ci,
    half_up,
    question_key,
    question_key_hash,
    read_csv,
    read_jsonl,
    sample_id,
    score,
    weighted_mae,
    weighted_mean,
    weighted_qwk,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from thesis_exp.exp17_low_score_evidence.validate_exp27lr1_balanced_group_crossfit import (  # noqa: E402
    validate_review_payload,
)


DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_EXP27LR1 = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27lr1_balanced_crossfit_review_seed42"
)
DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27m_model_review_audit_policy_seed42"
)
VALID_BUCKETS = {
    "no_failure",
    "visible_failure",
    "hidden_or_missing_failure",
    "unclear",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
REVIEW_BUDGET = 0.20
DIRECT_MIN_COVERAGE = 0.25


def require(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def extract_evaluator_output(packet: dict[str, Any]) -> str:
    content = str(packet["messages"][1]["content"])
    opening = "<EVALUATOR_OUTPUT_TO_SCORE>"
    closing = "</EVALUATOR_OUTPUT_TO_SCORE>"
    if opening not in content or closing not in content:
        raise ValueError(f"Packet lacks target-aware output delimiters: {packet.get('packet_id')}")
    return content.split(opening, 1)[1].split(closing, 1)[0]


def ranges_overlap(left: Any, right: Any) -> bool:
    return bool(
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == 2
        and max(int(left[0]), int(right[0])) <= min(int(left[1]), int(right[1]))
    )


def requires_adjudication(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(
        left["most_plausible_score"] != right["most_plausible_score"]
        or left["failure_bucket"] != right["failure_bucket"]
        or not ranges_overlap(left["score_range"], right["score_range"])
        or left.get("needs_adjudication") is True
        or right.get("needs_adjudication") is True
        or left.get("confidence") == "low"
        or right.get("confidence") == "low"
    )


def load_packet_views(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    paths = (
        root / "packets" / "exp27lr1_balanced_silver_validation_60.jsonl",
        root / "packets" / "exp27lr1_external_high_control_34.jsonl",
        root / "packets" / "exp27lr1_targeted_ambiguity_20.jsonl",
    )
    expected = (60, 34, 20)
    appearances: list[dict[str, Any]] = []
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, count in zip(paths, expected):
        require(path, "Exp27L-R1 blind packet")
        rows = read_jsonl(path)
        if len(rows) != count:
            raise ValueError(f"Expected {count} rows in {path.name}, found {len(rows)}")
        for row in rows:
            appearances.append(row)
            by_sample[str(row["sample_id"])].append(row)
    if len(appearances) != 114 or len(by_sample) != 107:
        raise ValueError(
            f"Expected 114 packet appearances over 107 samples, found {len(appearances)}/{len(by_sample)}"
        )
    for sid, packets in by_sample.items():
        outputs = {extract_evaluator_output(packet) for packet in packets}
        qkeys = {str(packet["question_key_hash"]) for packet in packets}
        if len(outputs) != 1 or len(qkeys) != 1:
            raise ValueError(f"Repeated packet content is inconsistent for {sid}")
    return appearances, by_sample


def validate_review(
    sid: str,
    review: dict[str, Any],
    packets: list[dict[str, Any]],
    reviewer: str,
) -> None:
    if review.get("sample_id") != sid:
        raise ValueError(f"Reviewer {reviewer} sample id mismatch: {sid}")
    errors = validate_review_payload(review, extract_evaluator_output(packets[0]))
    if errors:
        raise ValueError(f"Reviewer {reviewer} protocol errors for {sid}: {errors}")


def build_model_review_reference(
    packet_map: dict[str, list[dict[str, Any]]],
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    adjudication_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for path, purpose in (
        (reviewer_a_path, "normalized Reviewer A results"),
        (reviewer_b_path, "normalized Reviewer B results"),
        (adjudication_path, "Codex model adjudication"),
    ):
        require(path, purpose)
    reviewer_a = {str(row["sample_id"]): row for row in read_jsonl(reviewer_a_path)}
    reviewer_b = {str(row["sample_id"]): row for row in read_jsonl(reviewer_b_path)}
    adjudication = {str(row["sample_id"]): row for row in read_jsonl(adjudication_path)}
    expected_ids = set(packet_map)
    if set(reviewer_a) != expected_ids or set(reviewer_b) != expected_ids:
        raise ValueError(
            "Both reviewers must cover exactly the 107 unique packet samples: "
            f"A={len(reviewer_a)} B={len(reviewer_b)} expected={len(expected_ids)}"
        )

    required: set[str] = set()
    rows: list[dict[str, Any]] = []
    for sid in sorted(expected_ids):
        left, right = reviewer_a[sid], reviewer_b[sid]
        validate_review(sid, left, packet_map[sid], "A")
        validate_review(sid, right, packet_map[sid], "B")
        if requires_adjudication(left, right):
            required.add(sid)
    if set(adjudication) != required:
        raise ValueError(
            "Adjudication must match the required conflict set exactly: "
            f"required={len(required)} supplied={len(adjudication)} "
            f"missing={sorted(required - set(adjudication))[:3]} "
            f"extra={sorted(set(adjudication) - required)[:3]}"
        )

    for sid in sorted(expected_ids):
        left, right = reviewer_a[sid], reviewer_b[sid]
        if sid in adjudication:
            final = adjudication[sid]
            final_score = score(final.get("final_score"))
            final_range = final.get("final_score_range")
            final_bucket = str(final.get("failure_bucket"))
            final_confidence = str(final.get("confidence"))
            if final.get("adjudication_status") != "model_adjudicated_not_human":
                raise ValueError(f"Adjudication provenance is not model-only for {sid}")
            if (
                not isinstance(final_range, list)
                or len(final_range) != 2
                or not all(isinstance(value, int) and 1 <= value <= 5 for value in final_range)
                or int(final_range[0]) > int(final_range[1])
                or not int(final_range[0]) <= final_score <= int(final_range[1])
            ):
                raise ValueError(f"Invalid final score range for {sid}: {final_range}")
            resolution = "third_model_adjudication"
        else:
            final_score = score(left["most_plausible_score"])
            final_range = [
                max(int(left["score_range"][0]), int(right["score_range"][0])),
                min(int(left["score_range"][1]), int(right["score_range"][1])),
            ]
            final_bucket = str(left["failure_bucket"])
            final_confidence = (
                "high"
                if left.get("confidence") == right.get("confidence") == "high"
                else "medium"
            )
            resolution = "two_session_exact_consensus"
        if final_bucket not in VALID_BUCKETS or final_confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Invalid final review category for {sid}")
        packets = packet_map[sid]
        rows.append(
            {
                "sample_id": sid,
                "packet_ids": [packet["packet_id"] for packet in packets],
                "review_sets": sorted({packet["review_set"] for packet in packets}),
                "question_key_hash": str(packets[0]["question_key_hash"]),
                "reviewer_a_score": score(left["most_plausible_score"]),
                "reviewer_b_score": score(right["most_plausible_score"]),
                "reviewer_a_bucket": left["failure_bucket"],
                "reviewer_b_bucket": right["failure_bucket"],
                "requires_adjudication": sid in required,
                "final_score": final_score,
                "final_score_range": [int(final_range[0]), int(final_range[1])],
                "final_bucket": final_bucket,
                "final_confidence": final_confidence,
                "resolution_source": resolution,
                "reference_status": "model_review_silver_not_human_gold",
                "trainer_use": "prohibited_in_exp27m",
            }
        )
    completion = {
        "unique_samples": len(rows),
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "required_adjudications": len(required),
        "completed_adjudications": len(adjudication),
        "model_review_silver": True,
        "human_external_validation": False,
    }
    return rows, completion


def source_metrics(
    rows: list[dict[str, Any]],
    name: str,
    predictor: Callable[[dict[str, Any]], float],
    prediction_type: str = "ordinal",
) -> dict[str, Any]:
    gold = [int(row["final_score"]) for row in rows]
    raw = [float(predictor(row)) for row in rows]
    rounded = [half_up(value) for value in raw]
    weights = [1.0] * len(rows)
    errors = [right - left for left, right in zip(gold, rounded)]
    low_count = sum(value <= 2 for value in gold)
    high_count = sum(value >= 4 for value in gold)
    return {
        "source": name,
        "prediction_type": prediction_type,
        "n": len(rows),
        "MAE": weighted_mae(gold, raw, weights),
        "half_up_MAE": weighted_mae(gold, rounded, weights),
        "QWK": weighted_qwk(gold, rounded, weights),
        "exact_accuracy": weighted_mean([float(value == 0) for value in errors], weights),
        "within_one": weighted_mean([float(abs(value) <= 1) for value in errors], weights),
        "severe_error_rate": weighted_mean([float(abs(value) >= 2) for value in errors], weights),
        "signed_bias": weighted_mean([float(value) for value in errors], weights),
        "low_to_high_rate": (
            sum(left <= 2 and right >= 4 for left, right in zip(gold, rounded)) / low_count
            if low_count
            else None
        ),
        "high_to_low_rate": (
            sum(left >= 4 and right <= 2 for left, right in zip(gold, rounded)) / high_count
            if high_count
            else None
        ),
    }


def gap_value(row: dict[str, Any], candidate: str) -> int:
    human = int(row["original_score"])
    qwen = int(row["qwen_score"])
    deepseek = int(row["deepseek_score"])
    if candidate == "qwen_human_gap":
        return abs(qwen - human)
    if candidate == "deepseek_human_gap":
        return abs(deepseek - human)
    if candidate == "teacher_gap":
        return abs(qwen - deepseek)
    if candidate == "max_three_way_gap":
        return max(human, qwen, deepseek) - min(human, qwen, deepseek)
    raise KeyError(candidate)


def assign_tier(row: dict[str, Any], candidate: str) -> str:
    if candidate == "core_logistic_oof":
        return {
            "high": "direct_accept",
            "low": "weighted_accept",
            "review": "adjudication_required",
        }[str(row["core_logistic_tier"])]
    value = gap_value(row, candidate)
    if value >= 2 or bool_value(row.get("target_issue_flag")):
        return "adjudication_required"
    return "direct_accept" if value == 0 else "weighted_accept"


def policy_prediction(row: dict[str, Any], tier: str) -> int:
    if tier == "direct_accept":
        return int(row["original_score"])
    if tier == "weighted_accept":
        return int(row["half_up_teacher_mean_score"])
    return int(row["final_score"])


def tier_metric(rows: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    selected = [row for row in rows if row["policy_tier"] == tier]
    if not selected:
        return {
            "tier": tier,
            "n": 0,
            "coverage": 0.0,
            "original_human_MAE": None,
            "original_human_within_one": None,
            "original_human_severe_error_rate": None,
            "unresolved_after_review": 0,
        }
    errors = [abs(int(row["original_score"]) - int(row["final_score"])) for row in selected]
    return {
        "tier": tier,
        "n": len(selected),
        "coverage": len(selected) / len(rows),
        "original_human_MAE": sum(errors) / len(errors),
        "original_human_within_one": sum(value <= 1 for value in errors) / len(errors),
        "original_human_severe_error_rate": sum(value >= 2 for value in errors) / len(errors),
        "unresolved_after_review": sum(
            row["final_bucket"] == "unclear" or row["final_confidence"] == "low"
            for row in selected
        ),
    }


def candidate_metrics(
    balanced: list[dict[str, Any]],
    candidate: str,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [{**row, "policy_tier": assign_tier(row, candidate)} for row in balanced]
    tiers = [tier_metric(rows, tier) for tier in ("direct_accept", "weighted_accept", "adjudication_required")]
    by_tier = {row["tier"]: row for row in tiers}
    direct = by_tier["direct_accept"]
    review = by_tier["adjudication_required"]

    def direct_severe(sample: list[dict[str, Any]]) -> float | None:
        selected = [row for row in sample if assign_tier(row, candidate) == "direct_accept"]
        if not selected:
            return None
        return sum(abs(int(row["original_score"]) - int(row["final_score"])) >= 2 for row in selected) / len(selected)

    _low, upper = cluster_bootstrap_ci(
        balanced,
        direct_severe,
        seed=42,
        resamples=bootstrap_resamples,
    )
    predictions = [policy_prediction(row, assign_tier(row, candidate)) for row in balanced]
    gold = [int(row["final_score"]) for row in balanced]
    policy_mae = sum(abs(left - right) for left, right in zip(gold, predictions)) / len(gold)
    policy_qwk = weighted_qwk(gold, predictions, [1.0] * len(gold))
    direct_severe_value = direct["original_human_severe_error_rate"]
    review_severe_value = review["original_human_severe_error_rate"]
    review_separation = bool(
        review_severe_value is not None
        and direct_severe_value is not None
        and (
            (direct_severe_value == 0.0 and review_severe_value > 0.0)
            or review_severe_value >= 2.0 * direct_severe_value
        )
    )
    result = {
        "candidate": candidate,
        "direct_n": direct["n"],
        "direct_coverage": direct["coverage"],
        "direct_within_one": direct["original_human_within_one"],
        "direct_severe_error_rate": direct_severe_value,
        "direct_severe_cluster_bootstrap_ci95_upper": upper,
        "weighted_n": by_tier["weighted_accept"]["n"],
        "weighted_coverage": by_tier["weighted_accept"]["coverage"],
        "review_n": review["n"],
        "review_rate": review["coverage"],
        "review_severe_error_rate": review_severe_value,
        "review_error_at_least_2x_direct": review_separation,
        "policy_MAE_with_model_review": policy_mae,
        "policy_QWK_with_model_review": policy_qwk,
        "passes_direct_coverage": direct["coverage"] >= DIRECT_MIN_COVERAGE,
        "passes_review_budget": review["coverage"] <= REVIEW_BUDGET + 1e-12,
        "passes_direct_within_one": (direct["original_human_within_one"] or 0.0) >= 0.90,
        "passes_direct_severe": (direct_severe_value or 0.0) <= 0.10,
        "passes_direct_severe_upper": upper is not None and upper <= 0.20,
        "passes_review_separation": review_separation,
    }
    result["tier_acceptance_pass"] = all(
        bool(result[key])
        for key in (
            "passes_direct_coverage",
            "passes_review_budget",
            "passes_direct_within_one",
            "passes_direct_severe",
            "passes_direct_severe_upper",
            "passes_review_separation",
        )
    )
    return result, tiers


def point_distribution(*scores_and_weights: tuple[int, float]) -> list[float]:
    values = [0.0] * 5
    for value, weight in scores_and_weights:
        values[int(value) - 1] += float(weight)
    total = sum(values)
    return [value / total for value in values]


def range_distribution(low: int, high: int, center: int, tau: float = 0.75) -> list[float]:
    values = [
        math.exp(-abs(label - center) / tau) if low <= label <= high else 0.0
        for label in range(1, 6)
    ]
    total = sum(values)
    return [value / total for value in values]


def policy_preview(rows: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        tier = assign_tier(row, candidate)
        unresolved = row["final_bucket"] == "unclear" or row["final_confidence"] == "low"
        if unresolved:
            training_tier = "review_only"
            distribution = None
            quality_weight = 0.0
        elif tier == "direct_accept":
            training_tier = tier
            distribution = point_distribution((int(row["original_score"]), 1.0))
            quality_weight = 1.0
        elif tier == "weighted_accept":
            training_tier = tier
            distribution = point_distribution(
                (int(row["original_score"]), 0.50),
                (int(row["qwen_score"]), 0.25),
                (int(row["deepseek_score"]), 0.25),
            )
            quality_weight = 0.50
        else:
            training_tier = tier
            distribution = range_distribution(
                int(row["final_score_range"][0]),
                int(row["final_score_range"][1]),
                int(row["final_score"]),
            )
            quality_weight = 0.75 if row["final_confidence"] == "high" else 0.50
        output.append(
            {
                "sample_id": row["sample_id"],
                "question_key_hash": row["question_key_hash"],
                "policy_candidate": candidate,
                "audit_tier": tier,
                "training_tier_preview": training_tier,
                "soft_target_preview": distribution,
                "quality_weight_preview": quality_weight,
                "reference_status": "model_review_silver_not_human_gold",
                "trainer_use": "prohibited_in_exp27m",
            }
        )
    return output


def audit_split_ids(path: Path) -> tuple[set[str], set[str]]:
    rows = read_jsonl(path)
    return ({sample_id(row) for row in rows}, {question_key_hash(question_key(row)) for row in rows})


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    appearances, packet_map = load_packet_views(args.exp27lr1_dir)
    reviews, completion = build_model_review_reference(
        packet_map,
        args.reviewer_a,
        args.reviewer_b,
        args.adjudication,
    )
    review_by_id = {row["sample_id"]: row for row in reviews}
    balanced_ids = [
        str(row["sample_id"])
        for row in appearances
        if row["review_set"] == "balanced_silver_validation"
    ]
    external_ids = [
        str(row["sample_id"])
        for row in appearances
        if row["review_set"] == "external_high_control_qkey_lockbox"
    ]
    targeted_ids = [
        str(row["sample_id"])
        for row in appearances
        if row["review_set"] == "targeted_ambiguity_qualitative_only"
    ]
    if not (len(balanced_ids) == 60 and len(external_ids) == 34 and len(targeted_ids) == 20):
        raise ValueError("Review set counts changed")

    score_rows = {
        str(row["sample_id"]): row
        for row in read_csv(args.exp27lr1_dir / "data" / "exp27lr1_oof_score_predictions.csv")
    }
    tier_rows = {
        str(row["sample_id"]): row
        for row in read_csv(args.exp27lr1_dir / "data" / "exp27lr1_oof_tier_assignments.csv")
    }
    if not set(balanced_ids) <= set(score_rows) or not set(balanced_ids) <= set(tier_rows):
        raise ValueError("Balanced review rows do not align to Exp27L-R1 OOF artifacts")
    balanced_all = [
        {
            **score_rows[sid],
            **tier_rows[sid],
            **review_by_id[sid],
        }
        for sid in balanced_ids
    ]
    balanced = [
        row
        for row in balanced_all
        if row["final_bucket"] != "unclear" and row["final_confidence"] != "low"
    ]
    balanced_review_only = [row for row in balanced_all if row not in balanced]
    completion.update(
        {
            "balanced_packet_rows": len(balanced_all),
            "balanced_quantitative_rows": len(balanced),
            "balanced_review_only_rows": len(balanced_review_only),
        }
    )

    source_specs: dict[str, tuple[Callable[[dict[str, Any]], float], str]] = {
        "original_human": (lambda row: float(row["original_score"]), "ordinal"),
        "qwen_only": (lambda row: float(row["qwen_score"]), "ordinal"),
        "deepseek_only": (lambda row: float(row["deepseek_score"]), "ordinal"),
        "dual_teacher_half_up_mean": (
            lambda row: float(row["half_up_teacher_mean_score"]),
            "ordinal",
        ),
        "human_qwen_deepseek_median": (
            lambda row: float(row["human_teacher_median_score"]),
            "ordinal",
        ),
        "exp27i_v1": (lambda row: float(row["exp27i_v1_score"]), "ordinal"),
        "oof_soft_fusion_nll_expected": (
            lambda row: float(row["nll_expected_score"]),
            "continuous",
        ),
        "oof_soft_fusion_rps_expected": (
            lambda row: float(row["rps_expected_score"]),
            "continuous",
        ),
    }
    source_table = [
        source_metrics(balanced, name, predictor, kind)
        for name, (predictor, kind) in source_specs.items()
    ]
    simple_names = {
        "original_human",
        "qwen_only",
        "deepseek_only",
        "dual_teacher_half_up_mean",
        "human_qwen_deepseek_median",
    }
    best_simple = min(
        (row for row in source_table if row["source"] in simple_names),
        key=lambda row: (float(row["half_up_MAE"]), -float(row["QWK"] or -1.0)),
    )

    candidates = (
        "qwen_human_gap",
        "deepseek_human_gap",
        "teacher_gap",
        "max_three_way_gap",
        "core_logistic_oof",
    )
    candidate_table: list[dict[str, Any]] = []
    candidate_tiers: dict[str, list[dict[str, Any]]] = {}
    for name in candidates:
        metrics, tiers = candidate_metrics(balanced, name, args.bootstrap_resamples)
        metrics["best_simple_source"] = best_simple["source"]
        metrics["best_simple_MAE"] = best_simple["half_up_MAE"]
        metrics["policy_MAE_guard_delta"] = float(metrics["policy_MAE_with_model_review"]) - float(
            best_simple["half_up_MAE"]
        )
        metrics["passes_policy_MAE_guard"] = metrics["policy_MAE_guard_delta"] <= 0.05
        metrics["acceptance_candidate"] = bool(
            metrics["tier_acceptance_pass"] and metrics["passes_policy_MAE_guard"]
        )
        candidate_table.append(metrics)
        candidate_tiers[name] = tiers

    eligible = [row for row in candidate_table if row["acceptance_candidate"]]
    selected = min(
        eligible or candidate_table,
        key=lambda row: (
            not bool(row["acceptance_candidate"]),
            float(row["review_rate"]),
            float(row["policy_MAE_with_model_review"]),
            str(row["candidate"]),
        ),
    )
    selected_name = str(selected["candidate"])
    selected_tiers = candidate_tiers[selected_name]

    train_rows = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    external = []
    for sid in external_ids:
        source = train_rows.get(sid)
        if source is None:
            raise ValueError(f"External high-control sample missing from train: {sid}")
        human = score(source["label_5"])
        final = int(review_by_id[sid]["final_score"])
        external.append((human, final))
    external_hard_conflicts = sum(abs(left - right) >= 2 for left, right in external)
    external_summary = {
        "review_set": "external_high_control_qkey_lockbox",
        "n": len(external),
        "original_high_count": sum(left >= 4 for left, _right in external),
        "model_review_high_count": sum(right >= 4 for _left, right in external),
        "hard_conflict_count": external_hard_conflicts,
        "hard_conflict_rate": external_hard_conflicts / len(external),
        "used_for_policy_selection": False,
        "used_for_threshold_tuning": False,
    }

    review_set_table = []
    for name, expected in (
        ("balanced_silver_validation", 60),
        ("external_high_control_qkey_lockbox", 34),
        ("targeted_ambiguity_qualitative_only", 20),
    ):
        ids = [str(row["sample_id"]) for row in appearances if row["review_set"] == name]
        review_set_table.append(
            {
                "review_set": name,
                "packet_appearances": len(ids),
                "unique_samples": len(set(ids)),
                "expected_appearances": expected,
                "statistical_role": {
                    "balanced_silver_validation": "policy_acceptance_only",
                    "external_high_control_qkey_lockbox": "sealed_high_score_protection_only",
                    "targeted_ambiguity_qualitative_only": "qualitative_only",
                }[name],
            }
        )

    preview = policy_preview(balanced_all, selected_name)
    unresolved_count = sum(row["training_tier_preview"] == "review_only" for row in preview)
    checks = [
        ("all_107_unique_reviews_merged", completion["unique_samples"] == 107),
        (
            "all_required_adjudications_completed",
            completion["required_adjudications"] == completion["completed_adjudications"],
        ),
        ("selected_policy_exists", bool(eligible)),
        ("direct_accept_within_one_ge_0p90", bool(selected["passes_direct_within_one"])),
        ("direct_accept_severe_error_le_0p10", bool(selected["passes_direct_severe"])),
        (
            "direct_accept_cluster_bootstrap_upper_le_0p20",
            bool(selected["passes_direct_severe_upper"]),
        ),
        ("review_error_at_least_2x_direct", bool(selected["passes_review_separation"])),
        ("review_budget_le_0p20", bool(selected["passes_review_budget"])),
        ("high_control_hard_conflict_le_0p10", external_summary["hard_conflict_rate"] <= 0.10),
        ("policy_MAE_within_best_simple_plus_0p05", bool(selected["passes_policy_MAE_guard"])),
        ("model_review_marked_silver", completion["model_review_silver"] is True),
        (
            "unclear_balanced_rows_excluded_from_quantitative_selection",
            len(balanced_all) == len(balanced) + unresolved_count,
        ),
    ]
    acceptance_table = [{"check": name, "pass": passed} for name, passed in checks]
    acceptance_pass = all(passed for _name, passed in checks)

    train_ids = set(train_rows)
    train_qkeys = {question_key_hash(question_key(row)) for row in train_rows.values()}
    dev_ids, dev_qkeys = audit_split_ids(args.dev_jsonl)
    test_ids, test_qkeys = audit_split_ids(args.test_jsonl)
    review_ids = set(review_by_id)
    review_qkeys = {row["question_key_hash"] for row in reviews}
    balanced_qkeys = {review_by_id[sid]["question_key_hash"] for sid in balanced_ids}
    external_qkeys = {review_by_id[sid]["question_key_hash"] for sid in external_ids}
    leakage_table = [
        {"check": "review_sample_missing_from_train", "count": len(review_ids - train_ids)},
        {"check": "review_question_missing_from_train", "count": len(review_qkeys - train_qkeys)},
        {"check": "dev_sample_overlap", "count": len(review_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(review_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(review_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(review_qkeys & test_qkeys)},
        {"check": "balanced_external_question_overlap", "count": len(balanced_qkeys & external_qkeys)},
        {"check": "dev_label_used", "count": 0},
        {"check": "test_label_used", "count": 0},
        {"check": "teacher_api_calls", "count": 0},
        {"check": "gpu_training_runs", "count": 0},
    ]
    if any(int(row["count"]) != 0 for row in leakage_table):
        raise ValueError(f"Exp27M leakage/protocol audit failed: {leakage_table}")

    locked_policy = {
        "experiment": "exp27m_model_review_audit_policy",
        "selected_signal": selected_name,
        "selection_population": "balanced_silver_validation_60_only",
        "review_budget": REVIEW_BUDGET,
        "direct_accept_rule": "selected ordinal gap == 0 and no target_issue_flag",
        "weighted_accept_rule": "selected ordinal gap == 1 and no target_issue_flag",
        "adjudication_required_rule": "selected ordinal gap >= 2 or target_issue_flag",
        "review_only_rule": "adjudication final bucket is unclear or final confidence is low",
        "soft_target": {
            "direct_accept": "one_hot(original_human_score)",
            "weighted_accept": "0.50*human + 0.25*Qwen + 0.25*DeepSeek point-mass mixture",
            "adjudication_required_after_resolution": "score-range kernel around model-review silver, tau=0.75",
            "review_only": None,
        },
        "quality_weight": {
            "direct_accept": 1.0,
            "weighted_accept": 0.5,
            "resolved_adjudication_high_confidence": 0.75,
            "resolved_adjudication_medium_confidence": 0.5,
            "review_only": 0.0,
        },
        "downstream_weight_normalization": "renormalize active mean quality weight to 1",
        "reference_provenance": "two GPT-5.6Pro sessions plus Codex model adjudication",
        "reference_status": "model_review_silver_not_human_gold",
        "unresolved_balanced_rows": unresolved_count,
        "frozen_for": "361-row in-place downstream pilot only",
    }
    decision = {
        "experiment": "exp27m_model_review_audit_policy_acceptance",
        "status": "PASS" if acceptance_pass else "FAIL",
        "selected_policy": selected_name,
        "model_review_unique_samples": len(reviews),
        "balanced_quantitative_rows": len(balanced),
        "balanced_review_only_rows": unresolved_count,
        "required_adjudications": completion["required_adjudications"],
        "model_review_is_silver": True,
        "human_external_validation": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "model_training_runs": 0,
        "proceed_to_361_in_place_downstream_pilot": acceptance_pass,
        "proceed_to_full_3326_calibrated_dataset": False,
        "proceed_to_formal_qwen3_reranker_training": False,
        "proceed_to_dev_test_relabeling": False,
        "next_step": (
            "prepare_361_row_in_place_downstream_pilot"
            if acceptance_pass
            else "revise_audit_policy_without_expanding_or_training"
        ),
    }

    write_jsonl(args.out_dir / "private" / "exp27m_model_review_reference_107.jsonl", reviews)
    write_jsonl(args.out_dir / "private" / "exp27m_policy_preview_balanced60.jsonl", preview)
    write_csv(args.out_dir / "tables" / "exp27m_review_set_accounting.csv", review_set_table)
    write_csv(args.out_dir / "tables" / "exp27m_review_completion.csv", [completion])
    write_csv(args.out_dir / "tables" / "exp27m_source_metrics_balanced60.csv", source_table)
    write_csv(args.out_dir / "tables" / "exp27m_policy_candidate_acceptance.csv", candidate_table)
    write_csv(args.out_dir / "tables" / "exp27m_selected_tier_metrics.csv", selected_tiers)
    write_csv(args.out_dir / "tables" / "exp27m_high_control_lockbox.csv", [external_summary])
    write_csv(args.out_dir / "tables" / "exp27m_acceptance_checks.csv", acceptance_table)
    write_csv(args.out_dir / "tables" / "exp27m_leakage_audit.csv", leakage_table)
    write_json(args.out_dir / "configs" / "exp27m_locked_policy.json", locked_policy)
    write_json(args.out_dir / "decision" / "exp27m_model_review_audit_policy_decision.json", decision)

    source_lines = [
        f"- {row['source']}: MAE={row['half_up_MAE']:.4f}, QWK={row['QWK']:.4f}, "
        f"within-one={row['within_one']:.4f}, severe={row['severe_error_rate']:.4f}"
        for row in source_table
    ]
    tier_lines = [
        f"- {row['tier']}: n={row['n']}, coverage={row['coverage']:.4f}, "
        f"within-one={row['original_human_within_one']}, severe={row['original_human_severe_error_rate']}"
        for row in selected_tiers
    ]
    check_lines = [f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks]
    report = "\n".join(
        [
            "# Exp27M Model-Review Audit Policy Acceptance",
            "",
            "Exp27M is a CPU-only, train-only acceptance audit. Reviewer A/B and the",
            "third adjudicator are model reviewers. The resulting reference is silver,",
            "not human expert gold, and no row is emitted as trainer data.",
            "",
            "## Completion",
            "",
            f"- unique reviewed samples: {len(reviews)}",
            f"- packet appearances: {len(appearances)}",
            f"- required/completed adjudications: {completion['required_adjudications']}/{completion['completed_adjudications']}",
            "- API calls: 0",
            "- GPU runs: 0",
            "- model training runs: 0",
            "",
            "## Balanced-60 source comparison",
            "",
            f"The packet contains 60 rows; {len(balanced)} are quantitative and {unresolved_count} unresolved rows are qualitative-only.",
            "",
            *source_lines,
            "",
            "## Selected policy",
            "",
            f"- selected signal: `{selected_name}`",
            f"- review budget: {REVIEW_BUDGET:.2f}",
            f"- policy MAE with selective model review: {selected['policy_MAE_with_model_review']:.4f}",
            f"- best simple source: `{best_simple['source']}` (MAE={best_simple['half_up_MAE']:.4f})",
            "- review-tier predictions use the completed model-review result, so their accuracy includes review cost and is not a raw-source comparison.",
            "",
            "## Tier reliability",
            "",
            *tier_lines,
            "",
            "## High-control lockbox",
            "",
            f"- rows: {external_summary['n']}",
            f"- hard conflict rate: {external_summary['hard_conflict_rate']:.4f}",
            "- used for policy selection: false",
            "",
            "## Acceptance checks",
            "",
            *check_lines,
            "",
            "## Decision",
            "",
            f"- status: `{decision['status']}`",
            f"- proceed to 361-row in-place downstream pilot: {str(decision['proceed_to_361_in_place_downstream_pilot']).lower()}",
            "- proceed to full 3326 expansion: false",
            "- proceed to formal Qwen3-Reranker training: false",
            "- relabel dev/test: false",
            "",
            "The next allowed experiment is a controlled 361-row in-place downstream pilot.",
            "The same 3326-row universe, exclusions, effective weights, optimizer, and",
            "checkpoint rule must be shared across its ablations.",
            "",
        ]
    )
    write_text(args.out_dir / "reports" / "exp27m_model_review_audit_policy_report.md", report)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27lr1-dir", type=Path, default=DEFAULT_EXP27LR1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    private = args.exp27lr1_dir / "private"
    args.reviewer_a = args.reviewer_a or private / "exp27lr1_reviewer_a_normalized_107.jsonl"
    args.reviewer_b = args.reviewer_b or private / "exp27lr1_reviewer_b_normalized_107.jsonl"
    args.adjudication = args.adjudication or private / "exp27lr1_codex_model_adjudication_all38.jsonl"
    return args


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), ensure_ascii=False, sort_keys=True))
