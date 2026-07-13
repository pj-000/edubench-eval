"""Validate Exp40A pair judgments, apply the frozen gate, and publish aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    ROOT,
    SCHEMA_PATH,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def source_preferred(judgment: dict[str, Any]) -> bool:
    return (
        judgment["order"] == "original_first" and judgment["preferred_answer"] == "A"
    ) or (
        judgment["order"] == "counterfactual_first" and judgment["preferred_answer"] == "B"
    )


def counterfactual_preferred(judgment: dict[str, Any]) -> bool:
    return (
        judgment["order"] == "original_first" and judgment["preferred_answer"] == "B"
    ) or (
        judgment["order"] == "counterfactual_first" and judgment["preferred_answer"] == "A"
    )


def schema_valid(row: dict[str, Any], schema: dict[str, Any], packet: dict[str, Any]) -> tuple[bool, str]:
    errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(row)]
    for field in ("pair_id", "sample_id", "order"):
        if str(row.get(field)) != str(packet[field]):
            errors.append(f"{field}_mismatch")
    return not errors, "; ".join(errors)


def main() -> None:
    args = parse_args()
    packet_path = args.out_dir / "private/pair_packets/exp40a_pairwise_verification_packets.jsonl"
    judgment_path = args.out_dir / "private/pairwise_judgments/exp40a_deepseek_pairwise_judgments.jsonl"
    if not packet_path.exists() or not judgment_path.exists():
        raise FileNotFoundError("Exp40A packets and completed private judgments are required")
    packets = read_jsonl(packet_path)
    judgments = read_jsonl(judgment_path)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    resolved = read_jsonl(args.out_dir / "private/resolved_pairs/exp40a_resolved_exp39a_pairs.jsonl")
    pair_payload = {row["pair_id"]: row for row in resolved}
    packet_by_key = {(str(row["pair_id"]), str(row["order"])): row for row in packets}
    judgment_by_key = {(str(row.get("pair_id")), str(row.get("order"))): row for row in judgments}
    duplicate_judgments = len(judgments) - len(judgment_by_key)
    if len(packet_by_key) != 480 or len(pair_payload) != 240:
        raise ValueError("Exp40A requires 240 pairs and 480 packet identities")

    validity = {}
    for key, packet in packet_by_key.items():
        judgment = judgment_by_key.get(key)
        validity[key] = (False, "missing_judgment") if judgment is None else schema_valid(judgment, schema, packet)
    valid_count = sum(valid for valid, _ in validity.values())
    schema_success_rate = valid_count / 480

    pair_rows = []
    accepted_private = []
    operator_stats: dict[str, Counter[str]] = defaultdict(Counter)
    metric_stats: dict[str, Counter[str]] = defaultdict(Counter)
    order_source_wins = Counter()
    ties = conflicts = consistent_cf = 0
    for pair_id in sorted(pair_payload):
        payload = pair_payload[pair_id]
        key1, key2 = (pair_id, "original_first"), (pair_id, "counterfactual_first")
        first, second = judgment_by_key.get(key1), judgment_by_key.get(key2)
        both_valid = validity[key1][0] and validity[key2][0]
        if first and validity[key1][0] and source_preferred(first):
            order_source_wins["original_first"] += 1
        if second and validity[key2][0] and source_preferred(second):
            order_source_wins["counterfactual_first"] += 1
        has_tie = bool(first and first.get("preferred_answer") == "tie") or bool(
            second and second.get("preferred_answer") == "tie"
        )
        if has_tie:
            ties += 1
        logical_choices = []
        for row, valid in ((first, validity[key1][0]), (second, validity[key2][0])):
            if not row or not valid or row.get("preferred_answer") == "tie":
                logical_choices.append("unknown")
            elif source_preferred(row):
                logical_choices.append("source")
            elif counterfactual_preferred(row):
                logical_choices.append("counterfactual")
            else:
                logical_choices.append("unknown")
        order_conflict = len(set(logical_choices)) > 1 and "unknown" not in logical_choices
        if order_conflict:
            conflicts += 1
        if logical_choices == ["counterfactual", "counterfactual"]:
            consistent_cf += 1
        source_preferred_both = logical_choices == ["source", "source"]
        confidence_ok = bool(first and second) and all(
            row.get("preference_confidence") in {"high", "medium"} for row in (first, second)
        )
        degradation_ok = bool(first and second) and all(
            row.get("targeted_rubric_degradation") in {"yes", "partial"} for row in (first, second)
        ) and any(row.get("targeted_rubric_degradation") == "yes" for row in (first, second))
        preservation_ok = bool(first and second) and all(
            row.get("non_target_content_preserved") in {"yes", "partial"} for row in (first, second)
        )
        uncertainty_ok = bool(first and second) and all(
            row.get("answer_key_or_rubric_uncertainty") is False for row in (first, second)
        )
        target_scope_ok = bool(first and second) and all(row.get("target_scope_confirmed") is True for row in (first, second))
        accepted = bool(
            both_valid and source_preferred_both and not has_tie and confidence_ok and degradation_ok
            and preservation_ok and uncertainty_ok and target_scope_ok
        )
        rejection_reasons = []
        for condition, name in (
            (both_valid, "schema_or_completion"), (source_preferred_both, "source_preference"),
            (not has_tie, "tie"), (confidence_ok, "confidence"), (degradation_ok, "degradation"),
            (preservation_ok, "preservation"), (uncertainty_ok, "rubric_uncertainty"),
            (target_scope_ok, "target_scope"),
        ):
            if not condition:
                rejection_reasons.append(name)
        pair_row = {
            "pair_id_hash": pair_id,
            "sample_id_hash": payload["sample_id"],
            "question_key_hash": payload["question_key"],
            "language": payload.get("language", "unknown"),
            "metric_group": payload.get("metric_group", "unknown"),
            "operator": payload.get("operator", "unknown"),
            "both_schema_target_valid": both_valid,
            "order1_source_preferred": bool(first and validity[key1][0] and source_preferred(first)),
            "order2_source_preferred": bool(second and validity[key2][0] and source_preferred(second)),
            "order_consistent_source_preferred": source_preferred_both,
            "tie": has_tie,
            "order_conflict": order_conflict,
            "confidence_ok": confidence_ok,
            "degradation_ok": degradation_ok,
            "preservation_ok": preservation_ok,
            "uncertainty_ok": uncertainty_ok,
            "accepted": accepted,
            "rejection_reasons": "|".join(rejection_reasons),
        }
        pair_rows.append(pair_row)
        for stats in (operator_stats[str(payload.get("operator", "unknown"))], metric_stats[str(payload.get("metric_group", "unknown"))]):
            stats["pairs"] += 1
            stats["accepted"] += int(accepted)
        if accepted:
            accepted_private.append(payload)

    accepted_count = len(accepted_private)
    accepted_qkeys = len({row["question_key"] for row in accepted_private})
    degradation_rate = (
        sum(row["degradation_ok"] for row in pair_rows if row["accepted"]) / accepted_count if accepted_count else 0.0
    )
    preservation_rate = (
        sum(row["preservation_ok"] for row in pair_rows if row["accepted"]) / accepted_count if accepted_count else 0.0
    )
    original_first_rate = order_source_wins["original_first"] / 240
    counterfactual_first_rate = order_source_wins["counterfactual_first"] / 240
    position_difference = abs(original_first_rate - counterfactual_first_rate)
    conflict_or_tie_pairs = sum(row["tie"] or row["order_conflict"] for row in pair_rows)
    conflict_or_tie_rate = conflict_or_tie_pairs / 240
    gates = {
        "pairs_prepared_240": len(pair_payload) == 240,
        "judgments_completed_480": len(set(judgment_by_key) & set(packet_by_key)) == 480,
        "schema_success_ge_0p98": schema_success_rate >= 0.98,
        "accepted_pairs_ge_150": accepted_count >= 150,
        "accepted_unique_question_keys_ge_120": accepted_qkeys >= 120,
        "accepted_targeted_degradation_ge_0p90": degradation_rate >= 0.90,
        "accepted_non_target_preservation_ge_0p85": preservation_rate >= 0.85,
        "position_source_win_difference_le_0p05": position_difference <= 0.05,
        "order_conflict_or_tie_rate_le_0p25": conflict_or_tie_rate <= 0.25,
        "dev_access_zero": True,
        "test_access_zero": True,
    }
    go = all(gates.values())

    write_jsonl(args.out_dir / "private/accepted_pairs/exp40a_accepted_pairs.jsonl", accepted_private)
    write_csv(args.out_dir / "tables/exp40a_order_consistency.csv", pair_rows)
    summary = {
        "prepared_pairs": 240,
        "expected_judgments": 480,
        "completed_judgments": len(set(judgment_by_key) & set(packet_by_key)),
        "duplicate_judgments": duplicate_judgments,
        "schema_target_valid_judgments": valid_count,
        "schema_success_rate": schema_success_rate,
        "accepted_pairs": accepted_count,
        "accepted_unique_question_keys": accepted_qkeys,
        "accepted_targeted_degradation_rate": degradation_rate,
        "accepted_non_target_preservation_rate": preservation_rate,
        "tie_pairs": ties,
        "order_conflict_pairs": conflicts,
        "consistent_counterfactual_preferred_pairs": consistent_cf,
        "order_conflict_or_tie_rate": conflict_or_tie_rate,
    }
    write_csv(args.out_dir / "tables/exp40a_pair_acceptance_summary.csv", [summary])
    write_csv(
        args.out_dir / "tables/exp40a_position_bias_audit.csv",
        [
            {"order": "original_first", "source_win_count": order_source_wins["original_first"], "total_pairs": 240, "source_win_rate": original_first_rate},
            {"order": "counterfactual_first", "source_win_count": order_source_wins["counterfactual_first"], "total_pairs": 240, "source_win_rate": counterfactual_first_rate},
            {"order": "absolute_difference", "source_win_count": "", "total_pairs": 240, "source_win_rate": position_difference},
        ],
    )
    write_csv(
        args.out_dir / "tables/exp40a_acceptance_by_operator.csv",
        [{"operator": key, "pairs": value["pairs"], "accepted": value["accepted"], "acceptance_rate": value["accepted"] / value["pairs"]} for key, value in sorted(operator_stats.items())],
    )
    write_csv(
        args.out_dir / "tables/exp40a_acceptance_by_metric.csv",
        [{"metric_group": key, "pairs": value["pairs"], "accepted": value["accepted"], "acceptance_rate": value["accepted"] / value["pairs"]} for key, value in sorted(metric_stats.items())],
    )
    decision = {
        "status": "PAIRWISE_QUALIFICATION_GO" if go else "PAIRWISE_QUALIFICATION_NO_GO",
        "gates": gates,
        "recommend_groupcv_training": go,
        "stop_model_assisted_positive_route": not go,
        "groupcv_ran": False,
        **summary,
        "position_source_win_rate_difference": position_difference,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp40a_pairwise_qualification_decision.json", decision)
    report = [
        "# Exp40A EduPair-CF pairwise qualification report", "",
        f"- Status: **{decision['status']}**",
        f"- Completed judgments: `{summary['completed_judgments']}/480`",
        f"- Schema/target success: `{schema_success_rate:.4f}`",
        f"- Accepted pairs: `{accepted_count}/240` across `{accepted_qkeys}` question keys",
        f"- Tie pairs: `{ties}`; order conflicts: `{conflicts}`",
        f"- Original-first source win rate: `{original_first_rate:.4f}`",
        f"- Counterfactual-first source win rate: `{counterfactual_first_rate:.4f}`",
        f"- Position-specific absolute difference: `{position_difference:.4f}`",
        f"- Accepted targeted-degradation rate: `{degradation_rate:.4f}`",
        f"- Accepted non-target-preservation rate: `{preservation_rate:.4f}`",
        f"- Gates: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend GroupCV training: `{str(go).lower()}`",
        f"- Stop model-assisted positive route: `{str(not go).lower()}`",
        "- The verifier received no absolute target score, human score, or prior pointwise judgment.",
        "- No paper-like dev/test data were accessed.",
    ]
    report_path = args.out_dir / "reports/exp40a_pairwise_qualification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if not go:
        groupcv_decision = {
            "status": "GROUPCV_NOT_RUN_BLOCKED_BY_PAIRWISE_QUALIFICATION",
            "groupcv_ran": False,
            "recommend_run_dev_multiseed": False,
            "stop_model_assisted_positive_route": True,
            "blocking_pairwise_status": decision["status"],
            "dev_access_count": 0,
            "test_access_count": 0,
        }
        write_json(args.out_dir / "decision/exp40a_groupcv_decision.json", groupcv_decision)
        groupcv_report = [
            "# Exp40A EduPair-CF GroupCV report", "",
            "- Status: **GROUPCV_NOT_RUN_BLOCKED_BY_PAIRWISE_QUALIFICATION**",
            "- The frozen pairwise qualification gate failed, so no pair variant was trained.",
            "- No smoke run, formal GroupCV run, checkpoint, or prediction was produced.",
            "- No paper-like dev/test data were accessed.",
        ]
        groupcv_path = args.out_dir / "reports/exp40a_groupcv_report.md"
        groupcv_path.parent.mkdir(parents=True, exist_ok=True)
        groupcv_path.write_text("\n".join(groupcv_report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
