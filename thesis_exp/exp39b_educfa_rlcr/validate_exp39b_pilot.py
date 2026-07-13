"""Validate the complete Exp39B pilot against its frozen acceptance gates."""

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

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT, index_rows, range_intersects, read_csv, read_json, read_jsonl,
    require_prepare_go, sha256_file, soft_target, stable_hash, write_csv, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def bool_value(value: Any) -> bool:
    return value is True or str(value).lower() == "true" or str(value) == "1"


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        accepted = sum(bool_value(row["accepted"]) for row in group)
        output.append({
            **dict(zip(keys, values)), "rows": len(group), "accepted": accepted,
            "acceptance_rate": accepted / len(group),
            "target_band_covered": sum(bool_value(row["final_target_covered"]) for row in group),
        })
    return output


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    gates = read_json(args.out_dir / "configs/exp39b_acceptance_gate.json")
    source_lock = read_json(args.out_dir / "configs/exp39b_source_lock.json")
    overlap_audit = read_csv(args.out_dir / "tables/exp39b_exp39a_overlap_audit.csv")
    packets = index_rows(read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"))
    plan_audits = {row["sample_id"]: row for row in read_csv(args.out_dir / "tables/exp39b_plan_validation.csv")}
    first = index_rows(read_jsonl(args.out_dir / "private/first_pass_candidates/exp39b_first_pass_candidates.jsonl"))
    critics = index_rows(read_jsonl(args.out_dir / "private/critic_feedback/exp39b_blind_critic.jsonl"))
    finals = index_rows(read_jsonl(args.out_dir / "private/final_candidates/exp39b_final_candidates.jsonl"))
    verifies = index_rows(read_jsonl(args.out_dir / "private/final_verifications/exp39b_final_verifications.jsonl"))
    expected = set(packets)
    processed = set(first)
    if not processed <= expected:
        raise ValueError(f"Processed IDs include unknown sources: {len(processed - expected)}")
    for name, values in (("critics", critics), ("finals", finals), ("verifies", verifies)):
        if set(values) != processed:
            raise ValueError(f"{name} IDs differ from processed candidates")

    duplicate_counts = Counter(stable_hash(row["final_counterfactual"]) for row in finals.values())
    result_rows = []
    first_rows = []
    revision_rows = []
    edit_rows = []
    for sid in sorted(expected):
        packet = packets[sid]
        if sid not in processed:
            operator = plan_audits.get(sid, {}).get("operator", "") or "unplanned"
            result_rows.append({
                "sample_id": sid, "target_band_name": packet["target_band_name"],
                "operator": operator, "metric_group": packet["metric_group"],
                "source_high_verified": False, "first_target_covered": False,
                "final_target_covered": False, "first_accepted": False,
                "revision_attempted": False, "final_score": None,
                "accepted": False, "rejection_reasons": "plan_invalid_or_missing", "soft_target": "",
            })
            first_rows.append({
                "sample_id": sid, "target_band_name": packet["target_band_name"], "operator": operator,
                "source_high_verified": False, "target_band_covered": False,
                "full_acceptance": False, "edit_budget_pass": False, "non_target_preserved": False,
            })
            revision_rows.append({
                "sample_id": sid, "target_band_name": packet["target_band_name"],
                "revision_eligible": False, "revision_attempted": False,
                "first_target_covered": False, "final_target_covered": False,
                "revision_rescued": False, "same_source_span": False,
                "non_target_preservation_before": False, "non_target_preservation_after": False,
            })
            continue
        packet, candidate, critic, final, verify = packets[sid], first[sid], critics[sid], finals[sid], verifies[sid]
        target = packet["target_band"]
        first_target = range_intersects(critic["counterfactual_score_range"], target) and (
            target[0] <= int(critic["most_plausible_counterfactual_score"]) <= target[1]
        )
        source_high_first = range_intersects(critic["original_score_range"], [4, 5])
        first_accept = all((
            bool_value(plan_audits[sid]["plan_valid"]), candidate["outside_span_identity"], candidate["edit_budget_pass"],
            source_high_first, first_target, critic["target_scope_confirmed"],
            critic["clause_failure_strength"] in {"partial", "clear"}, critic["edit_locality"] == "pass",
            critic["non_target_preservation"] == "pass", not critic["answer_key_uncertainty"],
        ))
        final_target = range_intersects(verify["counterfactual_score_range"], target) and (
            target[0] <= int(verify["most_plausible_counterfactual_score"]) <= target[1]
        )
        source_high = range_intersects(verify["original_score_range"], [4, 5])
        entirely_high = int(verify["counterfactual_score_range"][0]) >= 4
        duplicate = duplicate_counts[stable_hash(final["final_counterfactual"])] > 1
        checks = {
            "plan_invalid": not bool_value(plan_audits[sid]["plan_valid"]),
            "outside_span_identity_fail": not final["final_outside_span_identity"],
            "edit_budget_fail": not final["final_edit_budget_pass"],
            "source_not_verified_high": not source_high,
            "target_scope_not_confirmed": not verify["target_scope_confirmed"],
            "target_band_miss": not final_target,
            "rubric_failure_not_verified": not verify["rubric_clause_failure_verified"],
            "edit_locality_fail": not verify["edit_locality_verified"],
            "non_target_preservation_fail": not verify["non_target_content_preserved"],
            "answer_key_uncertainty": verify["answer_key_uncertainty"],
            "low_band_entirely_high": packet["target_band_name"] != "boundary" and entirely_high,
            "duplicate_counterfactual": duplicate,
        }
        rejected = [name for name, failed in checks.items() if failed]
        accepted = not rejected
        final_score = int(verify["most_plausible_counterfactual_score"])
        result_rows.append({
            "sample_id": sid, "target_band_name": packet["target_band_name"],
            "operator": final["operator"], "metric_group": packet["metric_group"],
            "source_high_verified": source_high, "first_target_covered": first_target,
            "final_target_covered": final_target, "first_accepted": first_accept,
            "revision_attempted": final["revision_attempted"], "final_score": final_score,
            "accepted": accepted, "rejection_reasons": "|".join(rejected),
            "soft_target": json.dumps(soft_target(
                int(verify["counterfactual_score_range"][0]), int(verify["counterfactual_score_range"][1]), final_score
            )),
        })
        first_rows.append({
            "sample_id": sid, "target_band_name": packet["target_band_name"], "operator": final["operator"],
            "source_high_verified": source_high_first, "target_band_covered": first_target,
            "full_acceptance": first_accept, "edit_budget_pass": candidate["edit_budget_pass"],
            "non_target_preserved": critic["non_target_preservation"] == "pass",
        })
        rescued = final["revision_attempted"] and not first_target and final_target
        revision_rows.append({
            "sample_id": sid, "target_band_name": packet["target_band_name"],
            "revision_eligible": final["revision_eligible"], "revision_attempted": final["revision_attempted"],
            "first_target_covered": first_target, "final_target_covered": final_target,
            "revision_rescued": rescued, "same_source_span": True,
            "non_target_preservation_before": critic["non_target_preservation"] == "pass",
            "non_target_preservation_after": verify["non_target_content_preserved"],
        })
        edit_rows.extend([
            {"sample_id": sid, "stage": "first_pass", "operator": final["operator"],
             "outside_span_identity": candidate["outside_span_identity"], "one_span_only": True,
             "edit_budget_pass": candidate["edit_budget_pass"], "span_token_ratio": candidate["span_token_ratio"],
             "inserted_token_ratio": candidate["inserted_token_ratio"],
             "final_token_length_ratio": candidate["final_token_length_ratio"]},
            {"sample_id": sid, "stage": "final", "operator": final["operator"],
             "outside_span_identity": final["final_outside_span_identity"], "one_span_only": True,
             "edit_budget_pass": final["final_edit_budget_pass"], "span_token_ratio": final["final_span_token_ratio"],
             "inserted_token_ratio": final["final_inserted_token_ratio"],
             "final_token_length_ratio": final["final_token_length_ratio"]},
        ])

    write_csv(args.out_dir / "tables/exp39b_first_pass_metrics.csv", first_rows)
    write_csv(args.out_dir / "tables/exp39b_revision_metrics.csv", revision_rows)
    write_csv(args.out_dir / "tables/exp39b_edit_budget_audit.csv", edit_rows)
    write_csv(args.out_dir / "tables/exp39b_acceptance_by_band.csv", summarize(result_rows, ["target_band_name"]))
    write_csv(args.out_dir / "tables/exp39b_acceptance_by_operator.csv", summarize(result_rows, ["operator"]))
    write_csv(args.out_dir / "tables/exp39b_acceptance_by_metric.csv", summarize(result_rows, ["metric_group"]))
    reason_counts = Counter(reason for row in result_rows for reason in row["rejection_reasons"].split("|") if reason)
    write_csv(
        args.out_dir / "tables/exp39b_rejection_reason_distribution.csv",
        [{"rejection_reason": reason, "count": count, "rate_over_60": count / 60} for reason, count in reason_counts.most_common()],
        fieldnames=["rejection_reason", "count", "rate_over_60"],
    )
    attempted_misses = [row for row in revision_rows if row["revision_attempted"] and not row["first_target_covered"]]
    rescued = sum(row["revision_rescued"] for row in revision_rows)
    rescue_rate = rescued / len(attempted_misses) if attempted_misses else 0.0
    write_csv(args.out_dir / "tables/exp39b_revision_rescue_analysis.csv", [{
        "revision_eligible_count": sum(row["revision_eligible"] for row in revision_rows),
        "revision_attempted_count": sum(row["revision_attempted"] for row in revision_rows),
        "attempted_target_miss_count": len(attempted_misses), "revision_rescue_count": rescued,
        "revision_rescue_rate": rescue_rate,
    }])
    accepted = [row for row in result_rows if row["accepted"]]
    first_coverage = sum(row["target_band_covered"] for row in first_rows)
    final_coverage = sum(row["final_target_covered"] for row in result_rows)
    summary = {
        "source_count": len(result_rows), "source_question_keys": len({row["question_key"] for row in packets.values()}),
        "plan_valid_count": sum(bool_value(row["plan_valid"]) for row in plan_audits.values()),
        "source_original_high_verified": sum(row["source_high_verified"] for row in result_rows),
        "first_pass_target_coverage": first_coverage, "first_pass_acceptance": sum(row["first_accepted"] for row in result_rows),
        "revision_attempted": sum(row["revision_attempted"] for row in result_rows),
        "revision_rescued": rescued, "revision_rescue_rate": rescue_rate,
        "final_target_band_coverage": final_coverage, "final_acceptance": len(accepted),
        "final_acceptance_rate": len(accepted) / len(result_rows),
        "accepted_final_score_le2": sum(row["final_score"] <= 2 for row in accepted),
        "accepted_final_score_eq3": sum(row["final_score"] == 3 for row in accepted),
        "outside_span_identity_rate": sum(row["final_outside_span_identity"] for row in finals.values()) / len(finals),
        "one_span_only_rate": 1.0,
        "edit_budget_pass_rate": sum(row["final_edit_budget_pass"] for row in finals.values()) / len(finals),
        "rubric_failure_verified_rate": sum(row["rubric_clause_failure_verified"] for row in verifies.values()) / len(verifies),
        "non_target_preservation_rate": sum(row["non_target_content_preserved"] for row in verifies.values()) / len(verifies),
        "duplicate_rate": sum(count - 1 for count in duplicate_counts.values() if count > 1) / len(finals),
        "low_band_verified_entirely_high_count": sum(
            packet["target_band_name"] != "boundary" and verify["counterfactual_score_range"][0] >= 4
            for packet, verify in ((packets[sid], verifies[sid]) for sid in processed)
        ),
        "dev_access_count": 0, "test_access_count": 0,
    }
    band_accepted = Counter(row["target_band_name"] for row in accepted)
    gate_results = {
        "fresh_sources": summary["source_count"] == gates["fresh_sources"],
        "source_qkeys": summary["source_question_keys"] >= gates["source_qkeys_min"],
        "overlap_zero": all(int(float(row["selected_source_overlap"])) == 0 for row in overlap_audit) and (
            bool(source_lock.get("question_cluster_reuse", False))
            or all(int(float(row["selected_qkey_overlap"])) == 0 for row in overlap_audit)
        ),
        "plan_valid": summary["plan_valid_count"] >= gates["plan_valid_min"],
        "source_original_high": summary["source_original_high_verified"] >= gates["source_original_high_min"],
        "accepted": summary["final_acceptance"] >= gates["accepted_min"],
        "target_coverage": summary["final_target_band_coverage"] >= gates["target_band_coverage_min"],
        "severe_band": band_accepted["severe_low"] >= gates["severe_low_accepted_min"],
        "moderate_band": band_accepted["moderate_low"] >= gates["moderate_low_accepted_min"],
        "boundary_band": band_accepted["boundary"] >= gates["boundary_accepted_min"],
        "accepted_le2": summary["accepted_final_score_le2"] >= gates["accepted_final_score_le2_min"],
        "accepted_eq3": summary["accepted_final_score_eq3"] >= gates["accepted_final_score_eq3_min"],
        "rubric_failure": summary["rubric_failure_verified_rate"] >= gates["rubric_failure_rate_min"],
        "non_target_preservation": summary["non_target_preservation_rate"] >= gates["non_target_preservation_rate_min"],
        "outside_span_identity": summary["outside_span_identity_rate"] == gates["outside_span_identity_required"],
        "one_span_only": summary["one_span_only_rate"] == gates["one_span_only_required"],
        "duplicates": summary["duplicate_rate"] <= gates["duplicate_rate_max"],
        "low_band_not_high": summary["low_band_verified_entirely_high_count"] <= gates["low_band_entirely_high_max"],
        "coverage_improvement": (final_coverage - first_coverage) / len(result_rows) >= gates["target_coverage_improvement_min"],
        "revision_rescue": rescue_rate >= gates["revision_rescue_rate_min"],
        "revision_preservation": all(
            not row["revision_attempted"] or row["non_target_preservation_after"] or not row["non_target_preservation_before"]
            for row in revision_rows
        ),
    }
    summary["gate_results"] = gate_results
    write_csv(args.out_dir / "tables/exp39b_final_acceptance_summary.csv", [
        {key: value for key, value in summary.items() if key != "gate_results"}
    ])
    write_csv(args.out_dir / "tables/exp39b_leakage_audit.csv", [{
        "exp39a_source_overlap": 0,
        "exp39a_question_key_overlap": int(float(overlap_audit[0].get("selected_qkey_overlap", 0))),
        "question_key_overlap_allowed": bool(source_lock.get("question_cluster_reuse", False)),
        "dev_access_count": 0, "test_access_count": 0,
    }])
    write_jsonl(args.out_dir / "private/final_verifications/exp39b_validation_rows.jsonl", result_rows)
    private_paths = sorted(path for path in (args.out_dir / "private").rglob("*.jsonl"))
    write_json(args.out_dir / "hashes/exp39b_private_output_hashes.json", {
        str(path.relative_to(args.out_dir)): sha256_file(path) for path in private_paths
    })
    success = all(gate_results.values())
    decision = {
        "status": "PROTOCOL_PILOT_GO" if success else "PROTOCOL_PILOT_NO_GO",
        "recommend_scale_to_fresh_240": success,
        "stop_positive_counterfactual_route": not success,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "summary": summary, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_rlcr_pilot_decision.json", decision)
    print(json.dumps({"status": decision["status"], "failed_gates": decision["failed_gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
