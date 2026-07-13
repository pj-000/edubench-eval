"""Validate Exp39B target-aware clause plans before any editing request."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT,
    index_rows,
    locate_occurrence,
    read_jsonl,
    require_prepare_go,
    write_csv,
    write_json,
)

COMPATIBILITY = {
    "Factual & Reasoning Accuracy": {
        "delete_required_span", "replace_with_local_contradiction", "delete_supporting_reasoning"
    },
    "Pedagogical Application": {
        "delete_required_span", "delete_supporting_reasoning", "insert_local_scope_drift"
    },
    "Scenario Adaptability": {
        "delete_required_span", "insert_local_scope_drift", "replace_with_local_contradiction"
    },
}


def normalize(text: object) -> str:
    return re.sub(r"\s+", " ", json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text).strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def validate(plan: dict, packet: dict) -> tuple[list[str], dict]:
    errors: list[str] = []
    band_name = packet["target_band_name"]
    clause_type = str(plan.get("clause_type"))
    operator = str(plan.get("operator"))
    metric_group = str(packet.get("metric_group") or "")
    if plan.get("target_band") != packet["target_band"]:
        errors.append("target_band_mismatch")
    if plan.get("plan_confidence") == "low":
        errors.append("low_confidence")
    if band_name == "severe_low" and not (
        clause_type == "cap_setting" and int(plan.get("expected_score_cap_after_edit", 9)) <= 2
    ):
        errors.append("severe_band_clause_mismatch")
    if band_name == "moderate_low" and clause_type not in {"cap_setting", "major"}:
        errors.append("moderate_band_clause_mismatch")
    if band_name == "boundary" and clause_type not in {"major", "incremental"}:
        errors.append("boundary_clause_mismatch")
    rubric_text = normalize(packet.get("rubric"))
    rubric_clause = normalize(plan.get("rubric_clause", ""))
    rubric_linked = bool(rubric_clause and rubric_clause in rubric_text)
    if not rubric_linked:
        errors.append("rubric_clause_not_found")
    try:
        start, end = locate_occurrence(
            packet["original_answer"], str(plan.get("source_evidence_span") or ""),
            int(plan.get("source_span_occurrence", -1)),
        )
        span_valid = True
    except (TypeError, ValueError):
        start = end = -1
        span_valid = False
        errors.append("source_span_invalid")
    allowed = COMPATIBILITY.get(metric_group, set())
    metric_text = normalize(packet.get("metric")) + " " + rubric_text
    if operator == "violate_explicit_constraint":
        compatible = bool(re.search(r"format|json|instruction|constraint|格式|指令|约束|要求", metric_text))
    elif operator == "delete_supporting_reasoning" and metric_group == "Factual & Reasoning Accuracy":
        compatible = bool(re.search(r"reason|evidence|logic|推理|证据|论证", metric_text))
    elif operator == "insert_local_scope_drift" and metric_group == "Pedagogical Application":
        compatible = bool(re.search(r"personal|guidance|application|student|个性|指导|应用|学生", metric_text))
    elif operator == "replace_with_local_contradiction" and metric_group == "Scenario Adaptability":
        compatible = bool(re.search(r"scenario|context|场景|情境|背景", metric_text))
    else:
        compatible = operator in allowed
    if not compatible:
        errors.append("operator_incompatible")
    audit = {
        "sample_id": packet["sample_id"], "source_sample_id": packet["source_sample_id"],
        "target_band_name": band_name, "metric_group": metric_group, "clause_type": clause_type,
        "operator": operator, "rubric_clause_linked": rubric_linked, "source_span_valid": span_valid,
        "source_span_start": start, "source_span_end": end, "operator_compatible": compatible,
        "plan_confidence": plan.get("plan_confidence"), "plan_valid": not errors,
        "rejection_reasons": "|".join(errors),
    }
    return errors, audit


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    packets = index_rows(read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"))
    plans = read_jsonl(args.out_dir / "private/clause_plans/exp39b_clause_plans.jsonl")
    audits = []
    valid_ids = []
    for plan in plans:
        sid = str(plan["sample_id"])
        if sid not in packets:
            raise ValueError(f"Unknown plan sample_id: {sid}")
        errors, audit = validate(plan, packets[sid])
        audits.append(audit)
        if not errors:
            valid_ids.append(sid)
    received_ids = {str(plan["sample_id"]) for plan in plans}
    for sid in sorted(set(packets) - received_ids):
        packet = packets[sid]
        audits.append({
            "sample_id": sid, "source_sample_id": packet["source_sample_id"],
            "target_band_name": packet["target_band_name"], "metric_group": packet["metric_group"],
            "clause_type": "", "operator": "", "rubric_clause_linked": False,
            "source_span_valid": False, "source_span_start": -1, "source_span_end": -1,
            "operator_compatible": False, "plan_confidence": "",
            "plan_valid": False, "rejection_reasons": "missing_planner_output",
        })
    write_csv(args.out_dir / "tables/exp39b_plan_validation.csv", audits)
    write_csv(
        args.out_dir / "tables/exp39b_operator_compatibility.csv",
        [{"metric_group": key, "allowed_operators": "|".join(sorted(value))} for key, value in COMPATIBILITY.items()]
        + [{"metric_group": "explicit_constraint_only", "allowed_operators": "violate_explicit_constraint"}],
    )
    decision = {
        "status": "PLAN_VALIDATION_GO" if len(valid_ids) >= 54 else "PLAN_VALIDATION_NO_GO",
        "plans_received": len(plans), "plan_valid_count": len(valid_ids), "plan_valid_min": 54,
        "valid_sample_ids": valid_ids, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_plan_validation_decision.json", decision)
    print(json.dumps({key: value for key, value in decision.items() if key != "valid_sample_ids"}, sort_keys=True))
    if decision["status"] != "PLAN_VALIDATION_GO":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
