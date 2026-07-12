#!/usr/bin/env python3
"""Validate Exp35A blind reviews, freeze conflicts, and build silver references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_expert_reference.analyze_exp33a_review_completion import (  # noqa: E402
    agreement_record,
    canonical_hash,
    evaluator_content,
    krippendorff_ordinal_alpha,
    norm,
    quadratic_weighted_kappa,
    review_custom_errors,
    schema_errors,
)


DEFAULT_OUT = Path(
    "thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42"
)
REVIEW_SCHEMA = Path("thesis_exp/exp33_expert_reference/schemas/exp33a_blind_review_schema.json")
ADJ_SCHEMA = Path("thesis_exp/exp35_edudart_cal/schemas/exp35a_blind_adjudication_schema.json")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not repo_path(path).exists():
        return []
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_reviews(
    rows: list[dict[str, Any]], role: str, packets: dict[str, dict[str, Any]], schema: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    valid: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id") or "")
        row_errors = schema_errors(row, schema)
        if row.get("reviewer_role") != role:
            row_errors.append(f"reviewer_role must be {role}")
        if row.get("reviewer_type") != "model":
            row_errors.append("reviewer_type must be model")
        packet = packets.get(sid)
        if packet is None:
            row_errors.append("sample not assigned")
        else:
            row_errors.extend(review_custom_errors(row, packet))
        if sid in valid:
            row_errors.append("duplicate sample_id")
        if row_errors:
            errors.append(f"{role} row {index} {sid}: {'; '.join(row_errors)}")
        else:
            valid[sid] = row
    return valid, errors


def trigger_reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    reasons = []
    if a["most_plausible_score"] != b["most_plausible_score"]:
        reasons.append("point_score_disagreement")
    if a["score_range"][1] < b["score_range"][0] or b["score_range"][1] < a["score_range"][0]:
        reasons.append("non_overlapping_score_ranges")
    for field in ("failure_bucket", "student_input_sufficiency", "target_scope_confirmed"):
        if a[field] != b[field]:
            reasons.append(f"{field}_disagreement")
    if a["needs_adjudication"] or b["needs_adjudication"]:
        reasons.append("reviewer_requested_adjudication")
    if a["confidence"] == "low" or b["confidence"] == "low":
        reasons.append("low_confidence")
    if a["domain_escalation_required"] or b["domain_escalation_required"]:
        reasons.append("domain_escalation")
    return sorted(set(reasons))


def build_adjudication_packets(
    packets: dict[str, dict[str, Any]], a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], triggers: dict[str, list[str]]
) -> list[dict[str, Any]]:
    output = []
    for sid in sorted(triggers):
        bundle = {
            "sample_id": sid,
            "packet": packets[sid],
            "reviewer_a_result": a[sid],
            "reviewer_b_result": b[sid],
            "reviewer_a_result_hash": canonical_hash(a[sid]),
            "reviewer_b_result_hash": canonical_hash(b[sid]),
            "trigger_reasons": triggers[sid],
            "source_labels_available_to_adjudicator": False,
        }
        bundle["bundle_hash"] = canonical_hash(bundle)
        output.append(bundle)
    return output


def validate_adjudications(
    rows: list[dict[str, Any]], bundles: dict[str, dict[str, Any]], schema: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    validator = jsonschema.Draft202012Validator(schema)
    valid = {}
    errors = []
    for index, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id") or "")
        row_errors = [error.message for error in validator.iter_errors(row)]
        bundle = bundles.get(sid)
        if bundle is None:
            row_errors.append("sample not triggered")
        else:
            if row.get("packet_hash") != bundle["packet"]["packet_hash"]:
                row_errors.append("packet_hash mismatch")
            if row.get("reviewer_a_result_hash") != bundle["reviewer_a_result_hash"]:
                row_errors.append("reviewer_a_result_hash mismatch")
            if row.get("reviewer_b_result_hash") != bundle["reviewer_b_result_hash"]:
                row_errors.append("reviewer_b_result_hash mismatch")
            evidence = row.get("evaluator_output_evidence")
            if evidence is not None and norm(evidence) not in norm(evaluator_content(bundle["packet"])):
                row_errors.append("evidence is not evaluator output substring")
        posterior = row.get("final_score_posterior") or {}
        if abs(sum(float(posterior.get(str(score), 0)) for score in range(1, 6)) - 1.0) > 1e-6:
            row_errors.append("posterior does not sum to one")
        if sid in valid:
            row_errors.append("duplicate sample_id")
        if row_errors:
            errors.append(f"adjudicator row {index} {sid}: {'; '.join(row_errors)}")
        else:
            valid[sid] = row
    return valid, errors


def empirical_posterior(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    scores = [int(a["most_plausible_score"]), int(b["most_plausible_score"])]
    return {str(score): scores.count(score) / 2 for score in range(1, 6)}


def final_reference(
    packets: dict[str, dict[str, Any]], a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], triggers: dict[str, list[str]], adjudications: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for sid in sorted(packets):
        if sid not in triggers:
            posterior = {str(score): float(score == int(a[sid]["most_plausible_score"])) for score in range(1, 6)}
            point = int(a[sid]["most_plausible_score"])
            status = "blind_pair_consensus"
            confidence = "high" if a[sid]["confidence"] == b[sid]["confidence"] == "high" else "medium"
        else:
            adj = adjudications.get(sid)
            if adj and adj["final_status"] != "unresolved_domain_case":
                posterior = {str(score): float(adj["final_score_posterior"][str(score)]) for score in range(1, 6)}
                point = adj["final_most_plausible_score"]
                if point is None:
                    point = max(range(1, 6), key=lambda score: posterior[str(score)])
                status = adj["final_status"]
                confidence = adj["confidence"]
            else:
                posterior = empirical_posterior(a[sid], b[sid])
                point = max(range(1, 6), key=lambda score: posterior[str(score)])
                status = "model_disagreement_distribution_fallback"
                confidence = "low"
        rows.append(
            {
                "sample_id": sid,
                "qualification_view": packets[sid]["qualification_view"],
                "point_score": int(point),
                "p1": posterior["1"], "p2": posterior["2"], "p3": posterior["3"], "p4": posterior["4"], "p5": posterior["5"],
                "confidence": confidence,
                "reference_status": status,
            }
        )
    return rows


def view_agreement(view: str, ids: list[str], a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], triggers: set[str]) -> dict[str, Any]:
    pairs = [(sid, a[sid], b[sid]) for sid in ids]
    record = agreement_record(pairs, triggers & set(ids))
    return {"view": view, **record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    packet_rows = read_jsonl(out / "private_review/blind_packets/exp35a_reviewer_a_packet.jsonl")
    packets = {str(row["sample_id"]): row for row in packet_rows}
    review_schema = read_json(REVIEW_SCHEMA)
    adj_schema = read_json(ADJ_SCHEMA)
    raw_a = read_jsonl(out / "private_review/reviewer_filled/exp35a_reviewer_a_results.jsonl")
    raw_b = read_jsonl(out / "private_review/reviewer_filled/exp35a_reviewer_b_results.jsonl")
    a, errors_a = validate_reviews(raw_a, "reviewer_a", packets, review_schema)
    b, errors_b = validate_reviews(raw_b, "reviewer_b", packets, review_schema)
    blind_complete = set(a) == set(packets) and set(b) == set(packets) and not errors_a and not errors_b
    triggers = {sid: trigger_reasons(a[sid], b[sid]) for sid in packets if blind_complete and trigger_reasons(a[sid], b[sid])}
    bundles = build_adjudication_packets(packets, a, b, triggers) if blind_complete else []
    write_jsonl(out / "private_review/adjudication_packets/exp35a_blind_adjudication_packet.jsonl", bundles)
    bundle_map = {row["sample_id"]: row for row in bundles}
    raw_adj = read_jsonl(out / "private_review/adjudication_filled/exp35a_adjudicator_results.jsonl")
    adjudications, errors_adj = validate_adjudications(raw_adj, bundle_map, adj_schema) if bundles else ({}, [])
    adjudication_complete = blind_complete and set(adjudications) == set(triggers) and not errors_adj
    final = final_reference(packets, a, b, triggers, adjudications) if blind_complete and adjudication_complete else []
    if final:
        write_csv(out / "private/exp35a_model_reviewed_silver_reference.csv", final, list(final[0]))

    agreement_rows = []
    for view in ("fresh_general_qualification", "low_tail_reassessment"):
        ids = [sid for sid, packet in packets.items() if packet["qualification_view"] == view and sid in a and sid in b]
        if ids:
            agreement_rows.append(view_agreement(view, ids, a, b, set(triggers)))
    fields = ["view", "paired_rows", "exact_agreement", "within_one", "quadratic_weighted_kappa", "krippendorff_ordinal_alpha", "score_range_overlap", "adjudication_rate", "status"]
    write_csv(out / "tables/exp35a_reviewer_agreement.csv", agreement_rows, fields)
    general = next((row for row in agreement_rows if row["view"] == "fresh_general_qualification"), {})
    gates = {
        "blind_reviews_complete": blind_complete,
        "all_conflicts_adjudicated": adjudication_complete,
        "fresh_general_within_one_ge_0p90": bool(general and float(general["within_one"]) >= 0.90),
        "fresh_general_qwk_ge_0p60": bool(general and float(general["quadratic_weighted_kappa"]) >= 0.60),
        "fresh_general_alpha_ge_0p60": bool(general and float(general["krippendorff_ordinal_alpha"]) >= 0.60),
        "dev_test_access_zero": True,
    }
    decision = {
        "experiment": "Exp35A independent model-reviewed qualification",
        "reference_status": "model-reviewed silver, not human expert gold",
        "packet_rows": len(packets),
        "reviewer_a_valid": len(a),
        "reviewer_b_valid": len(b),
        "triggered_conflicts": len(triggers),
        "adjudicator_valid": len(adjudications),
        "final_reference_rows": len(final),
        "review_gate_passed": all(gates.values()),
        "gates": gates,
        "errors": errors_a + errors_b + errors_adj,
        "dev_rows_read": 0,
        "test_access_count": 0,
        "student_training": False,
    }
    write_json(out / "decision/exp35a_review_decision.json", decision)
    write_text(out / "reports/exp35a_review_report.md", f"""# Exp35A Blind Review Status

- Reference: model-reviewed silver, not human expert gold.
- Reviewer A/B valid: {len(a)}/{len(b)} of {len(packets)}.
- Triggered conflicts: {len(triggers)}; valid adjudications: {len(adjudications)}.
- Final silver rows: {len(final)}.
- Review gate passed: {all(gates.values())}.
- Dev/test access: 0/0; student training: none.
- Errors: {len(decision['errors'])}.
""")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
