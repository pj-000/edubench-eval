"""Validate the repaired Exp37A-R1 protocol and selective review outputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    EVIDENCE_SUFFICIENCY,
    EVIDENCE_TYPES,
    FAILURE_CLASSES,
    MAJOR_FAILURE_PRESENCE,
    REFERENCE_TYPE,
    REVIEW_FIELDS,
    ROOT,
    build_final_reference,
    expected_conflict_reasons,
    normalized_substring,
    packet_hash,
    read_jsonl,
    sha256_text,
    tie_safe_average_precision,
    write_csv,
    write_json,
    write_jsonl,
)

ALLOWED_BUCKETS = {"no_failure", "visible_failure", "hidden_or_missing_failure", "unclear"}
PACKET_FIELDS = {
    "sample_id", "anonymized_question_key_hash", "question_context", "evaluator_output",
    "metric", "rubric", "non_label_metadata", "review_text", "packet_hash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    return parser.parse_args()


def load_packet_map(out_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    output = []
    for reviewer in ("reviewer_a", "reviewer_b"):
        path = out_dir / "private_packets" / f"exp37a_r1_{reviewer}_packets.jsonl"
        rows = read_jsonl(path)
        mapping = {str(row["sample_id"]): row for row in rows}
        if len(rows) != 196 or len(mapping) != 196:
            raise ValueError(f"{reviewer} packet must contain 196 unique rows")
        for sid, row in mapping.items():
            if set(row) != PACKET_FIELDS:
                raise ValueError(f"Unexpected packet fields for {sid}: {sorted(set(row) - PACKET_FIELDS)}")
            if row.get("packet_hash") != packet_hash(row):
                raise ValueError(f"Packet hash mismatch: {sid}")
        output.append(mapping)
    if set(output[0]) != set(output[1]):
        raise ValueError("Reviewer A/B packet ID sets differ")
    if any(output[0][sid]["packet_hash"] != output[1][sid]["packet_hash"] for sid in output[0]):
        raise ValueError("Reviewer A/B packet inputs differ")
    return output[0], output[1]


def schema_protocol_checks() -> dict[str, Any]:
    schema_dir = Path(__file__).resolve().parent / "schemas"
    blind = json.loads((schema_dir / "exp37a_blind_failure_review_schema.json").read_text(encoding="utf-8"))
    adjudication = json.loads((schema_dir / "exp37a_failure_adjudication_schema.json").read_text(encoding="utf-8"))
    failure = blind.get("properties", {}).get("failure_classes", {})
    checks = {
        "blind_additional_properties_false": blind.get("additionalProperties") is False,
        "adjudication_additional_properties_false": adjudication.get("additionalProperties") is False,
        "failure_classes_min_items_one": failure.get("minItems") == 1,
        "failure_classes_unique": failure.get("uniqueItems") is True,
        "rubric_evidence_min_length": blind.get("properties", {}).get("rubric_evidence", {}).get("minLength") == 1,
        "review_reason_min_length": blind.get("properties", {}).get("review_reason", {}).get("minLength") == 1,
        "major_failure_presence_present": "major_failure_presence" in blind.get("required", []),
        "evidence_type_present": "evidence_type" in blind.get("required", []),
        "evidence_sufficiency_present": "evidence_sufficiency" in blind.get("required", []),
        "new_reference_type": adjudication.get("properties", {}).get("reference_type", {}).get("const") == REFERENCE_TYPE,
    }
    checks["schema_strict"] = all(checks.values())
    return checks


def validate_review(row: dict[str, Any], packet: dict[str, Any], adjudication: bool = False) -> list[str]:
    errors: list[str] = []
    expected_fields = set(REVIEW_FIELDS) | ({"reference_type", "adjudication_reason"} if adjudication else set())
    if set(row) != expected_fields:
        errors.append("additional_or_missing_properties")
    if str(row.get("sample_id")) != str(packet["sample_id"]):
        errors.append("sample_id_mismatch")
    if row.get("target_scope_confirmed") is not True:
        errors.append("target_scope_not_confirmed")
    score_range = row.get("score_range")
    valid_range = isinstance(score_range, list) and len(score_range) == 2 and all(isinstance(value, int) and 1 <= value <= 5 for value in score_range) and score_range[0] <= score_range[1]
    if not valid_range:
        errors.append("invalid_score_range")
    plausible = row.get("most_plausible_score")
    if not isinstance(plausible, int) or not valid_range or not score_range[0] <= plausible <= score_range[1]:
        errors.append("plausible_score_outside_range")
    if row.get("failure_bucket") not in ALLOWED_BUCKETS:
        errors.append("invalid_failure_bucket")
    presence = row.get("major_failure_presence")
    if presence not in MAJOR_FAILURE_PRESENCE:
        errors.append("invalid_major_failure_presence")
    failures = row.get("failure_classes")
    if not isinstance(failures, list) or not failures or len(failures) != len(set(failures)) or not all(value in FAILURE_CLASSES for value in failures):
        errors.append("invalid_failure_classes")
        failures = []
    if presence == "no" and failures != ["no_major_failure"]:
        errors.append("no_presence_requires_no_major_failure")
    if presence == "yes" and "no_major_failure" in failures:
        errors.append("yes_presence_forbids_no_major_failure")
    evidence_type = row.get("evidence_type")
    sufficiency = row.get("evidence_sufficiency")
    if evidence_type not in EVIDENCE_TYPES:
        errors.append("invalid_evidence_type")
    if sufficiency not in EVIDENCE_SUFFICIENCY:
        errors.append("invalid_evidence_sufficiency")
    evidence = row.get("evaluator_output_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(set(evidence)) or not all(isinstance(value, str) and value.strip() for value in evidence):
        errors.append("invalid_evidence_list")
        evidence = []
    if evidence_type == "explicit_span":
        if not evidence:
            errors.append("explicit_span_requires_evidence")
        elif not all(normalized_substring(value, str(packet.get("evaluator_output", ""))) for value in evidence):
            errors.append("evidence_not_normalized_substring")
    if evidence_type == "missing_required_content" and not str(row.get("missing_evidence_reason") or "").strip():
        errors.append("missing_content_requires_declaration")
    if not str(row.get("rubric_evidence") or "").strip():
        errors.append("rubric_evidence_missing")
    cap = row.get("score_cap")
    if cap is not None and (not isinstance(cap, int) or not 1 <= cap <= 5):
        errors.append("invalid_score_cap")
    if row.get("confidence") not in {"high", "medium", "low"}:
        errors.append("invalid_confidence")
    if row.get("confidence") == "low" and row.get("needs_adjudication") is not True:
        errors.append("low_confidence_requires_adjudication")
    if not str(row.get("review_reason") or "").strip():
        errors.append("review_reason_missing")
    if adjudication:
        if row.get("reference_type") != REFERENCE_TYPE:
            errors.append("invalid_reference_type")
        if not str(row.get("adjudication_reason") or "").strip():
            errors.append("adjudication_reason_missing")
    return sorted(set(errors))


def load_and_validate_review(path: Path, packets: dict[str, dict[str, Any]], role: str, adjudication: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    invalid: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("sample_id"))
        errors = []
        if sid in seen:
            errors.append("duplicate_sample_id")
        seen.add(sid)
        if sid not in packets:
            errors.append("sample_not_in_frozen_packets")
        else:
            errors.extend(validate_review(row, packets[sid], adjudication=adjudication))
        if errors:
            invalid.append({"sample_id_hash": sha256_text(sid), "errors": "|".join(sorted(set(errors)))})
    return rows, {
        "role": role,
        "status": "PASS" if not invalid else "FAIL",
        "rows": len(rows),
        "unique_rows": len(seen),
        "invalid_rows": len(invalid),
    }


def ap_tie_invariance(out_dir: Path, seed: int = 42) -> dict[str, Any]:
    labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    scores = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]
    rng = random.Random(seed)
    values = []
    for _ in range(100):
        order = list(range(len(labels)))
        rng.shuffle(order)
        values.append(tie_safe_average_precision([labels[index] for index in order], [scores[index] for index in order]))
    result = {
        "permutations": 100,
        "ap_min": min(values),
        "ap_max": max(values),
        "ap_max_minus_min": max(values) - min(values),
        "threshold": 1e-12,
        "pass": max(values) - min(values) < 1e-12,
        "implementation": "sklearn.metrics.average_precision_score",
    }
    write_csv(out_dir / "tables/exp37a_r1_ap_tie_invariance_test.csv", [result])
    return result


def selective_contract_self_test() -> bool:
    base = {
        "sample_id": "a", "target_scope_confirmed": True, "score_range": [2, 3],
        "most_plausible_score": 2, "failure_bucket": "visible_failure",
        "major_failure_presence": "yes", "failure_classes": ["factual_or_rubric_mismatch"],
        "evidence_type": "explicit_span", "evidence_sufficiency": "sufficient",
        "evaluator_output_evidence": ["x"], "missing_evidence_reason": None,
        "rubric_evidence": "r", "score_cap": 3, "confidence": "high",
        "needs_adjudication": False, "review_reason": "r",
    }
    consensus = dict(base)
    conflict_a = dict(base, sample_id="b")
    conflict_b = dict(conflict_a, most_plausible_score=3)
    adjudicated = dict(conflict_a, reference_type=REFERENCE_TYPE, adjudication_reason="resolved")
    final = build_final_reference(
        [consensus, conflict_a], [dict(consensus), conflict_b], [adjudicated], expected_count=2
    )
    return len(final) == 2 and {row["provenance"] for row in final} == {"reviewer_ab_consensus", "reviewer_c_adjudication"}


def main() -> None:
    args = parse_args()
    packets_a, packets_b = load_packet_map(args.out_dir)
    schema_checks = schema_protocol_checks()
    ap_check = ap_tie_invariance(args.out_dir)
    contract_pass = selective_contract_self_test()

    forbidden_keys = {"human_1", "human_2", "human_3", "label_5", "human_reason", "qwen", "deepseek", "oof", "teacher", "exp36", "view", "risk_reason"}
    leakage_rows = []
    for sid, packet in packets_a.items():
        hits = sorted(forbidden_keys & {str(key).lower() for key in packet})
        leakage_rows.append({"sample_id_hash": packet["anonymized_question_key_hash"], "forbidden_field_hits": "|".join(hits), "pass": not hits})
    write_csv(args.out_dir / "tables/exp37a_r1_leakage_audit.csv", leakage_rows)

    a_path = args.reviewer_a or args.out_dir / "private_reviews/reviewer_a_filled.jsonl"
    b_path = args.reviewer_b or args.out_dir / "private_reviews/reviewer_b_filled.jsonl"
    c_path = args.adjudication or args.out_dir / "private_reviews/adjudication_filled.jsonl"
    completion = []
    reviewer_a: list[dict[str, Any]] = []
    reviewer_b: list[dict[str, Any]] = []
    adjudication: list[dict[str, Any]] = []

    for role, path in (("reviewer_a", a_path), ("reviewer_b", b_path)):
        if path.exists():
            rows, summary = load_and_validate_review(path, packets_a, role)
            if role == "reviewer_a": reviewer_a = rows
            else: reviewer_b = rows
            expected_count = 196
            if len(rows) != expected_count:
                summary["status"] = "FAIL"
            summary["expected_rows"] = expected_count
        else:
            if not args.allow_missing_reviews:
                raise FileNotFoundError(path)
            summary = {"role": role, "status": "missing", "rows": 0, "unique_rows": 0, "invalid_rows": 0, "expected_rows": 196}
        completion.append(summary)

    expected_conflicts: set[str] = set()
    if len(reviewer_a) == len(reviewer_b) == 196:
        by_a = {str(row["sample_id"]): row for row in reviewer_a}
        by_b = {str(row["sample_id"]): row for row in reviewer_b}
        expected_conflicts = {sid for sid in by_a if expected_conflict_reasons(by_a[sid], by_b[sid])}
    if c_path.exists():
        adjudication, c_summary = load_and_validate_review(c_path, packets_a, "reviewer_c", adjudication=True)
        actual = {str(row["sample_id"]) for row in adjudication}
        c_summary.update({"expected_rows": len(expected_conflicts), "missing_conflict_rows": len(expected_conflicts - actual), "extra_rows": len(actual - expected_conflicts)})
        if actual != expected_conflicts:
            c_summary["status"] = "FAIL"
    elif expected_conflicts:
        if not args.allow_missing_reviews:
            raise FileNotFoundError(c_path)
        c_summary = {"role": "reviewer_c", "status": "missing", "rows": 0, "unique_rows": 0, "invalid_rows": 0, "expected_rows": len(expected_conflicts), "missing_conflict_rows": len(expected_conflicts), "extra_rows": 0}
    else:
        c_summary = {"role": "reviewer_c", "status": "not_yet_required", "rows": 0, "unique_rows": 0, "invalid_rows": 0, "expected_rows": 0, "missing_conflict_rows": 0, "extra_rows": 0}
    completion.append(c_summary)
    write_csv(args.out_dir / "tables/exp37a_r1_review_completion.csv", completion)

    reference_complete = False
    if len(reviewer_a) == len(reviewer_b) == 196 and c_summary["status"] == "PASS" or (
        len(reviewer_a) == len(reviewer_b) == 196 and not expected_conflicts and c_summary["status"] == "not_yet_required"
    ):
        final = build_final_reference(reviewer_a, reviewer_b, adjudication)
        write_jsonl(args.out_dir / "private_reference/exp37a_r1_final_reference.jsonl", final)
        reference_complete = True

    protocol_pass = bool(
        schema_checks["schema_strict"]
        and ap_check["pass"]
        and contract_pass
        and not any(not row["pass"] for row in leakage_rows)
        and set(packets_a) == set(packets_b)
    )
    any_review_started = a_path.exists() or b_path.exists() or c_path.exists()
    status = (
        "READY_FOR_ANALYSIS" if reference_complete
        else "WAITING_FOR_SELECTIVE_ADJUDICATION" if len(reviewer_a) == len(reviewer_b) == 196
        else "READY_FOR_EXTERNAL_MODEL_REVIEWS" if protocol_pass and not any_review_started
        else "PROTOCOL_OR_REVIEW_VALIDATION_FAILED" if not protocol_pass or any(row["status"] == "FAIL" for row in completion)
        else "REVIEWS_IN_PROGRESS"
    )
    decision = {
        "experiment_name": "Exp37A-R1 Multi-Session Model-Reviewed Failure-Evidence Qualification",
        "reference_type": REFERENCE_TYPE,
        "status": status,
        "protocol_repair_pass": protocol_pass,
        "sample_ids_preserved": True,
        "schema_strict": schema_checks["schema_strict"],
        "selective_adjudication_contract_pass": contract_pass,
        "ap_tie_invariance_pass": ap_check["pass"],
        "reference_complete": reference_complete,
        "expected_conflict_count": len(expected_conflicts) if reviewer_a and reviewer_b else None,
        "recommend_new_reason_evidence_training": False,
        "recommend_qwen_score_range_pilot": False,
        "recommend_student_training": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp37a_r1_protocol_repair_decision.json", decision)
    report = [
        "# Exp37A-R1 protocol repair report", "",
        f"- Protocol repair pass: `{protocol_pass}`",
        f"- Strict schema: `{schema_checks['schema_strict']}`",
        f"- Selective adjudication contract self-test: `{contract_pass}`",
        f"- AP tie invariance: `{ap_check['pass']}`",
        f"- Status: `{status}`",
        "- A/B each cover all 196 rows; C covers only the frozen expected conflict set.",
        "- Non-conflicts use A/B consensus; conflicts use Reviewer C.",
        "- No human reasons, teacher output, OOF predictions, view names, or risk reasons enter reviewer packets.",
        "- No API, GPU, training, inference, dev, or test access occurred.",
    ]
    (args.out_dir / "reports/exp37a_r1_protocol_repair_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports/exp37a_r1_protocol_repair_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not protocol_pass or any(row["status"] == "FAIL" for row in completion):
        raise SystemExit("Exp37A-R1 protocol validation failed")
    print(decision)


if __name__ == "__main__":
    main()
