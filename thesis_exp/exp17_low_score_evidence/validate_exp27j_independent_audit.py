"""Validate Exp27J sampling, blindness, and filled review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27j_independent_audit_seed42"
)
FORBIDDEN_PACKET_KEYS = {
    "label",
    "label_5",
    "original_label",
    "original_score",
    "original_human_score",
    "human_1",
    "human_2",
    "human_3",
    "human_reason",
    "qwen_score",
    "qwen_reason",
    "deepseek_score",
    "deepseek_reason",
    "calibrated_score",
    "recommended_training_use",
    "sample_weight",
    "calibration_source",
    "conflict_type",
    "pilot_group",
}
FAILURE_BUCKETS = {"no_failure", "visible_failure", "hidden_or_missing_failure", "unclear"}
CONFIDENCES = {"high", "medium", "low"}
BLIND_REVIEW_KEYS = {
    "sample_id", "review_packet_hash", "score_range", "most_plausible_score",
    "failure_bucket", "major_failures", "rubric_evidence", "evaluator_output_evidence",
    "score_cap", "confidence", "needs_adjudication", "review_reason",
}
FINAL_REVIEW_KEYS = {
    "sample_id", "review_packet_hash", "final_score_range", "final_score",
    "final_failure_bucket", "final_major_failures", "final_rubric_evidence",
    "final_output_evidence", "final_confidence", "ambiguity_type", "adjudication_reason",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def validate_review(row: dict[str, Any], packet: dict[str, Any], prefix: str = "") -> list[str]:
    errors: list[str] = []
    sid = str(row.get("sample_id") or "")
    if set(row) != BLIND_REVIEW_KEYS:
        errors.append(
            f"{prefix}{sid}: schema keys differ missing={sorted(BLIND_REVIEW_KEYS - set(row))} "
            f"extra={sorted(set(row) - BLIND_REVIEW_KEYS)}"
        )
    if row.get("review_packet_hash") != packet.get("review_packet_hash"):
        errors.append(f"{prefix}{sid}: review_packet_hash mismatch")
    score_range = row.get("score_range")
    score = row.get("most_plausible_score")
    if not (
        isinstance(score_range, list)
        and len(score_range) == 2
        and all(isinstance(value, int) and 1 <= value <= 5 for value in score_range)
        and score_range[0] <= score_range[1]
    ):
        errors.append(f"{prefix}{sid}: invalid score_range")
    elif not isinstance(score, int) or not score_range[0] <= score <= score_range[1]:
        errors.append(f"{prefix}{sid}: most_plausible_score outside score_range")
    if row.get("failure_bucket") not in FAILURE_BUCKETS:
        errors.append(f"{prefix}{sid}: invalid failure_bucket")
    failures = row.get("major_failures")
    if not isinstance(failures, list) or not failures or not all(isinstance(value, str) and value for value in failures):
        errors.append(f"{prefix}{sid}: invalid major_failures")
    if not str(row.get("rubric_evidence") or "").strip():
        errors.append(f"{prefix}{sid}: missing rubric_evidence")
    evidence = row.get("evaluator_output_evidence")
    if evidence is not None and str(evidence) not in str(packet.get("evaluator_output") or ""):
        errors.append(f"{prefix}{sid}: evaluator_output_evidence is not a substring")
    cap = row.get("score_cap")
    if cap is not None and (not isinstance(cap, int) or not 1 <= cap <= 5):
        errors.append(f"{prefix}{sid}: invalid score_cap")
    if row.get("confidence") not in CONFIDENCES:
        errors.append(f"{prefix}{sid}: invalid confidence")
    if not isinstance(row.get("needs_adjudication"), bool):
        errors.append(f"{prefix}{sid}: needs_adjudication must be boolean")
    if not str(row.get("review_reason") or "").strip():
        errors.append(f"{prefix}{sid}: missing review_reason")
    return errors


def validate_final(row: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    key_errors: list[str] = []
    if set(row) != FINAL_REVIEW_KEYS:
        key_errors.append(
            f"final:{row.get('sample_id')}: schema keys differ "
            f"missing={sorted(FINAL_REVIEW_KEYS - set(row))} extra={sorted(set(row) - FINAL_REVIEW_KEYS)}"
        )
    translated = {
        "sample_id": row.get("sample_id"),
        "review_packet_hash": row.get("review_packet_hash"),
        "score_range": row.get("final_score_range"),
        "most_plausible_score": row.get("final_score"),
        "failure_bucket": row.get("final_failure_bucket"),
        "major_failures": row.get("final_major_failures"),
        "rubric_evidence": row.get("final_rubric_evidence"),
        "evaluator_output_evidence": row.get("final_output_evidence"),
        "score_cap": None,
        "confidence": row.get("final_confidence"),
        "needs_adjudication": False,
        "review_reason": row.get("adjudication_reason"),
    }
    errors = key_errors + validate_review(translated, packet, "final:")
    if row.get("ambiguity_type") not in {
        "none",
        "rubric_ambiguity",
        "answer_key_ambiguity",
        "insufficient_context",
        "metric_scope_ambiguity",
        "annotator_disagreement",
        "other",
    }:
        errors.append(f"final:{row.get('sample_id')}: invalid ambiguity_type")
    return errors


def review_requires_adjudication(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_range = a.get("score_range", [1, 5])
    b_range = b.get("score_range", [1, 5])
    return any(
        [
            a.get("most_plausible_score") != b.get("most_plausible_score"),
            max(a_range[0], b_range[0]) > min(a_range[1], b_range[1]),
            a.get("failure_bucket") != b.get("failure_bucket"),
            bool(a.get("needs_adjudication")),
            bool(b.get("needs_adjudication")),
        ]
    )


def validate(out_dir: Path) -> dict[str, Any]:
    rep = read_jsonl(out_dir / "packets" / "exp27j_representative_blind_packets.jsonl")
    risk = read_jsonl(out_dir / "packets" / "exp27j_risk_enriched_blind_packets.jsonl")
    packets = read_jsonl(out_dir / "packets" / "exp27j_all_blind_packets.jsonl")
    packet_by_id = {str(row.get("sample_id")): row for row in packets}
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": str(passed).lower(), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    ids = [str(row.get("sample_id")) for row in packets]
    qkeys = [str(row.get("question_key")) for row in packets]
    check("representative_rows", len(rep) == 120, str(len(rep)))
    check("risk_enriched_rows", len(risk) == 60, str(len(risk)))
    check("total_rows", len(packets) == 180, str(len(packets)))
    check("unique_sample_ids", len(set(ids)) == 180, str(len(set(ids))))
    check(
        "question_key_design_feasibility",
        len(set(qkeys)) <= 118,
        f"unique={len(set(qkeys))}; train has 118, so 180 unique is impossible and clusters are retained",
    )
    forbidden_count = sum(bool(nested_keys(row) & FORBIDDEN_PACKET_KEYS) for row in packets)
    check("blind_label_leakage", forbidden_count == 0, str(forbidden_count))
    check(
        "packet_hashes_present",
        all(str(row.get("review_packet_hash") or "") for row in packets),
        "all packets require hashes",
    )

    leakage = {row.get("check", ""): int(float(row.get("count") or 0)) for row in read_csv_rows(out_dir / "tables" / "exp27j_leakage_audit.csv")}
    for name in [
        "dev_sample_overlap",
        "dev_question_overlap",
        "test_sample_overlap",
        "test_question_overlap",
        "test_label_read",
        "dev_label_used",
        "blind_label_leakage",
        "human_reason_leakage",
    ]:
        check(name, leakage.get(name, -1) == 0, str(leakage.get(name, "missing")))

    completion: dict[str, int] = {}
    review_rows_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer in ["a", "b"]:
        path = out_dir / "annotation" / f"exp27j_reviewer_{reviewer}_filled.jsonl"
        rows = read_jsonl(path)
        completion[f"reviewer_{reviewer}_rows"] = len(rows)
        if rows:
            review_rows_by_name[reviewer] = {str(row.get("sample_id")): row for row in rows}
            row_ids = {str(row.get("sample_id")) for row in rows}
            check(f"reviewer_{reviewer}_coverage", row_ids == set(ids), f"rows={len(rows)} unique={len(row_ids)}")
            for row in rows:
                sid = str(row.get("sample_id"))
                if sid not in packet_by_id:
                    errors.append(f"reviewer {reviewer}: unknown sample_id {sid}")
                    continue
                errors.extend(validate_review(row, packet_by_id[sid], f"reviewer_{reviewer}:"))

    final_rows = read_jsonl(out_dir / "annotation" / "exp27j_final_adjudication_filled.jsonl")
    completion["final_adjudication_rows"] = len(final_rows)
    for row in final_rows:
        sid = str(row.get("sample_id"))
        if sid not in packet_by_id:
            errors.append(f"final: unknown sample_id {sid}")
            continue
        errors.extend(validate_final(row, packet_by_id[sid]))

    required_ids: set[str] = set()
    if set(review_rows_by_name) == {"a", "b"}:
        required_ids = {
            sid
            for sid in set(review_rows_by_name["a"]) & set(review_rows_by_name["b"])
            if review_requires_adjudication(review_rows_by_name["a"][sid], review_rows_by_name["b"][sid])
        }
        final_ids = {str(row.get("sample_id")) for row in final_rows}
        check(
            "final_adjudication_exact_required_set",
            final_ids == required_ids,
            f"required={len(required_ids)} final={len(final_ids)} "
            f"missing={len(required_ids - final_ids)} extra={len(final_ids - required_ids)}",
        )

    write_csv(out_dir / "tables" / "exp27j_validation_checks.csv", checks)
    decision = {
        "errors": errors,
        "error_count": len(errors),
        "validation_pass": not errors,
        "representative_rows": len(rep),
        "risk_enriched_rows": len(risk),
        "total_rows": len(packets),
        "unique_sample_ids": len(set(ids)),
        "unique_question_keys": len(set(qkeys)),
        "question_key_cluster_analysis_required": len(set(qkeys)) < len(packets),
        "required_adjudication_rows": len(required_ids),
        **completion,
    }
    write_json(out_dir / "decision" / "exp27j_validation_decision.json", decision)
    if errors:
        raise SystemExit("Exp27J validation failed:\n" + "\n".join(errors[:30]))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp27J independent audit artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate(args.out_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
