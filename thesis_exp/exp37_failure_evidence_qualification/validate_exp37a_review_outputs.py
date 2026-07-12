"""Validate Exp37A packets and externally filled structured reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    FAILURE_CLASSES, ROOT, normalized_substring, packet_hash, read_jsonl, write_csv, write_json,
)

ALLOWED_BUCKETS = {"no_failure", "visible_failure", "hidden_or_missing_failure", "unclear"}
ALLOWED_FAILURES = set(FAILURE_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    return parser.parse_args()


def packet_map(out_dir: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted((out_dir / "private_packets").glob("exp37a_*_blind_packet.jsonl")):
        if "adjudication" in path.name:
            continue
        for packet in read_jsonl(path):
            sid = str(packet["sample_id"])
            if sid in output:
                raise ValueError(f"Duplicate sample across packets: {sid}")
            if packet.get("packet_hash") != packet_hash(packet):
                raise ValueError(f"Packet hash mismatch: {sid}")
            output[sid] = packet
    if len(output) != 196:
        raise ValueError(f"Expected 196 exact-disjoint packet rows, found {len(output)}")
    return output


def validate_review(row: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(row.get("sample_id")) != str(packet["sample_id"]):
        errors.append("sample_id_mismatch")
    if row.get("target_scope_confirmed") is not True:
        errors.append("target_scope_not_confirmed")
    score_range = row.get("score_range")
    if not isinstance(score_range, list) or len(score_range) != 2 or not all(isinstance(v, int) and 1 <= v <= 5 for v in score_range) or score_range[0] > score_range[1]:
        errors.append("invalid_score_range")
    plausible = row.get("most_plausible_score")
    if not isinstance(plausible, int) or not isinstance(score_range, list) or len(score_range) != 2 or not (score_range[0] <= plausible <= score_range[1]):
        errors.append("plausible_score_outside_range")
    if row.get("failure_bucket") not in ALLOWED_BUCKETS:
        errors.append("invalid_failure_bucket")
    failures = row.get("failure_classes")
    if not isinstance(failures, list) or not all(value in ALLOWED_FAILURES for value in failures):
        errors.append("invalid_failure_classes")
    if "no_major_failure" in failures and len(failures) != 1:
        errors.append("no_major_failure_not_exclusive")
    evidence = row.get("evaluator_output_evidence")
    if not isinstance(evidence, list) or not all(isinstance(value, str) and normalized_substring(value, str(packet.get("evaluator_output", ""))) for value in evidence):
        errors.append("evidence_not_normalized_substring")
    if not evidence and not str(row.get("missing_evidence_reason") or "").strip():
        errors.append("missing_evidence_reason_required")
    if row.get("confidence") not in {"high", "medium", "low"}:
        errors.append("invalid_confidence")
    if row.get("confidence") == "low" and row.get("needs_adjudication") is not True:
        errors.append("low_confidence_requires_adjudication")
    if not str(row.get("review_reason") or "").strip():
        errors.append("review_reason_missing")
    return errors


def validate_file(path: Path | None, packets: dict[str, dict[str, Any]], role: str, allow_missing: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        path = packets and Path("__missing__")
    if not path.exists():
        if not allow_missing:
            raise FileNotFoundError(f"Missing {role} review: {path}")
        return [], {"role": role, "status": "missing", "rows": 0, "valid_rows": 0, "invalid_rows": 0, "duplicate_rows": 0}
    rows = read_jsonl(path)
    seen: set[str] = set()
    invalid = []
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in seen:
            invalid.append({"sample_id": sid, "errors": ["duplicate_sample_id"]})
        seen.add(sid)
        if sid not in packets:
            invalid.append({"sample_id": sid, "errors": ["sample_not_in_packet"]})
        else:
            errors = validate_review(row, packets[sid])
            if role == "adjudication":
                if row.get("reference_type") != "human_rationale_grounded_model_reviewed_silver":
                    errors.append("invalid_reference_type")
                if not str(row.get("adjudication_reason") or "").strip():
                    errors.append("adjudication_reason_missing")
            if errors:
                invalid.append({"sample_id": sid, "errors": errors})
    summary = {"role": role, "status": "PASS" if not invalid and len(rows) == len(packets) else "FAIL", "rows": len(rows), "valid_rows": len(rows) - len(invalid), "invalid_rows": len(invalid), "duplicate_rows": sum("duplicate_sample_id" in x["errors"] for x in invalid)}
    return rows, summary


def main() -> None:
    args = parse_args()
    packets = packet_map(args.out_dir)
    a, sa = validate_file(args.reviewer_a or args.out_dir / "private_reviews/reviewer_a_filled.jsonl", packets, "reviewer_a", args.allow_missing_reviews)
    b, sb = validate_file(args.reviewer_b or args.out_dir / "private_reviews/reviewer_b_filled.jsonl", packets, "reviewer_b", args.allow_missing_reviews)
    c, sc = validate_file(args.adjudication or args.out_dir / "private_reviews/adjudication_filled.jsonl", packets, "adjudication", args.allow_missing_reviews)
    completion = [sa, sb, sc]
    write_csv(args.out_dir / "tables/exp37a_review_completion.csv", completion)
    forbidden = {"human_1", "human_2", "human_3", "label_5", "human_reason", "qwen", "deepseek", "oof", "teacher", "exp36"}
    def key_hits(value: Any) -> set[str]:
        hits: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden:
                    hits.add(str(key).lower())
                hits.update(key_hits(child))
        elif isinstance(value, list):
            for child in value:
                hits.update(key_hits(child))
        return hits
    leakage = []
    for sid, packet in packets.items():
        hits = sorted(key_hits(packet))
        leakage.append({"sample_id_hash": packet.get("anonymized_question_key_hash"), "forbidden_field_hits": "|".join(hits), "pass": not hits})
    write_csv(args.out_dir / "tables/exp37a_leakage_audit.csv", leakage)
    status = "PASS" if all(row["status"] in {"PASS", "missing"} for row in completion) and not any(not row["pass"] for row in leakage) else "FAIL"
    write_json(args.out_dir / "decision/exp37a_validation.json", {
        "status": status, "allow_missing_reviews": args.allow_missing_reviews,
        "review_completion": completion, "blind_packet_leakage_count": sum(not row["pass"] for row in leakage),
        "reference_complete": bool(sc["status"] == "PASS"), "test_access_count": 0,
    })
    if status == "FAIL":
        raise SystemExit("Exp37A validation failed")
    print({"status": status, "review_completion": completion, "test_access_count": 0})


if __name__ == "__main__":
    main()
