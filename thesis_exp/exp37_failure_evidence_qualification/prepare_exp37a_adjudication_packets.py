"""Build private Reviewer-C packets from blind A/B reviews and strict reasons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import ROOT, read_jsonl, write_jsonl  # noqa: E402


def needs_adjudication(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("most_plausible_score") != right.get("most_plausible_score"):
        return True
    if left.get("score_range") != right.get("score_range"):
        return True
    if left.get("failure_bucket") != right.get("failure_bucket"):
        return True
    if set(left.get("failure_classes", [])) != set(right.get("failure_classes", [])):
        return True
    if set(left.get("evaluator_output_evidence", [])) & set(right.get("evaluator_output_evidence", [])) == set() and (left.get("evaluator_output_evidence") or right.get("evaluator_output_evidence")):
        return True
    return left.get("confidence") == "low" or right.get("confidence") == "low" or left.get("needs_adjudication") is True or right.get("needs_adjudication") is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir
    a_path = args.reviewer_a or out_dir / "private_reviews/reviewer_a_filled.jsonl"
    b_path = args.reviewer_b or out_dir / "private_reviews/reviewer_b_filled.jsonl"
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Reviewer A and B files are required before adjudication packets can be built")
    a = {str(row["sample_id"]): row for row in read_jsonl(a_path)}
    b = {str(row["sample_id"]): row for row in read_jsonl(b_path)}
    reasons: dict[str, list[dict[str, Any]]] = {}
    reason_path = out_dir / "private_reference/recovered_human_reasons.jsonl"
    if reason_path.exists():
        for row in read_jsonl(reason_path):
            reasons.setdefault(str(row["sample_id"]), []).append(row)
    packets = []
    for packet_path in sorted((out_dir / "private_packets").glob("exp37a_*_blind_packet.jsonl")):
        for packet in read_jsonl(packet_path):
            sid = str(packet["sample_id"])
            if sid not in a or sid not in b or not needs_adjudication(a[sid], b[sid]):
                continue
            packets.append({
                "sample_id": sid,
                "packet_hash": packet["packet_hash"],
                "review_text": packet.get("review_text"),
                "reviewer_a": a[sid],
                "reviewer_b": b[sid],
                "strict_recovered_human_reasons": reasons.get(sid, []),
                "reference_type_required": "human_rationale_grounded_model_reviewed_silver",
            })
    write_jsonl(out_dir / "private_packets/exp37a_adjudication_packets.jsonl", packets)
    print({"adjudication_packet_count": len(packets), "test_access_count": 0})


if __name__ == "__main__":
    main()
