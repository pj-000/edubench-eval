"""Build Exp37A-R1 selective Reviewer-C packets from complete A/B reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    ROOT, expected_conflict_reasons, read_jsonl, sha256_text, write_csv, write_jsonl,
)


def load_packet_map(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "private_packets/exp37a_r1_reviewer_a_packets.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    args = parser.parse_args()
    a_path = args.reviewer_a or args.out_dir / "private_reviews/reviewer_a_filled.jsonl"
    b_path = args.reviewer_b or args.out_dir / "private_reviews/reviewer_b_filled.jsonl"
    public_path = args.out_dir / "tables/exp37a_r1_expected_adjudication_set.csv"
    if not a_path.exists() or not b_path.exists():
        if not args.allow_missing_reviews:
            raise FileNotFoundError("Complete Reviewer A/B files are required")
        write_csv(public_path, [{"status": "WAITING_FOR_REVIEWER_AB", "expected_conflict_count": "", "test_access_count": 0}])
        print({"status": "WAITING_FOR_REVIEWER_AB", "test_access_count": 0})
        return

    reviewer_a = {str(row["sample_id"]): row for row in read_jsonl(a_path)}
    reviewer_b = {str(row["sample_id"]): row for row in read_jsonl(b_path)}
    if len(reviewer_a) != 196 or len(reviewer_b) != 196 or set(reviewer_a) != set(reviewer_b):
        raise ValueError(f"Reviewer A/B must cover the same 196 IDs: A={len(reviewer_a)} B={len(reviewer_b)}")
    packets = load_packet_map(args.out_dir)
    if set(packets) != set(reviewer_a):
        raise ValueError("Reviewer IDs differ from frozen packet IDs")

    conflicts = []
    adjudication_packets = []
    for sid in sorted(reviewer_a):
        reasons = expected_conflict_reasons(reviewer_a[sid], reviewer_b[sid])
        if not reasons:
            continue
        conflicts.append({"sample_id": sid, "conflict_reasons": reasons})
        adjudication_packets.append({
            "sample_id": sid,
            "packet_hash": packets[sid]["packet_hash"],
            "review_text": packets[sid]["review_text"],
            "reviewer_a": reviewer_a[sid],
            "reviewer_b": reviewer_b[sid],
            "expected_conflict_reasons": reasons,
            "reference_type_required": "multi_session_model_reviewed_silver",
        })
    write_jsonl(args.out_dir / "private_reference/exp37a_r1_expected_conflicts.jsonl", conflicts)
    write_jsonl(args.out_dir / "private_packets/exp37a_r1_adjudication_packets.jsonl", adjudication_packets)
    write_jsonl(args.out_dir / "annotation_templates/exp37a_r1_reviewer_c_template.jsonl", [
        {"sample_id": row["sample_id"], "packet_hash": row["packet_hash"], "review_status": "pending"}
        for row in adjudication_packets
    ])
    write_csv(public_path, [
        {"sample_id_hash": sha256_text(row["sample_id"]), "conflict_reasons": "|".join(row["conflict_reasons"]), "expected_conflict": 1}
        for row in conflicts
    ] or [{"status": "NO_EXPECTED_CONFLICTS", "expected_conflict_count": 0, "test_access_count": 0}])
    print({"status": "ADJUDICATION_PACKETS_READY", "expected_conflict_count": len(conflicts), "test_access_count": 0})


if __name__ == "__main__":
    main()
