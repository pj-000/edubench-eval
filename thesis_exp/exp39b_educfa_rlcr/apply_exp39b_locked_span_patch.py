"""Programmatically apply exactly one validated Exp39B source-span replacement."""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT, edit_budget, index_rows, patch_locked_span, read_json, read_jsonl,
    require_prepare_go, write_csv, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    decision = read_json(args.out_dir / "decision/exp39b_plan_validation_decision.json")
    valid = set(decision.get("valid_sample_ids", []))
    packets = index_rows(read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"))
    plans = index_rows(read_jsonl(args.out_dir / "private/clause_plans/exp39b_clause_plans.jsonl"))
    edits = index_rows(read_jsonl(args.out_dir / "private/first_pass_edits/exp39b_first_pass_edits.jsonl"))
    rows = []
    audits = []
    for sid in sorted(valid):
        packet, plan, edit = packets[sid], plans[sid], edits[sid]
        replacement = str(edit["replacement_text"])
        patched = patch_locked_span(
            packet["original_answer"], plan["source_evidence_span"], int(plan["source_span_occurrence"]), replacement
        )
        budget = edit_budget(packet["original_answer"], plan["source_evidence_span"], replacement, plan["operator"])
        row = {
            **packet, "rubric_clause": plan["rubric_clause"], "clause_type": plan["clause_type"],
            "source_evidence_span": plan["source_evidence_span"],
            "source_span_occurrence": plan["source_span_occurrence"], "operator": plan["operator"],
            "replacement_text": replacement, "first_pass_counterfactual": patched["candidate"],
            "patch_start": patched["start"], "patch_end": patched["end"],
            "outside_span_identity": patched["outside_span_identity"], **budget,
        }
        rows.append(row)
        audits.append({
            "sample_id": sid, "stage": "first_pass", "target_band_name": packet["target_band_name"],
            "operator": plan["operator"], "outside_span_identity": patched["outside_span_identity"],
            "one_span_only": True, "sequence_similarity": SequenceMatcher(
                None, packet["original_answer"], patched["candidate"]
            ).ratio(), **budget,
        })
    write_jsonl(args.out_dir / "private/first_pass_candidates/exp39b_first_pass_candidates.jsonl", rows)
    write_csv(args.out_dir / "tables/exp39b_edit_budget_audit.csv", audits)
    summary = {
        "rows": len(rows), "outside_span_identity_count": sum(row["outside_span_identity"] for row in rows),
        "edit_budget_pass_count": sum(row["edit_budget_pass"] for row in rows),
        "dev_access_count": 0, "test_access_count": 0,
    }
    print(json.dumps(summary, sort_keys=True))
    if len(rows) != len(valid):
        raise SystemExit("First-pass patch row count mismatch")


if __name__ == "__main__":
    main()
