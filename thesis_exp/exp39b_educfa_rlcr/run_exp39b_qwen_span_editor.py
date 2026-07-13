"""Ask Qwen for replacement payloads for validated Exp39B locked spans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT, compact_json, index_rows, model_from_protocol, read_json, read_jsonl,
    require_prepare_go, run_json_stage, write_stage_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def build_user(row: dict, _: dict) -> str:
    payload = {
        "sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
        "source_evidence_span": row["source_evidence_span"], "operator": row["operator"],
        "rubric_clause": row["rubric_clause"], "target_band": row["target_band"],
        "local_context": row["local_context"],
    }
    return "Return only the replacement payload for the locked span.\n" + compact_json(payload)


def semantic_errors(value: dict, row: dict) -> list[str]:
    errors = []
    for key in ("sample_id", "source_sample_id", "source_evidence_span", "operator"):
        if value.get(key) != row.get(key):
            errors.append(f"{key} mismatch")
    replacement = str(value.get("replacement_text") or "")
    if replacement and replacement == row.get("original_answer"):
        errors.append("replacement contains full original answer")
    return errors


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    decision = read_json(args.out_dir / "decision/exp39b_plan_validation_decision.json")
    if decision.get("status") != "PLAN_VALIDATION_GO":
        raise SystemExit("Exp39B editor blocked: plan validation is not GO")
    valid = set(decision["valid_sample_ids"])
    packets = index_rows(read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"))
    plans = index_rows(read_jsonl(args.out_dir / "private/clause_plans/exp39b_clause_plans.jsonl"))
    rows = []
    for sid in sorted(valid):
        packet = packets[sid]
        plan = plans[sid]
        span = plan["source_evidence_span"]
        start = packet["original_answer"].find(span)
        rows.append({
            **plan,
            "original_answer": packet["original_answer"],
            "local_context": packet["original_answer"][max(0, start - 240): start + len(span) + 240],
        })
    summary = run_json_stage(
        rows=rows,
        prompt_path=Path("thesis_exp/exp39b_educfa_rlcr/prompts/exp39b_qwen_span_editor.md"),
        schema_path=Path("thesis_exp/exp39b_educfa_rlcr/schemas/exp39b_span_edit_schema.json"),
        parsed_path=args.out_dir / "private/first_pass_edits/exp39b_first_pass_edits.jsonl",
        raw_path=args.out_dir / "raw_api/qwen_span_editor.jsonl",
        provider="qwen", model=model_from_protocol(args.out_dir, "qwen_model"), max_tokens=1000,
        build_user=build_user, semantic_errors=semantic_errors, dry_run=args.dry_run,
        max_rows=args.max_rows, workers=args.workers, timeout=args.timeout, retries=args.retries,
        stage_name="qwen_span_editor",
    )
    write_stage_summary(args.out_dir, "qwen_span_editor", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
