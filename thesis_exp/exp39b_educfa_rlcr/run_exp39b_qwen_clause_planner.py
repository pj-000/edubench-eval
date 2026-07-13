"""Run the target-aware Qwen rubric-clause planner after the fresh-source gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT,
    compact_json,
    locate_occurrence,
    model_from_protocol,
    read_jsonl,
    require_prepare_go,
    run_json_stage,
    sample_id,
    write_stage_summary,
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
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "question": row["question"],
        "original_high_score_answer": row["original_answer"],
        "metric": row["metric"],
        "rubric": row["rubric"],
        "metadata": row["metadata"],
        "assigned_target_band": row["target_band"],
        "assigned_target_band_name": row["target_band_name"],
    }
    return "Plan one rubric-locked edit for this record.\n" + compact_json(payload)


def semantic_errors(value: dict, row: dict) -> list[str]:
    errors: list[str] = []
    for key in ("sample_id", "source_sample_id"):
        if str(value.get(key)) != str(row[key]):
            errors.append(f"{key} mismatch")
    if value.get("target_band") != row["target_band"]:
        errors.append("target_band mismatch")
    try:
        locate_occurrence(
            row["original_answer"],
            str(value.get("source_evidence_span") or ""),
            int(value.get("source_span_occurrence", -1)),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    packets = read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl")
    summary = run_json_stage(
        rows=packets,
        prompt_path=Path("thesis_exp/exp39b_educfa_rlcr/prompts/exp39b_qwen_clause_planner.md"),
        schema_path=Path("thesis_exp/exp39b_educfa_rlcr/schemas/exp39b_clause_plan_schema.json"),
        parsed_path=args.out_dir / "private/clause_plans/exp39b_clause_plans.jsonl",
        raw_path=args.out_dir / "raw_api/qwen_clause_planner.jsonl",
        provider="qwen",
        model=model_from_protocol(args.out_dir, "qwen_model"),
        max_tokens=1800,
        build_user=build_user,
        semantic_errors=semantic_errors,
        dry_run=args.dry_run,
        max_rows=args.max_rows,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        stage_name="qwen_clause_planner",
    )
    write_stage_summary(args.out_dir, "qwen_clause_planner", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
