"""Apply the frozen controller and allow at most one same-span Qwen revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT, compact_json, edit_budget, index_rows, model_from_protocol, patch_locked_span,
    range_intersects, read_jsonl, require_prepare_go, run_json_stage, write_jsonl,
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


def controller(row: dict, critic: dict) -> dict:
    target = row["target_band"]
    predicted = critic["counterfactual_score_range"]
    within = range_intersects(predicted, target) and target[0] <= critic["most_plausible_counterfactual_score"] <= target[1]
    if within:
        relation = "within_band"
    elif predicted[0] > target[1]:
        relation = "too_high"
    elif predicted[1] < target[0]:
        relation = "too_low"
    else:
        relation = "range_overlap_center_miss"
    blockers = []
    if critic["original_score_range"][1] < 4:
        blockers.append("source_not_verified_high")
    if critic["answer_key_uncertainty"]:
        blockers.append("answer_key_uncertainty")
    if critic["feedback_code"] == "invalid_clause":
        blockers.append("invalid_clause")
    if not row["outside_span_identity"] or not row["edit_budget_pass"]:
        blockers.append("invalid_first_patch")
    if critic["feedback_code"] == "non_target_damage" and critic["edit_locality"] == "fail":
        # A same-span revision can repair local damage, but not damage attributed outside the lock.
        blockers.append("non_target_damage_not_local")
    repair_issue = critic["edit_locality"] == "fail" or critic["non_target_preservation"] == "fail"
    eligible = not blockers and critic["clause_failure_strength"] in {"partial", "clear"} and (not within or repair_issue)
    return {
        "critic_target_relation": relation, "first_pass_within_band": within,
        "revision_eligible": eligible, "revision_blockers": blockers,
    }


def build_user(row: dict, _: dict) -> str:
    payload = {
        "sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
        "source_evidence_span": row["source_evidence_span"], "current_replacement": row["replacement_text"],
        "target_band": row["target_band"], "rubric_clause": row["rubric_clause"],
        "operator": row["operator"], "critic_feedback_code": row["critic"]["feedback_code"],
        "critic_feedback": row["critic"]["feedback"],
        "critic_score_range": row["critic"]["counterfactual_score_range"],
    }
    return "Revise only the replacement payload for the exact same locked span.\n" + compact_json(payload)


def semantic_errors(value: dict, row: dict) -> list[str]:
    errors = []
    for key in ("sample_id", "source_sample_id", "source_evidence_span", "operator"):
        if value.get(key) != row.get(key):
            errors.append(f"{key} mismatch")
    if str(value.get("replacement_text") or "") == row.get("original_answer"):
        errors.append("replacement contains full original answer")
    return errors


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    first = index_rows(read_jsonl(args.out_dir / "private/first_pass_candidates/exp39b_first_pass_candidates.jsonl"))
    critics = index_rows(read_jsonl(args.out_dir / "private/critic_feedback/exp39b_blind_critic.jsonl"))
    eligible_rows = []
    controls = {}
    for sid, row in first.items():
        control = controller(row, critics[sid])
        controls[sid] = control
        if control["revision_eligible"]:
            eligible_rows.append({**row, "critic": critics[sid], **control})
    if args.max_rows:
        eligible_rows = eligible_rows[: args.max_rows]
    if args.dry_run:
        if not eligible_rows:
            print(json.dumps({"status": "DRY_RUN", "stage": "qwen_restricted_revision", "rows": 0, "api_called": False}))
            return
    if eligible_rows:
        summary = run_json_stage(
            rows=eligible_rows,
            prompt_path=Path("thesis_exp/exp39b_educfa_rlcr/prompts/exp39b_qwen_restricted_revision.md"),
            schema_path=Path("thesis_exp/exp39b_educfa_rlcr/schemas/exp39b_restricted_revision_schema.json"),
            parsed_path=args.out_dir / "private/revisions/exp39b_restricted_revisions_api.jsonl",
            raw_path=args.out_dir / "raw_api/qwen_restricted_revision.jsonl",
            provider="qwen", model=model_from_protocol(args.out_dir, "qwen_model"), max_tokens=1000,
            build_user=build_user, semantic_errors=semantic_errors, dry_run=args.dry_run,
            max_rows=None, workers=args.workers, timeout=args.timeout, retries=args.retries,
            stage_name="qwen_restricted_revision",
        )
        if args.dry_run:
            write_stage_summary(args.out_dir, "qwen_restricted_revision", summary)
            print(json.dumps(summary, sort_keys=True))
            return
    else:
        summary = {"status": "COMPLETED", "stage": "qwen_restricted_revision", "rows": 0, "api_called": False}
    revised = index_rows(read_jsonl(
        args.out_dir / "private/revisions/exp39b_restricted_revisions_api.jsonl"
    )) if (args.out_dir / "private/revisions/exp39b_restricted_revisions_api.jsonl").exists() else {}
    final_rows = []
    revision_rows = []
    for sid, row in first.items():
        attempted = controls[sid]["revision_eligible"] and sid in revised
        replacement = str(revised[sid]["replacement_text"] if attempted else row["replacement_text"])
        patched = patch_locked_span(
            row["original_answer"], row["source_evidence_span"], int(row["source_span_occurrence"]), replacement
        )
        budget = edit_budget(row["original_answer"], row["source_evidence_span"], replacement, row["operator"])
        final_rows.append({
            **row, "final_replacement_text": replacement, "final_counterfactual": patched["candidate"],
            "final_outside_span_identity": patched["outside_span_identity"],
            "final_patch_start": patched["start"], "final_patch_end": patched["end"],
            "final_edit_budget_pass": budget["edit_budget_pass"],
            "final_span_token_ratio": budget["span_token_ratio"],
            "final_inserted_token_ratio": budget["inserted_token_ratio"],
            "final_token_length_ratio": budget["final_token_length_ratio"],
            "revision_attempted": attempted, **controls[sid],
        })
        revision_rows.append({
            "sample_id": sid, "source_sample_id": row["source_sample_id"], "revision_attempted": attempted,
            "revision_eligible": controls[sid]["revision_eligible"],
            "revision_blockers": controls[sid]["revision_blockers"],
            "critic_target_relation": controls[sid]["critic_target_relation"],
            "source_span_unchanged": True, "outside_span_identity": patched["outside_span_identity"], **budget,
        })
    write_jsonl(args.out_dir / "private/revisions/exp39b_revision_controller.jsonl", revision_rows)
    write_jsonl(args.out_dir / "private/final_candidates/exp39b_final_candidates.jsonl", final_rows)
    summary.update({
        "total_rows": len(final_rows), "revision_eligible_count": sum(row["revision_eligible"] for row in final_rows),
        "revision_attempted_count": sum(row["revision_attempted"] for row in final_rows),
        "dev_access_count": 0, "test_access_count": 0,
    })
    write_stage_summary(args.out_dir, "qwen_restricted_revision", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
