"""Run the target-blind first-pass DeepSeek critic for Exp39B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (
    ROOT, compact_json, model_from_protocol, read_jsonl, require_prepare_go,
    run_json_stage, write_stage_summary,
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
    # Target band, human scores, and generator expectations are deliberately omitted.
    payload = {
        "sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
        "original_answer": row["original_answer"],
        "counterfactual_answer": row["first_pass_counterfactual"],
        "metric": row["metric"], "rubric": row["rubric"],
        "selected_rubric_clause": row["rubric_clause"], "operator": row["operator"],
    }
    return "Independently score and critique this locked-span edit.\n" + compact_json(payload)


def semantic_errors(value: dict, row: dict) -> list[str]:
    errors = []
    for key in ("sample_id", "source_sample_id"):
        if value.get(key) != row.get(key):
            errors.append(f"{key} mismatch")
    for key in ("original_score_range", "counterfactual_score_range"):
        values = value.get(key, [])
        if len(values) == 2 and int(values[0]) > int(values[1]):
            errors.append(f"{key} not ordered")
    score_range = value.get("counterfactual_score_range", [])
    score = int(value.get("most_plausible_counterfactual_score", 0))
    if len(score_range) == 2 and not int(score_range[0]) <= score <= int(score_range[1]):
        errors.append("most plausible score outside counterfactual range")
    return errors


def main() -> None:
    args = parse_args()
    require_prepare_go(args.out_dir)
    rows = read_jsonl(args.out_dir / "private/first_pass_candidates/exp39b_first_pass_candidates.jsonl")
    summary = run_json_stage(
        rows=rows,
        prompt_path=Path("thesis_exp/exp39b_educfa_rlcr/prompts/exp39b_deepseek_blind_critic.md"),
        schema_path=Path("thesis_exp/exp39b_educfa_rlcr/schemas/exp39b_blind_critic_schema.json"),
        parsed_path=args.out_dir / "private/critic_feedback/exp39b_blind_critic.jsonl",
        raw_path=args.out_dir / "raw_api/deepseek_blind_critic.jsonl",
        provider="deepseek", model=model_from_protocol(args.out_dir, "deepseek_model"), max_tokens=1500,
        build_user=build_user, semantic_errors=semantic_errors, dry_run=args.dry_run,
        max_rows=args.max_rows, workers=args.workers, timeout=args.timeout, retries=args.retries,
        stage_name="deepseek_blind_critic",
    )
    write_stage_summary(args.out_dir, "deepseek_blind_critic", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
