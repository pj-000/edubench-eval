"""Render the lightweight Exp39B protocol or pilot decision report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import ROOT, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def source_no_go(args: argparse.Namespace, prepare: dict) -> dict:
    decision = {
        "status": "SOURCE_POOL_NO_GO", "recommend_scale_to_fresh_240": False,
        "stop_positive_counterfactual_route": False,
        "reason": prepare.get("failure_reason"), "pilot_not_executed": True,
        "api_calls_made": False, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_rlcr_pilot_decision.json", decision)
    report = [
        "# Exp39B EduCFA-RLCR Pilot", "",
        "## Decision", "", "- Status: **SOURCE_POOL_NO_GO**",
        "- The generation method was not evaluated because the frozen source-isolation gate failed.",
        f"- Fresh sources/question keys: `{prepare.get('fresh_source_count', 0)} / {prepare.get('fresh_question_key_count', 0)}`",
        f"- Reason: {prepare.get('failure_reason')}", "",
        "## Interpretation", "",
        "Exp39A used at least one source from every paper-like train question key. Therefore no row can satisfy both the fresh-source-ID and fresh-question-key requirements.",
        "The protocol did not relax isolation, did not call Qwen or DeepSeek, and did not access dev/test.",
        "This is a source-pool incompatibility, not evidence that RLCR generation failed.", "",
        "## Compliance", "",
        "- No GPU.", "- No training or GroupCV.", "- No dev/test access.",
        "- No API call and no private/heavy artifact committed.",
    ]
    (args.out_dir / "reports/exp39b_rlcr_pilot_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports/exp39b_rlcr_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return decision


def completed_report(args: argparse.Namespace, decision: dict) -> None:
    summary = decision["summary"]
    report = [
        "# Exp39B EduCFA-RLCR Pilot", "", "## Decision", "",
        f"- Status: **{decision['status']}**",
        f"- Scale to fresh 240: `{decision['recommend_scale_to_fresh_240']}`",
        f"- Failed gates: `{', '.join(decision['failed_gates']) or 'none'}`", "",
        "## Core results", "",
    ]
    labels = [
        ("source_count", "Fresh sources"), ("source_question_keys", "Fresh question keys"),
        ("plan_valid_count", "Plan-valid"), ("first_pass_target_coverage", "First-pass target coverage"),
        ("first_pass_acceptance", "First-pass acceptance"), ("revision_attempted", "Revisions attempted"),
        ("revision_rescued", "Revisions rescued"), ("revision_rescue_rate", "Revision rescue rate"),
        ("final_target_band_coverage", "Final target coverage"), ("final_acceptance", "Final acceptance"),
        ("final_acceptance_rate", "Final acceptance rate"),
        ("outside_span_identity_rate", "Outside-span identity"),
        ("edit_budget_pass_rate", "Edit-budget pass rate"),
        ("rubric_failure_verified_rate", "Rubric-failure verified"),
        ("non_target_preservation_rate", "Non-target preservation"),
    ]
    report.extend(f"- {label}: `{summary.get(key)}`" for key, label in labels)
    report.extend(["", "## Compliance", "", "- No GPU or training.", "- No dev/test access.", "- Private text and API artifacts remain ignored."])
    (args.out_dir / "reports/exp39b_rlcr_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare = read_json(args.out_dir / "decision/exp39b_protocol_prepare_decision.json")
    if prepare.get("status") != "PROTOCOL_PREPARE_GO":
        decision = source_no_go(args, prepare)
    else:
        decision = read_json(args.out_dir / "decision/exp39b_rlcr_pilot_decision.json")
        completed_report(args, decision)
    print(json.dumps({"status": decision["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
