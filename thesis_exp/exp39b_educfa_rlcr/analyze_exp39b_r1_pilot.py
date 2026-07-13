"""Create R1-named outputs, confidence intervals, report, and final decision."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (  # noqa: E402
    R1_ROOT, read_csv, read_json, read_jsonl, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=R1_ROOT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, center - margin), min(1.0, center + margin)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def copy_tables(out_dir: Path) -> None:
    names = (
        "plan_validation", "first_pass_metrics", "revision_metrics", "final_acceptance_summary",
        "acceptance_by_band", "acceptance_by_operator", "acceptance_by_metric", "edit_budget_audit",
        "rejection_reason_distribution", "revision_rescue_analysis", "leakage_audit",
    )
    for name in names:
        source = out_dir / f"tables/exp39b_{name}.csv"
        if source.exists():
            shutil.copy2(source, out_dir / f"tables/exp39b_r1_{name}.csv")


def main() -> None:
    args = parse_args()
    source_lock = read_json(args.out_dir / "configs/exp39b_r1_source_lock.json")
    novelty = read_csv(args.out_dir / "tables/exp39b_r1_source_novelty_audit.csv")[0]
    plan_decision_path = args.out_dir / "decision/exp39b_plan_validation_decision.json"
    plan_decision = read_json(plan_decision_path)
    if plan_decision.get("status") != "PLAN_VALIDATION_GO":
        plan_rows = read_csv(args.out_dir / "tables/exp39b_plan_validation.csv")
        shutil.copy2(
            args.out_dir / "tables/exp39b_plan_validation.csv",
            args.out_dir / "tables/exp39b_r1_plan_validation.csv",
        )
        reason_counts = Counter(
            reason for row in plan_rows for reason in row["rejection_reasons"].split("|") if reason
        )
        write_csv(
            args.out_dir / "tables/exp39b_r1_rejection_reason_distribution.csv",
            [{"stage": "planner", "rejection_reason": reason, "count": count, "rate_over_60": count / 60}
             for reason, count in reason_counts.most_common()],
        )
        decision = {
            "status": "PROTOCOL_PILOT_NO_GO",
            "stopped_at": "plan_validation",
            "plan_valid_count": int(plan_decision.get("plan_valid_count", 0)),
            "plan_valid_min": int(plan_decision.get("plan_valid_min", 54)),
            "recommend_scale_to_response_disjoint_240": False,
            "stop_positive_counterfactual_route": True,
            "inference_scope": source_lock["inference_scope"],
            "failed_gates": ["plan_valid"],
            "downstream_api_stages_executed": False,
            "dev_access_count": 0, "test_access_count": 0,
        }
        write_json(args.out_dir / "decision/exp39b_r1_pilot_decision.json", decision)
        report = [
            "# Exp39B-R1 EduCFA-RLCR Response-Disjoint Pilot", "",
            "## Decision", "", "- Status: **PROTOCOL_PILOT_NO_GO**",
            "- Stopped at: `plan_validation`",
            f"- Plan-valid: `{decision['plan_valid_count']}/60` (required `{decision['plan_valid_min']}/60`)",
            "- Qwen editor, DeepSeek critic, revision, and final verifier were not run.", "",
            "## Source preflight", "",
            f"- Selected rows / qkeys: `{novelty['selected_rows']} / {novelty['selected_question_keys']}`",
            f"- Source-ID / answer-hash / assessment-key overlap: `{novelty['source_id_overlap']} / {novelty['exact_answer_hash_overlap']} / {novelty['assessment_key_overlap']}`",
            f"- Question-key overlap with Exp39A: `{novelty['question_key_overlap_count']}` (expected and allowed)",
            f"- Max character/token Jaccard: `{float(novelty['max_char_5gram_similarity']):.4f} / {float(novelty['max_token_jaccard']):.4f}`", "",
            "## Planner rejection reasons", "",
        ]
        report.extend(f"- {reason}: `{count}`" for reason, count in reason_counts.most_common())
        report.extend([
            "", "## Interpretation", "",
            "The response-disjoint source amendment passed. The frozen planner protocol did not reach its pre-registered compatibility gate.",
            "This is a planner-level protocol failure; no downstream generation-quality claim is made.",
            "The result is not unseen-question validation.", "",
            "## Compliance", "", "- No GPU or training.", "- No dev/test access.",
            "- Raw/private API artifacts remain ignored.",
        ])
        (args.out_dir / "reports/exp39b_r1_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps({"status": decision["status"], "stopped_at": decision["stopped_at"]}, sort_keys=True))
        return

    generic = read_json(args.out_dir / "decision/exp39b_rlcr_pilot_decision.json")
    validation_rows = read_jsonl(args.out_dir / "private/final_verifications/exp39b_validation_rows.jsonl")
    copy_tables(args.out_dir)
    accepted_flags = [bool(row["accepted"]) for row in validation_rows]
    successes = sum(accepted_flags)
    rate, lower, upper = wilson(successes, len(accepted_flags))
    write_csv(args.out_dir / "tables/exp39b_r1_acceptance_wilson_ci.csv", [{
        "scope": "overall", "accepted": successes, "n": len(accepted_flags),
        "acceptance_rate": rate, "wilson_95_low": lower, "wilson_95_high": upper,
    }])
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap_resamples):
        sample = [accepted_flags[rng.randrange(len(accepted_flags))] for _ in accepted_flags]
        boot.append(sum(sample) / len(sample))
    write_csv(args.out_dir / "tables/exp39b_r1_question_key_bootstrap_ci.csv", [{
        "resamples": args.bootstrap_resamples, "point_estimate": rate,
        "bootstrap_95_low": percentile(boot, 0.025), "bootstrap_95_high": percentile(boot, 0.975),
        "resampling_unit": "question_key_one_row_each",
    }])
    summary = generic["summary"]
    success = generic["status"] == "PROTOCOL_PILOT_GO"
    decision = {
        "status": generic["status"],
        "recommend_scale_to_response_disjoint_240": success,
        "stop_positive_counterfactual_route": not success,
        "inference_scope": source_lock["inference_scope"],
        "question_key_overlap_with_exp39a_expected": True,
        "failed_gates": generic.get("failed_gates", []), "summary": summary,
        "acceptance_wilson_95": [lower, upper],
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_r1_pilot_decision.json", decision)
    final_scores = Counter(int(row["final_score"]) for row in validation_rows if row["accepted"])
    report = [
        "# Exp39B-R1 EduCFA-RLCR Response-Disjoint Pilot", "",
        "## Decision", "", f"- Status: **{decision['status']}**",
        f"- Recommend scale to response-disjoint 240: `{success}`",
        f"- Failed gates: `{', '.join(decision['failed_gates']) or 'none'}`", "",
        "## Source novelty", "",
        f"- Selected rows / qkeys: `{novelty['selected_rows']} / {novelty['selected_question_keys']}`",
        f"- Source-ID / answer-hash / assessment-key overlap: `{novelty['source_id_overlap']} / {novelty['exact_answer_hash_overlap']} / {novelty['assessment_key_overlap']}`",
        f"- Question-key overlap with Exp39A: `{novelty['question_key_overlap_count']}` (expected and allowed)",
        f"- Max character/token Jaccard: `{float(novelty['max_char_5gram_similarity']):.4f} / {float(novelty['max_token_jaccard']):.4f}`",
        f"- Inference scope: `{source_lock['inference_scope']}`", "",
        "## Protocol results", "",
        f"- Plan-valid: `{summary['plan_valid_count']}/60`",
        f"- First-pass target coverage / acceptance: `{summary['first_pass_target_coverage']} / {summary['first_pass_acceptance']}`",
        f"- Revision attempted / rescued / rate: `{summary['revision_attempted']} / {summary['revision_rescued']} / {summary['revision_rescue_rate']:.4f}`",
        f"- Final target coverage: `{summary['final_target_band_coverage']}/60`",
        f"- Final acceptance: `{successes}/60 = {rate:.4f}`; Wilson 95% CI `[{lower:.4f}, {upper:.4f}]`",
        f"- Accepted final score distribution: `{dict(sorted(final_scores.items()))}`",
        f"- Outside-span identity / edit budget: `{summary['outside_span_identity_rate']:.4f} / {summary['edit_budget_pass_rate']:.4f}`",
        f"- Rubric failure / non-target preservation: `{summary['rubric_failure_verified_rate']:.4f} / {summary['non_target_preservation_rate']:.4f}`", "",
        "## Conclusion boundary", "",
        "This pilot evaluates response-level feasibility within previously seen question clusters. It is not unseen-question validation.",
        "Downstream GroupCV must remain question-key disjoint. No dev/test data was accessed.", "",
        "## Compliance", "", "- No GPU or training.", "- API raw/private text remains ignored.",
        "- No dev/test access and no private/heavy artifact committed.",
    ]
    (args.out_dir / "reports/exp39b_r1_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    generic_hash = args.out_dir / "hashes/exp39b_private_output_hashes.json"
    if generic_hash.exists():
        shutil.copy2(generic_hash, args.out_dir / "hashes/exp39b_r1_private_output_hashes.json")
    print(json.dumps({"status": decision["status"], "accepted": successes, "wilson_95": [lower, upper]}, sort_keys=True))


if __name__ == "__main__":
    main()
