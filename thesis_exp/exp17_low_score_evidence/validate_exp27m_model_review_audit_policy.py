"""Validate Exp27M model-review acceptance artifacts and protocol gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    read_csv,
    read_jsonl,
)


DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27m_model_review_audit_policy_seed42"
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def true_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    required = (
        out / "tables" / "exp27m_review_set_accounting.csv",
        out / "tables" / "exp27m_review_completion.csv",
        out / "tables" / "exp27m_source_metrics_balanced60.csv",
        out / "tables" / "exp27m_policy_candidate_acceptance.csv",
        out / "tables" / "exp27m_selected_tier_metrics.csv",
        out / "tables" / "exp27m_high_control_lockbox.csv",
        out / "tables" / "exp27m_acceptance_checks.csv",
        out / "tables" / "exp27m_leakage_audit.csv",
        out / "configs" / "exp27m_locked_policy.json",
        out / "decision" / "exp27m_model_review_audit_policy_decision.json",
        out / "reports" / "exp27m_model_review_audit_policy_report.md",
        out / "private" / "exp27m_model_review_reference_107.jsonl",
        out / "private" / "exp27m_policy_preview_balanced60.jsonl",
    )
    for path in required:
        require(path)

    completion = read_csv(out / "tables" / "exp27m_review_completion.csv")
    if len(completion) != 1:
        raise ValueError("Exp27M completion table must have one row")
    complete = completion[0]
    if (
        int(complete["unique_samples"]) != 107
        or int(complete["reviewer_a_rows"]) != 107
        or int(complete["reviewer_b_rows"]) != 107
        or int(complete["required_adjudications"]) != 38
        or int(complete["completed_adjudications"]) != 38
        or int(complete["balanced_packet_rows"]) != 60
        or int(complete["balanced_quantitative_rows"]) != 58
        or int(complete["balanced_review_only_rows"]) != 2
        or not true_value(complete["model_review_silver"])
        or true_value(complete["human_external_validation"])
    ):
        raise ValueError(f"Invalid model-review completion state: {complete}")

    reference = read_jsonl(out / "private" / "exp27m_model_review_reference_107.jsonl")
    preview = read_jsonl(out / "private" / "exp27m_policy_preview_balanced60.jsonl")
    if len(reference) != 107 or len({row["sample_id"] for row in reference}) != 107:
        raise ValueError("Private model-review reference must contain 107 unique samples")
    if len(preview) != 60 or len({row["sample_id"] for row in preview}) != 60:
        raise ValueError("Private policy preview must contain the balanced 60 once each")
    if any(row.get("trainer_use") != "prohibited_in_exp27m" for row in reference + preview):
        raise ValueError("Exp27M private rows incorrectly allow trainer use")

    accounting = {row["review_set"]: row for row in read_csv(out / "tables" / "exp27m_review_set_accounting.csv")}
    expected = {
        "balanced_silver_validation": (60, "policy_acceptance_only"),
        "external_high_control_qkey_lockbox": (34, "sealed_high_score_protection_only"),
        "targeted_ambiguity_qualitative_only": (20, "qualitative_only"),
    }
    for name, (count, role) in expected.items():
        row = accounting.get(name)
        if row is None or int(row["packet_appearances"]) != count or row["statistical_role"] != role:
            raise ValueError(f"Review-set accounting failed for {name}: {row}")

    source = read_csv(out / "tables" / "exp27m_source_metrics_balanced60.csv")
    required_sources = {
        "original_human",
        "qwen_only",
        "deepseek_only",
        "dual_teacher_half_up_mean",
        "human_qwen_deepseek_median",
    }
    if not required_sources <= {row["source"] for row in source} or any(int(row["n"]) != 58 for row in source):
        raise ValueError("Balanced source comparison must exclude the two unresolved rows")

    candidates = read_csv(out / "tables" / "exp27m_policy_candidate_acceptance.csv")
    selected_tiers = read_csv(out / "tables" / "exp27m_selected_tier_metrics.csv")
    if len(candidates) != 5 or {row["tier"] for row in selected_tiers} != {
        "direct_accept",
        "weighted_accept",
        "adjudication_required",
    }:
        raise ValueError("Exp27M candidate or tier table is incomplete")
    if sum(int(row["n"]) for row in selected_tiers) != 58:
        raise ValueError("Selected tier metrics must partition the 58 quantitative balanced rows")

    lockbox = read_csv(out / "tables" / "exp27m_high_control_lockbox.csv")
    if (
        len(lockbox) != 1
        or int(lockbox[0]["n"]) != 34
        or true_value(lockbox[0]["used_for_policy_selection"])
        or true_value(lockbox[0]["used_for_threshold_tuning"])
    ):
        raise ValueError("High-control lockbox was not kept sealed")

    checks = read_csv(out / "tables" / "exp27m_acceptance_checks.csv")
    if not checks or not all(true_value(row["pass"]) for row in checks):
        failed = [row["check"] for row in checks if not true_value(row["pass"])]
        raise ValueError(f"Exp27M acceptance checks failed: {failed}")
    leakage = read_csv(out / "tables" / "exp27m_leakage_audit.csv")
    if not leakage or any(int(row["count"]) != 0 for row in leakage):
        raise ValueError(f"Exp27M leakage audit failed: {leakage}")

    policy = json.loads((out / "configs" / "exp27m_locked_policy.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (out / "decision" / "exp27m_model_review_audit_policy_decision.json").read_text(encoding="utf-8")
    )
    if policy.get("reference_status") != "model_review_silver_not_human_gold":
        raise ValueError("Locked policy overstates model-review provenance")
    if decision.get("status") != "PASS":
        raise ValueError(f"Exp27M did not pass: {decision}")
    if (
        decision.get("human_external_validation") is not False
        or decision.get("teacher_api_calls") != 0
        or decision.get("gpu_required") is not False
        or decision.get("model_training_runs") != 0
        or decision.get("balanced_quantitative_rows") != 58
        or decision.get("balanced_review_only_rows") != 2
        or decision.get("proceed_to_361_in_place_downstream_pilot") is not True
        or decision.get("proceed_to_full_3326_calibrated_dataset") is not False
        or decision.get("proceed_to_formal_qwen3_reranker_training") is not False
        or decision.get("proceed_to_dev_test_relabeling") is not False
    ):
        raise ValueError(f"Exp27M decision gates are invalid: {decision}")

    report = (out / "reports" / "exp27m_model_review_audit_policy_report.md").read_text(
        encoding="utf-8"
    )
    for forbidden in ("human expert validation completed", "human gold reference", "proves label correction"):
        if forbidden in report.lower():
            raise ValueError(f"Exp27M report contains an overclaim: {forbidden}")
    return {
        "status": "PASS",
        "unique_model_reviews": 107,
        "required_adjudications": 38,
        "selected_policy": policy["selected_signal"],
        "model_review_is_silver": True,
        "human_external_validation": False,
        "training_gate": "361_row_pilot_only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
