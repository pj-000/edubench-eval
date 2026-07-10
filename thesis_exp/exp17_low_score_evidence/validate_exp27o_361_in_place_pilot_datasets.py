"""Validate Exp27O public summaries and optional private datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import read_csv, read_jsonl
from thesis_exp.exp17_low_score_evidence.prepare_exp27o_361_in_place_pilot_datasets import VARIANTS


DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
)


def validate(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "design": args.out_dir / "configs" / "exp27o_locked_pilot_design.json",
        "tiers": args.out_dir / "tables" / "exp27o_tier_summary.csv",
        "summary": args.out_dir / "tables" / "exp27o_variant_summary.csv",
        "labels": args.out_dir / "tables" / "exp27o_label_distribution.csv",
        "transitions": args.out_dir / "tables" / "exp27o_hard_label_transitions.csv",
        "transition_risk": args.out_dir / "tables" / "exp27o_transition_risk_summary.csv",
        "fingerprints": args.out_dir / "tables" / "exp27o_dataset_fingerprints.csv",
        "leakage": args.out_dir / "tables" / "exp27o_leakage_audit.csv",
        "decision": args.out_dir / "decision" / "exp27o_pilot_dataset_prepare_decision.json",
        "report": args.out_dir / "reports" / "exp27o_pilot_dataset_prepare_report.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    design = json.loads(paths["design"].read_text(encoding="utf-8"))
    if (
        design.get("train_universe_rows") != 3326
        or design.get("audited_in_place_rows") != 361
        or design.get("variants") != list(VARIANTS)
        or design.get("dev_test_labels_used_for_construction") is not False
    ):
        raise ValueError(f"Invalid Exp27O locked design: {design}")
    tiers = {(row["dimension"], row["value"]): int(row["count"]) for row in read_csv(paths["tiers"])}
    expected = {
        ("audit_tier", "unaudited"): 2965,
        ("audit_tier", "direct_accept"): 173,
        ("audit_tier", "weighted_accept"): 118,
        ("audit_tier", "adjudication_required"): 70,
        ("training_tier", "original_train"): 2965,
        ("training_tier", "direct_accept"): 173,
        ("training_tier", "weighted_accept"): 118,
        ("training_tier", "resolved_adjudication"): 62,
        ("training_tier", "review_only"): 8,
    }
    if tiers != expected:
        raise ValueError(f"Exp27O tier counts changed: {tiers}")
    summaries = {row["variant"]: row for row in read_csv(paths["summary"])}
    if set(summaries) != set(VARIANTS):
        raise ValueError("Exp27O variant summary is incomplete")
    for variant, row in summaries.items():
        if int(row["rows"]) != 3326:
            raise ValueError(f"Wrong row count for {variant}")
        expected_active = 3326 if variant == "v0_original_unweighted" else 3318
        if int(row["active_rows"]) != expected_active:
            raise ValueError(f"Wrong active row count for {variant}")
    if int(summaries["v0_original_unweighted"]["changed_hard_labels"]) != 0:
        raise ValueError("Original baseline labels changed")
    if int(summaries["v1_original_label_matched_weight"]["changed_hard_labels"]) != 0:
        raise ValueError("Weight-only control labels changed")
    if int(summaries["v2_selective_hard_relabel"]["non_one_hot_soft_targets"]) != 0:
        raise ValueError("Hard-relabel control contains soft targets")
    if int(summaries["v3_selective_soft_audit"]["non_one_hot_soft_targets"]) <= 0:
        raise ValueError("Soft-audit main variant has no soft targets")
    transition_risk = {
        row["item"]: int(row["count"]) for row in read_csv(paths["transition_risk"])
    }
    expected_transition_risk = {
        "audited_rows": 361,
        "changed_hard_labels": 109,
        "upward_changes": 84,
        "downward_changes": 25,
        "original_low_to_hard_high_escalations": 16,
        "original_high_to_hard_low_changes": 4,
        "low_to_high_escalations_resolved_adjudication": 16,
    }
    if transition_risk != expected_transition_risk:
        raise ValueError(f"Exp27O transition-risk counts changed: {transition_risk}")
    leakage = read_csv(paths["leakage"])
    if not leakage or any(int(row["count"]) != 0 for row in leakage):
        raise ValueError(f"Exp27O leakage audit failed: {leakage}")
    fingerprints = read_csv(paths["fingerprints"])
    if len(fingerprints) != 4 or any(len(row["sha256"]) != 64 for row in fingerprints):
        raise ValueError("Exp27O dataset fingerprints are incomplete")
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    if (
        decision.get("status") != "PASS"
        or decision.get("datasets_constructed") != 4
        or decision.get("rows_per_dataset") != 3326
        or decision.get("dev_test_labels_used") is not False
        or decision.get("proceed_to_training_workflow_implementation") is not True
        or decision.get("proceed_to_training") is not False
        or decision.get("requires_high_impact_transition_monitoring") is not True
        or decision.get("high_impact_transition_rows") != 16
        or decision.get("proceed_to_full_3326_teacher_relabeling") is not False
    ):
        raise ValueError(f"Invalid Exp27O decision: {decision}")
    if args.require_private:
        for variant in VARIANTS:
            path = args.out_dir / "private" / "data" / f"exp27o_{variant}_train.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 3326 or len({row["sample_id"] for row in rows}) != 3326:
                raise ValueError(f"Private dataset is incomplete: {path}")
            for row in rows:
                target = row["soft_target_5"]
                if len(target) != 5 or abs(sum(target) - 1.0) > 1e-8:
                    raise ValueError(f"Invalid soft target in {variant}: {row['sample_id']}")
                if row["sample_weight"] < 0:
                    raise ValueError(f"Negative sample weight in {variant}: {row['sample_id']}")
    return {
        "status": "PASS",
        "variants": 4,
        "rows_per_variant": 3326,
        "audited_in_place": 361,
        "review_only": 8,
        "next_step": "implement_shared_qwen3_reranker_pilot_trainer",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
