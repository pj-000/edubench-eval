"""Validate Exp27O public summaries and optional private datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import read_csv, read_jsonl
from thesis_exp.exp17_low_score_evidence.prepare_exp27o_361_in_place_pilot_datasets import (
    CORE_INPUT_KEYS,
    VARIANTS,
)


DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
)
EXPERIMENT_FIELDS = {
    "sample_id",
    "original_label_5",
    "label_5",
    "soft_target_5",
    "sample_weight",
    "raw_quality_weight",
    "exp27o_variant",
    "exp27o_target_type",
    "exp27o_target_source",
    "exp27o_audit_scope",
    "exp27o_audit_tier",
    "exp27o_training_tier",
    "exp27o_reference_status",
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def one_hot(label: int) -> list[float]:
    return [1.0 if index == label else 0.0 for index in range(1, 6)]


def validate(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "design": args.out_dir / "configs" / "exp27o_locked_pilot_design.json",
        "tiers": args.out_dir / "tables" / "exp27o_tier_summary.csv",
        "summary": args.out_dir / "tables" / "exp27o_variant_summary.csv",
        "labels": args.out_dir / "tables" / "exp27o_label_distribution.csv",
        "transitions": args.out_dir / "tables" / "exp27o_hard_label_transitions.csv",
        "transition_risk": args.out_dir / "tables" / "exp27o_transition_risk_summary.csv",
        "equivalence": args.out_dir / "tables" / "exp27o_variant_pairwise_equivalence.csv",
        "effective_mass": args.out_dir / "tables" / "exp27o_effective_supervision_mass.csv",
        "tier_safety": args.out_dir / "tables" / "exp27o_tier_gap_safety_audit.csv",
        "high_impact": args.out_dir / "tables" / "exp27o_high_impact16_manifest_light.csv",
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
        or design.get("primary_comparisons")
        != [
            "v0_vs_v1_weight_and_exclusion_effect",
            "v1_vs_v2_hard_relabel_effect",
            "v2_vs_v3_soft_target_effect",
            "v0_vs_v3_end_to_end_effect",
        ]
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
    equivalence = read_csv(paths["equivalence"])
    if not equivalence or any(row["status"] != "PASS" or int(row["mismatch_count"]) != 0 for row in equivalence):
        raise ValueError(f"Exp27O public equivalence audit failed: {equivalence}")
    effective_mass = read_csv(paths["effective_mass"])
    if len(effective_mass) != len(VARIANTS) * 5:
        raise ValueError("Exp27O effective supervision mass table is incomplete")
    tier_safety = {row["check"]: int(row["count"]) for row in read_csv(paths["tier_safety"])}
    expected_safety = {
        "direct_accept_deepseek_human_gap_ge_2": 8,
        "direct_accept_qwen_deepseek_gap_ge_2": 8,
        "weighted_accept_qwen_deepseek_gap_ge_2": 12,
        "original_low_direct_accept_any_teacher_high": 1,
        "original_high_direct_accept_any_teacher_low": 4,
    }
    if tier_safety != expected_safety:
        raise ValueError(f"Exp27O tier-gap safety counts changed: {tier_safety}")
    high_impact = read_csv(paths["high_impact"])
    if len(high_impact) != 16 or len({row["sample_id"] for row in high_impact}) != 16:
        raise ValueError("Exp27O high-impact manifest must contain 16 unique rows")
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
        or decision.get("protocol_exception_count") != 5
        or decision.get("requires_protocol_review") is not True
        or decision.get("proceed_to_full_3326_teacher_relabeling") is not False
    ):
        raise ValueError(f"Invalid Exp27O decision: {decision}")
    if args.require_private:
        datasets: dict[str, list[dict[str, Any]]] = {}
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
            datasets[variant] = rows
        base = datasets[VARIANTS[0]]
        base_ids = [row["sample_id"] for row in base]
        base_hashes = [stable_hash({key: row.get(key) for key in CORE_INPUT_KEYS}) for row in base]
        for variant in VARIANTS[1:]:
            rows = datasets[variant]
            if [row["sample_id"] for row in rows] != base_ids:
                raise ValueError(f"sample_id order differs: {variant}")
            hashes = [stable_hash({key: row.get(key) for key in CORE_INPUT_KEYS}) for row in rows]
            if hashes != base_hashes:
                raise ValueError(f"Core model inputs differ: {variant}")
        v0, v1, v2, v3 = (datasets[variant] for variant in VARIANTS)
        if any(a["label_5"] != b["label_5"] for a, b in zip(v0, v1)):
            raise ValueError("v0/v1 labels differ")
        if any(a["label_5"] != b["label_5"] for a, b in zip(v2, v3)):
            raise ValueError("v2/v3 hard labels differ")
        for left, right, name in ((v1, v2, "v1/v2"), (v2, v3, "v2/v3"), (v1, v3, "v1/v3")):
            if any(abs(float(a["sample_weight"]) - float(b["sample_weight"])) > 1e-12 for a, b in zip(left, right)):
                raise ValueError(f"{name} sample weights differ")
        if any(row["soft_target_5"] != one_hot(int(row["label_5"])) for row in v2):
            raise ValueError("v2 soft targets are not exact one-hot labels")
        for index, base_row in enumerate(base):
            if base_row["exp27o_audit_scope"] != "unaudited_train":
                continue
            stripped = {key: value for key, value in base_row.items() if key not in EXPERIMENT_FIELDS}
            for variant in VARIANTS[1:]:
                candidate = {key: value for key, value in datasets[variant][index].items() if key not in EXPERIMENT_FIELDS}
                if candidate != stripped:
                    raise ValueError(f"Unaudited source row differs: {base_row['sample_id']} {variant}")
        review_sets = []
        for rows in (v1, v2, v3):
            ids = {
                row["sample_id"]
                for row in rows
                if row["exp27o_training_tier"] == "review_only" and float(row["sample_weight"]) == 0.0
            }
            review_sets.append(ids)
        if any(ids != review_sets[0] for ids in review_sets[1:]) or len(review_sets[0]) != 8:
            raise ValueError("Review-only zero-weight IDs differ across v1/v2/v3")
        for row in v3:
            if row["exp27o_training_tier"] != "resolved_adjudication":
                continue
            target = [float(value) for value in row["soft_target_5"]]
            support = [index + 1 for index, value in enumerate(target) if value > 1e-12]
            if support != list(range(min(support), max(support) + 1)):
                raise ValueError(f"Resolved soft-target support is not contiguous: {row['sample_id']}")
            center = int(row["label_5"])
            if target[center - 1] != max(target):
                raise ValueError(f"Resolved soft-target center is not maximal: {row['sample_id']}")
            ordered = sorted(((abs(label - center), target[label - 1]) for label in support))
            for (left_distance, left_prob), (right_distance, right_prob) in zip(ordered, ordered[1:]):
                if right_distance > left_distance and right_prob > left_prob + 1e-12:
                    raise ValueError(f"Resolved soft target is not ordinal-monotone: {row['sample_id']}")
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
