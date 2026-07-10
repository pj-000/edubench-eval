"""Build controlled Exp27O 361-row in-place training pilot datasets.

All variants retain the same locked 3,326-row train universe. Only supervision
for the 361 train-only audited rows may differ. Dev and test are read solely
for sample/question-key leakage checks; their labels are never consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    bool_value,
    question_key,
    question_key_hash,
    read_jsonl,
    sample_id,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


DEFAULT_TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_EXP27I = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_EXP27M = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27m_model_review_audit_policy_seed42"
)
DEFAULT_EXP27N = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42"
)
DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
)
VARIANTS = (
    "v0_original_unweighted",
    "v1_original_label_matched_weight",
    "v2_selective_hard_relabel",
    "v3_selective_soft_audit",
)


def require(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def ordinal(value: Any) -> int:
    parsed = int(float(value))
    if not 1 <= parsed <= 5:
        raise ValueError(f"Invalid ordinal score: {value!r}")
    return parsed


def half_up(value: float) -> int:
    return min(5, max(1, int(math.floor(value + 0.5))))


def one_hot(value: int) -> list[float]:
    return [1.0 if label == value else 0.0 for label in range(1, 6)]


def point_distribution(*scores_and_weights: tuple[int, float]) -> list[float]:
    values = [0.0] * 5
    for value, weight in scores_and_weights:
        values[value - 1] += weight
    total = sum(values)
    return [value / total for value in values]


def range_distribution(low: int, high: int, center: int, tau: float = 0.75) -> list[float]:
    values = [
        math.exp(-abs(label - center) / tau) if low <= label <= high else 0.0
        for label in range(1, 6)
    ]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"Invalid score-range kernel: low={low}, high={high}, center={center}")
    return [value / total for value in values]


def expected_score(distribution: list[float]) -> float:
    return sum((index + 1) * value for index, value in enumerate(distribution))


def entropy(distribution: list[float]) -> float:
    return -sum(value * math.log(value) for value in distribution if value > 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_keys(path: Path) -> tuple[set[str], set[str]]:
    rows = read_jsonl(path)
    return (
        {sample_id(row) for row in rows},
        {question_key_hash(question_key(row)) for row in rows},
    )


def tier_for(audit: dict[str, Any]) -> str:
    human = ordinal(audit["original_human_score"])
    qwen = ordinal(audit["qwen_score"])
    gap = abs(qwen - human)
    if gap >= 2 or bool_value(audit.get("target_issue_flag")):
        return "adjudication_required"
    if gap == 0:
        return "direct_accept"
    return "weighted_accept"


def policy_supervision(
    source: dict[str, Any],
    audit: dict[str, Any] | None,
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    original = ordinal(source["label_5"])
    if audit is None:
        return {
            "audit_scope": "unaudited_train",
            "audit_tier": "unaudited",
            "training_tier": "original_train",
            "soft_target": one_hot(original),
            "hard_target": original,
            "raw_quality_weight": 1.0,
            "target_source": "original_human_label",
            "reference_status": "original_benchmark_label",
        }
    tier = tier_for(audit)
    human = ordinal(audit["original_human_score"])
    qwen = ordinal(audit["qwen_score"])
    deepseek = ordinal(audit["deepseek_score"])
    if human != original:
        raise ValueError(f"Audit/original label mismatch: {sample_id(source)}")
    if tier == "direct_accept":
        distribution = one_hot(human)
        hard_target = human
        training_tier = tier
        weight = 1.0
        target_source = "original_human_teacher_confirmed"
        status = "teacher_audited_no_model_adjudication"
    elif tier == "weighted_accept":
        distribution = point_distribution((human, 0.50), (qwen, 0.25), (deepseek, 0.25))
        hard_target = half_up(expected_score(distribution))
        training_tier = tier
        weight = 0.50
        target_source = "human_qwen_deepseek_weighted_mixture"
        status = "teacher_audited_no_model_adjudication"
    else:
        if review is None:
            raise ValueError(f"Required adjudication missing: {sample_id(source)}")
        unresolved = review["failure_bucket"] == "unclear" or review["confidence"] == "low"
        if unresolved:
            distribution = one_hot(original)
            hard_target = original
            training_tier = "review_only"
            weight = 0.0
            target_source = "isolated_unresolved_model_review"
        else:
            low, high = (ordinal(value) for value in review["final_score_range"])
            center = ordinal(review["final_score"])
            if not low <= center <= high:
                raise ValueError(f"Review center outside range: {sample_id(source)}")
            distribution = range_distribution(low, high, center)
            hard_target = center
            training_tier = "resolved_adjudication"
            weight = 0.75 if review["confidence"] == "high" else 0.50
            target_source = "model_review_score_range_kernel_tau_0p75"
        status = "model_review_silver_not_human_gold"
    return {
        "audit_scope": "audited_361",
        "audit_tier": tier,
        "training_tier": training_tier,
        "soft_target": distribution,
        "hard_target": hard_target,
        "raw_quality_weight": weight,
        "target_source": target_source,
        "reference_status": status,
    }


def variant_row(
    source: dict[str, Any],
    supervision: dict[str, Any],
    variant: str,
    normalization: float,
) -> dict[str, Any]:
    original = ordinal(source["label_5"])
    if variant == "v0_original_unweighted":
        target = one_hot(original)
        hard = original
        raw_weight = 1.0
        weight = 1.0
        source_name = "original_human_label"
        target_type = "hard_one_hot"
    elif variant == "v1_original_label_matched_weight":
        target = one_hot(original)
        hard = original
        raw_weight = supervision["raw_quality_weight"]
        weight = raw_weight * normalization
        source_name = "original_human_label_with_locked_audit_weight"
        target_type = "hard_one_hot"
    elif variant == "v2_selective_hard_relabel":
        hard = supervision["hard_target"]
        target = one_hot(hard)
        raw_weight = supervision["raw_quality_weight"]
        weight = raw_weight * normalization
        source_name = supervision["target_source"] + "_hard"
        target_type = "hard_one_hot"
    elif variant == "v3_selective_soft_audit":
        target = supervision["soft_target"]
        hard = half_up(expected_score(target))
        raw_weight = supervision["raw_quality_weight"]
        weight = raw_weight * normalization
        source_name = supervision["target_source"]
        target_type = "soft_distribution"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    row = dict(source)
    row.update(
        {
            "sample_id": sample_id(source),
            "original_label_5": original,
            "label_5": hard,
            "soft_target_5": target,
            "sample_weight": weight,
            "raw_quality_weight": raw_weight,
            "exp27o_variant": variant,
            "exp27o_target_type": target_type,
            "exp27o_target_source": source_name,
            "exp27o_audit_scope": supervision["audit_scope"],
            "exp27o_audit_tier": supervision["audit_tier"],
            "exp27o_training_tier": supervision["training_tier"],
            "exp27o_reference_status": supervision["reference_status"],
        }
    )
    return row


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    audited_path = args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    policy_path = args.exp27m_dir / "configs" / "exp27m_locked_policy.json"
    reviews_path = args.exp27n_dir / "private" / "exp27n_adjudication_reference_70.jsonl"
    for path, purpose in (
        (args.train_jsonl, "locked train split"),
        (args.dev_jsonl, "locked dev split"),
        (args.test_jsonl, "locked test split"),
        (audited_path, "361-row teacher audit"),
        (policy_path, "Exp27M locked policy"),
        (reviews_path, "Exp27N consolidated adjudications"),
    ):
        require(path, purpose)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("selected_signal") != "qwen_human_gap" or policy.get("frozen_for") != (
        "361-row in-place downstream pilot only"
    ):
        raise ValueError(f"Unexpected Exp27M locked policy: {policy}")
    train = read_jsonl(args.train_jsonl)
    if len(train) != 3326 or len({sample_id(row) for row in train}) != 3326:
        raise ValueError("Exp27O requires exactly 3,326 unique train rows")
    train_by_id = {sample_id(row): row for row in train}
    audited_rows = read_jsonl(audited_path)
    audited = {str(row["sample_id"]): row for row in audited_rows}
    reviews = {str(row["sample_id"]): row for row in read_jsonl(reviews_path)}
    if len(audited) != 361 or not set(audited) <= set(train_by_id):
        raise ValueError("The 361 audited rows must be a unique subset of train")
    if len(reviews) != 70:
        raise ValueError("Exp27O requires all 70 consolidated adjudications")
    for sid, audit in audited.items():
        source = train_by_id[sid]
        if question_key(source) != str(audit["question_key"]):
            raise ValueError(f"Question-key mismatch for audited sample: {sid}")

    supervision = {
        sid: policy_supervision(row, audited.get(sid), reviews.get(sid))
        for sid, row in train_by_id.items()
    }
    tier_counts = Counter(value["audit_tier"] for value in supervision.values())
    training_tier_counts = Counter(value["training_tier"] for value in supervision.values())
    expected_tiers = Counter(
        {
            "unaudited": 2965,
            "direct_accept": 173,
            "weighted_accept": 118,
            "adjudication_required": 70,
        }
    )
    if tier_counts != expected_tiers:
        raise ValueError(f"Locked Exp27O tier counts changed: {tier_counts}")
    active = [value["raw_quality_weight"] for value in supervision.values() if value["raw_quality_weight"] > 0]
    normalization = len(active) / sum(active)
    if training_tier_counts["review_only"] != 8 or len(active) != 3318:
        raise ValueError(f"Unexpected active/review-only counts: {training_tier_counts}")

    data_dir = args.out_dir / "private" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    fingerprints = []
    for variant in VARIANTS:
        rows = [variant_row(row, supervision[sample_id(row)], variant, normalization) for row in train]
        path = data_dir / f"exp27o_{variant}_train.jsonl"
        write_jsonl(path, rows)
        variant_rows[variant] = rows
        fingerprints.append(
            {
                "variant": variant,
                "path": str(path),
                "rows": len(rows),
                "unique_sample_ids": len({row["sample_id"] for row in rows}),
                "sha256": sha256(path),
            }
        )

    tier_rows = [
        {"dimension": "audit_tier", "value": key, "count": value}
        for key, value in sorted(tier_counts.items())
    ] + [
        {"dimension": "training_tier", "value": key, "count": value}
        for key, value in sorted(training_tier_counts.items())
    ]
    summary_rows = []
    label_rows = []
    for variant, rows in variant_rows.items():
        active_rows = [row for row in rows if row["sample_weight"] > 0]
        changed = sum(row["label_5"] != row["original_label_5"] for row in rows)
        non_one_hot = sum(
            sum(value > 1e-12 for value in row["soft_target_5"]) > 1 for row in rows
        )
        summary_rows.append(
            {
                "variant": variant,
                "rows": len(rows),
                "active_rows": len(active_rows),
                "zero_weight_rows": len(rows) - len(active_rows),
                "changed_hard_labels": changed,
                "non_one_hot_soft_targets": non_one_hot,
                "mean_weight_active": sum(row["sample_weight"] for row in active_rows) / len(active_rows),
                "mean_weight_all": sum(row["sample_weight"] for row in rows) / len(rows),
                "mean_target_entropy": sum(entropy(row["soft_target_5"]) for row in active_rows) / len(active_rows),
            }
        )
        counts = Counter(row["label_5"] for row in rows)
        label_rows.extend(
            {"variant": variant, "label_5": label, "count": counts[label]}
            for label in range(1, 6)
        )

    hard_rows = variant_rows["v2_selective_hard_relabel"]
    transition_counts = Counter(
        (
            row["exp27o_audit_tier"],
            row["exp27o_training_tier"],
            row["original_label_5"],
            row["label_5"],
            "up" if row["label_5"] > row["original_label_5"] else (
                "down" if row["label_5"] < row["original_label_5"] else "same"
            ),
        )
        for row in hard_rows
        if row["exp27o_audit_scope"] == "audited_361"
    )
    transition_rows = [
        {
            "audit_tier": key[0],
            "training_tier": key[1],
            "original_label_5": key[2],
            "hard_target_5": key[3],
            "direction": key[4],
            "count": count,
        }
        for key, count in sorted(transition_counts.items())
    ]
    changed_rows = [row for row in hard_rows if row["label_5"] != row["original_label_5"]]
    low_to_high_escalations = [
        row for row in hard_rows if row["original_label_5"] <= 2 and row["label_5"] >= 4
    ]
    transition_risk_rows = [
        {"item": "audited_rows", "count": 361},
        {"item": "changed_hard_labels", "count": len(changed_rows)},
        {"item": "upward_changes", "count": sum(row["label_5"] > row["original_label_5"] for row in changed_rows)},
        {"item": "downward_changes", "count": sum(row["label_5"] < row["original_label_5"] for row in changed_rows)},
        {"item": "original_low_to_hard_high_escalations", "count": len(low_to_high_escalations)},
        {
            "item": "original_high_to_hard_low_changes",
            "count": sum(row["original_label_5"] >= 4 and row["label_5"] <= 2 for row in changed_rows),
        },
        {
            "item": "low_to_high_escalations_resolved_adjudication",
            "count": sum(row["exp27o_training_tier"] == "resolved_adjudication" for row in low_to_high_escalations),
        },
    ]

    train_ids = set(train_by_id)
    train_qkeys = {question_key_hash(question_key(row)) for row in train}
    dev_ids, dev_qkeys = split_keys(args.dev_jsonl)
    test_ids, test_qkeys = split_keys(args.test_jsonl)
    audit_ids = set(audited)
    audit_qkeys = {question_key_hash(question_key(train_by_id[sid])) for sid in audit_ids}
    leakage_rows = [
        {"check": "audited_rows_missing_from_train", "count": len(audit_ids - train_ids)},
        {"check": "audited_sample_overlap_dev", "count": len(audit_ids & dev_ids)},
        {"check": "audited_sample_overlap_test", "count": len(audit_ids & test_ids)},
        {"check": "audited_question_overlap_dev", "count": len(audit_qkeys & dev_qkeys)},
        {"check": "audited_question_overlap_test", "count": len(audit_qkeys & test_qkeys)},
        {"check": "train_question_overlap_dev", "count": len(train_qkeys & dev_qkeys)},
        {"check": "train_question_overlap_test", "count": len(train_qkeys & test_qkeys)},
        {"check": "dev_label_used", "count": 0},
        {"check": "test_label_used", "count": 0},
        {"check": "teacher_api_calls", "count": 0},
        {"check": "gpu_training_runs", "count": 0},
    ]
    if any(row["count"] for row in leakage_rows):
        raise ValueError(f"Exp27O leakage/protocol audit failed: {leakage_rows}")

    locked_design = {
        "experiment": "exp27o_361_in_place_downstream_pilot",
        "train_universe_rows": 3326,
        "audited_in_place_rows": 361,
        "unaudited_original_rows": 2965,
        "variants": list(VARIANTS),
        "primary_comparisons": [
            "v0_original_unweighted_vs_v2_selective_hard_relabel",
            "v2_selective_hard_relabel_vs_v3_selective_soft_audit",
        ],
        "weight_ablation": "v1_original_label_matched_weight",
        "soft_target_tau": 0.75,
        "active_weight_normalization": normalization,
        "dev_test_labels_used_for_construction": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "model_training_runs": 0,
        "high_impact_low_to_high_transition_monitoring_required": True,
        "high_impact_transition_rows": len(low_to_high_escalations),
    }
    decision = {
        "experiment": "exp27o_361_in_place_pilot_dataset_preparation",
        "status": "PASS",
        "datasets_constructed": len(VARIANTS),
        "rows_per_dataset": 3326,
        "audited_rows": 361,
        "resolved_model_silver_rows": 62,
        "review_only_zero_weight_rows": 8,
        "dev_test_labels_used": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "model_training_runs": 0,
        "proceed_to_training_workflow_implementation": True,
        "proceed_to_training": False,
        "requires_high_impact_transition_monitoring": True,
        "high_impact_transition_rows": len(low_to_high_escalations),
        "proceed_to_full_3326_teacher_relabeling": False,
        "proceed_to_dev_test_relabeling": False,
        "next_step": "implement_shared_config_qwen3_reranker_pilot_trainer_with_soft_targets",
    }
    write_json(args.out_dir / "configs" / "exp27o_locked_pilot_design.json", locked_design)
    write_csv(args.out_dir / "tables" / "exp27o_tier_summary.csv", tier_rows)
    write_csv(args.out_dir / "tables" / "exp27o_variant_summary.csv", summary_rows)
    write_csv(args.out_dir / "tables" / "exp27o_label_distribution.csv", label_rows)
    write_csv(args.out_dir / "tables" / "exp27o_hard_label_transitions.csv", transition_rows)
    write_csv(args.out_dir / "tables" / "exp27o_transition_risk_summary.csv", transition_risk_rows)
    write_csv(args.out_dir / "tables" / "exp27o_dataset_fingerprints.csv", fingerprints)
    write_csv(args.out_dir / "tables" / "exp27o_leakage_audit.csv", leakage_rows)
    write_json(args.out_dir / "decision" / "exp27o_pilot_dataset_prepare_decision.json", decision)
    report = "\n".join(
        [
            "# Exp27O 361-Row In-Place Pilot Dataset Preparation",
            "",
            "Exp27O keeps the same 3,326-row question-disjoint train universe in every",
            "variant. The existing audit changes supervision only for 361 train rows;",
            "the remaining 2,965 rows retain their original human labels.",
            "",
            "## Variants",
            "",
            "- `v0_original_unweighted`: original benchmark labels and unit weights.",
            "- `v1_original_label_matched_weight`: original labels with the locked audit weights.",
            "- `v2_selective_hard_relabel`: hard labels derived from the locked audit policy.",
            "- `v3_selective_soft_audit`: tier-specific soft targets and confidence weights.",
            "",
            "The extra v1 weight-only ablation separates the effect of sample weighting",
            "from the effect of teacher/adjudication targets.",
            "",
            "## Locked counts",
            "",
            "- unaudited original rows: 2,965",
            "- direct accept: 173",
            "- weighted accept: 118",
            "- adjudication required: 70",
            "- resolved model-silver adjudications: 62",
            "- review-only rows with zero weight: 8",
            f"- active-weight normalization factor: {normalization:.8f}",
            f"- changed hard labels: {len(changed_rows)}",
            f"- upward/downward changes: {transition_risk_rows[2]['count']}/{transition_risk_rows[3]['count']}",
            f"- original low labels changed to hard 4/5: {len(low_to_high_escalations)}",
            "",
            "## Protocol",
            "",
            "Full JSONL datasets remain private/local because they repeat original answer",
            "text. Public artifacts contain only aggregate summaries and SHA-256 hashes.",
            "No teacher API, GPU, training, dev label, or test label was used.",
            "",
            "## High-impact transition gate",
            "",
            "Sixteen original label-1/2 rows become hard label 4/5 under resolved",
            "model-review silver. This may reflect genuine correction of noisy original",
            "low labels, but it is also the highest-impact case group for the low-to-high",
            "research objective. The pilot must report this group separately. No new API",
            "annotation is required before the controlled pilot.",
            "",
            "## Decision",
            "",
            "Dataset construction passes. The next step is implementation and preflight",
            "validation of one shared Qwen3-Reranker-0.6B pilot trainer that supports",
            "per-example weights and five-class soft targets. Training remains gated until",
            "that workflow passes review and smoke tests; the 16 high-impact upward",
            "transitions remain a mandatory diagnostic slice.",
            "",
        ]
    )
    write_text(args.out_dir / "reports" / "exp27o_pilot_dataset_prepare_report.md", report)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27m-dir", type=Path, default=DEFAULT_EXP27M)
    parser.add_argument("--exp27n-dir", type=Path, default=DEFAULT_EXP27N)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
