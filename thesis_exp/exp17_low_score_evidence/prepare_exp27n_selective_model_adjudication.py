"""Prepare the remaining Exp27N selective model-adjudication packets.

The locked Exp27M policy marks 70 of the existing 361 teacher-audited train
rows for adjudication. Sixteen already have a completed model review, so this
CPU-only step emits the remaining 54 as target-aware blind packets. Human,
Qwen, and DeepSeek scores are deliberately absent from every packet.
"""

from __future__ import annotations

import argparse
import json
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
    row_metadata,
    sample_id,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp27lr1_blind_review_sets import (  # noqa: E402
    packet_content,
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
DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42"
)
SYSTEM_PROMPT = (
    "You are the final independent model reviewer for an educational assessment audit. "
    "Score only <EVALUATOR_OUTPUT_TO_SCORE> under the supplied metric and rubric. "
    "The original task is context only. Return one compact JSON object matching the "
    "Exp27N schema without chain-of-thought."
)


def require(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def score_value(value: Any) -> int:
    parsed = int(float(value))
    if not 1 <= parsed <= 5:
        raise ValueError(f"Invalid ordinal score: {value!r}")
    return parsed


def audit_split(path: Path) -> tuple[set[str], set[str]]:
    rows = read_jsonl(path)
    return ({sample_id(row) for row in rows}, {question_key_hash(question_key(row)) for row in rows})


def review_template(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": packet["sample_id"],
        "target_scope_confirmed": None,
        "final_score_range": None,
        "final_score": None,
        "failure_bucket": None,
        "major_failures": None,
        "rubric_evidence": None,
        "evaluator_output_evidence": None,
        "missing_evidence_reason": None,
        "score_cap": None,
        "confidence": None,
        "training_use": None,
        "review_status": "pending",
    }


def output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Exp27N selective model adjudication",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "sample_id",
            "target_scope_confirmed",
            "final_score_range",
            "final_score",
            "failure_bucket",
            "major_failures",
            "rubric_evidence",
            "evaluator_output_evidence",
            "missing_evidence_reason",
            "score_cap",
            "confidence",
            "training_use",
            "review_status",
        ],
        "properties": {
            "sample_id": {"type": "string", "minLength": 1},
            "target_scope_confirmed": {"const": True},
            "final_score_range": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer", "minimum": 1, "maximum": 5},
                    {"type": "integer", "minimum": 1, "maximum": 5},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "final_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "failure_bucket": {
                "enum": [
                    "no_failure",
                    "visible_failure",
                    "hidden_or_missing_failure",
                    "unclear",
                ]
            },
            "major_failures": {"type": "array", "items": {"type": "string"}},
            "rubric_evidence": {"type": "string", "minLength": 1},
            "evaluator_output_evidence": {"type": ["string", "null"]},
            "missing_evidence_reason": {"type": ["string", "null"]},
            "score_cap": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
            "confidence": {"enum": ["high", "medium", "low"]},
            "training_use": {"enum": ["resolved_model_silver", "review_only"]},
            "review_status": {"const": "completed"},
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    audited_path = args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    policy_path = args.exp27m_dir / "configs" / "exp27m_locked_policy.json"
    reviewed_path = args.exp27m_dir / "private" / "exp27m_model_review_reference_107.jsonl"
    for path, purpose in (
        (audited_path, "361-row teacher audit"),
        (policy_path, "locked Exp27M policy"),
        (reviewed_path, "private Exp27M model-review reference"),
        (args.train_jsonl, "question-disjoint train split"),
        (args.dev_jsonl, "question-disjoint dev split"),
        (args.test_jsonl, "question-disjoint test split"),
    ):
        require(path, purpose)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("selected_signal") != "qwen_human_gap":
        raise ValueError(f"Exp27N expects the locked qwen_human_gap policy, found {policy}")
    audited = read_jsonl(audited_path)
    if len(audited) != 361 or len({row["sample_id"] for row in audited}) != 361:
        raise ValueError("Exp27I teacher audit must contain 361 unique samples")
    already_reviewed = {str(row["sample_id"]) for row in read_jsonl(reviewed_path)}

    tiers: dict[str, str] = {}
    for row in audited:
        sid = str(row["sample_id"])
        human = score_value(row["original_human_score"])
        qwen = score_value(row["qwen_score"])
        gap = abs(qwen - human)
        if gap >= 2 or bool_value(row.get("target_issue_flag")):
            tiers[sid] = "adjudication_required"
        elif gap == 0:
            tiers[sid] = "direct_accept"
        else:
            tiers[sid] = "weighted_accept"
    tier_counts = Counter(tiers.values())
    required_ids = {sid for sid, tier in tiers.items() if tier == "adjudication_required"}
    completed_ids = required_ids & already_reviewed
    remaining_ids = required_ids - already_reviewed
    if tier_counts != Counter(
        {"direct_accept": 173, "weighted_accept": 118, "adjudication_required": 70}
    ):
        raise ValueError(f"Locked Exp27M tier counts changed: {tier_counts}")
    if len(completed_ids) != 16 or len(remaining_ids) != 54:
        raise ValueError(
            f"Expected 16 reviewed and 54 remaining high-risk samples, found {len(completed_ids)}/{len(remaining_ids)}"
        )

    train_rows = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    missing_train = remaining_ids - set(train_rows)
    if missing_train:
        raise ValueError(f"Exp27N samples missing from train: {sorted(missing_train)[:3]}")
    selected = [train_rows[sid] for sid in sorted(remaining_ids)]
    packets = []
    for index, row in enumerate(selected, 1):
        packets.append(
            {
                "packet_id": f"exp27n-selective-adjudication-{index:03d}",
                "sample_id": sample_id(row),
                "question_key_hash": question_key_hash(question_key(row)),
                "review_set": "exp27n_selective_adjudication_remaining_54",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": packet_content(row)},
                ],
            }
        )
    forbidden = {
        "original_human_score",
        "human_score",
        "qwen_score",
        "deepseek_score",
        "calibrated_score",
        "teacher_gap",
        "risk_probability",
    }
    if any(forbidden & set(packet) for packet in packets):
        raise ValueError("A blind packet exposes hidden score fields")
    if len(packets) != 54 or len({row["sample_id"] for row in packets}) != 54:
        raise ValueError("Exp27N packet count or sample uniqueness failed")

    review_ids = {row["sample_id"] for row in packets}
    review_qkeys = {row["question_key_hash"] for row in packets}
    dev_ids, dev_qkeys = audit_split(args.dev_jsonl)
    test_ids, test_qkeys = audit_split(args.test_jsonl)
    leakage = [
        {"check": "dev_sample_overlap", "count": len(review_ids & dev_ids)},
        {"check": "dev_question_overlap", "count": len(review_qkeys & dev_qkeys)},
        {"check": "test_sample_overlap", "count": len(review_ids & test_ids)},
        {"check": "test_question_overlap", "count": len(review_qkeys & test_qkeys)},
        {"check": "dev_label_used", "count": 0},
        {"check": "test_label_used", "count": 0},
        {"check": "teacher_api_calls", "count": 0},
        {"check": "gpu_training_runs", "count": 0},
    ]
    if any(row["count"] != 0 for row in leakage):
        raise ValueError(f"Exp27N leakage/protocol audit failed: {leakage}")

    language_counts = Counter(row_metadata(row)[0] for row in selected)
    metric_group_counts = Counter(row_metadata(row)[1] for row in selected)
    summary_rows = [
        {"item": "teacher_audited_rows", "count": len(audited)},
        {"item": "direct_accept", "count": tier_counts["direct_accept"]},
        {"item": "weighted_accept", "count": tier_counts["weighted_accept"]},
        {"item": "adjudication_required", "count": tier_counts["adjudication_required"]},
        {"item": "adjudication_already_reviewed", "count": len(completed_ids)},
        {"item": "adjudication_remaining_packet", "count": len(packets)},
    ]
    distribution = [
        {"dimension": "language", "value": key, "count": value}
        for key, value in sorted(language_counts.items())
    ] + [
        {"dimension": "metric_group", "value": key, "count": value}
        for key, value in sorted(metric_group_counts.items())
    ]
    decision = {
        "experiment": "exp27n_selective_model_adjudication_prepare",
        "status": "READY_FOR_ONE_GPT56PRO_SESSION",
        "packet_rows": len(packets),
        "adjudication_required_total": len(required_ids),
        "already_reviewed": len(completed_ids),
        "remaining_to_review": len(remaining_ids),
        "reviewer_plan": "one GPT-5.6Pro session for all 54; Codex performs structural QC and low-confidence isolation",
        "model_review_is_silver": True,
        "human_external_validation": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "model_training_runs": 0,
        "proceed_to_361_downstream_dataset_construction": False,
        "next_step": "attach_packet_and_prompt_to_one_gpt56pro_session",
    }
    write_jsonl(args.out_dir / "packets" / "exp27n_selective_adjudication_packets_54.jsonl", packets)
    write_jsonl(
        args.out_dir / "annotation_templates" / "exp27n_selective_adjudication_template_54.jsonl",
        [review_template(packet) for packet in packets],
    )
    write_json(args.out_dir / "schemas" / "exp27n_selective_adjudication_schema.json", output_schema())
    write_csv(args.out_dir / "tables" / "exp27n_selection_summary.csv", summary_rows)
    write_csv(args.out_dir / "tables" / "exp27n_packet_distribution.csv", distribution)
    write_csv(args.out_dir / "tables" / "exp27n_leakage_audit.csv", leakage)
    write_json(args.out_dir / "decision" / "exp27n_selective_adjudication_prepare_decision.json", decision)
    report = "\n".join(
        [
            "# Exp27N Selective Model Adjudication Preparation",
            "",
            "Exp27N applies the Exp27M locked qwen-human gap policy to the existing 361",
            "train-only teacher-audited rows. It emits only the 54 high-risk rows that",
            "still lack model review.",
            "",
            f"- direct accept: {tier_counts['direct_accept']}",
            f"- weighted accept: {tier_counts['weighted_accept']}",
            f"- adjudication required: {tier_counts['adjudication_required']}",
            f"- already reviewed: {len(completed_ids)}",
            f"- remaining blind packet: {len(packets)}",
            "- reviewer plan: one GPT-5.6Pro session for all 54",
            "- Qwen/DeepSeek/human scores in packet: no",
            "- API calls: 0",
            "- GPU/training: 0",
            "- dev/test labels used: no",
            "",
            "The returned annotations remain model-reviewed silver. Low-confidence or",
            "unclear rows must become review_only. Dataset construction and training stay",
            "blocked until all 54 outputs pass schema and evidence-substring validation.",
            "",
        ]
    )
    write_text(args.out_dir / "reports" / "exp27n_selective_adjudication_prepare_report.md", report)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27m-dir", type=Path, default=DEFAULT_EXP27M)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
