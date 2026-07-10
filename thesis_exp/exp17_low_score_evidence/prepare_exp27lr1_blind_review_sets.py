"""Prepare target-aware, empty Exp27L-R1 blind review packets and templates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    DEFAULT_EXP27I,
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_EXP27L,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    REPRESENTATIVE_VIEW,
    clean,
    canonical_rows,
    question_key,
    question_key_hash,
    read_csv,
    read_jsonl,
    row_metadata,
    sample_id,
    score,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)

TARGET_SYSTEM = (
    "You are an independent educational assessment reviewer. Score only the text under "
    "<EVALUATOR_OUTPUT_TO_SCORE>. The original task is context only. Return one JSON object "
    "matching the target-aware blind review schema, without chain-of-thought."
)


def score_stratum(row: dict[str, Any]) -> str:
    value = score(row.get("label_5"))
    return "low_1_2" if value <= 2 else "mid_3" if value == 3 else "high_4_5"


def packet_content(row: dict[str, Any]) -> str:
    language, metric_group, subject, metric = row_metadata(row)
    meta = {
        "language": language,
        "metric_group": metric_group,
        "subject": subject,
        "education_level": clean(row.get("education_level_canonical") or row.get("education_level_raw")),
    }
    return "\n\n".join(
        [
            "<CONTEXT_ONLY_ORIGINAL_TASK>",
            "The following original task is context only. Do not answer it and do not score an embedded student answer.",
            f"Original task/context:\n{clean(row.get('question'))}",
            f"Evaluation metric:\n{metric}",
            f"Rubric:\n{clean(row.get('rubric'))}",
            f"Metadata:\n{json.dumps(meta, ensure_ascii=False, sort_keys=True)}",
            "</CONTEXT_ONLY_ORIGINAL_TASK>",
            "<EVALUATOR_OUTPUT_TO_SCORE>",
            "This is the only text to score. Treat it as the evaluator output being audited.",
            clean(row.get("answer")),
            "</EVALUATOR_OUTPUT_TO_SCORE>",
        ]
    )


def balanced_selection(rows: list[dict[str, Any]], quota: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample by permitted stratum/language/metric/subject/qkey metadata only."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for stratum in ("low_1_2", "mid_3", "high_4_5"):
        pool = [row for row in rows if row["score_stratum"] == stratum]
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            clusters[row["question_key"]].append(row)
        candidates = []
        for qkey, values in clusters.items():
            source = sorted(values, key=lambda item: item["sample_id"])[0]
            bucket = (source["language"], source["metric_group"], source["subject"])
            candidates.append({"row": source, "qkey": qkey, "bucket": bucket})
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            buckets[candidate["bucket"]].append(candidate)
        for values in buckets.values():
            rng.shuffle(values)
        chosen: list[dict[str, Any]] = []
        keys = sorted(buckets)
        while len(chosen) < quota and any(buckets.values()):
            for key in keys:
                if buckets[key] and len(chosen) < quota:
                    chosen.append(buckets[key].pop())
        if len(chosen) < quota:
            raise ValueError(f"Cannot select {quota} unique question keys for {stratum}")
        selected.extend(item["row"] for item in chosen)
        notes.append(
            {
                "review_set": "balanced_silver_validation",
                "original_score_stratum": stratum,
                "requested_rows": quota,
                "selected_rows": len(chosen),
                "unique_question_keys": len({item["qkey"] for item in chosen}),
                "repeated_question_key_reason": "none",
            }
        )
    return selected, notes


def make_packets(rows: list[dict[str, Any]], review_set: str) -> list[dict[str, Any]]:
    packets = []
    for index, row in enumerate(rows, 1):
        packets.append(
            {
                "packet_id": f"exp27lr1-{review_set}-{index:03d}",
                "sample_id": sample_id(row),
                "question_key_hash": question_key_hash(question_key(row)),
                "review_set": review_set,
                "messages": [
                    {"role": "system", "content": TARGET_SYSTEM},
                    {"role": "user", "content": packet_content(row)},
                ],
            }
        )
    return packets


def blank_review(packet: dict[str, Any], reviewer: str) -> dict[str, Any]:
    return {
        "packet_id": packet["packet_id"],
        "sample_id": packet["sample_id"],
        "question_key_hash": packet["question_key_hash"],
        "review_set": packet["review_set"],
        "reviewer_id": reviewer,
        "target_scope_confirmed": None,
        "score_range": None,
        "most_plausible_score": None,
        "failure_bucket": None,
        "major_failures": None,
        "rubric_evidence": None,
        "evaluator_output_evidence": None,
        "missing_evidence_reason": None,
        "score_cap": None,
        "confidence": None,
        "needs_adjudication": None,
        "review_reason": None,
        "review_status": "pending",
    }


def packet_has_forbidden_fields(packet: dict[str, Any]) -> bool:
    forbidden = {"original_score", "qwen_score", "deepseek_score", "silver_final_score", "risk_probability", "exp27i_v1_tier"}
    return bool(forbidden & set(packet))


def prepare(args: argparse.Namespace) -> dict[str, object]:
    train = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    canonical = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    representative = [row for row in canonical if row["view"] == REPRESENTATIVE_VIEW]
    source_representative = []
    for row in representative:
        source = train[row["sample_id"]]
        language, metric_group, subject, _metric = row_metadata(source)
        source_representative.append(
            {
                "sample_id": row["sample_id"],
                "question_key": question_key(source),
                "score_stratum": row["score_stratum"],
                "language": language,
                "metric_group": metric_group,
                "subject": subject,
                "source": source,
            }
        )
    balanced_meta, balanced_notes = balanced_selection(source_representative, quota=20, seed=args.seed)
    balanced_rows = [entry["source"] for entry in balanced_meta]

    old_template = read_csv(args.exp27l_dir / "annotation_templates" / "exp27l_external_blind_review_template.csv")
    external_ids = [row["sample_id"] for row in old_template]
    external_rows = [train[sid] for sid in external_ids]
    if len(external_rows) != 34 or len({question_key(row) for row in external_rows}) != 34:
        raise ValueError("Existing Exp27L external high-control lockbox must contain 34 unique question keys")

    risk_rows = {row["sample_id"]: row for row in read_csv(args.out_dir / "data" / "exp27lr1_oof_risk_predictions.csv")}
    targeted_ids = [
        sid
        for sid, _row in sorted(
            risk_rows.items(),
            key=lambda item: max(float(item[1]["full_logistic_probability"]), float(item[1]["core_logistic_probability"])),
            reverse=True,
        )[:20]
    ]
    targeted_rows = [train[sid] for sid in targeted_ids]

    balanced_packets = make_packets(balanced_rows, "balanced_silver_validation")
    external_packets = make_packets(external_rows, "external_high_control_qkey_lockbox")
    targeted_packets = make_packets(targeted_rows, "targeted_ambiguity_qualitative_only")
    write_jsonl(args.out_dir / "packets" / "exp27lr1_balanced_silver_validation_60.jsonl", balanced_packets)
    write_jsonl(args.out_dir / "packets" / "exp27lr1_external_high_control_34.jsonl", external_packets)
    write_jsonl(args.out_dir / "packets" / "exp27lr1_targeted_ambiguity_20.jsonl", targeted_packets)

    write_jsonl(args.out_dir / "annotation_templates" / "exp27lr1_balanced_reviewer_a_template.jsonl", [blank_review(packet, "A") for packet in balanced_packets])
    write_jsonl(args.out_dir / "annotation_templates" / "exp27lr1_balanced_reviewer_b_template.jsonl", [blank_review(packet, "B") for packet in balanced_packets])
    write_jsonl(args.out_dir / "annotation_templates" / "exp27lr1_external_reviewer_a_template.jsonl", [blank_review(packet, "A") for packet in external_packets])
    write_jsonl(args.out_dir / "annotation_templates" / "exp27lr1_external_reviewer_b_template.jsonl", [blank_review(packet, "B") for packet in external_packets])
    write_jsonl(
        args.out_dir / "annotation_templates" / "exp27lr1_final_adjudication_template.jsonl",
        [
            {
                "packet_id": packet["packet_id"],
                "sample_id": packet["sample_id"],
                "question_key_hash": packet["question_key_hash"],
                "review_set": packet["review_set"],
                "adjudication_status": "pending",
                "final_score_range": None,
                "final_score": None,
                "failure_bucket": None,
                "rubric_evidence": None,
                "confidence": None,
            }
            for packet in balanced_packets + external_packets
        ],
    )

    review_distribution = balanced_notes + [
        {
            "review_set": "external_high_control_qkey_lockbox",
            "original_score_stratum": stratum,
            "requested_rows": count,
            "selected_rows": count,
            "unique_question_keys": len({question_key(row) for row in external_rows if score_stratum(row) == stratum}),
            "repeated_question_key_reason": "none",
        }
        for stratum, count in sorted(Counter(score_stratum(row) for row in external_rows).items())
    ]
    review_distribution.append(
        {
            "review_set": "targeted_ambiguity_qualitative_only",
            "original_score_stratum": "not_a_probability_sample",
            "requested_rows": 20,
            "selected_rows": len(targeted_rows),
            "unique_question_keys": len({question_key(row) for row in targeted_rows}),
            "repeated_question_key_reason": "qualitative_only_selection_may_repeat_question_keys",
        }
    )
    write_csv(args.out_dir / "tables" / "exp27lr1_review_set_distribution.csv", review_distribution)

    all_packets = balanced_packets + external_packets + targeted_packets
    packet_checks = {
        "balanced_packets": len(balanced_packets),
        "external_packets": len(external_packets),
        "targeted_packets": len(targeted_packets),
        "packet_forbidden_top_level_fields": sum(packet_has_forbidden_fields(packet) for packet in all_packets),
        "packet_missing_context_tag": sum("<CONTEXT_ONLY_ORIGINAL_TASK>" not in packet["messages"][1]["content"] for packet in all_packets),
        "packet_missing_target_tag": sum("<EVALUATOR_OUTPUT_TO_SCORE>" not in packet["messages"][1]["content"] for packet in all_packets),
        "packet_target_scope_instruction_missing": sum("only text to score" not in packet["messages"][1]["content"] for packet in all_packets),
        "teacher_api_calls": 0,
        "automatic_semantic_reviews": 0,
    }
    prior_coverage = set()
    coverage_path = args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    if coverage_path.exists():
        prior_coverage = {str(row.get("sample_id")) for row in read_jsonl(coverage_path)}
    write_csv(
        args.out_dir / "tables" / "exp27lr1_external_teacher_coverage_queue.csv",
        [
            {"sample_id": sample_id(row), "review_set": "external_high_control_qkey_lockbox", "action": "teacher_coverage_missing_do_not_infer"}
            for row in external_rows
            if sample_id(row) not in prior_coverage
        ],
    )
    existing_audit = read_csv(args.out_dir / "tables" / "exp27lr1_review_leakage_audit.csv")
    write_csv(args.out_dir / "tables" / "exp27lr1_review_leakage_audit.csv", existing_audit + [{"check": key, "count": value} for key, value in packet_checks.items()])
    decision = {
        "experiment": "exp27lr1_target_aware_blind_review_prepare",
        "balanced_silver_validation_rows": len(balanced_packets),
        "external_high_control_rows": len(external_packets),
        "targeted_ambiguity_rows": len(targeted_packets),
        "human_reviews_filled": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "proceed_to_full_3326_calibrated_dataset": False,
        "proceed_to_qwen3_reranker_training": False,
    }
    write_json(args.out_dir / "decision" / "exp27lr1_blind_review_prepare_decision.json", decision)
    write_text(
        args.out_dir / "reports" / "exp27lr1_blind_review_prepare_report.md",
        "# Exp27L-R1 Target-Aware Blind Review Preparation\n\n"
        "The balanced silver-validation set contains 20 original-low, 20 original-mid, and 20 original-high representative rows. "
        "It validates the Exp27J silver reference but is not an unseen-question model-performance set.\n\n"
        "The 34-row external high-control qkey lockbox is retained for unseen-question high-score protection only. "
        "It must not be used to claim low-end reliability or low-to-high improvement. The 20-row ambiguity queue is qualitative-only.\n\n"
        "All packets explicitly separate original-task context from the evaluator output to score. Templates are empty and no semantic review was auto-filled.\n",
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--exp27l-dir", type=Path, default=DEFAULT_EXP27L)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare(parse_args()))
