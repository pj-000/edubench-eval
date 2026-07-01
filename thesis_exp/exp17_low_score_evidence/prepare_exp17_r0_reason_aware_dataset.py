"""Prepare redacted Exp17-R0 reason-aware evaluator dataset samples.

R0 is a future causal/instruction-model experiment. This script prepares QC
artifacts only; it does not train a model and does not write full raw SFT data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.io import read_jsonl, write_csv, write_jsonl, write_text


DEFAULT_TRAIN_PATH = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_A0_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42")
DEFAULT_D1_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/d1_hidden_failure_audit_seed42_dev")
DEFAULT_OUTPUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp17_r0_reason_aware_dataset_seed42")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return clean(row.get("sample_id") or row.get("record_id") or row.get("id"))


def truncate(text: str, limit: int = 160) -> str:
    text = " ".join(clean(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def failure_types_from_row(row: dict[str, str]) -> list[str]:
    mode = clean(row.get("failure_mode_auto") or row.get("primary_failure_mode_manual"))
    if mode in {"", "unclear", "possible_label_conflict"}:
        return []
    return [mode]


def target_from_a0(row: dict[str, str]) -> dict[str, Any]:
    reason = clean(row.get("recovered_reason_summary"))
    if not reason:
        reason = f"Low-score rationale type: {clean(row.get('failure_mode_auto') or 'unclear')}."
    return {
        "score": int(float(row.get("gold_label") or 0)),
        "failure_types": failure_types_from_row(row),
        "short_rationale": truncate(reason, 240),
        "rubric_linked": clean(row.get("failure_mode_auto")) not in {"", "unclear"},
    }


def target_from_d1(row: dict[str, str]) -> dict[str, Any]:
    reason = clean(row.get("defect_notes_manual") or row.get("evidence_span_manual") or row.get("rubric_clause_manual"))
    if not reason:
        reason = f"Manual failure type: {clean(row.get('primary_failure_mode_manual') or 'unclear')}."
    types = [clean(row.get("primary_failure_mode_manual"))]
    secondary = clean(row.get("secondary_failure_mode_manual"))
    if secondary and secondary not in types:
        types.append(secondary)
    types = [item for item in types if item and item not in {"unclear", "possible_label_conflict"}]
    return {
        "score": int(float(row.get("gold_label") or 0)),
        "failure_types": types,
        "short_rationale": truncate(reason, 240),
        "rubric_linked": clean(row.get("rubric_link_level_manual")) not in {"", "none"},
    }


def redacted_input(row: dict[str, Any]) -> dict[str, Any]:
    question = clean(row.get("question"))
    answer = clean(row.get("answer"))
    rubric = clean(row.get("rubric") or row.get("rubric_text"))
    metadata = clean(row.get("metadata") or row.get("metadata_raw"))
    return {
        "question_hash": sha(question),
        "answer_hash": sha(answer),
        "rubric_hash": sha(rubric),
        "question_preview": truncate(question, 120),
        "answer_preview": truncate(answer, 120),
        "metric": clean(row.get("metric") or row.get("metric_canonical") or row.get("metric_raw")),
        "metadata_preview": truncate(metadata, 120),
    }


def build_train_examples(a0_dir: Path, max_examples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = read_csv_rows(a0_dir / "train_hidden_failure_candidates.csv")
    usable = [
        row
        for row in candidates
        if clean(row.get("recommended_training_use")) in {"weak_evidence_positive", "pairwise_low", "format_auxiliary"}
        and clean(row.get("hidden_failure_candidate_type")) not in {"unclear", "conflict_or_exclude"}
    ]
    examples = []
    for row in usable[:max_examples]:
        examples.append(
            {
                "sample_id": row.get("sample_id", ""),
                "question_group_id": row.get("question_group_id", ""),
                "input_redacted": redacted_input(row),
                "target": target_from_a0(row),
                "source": "train_a0_human_rationale_derived",
                "note": "Human rationale is supervision target only, not input.",
            }
        )
    conflicts = sum(1 for row in candidates if clean(row.get("score_phrase_conflict_flag")) in {"1", "True", "true"})
    return examples, {
        "train_candidate_count": len(candidates),
        "train_usable_count": len(usable),
        "score_phrase_conflict_count": conflicts,
    }


def build_dev_examples(d1_dir: Path, max_examples: int) -> list[dict[str, Any]]:
    path = d1_dir / "d1_hidden_failure_annotation_template_filled_human_rationale_recovered.csv"
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    out = []
    for row in rows[:max_examples]:
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "question_group_id": row.get("question_group_id", ""),
                "input_redacted": redacted_input(row),
                "target": target_from_d1(row),
                "source": "dev_d1_eval_only",
                "note": "Dev D1 annotations are for evaluation only and must not be used as train labels.",
            }
        )
    return out


def length_stats(values: list[str]) -> dict[str, Any]:
    lengths = [len(value.split()) for value in values if clean(value)]
    if not lengths:
        return {"mean": 0, "p50": 0, "max": 0}
    return {
        "mean": sum(lengths) / len(lengths),
        "p50": sorted(lengths)[len(lengths) // 2],
        "max": max(lengths),
    }


def write_qc(out_dir: Path, train_examples: list[dict[str, Any]], dev_examples: list[dict[str, Any]], train_meta: dict[str, Any]) -> None:
    failure_counter: Counter[str] = Counter()
    qgroup_counter: Counter[str] = Counter()
    rationales: list[str] = []
    for example in train_examples:
        qgroup_counter.update([clean(example.get("question_group_id"))])
        rationales.append(clean(example["target"].get("short_rationale")))
        failure_counter.update(example["target"].get("failure_types") or ["none"])
    max_qgroup_rate = max(qgroup_counter.values()) / len(train_examples) if train_examples else 0.0
    stats = length_stats(rationales)
    safe_for_sft = bool(train_examples) and max_qgroup_rate <= 0.5 and train_meta.get("score_phrase_conflict_count", 0) <= max(5, len(train_examples) * 0.2)
    rows = [
        {"metric": "train_redacted_sample_count", "value": len(train_examples)},
        {"metric": "dev_redacted_eval_sample_count", "value": len(dev_examples)},
        {"metric": "train_candidate_count", "value": train_meta.get("train_candidate_count", 0)},
        {"metric": "train_usable_count", "value": train_meta.get("train_usable_count", 0)},
        {"metric": "max_question_group_rate", "value": max_qgroup_rate},
        {"metric": "score_phrase_conflict_count", "value": train_meta.get("score_phrase_conflict_count", 0)},
        {"metric": "rationale_length_mean_words", "value": stats["mean"]},
        {"metric": "rationale_length_p50_words", "value": stats["p50"]},
        {"metric": "rationale_length_max_words", "value": stats["max"]},
        {"metric": "safe_for_sft", "value": safe_for_sft},
    ]
    for failure_type, count in sorted(failure_counter.items()):
        rows.append({"metric": f"failure_type::{failure_type}", "value": count})
    write_csv(out_dir / "tables" / "reason_aware_dataset_stats.csv", rows, fieldnames=["metric", "value"])
    report = [
        "# Exp17-R0 Reason-Aware Dataset QC Report",
        "",
        "R0 is a future causal/instruction-model experiment. This run prepares redacted samples and QC only; no model is trained.",
        "",
        f"- train redacted sample count: {len(train_examples)}",
        f"- dev eval redacted sample count: {len(dev_examples)}",
        f"- train candidate count: {train_meta.get('train_candidate_count', 0)}",
        f"- train usable count: {train_meta.get('train_usable_count', 0)}",
        f"- max question group rate: {max_qgroup_rate:.4f}",
        f"- score phrase conflict count: {train_meta.get('score_phrase_conflict_count', 0)}",
        f"- rationale length mean/p50/max words: {stats['mean']:.2f}/{stats['p50']}/{stats['max']}",
        f"- safe for SFT: `{safe_for_sft}`",
        "",
        "## Guardrails",
        "",
        "- Human rationale is not included in the input.",
        "- Train rationales are represented only as target fields.",
        "- Dev D1 annotations are evaluation-only and must not be used as train labels.",
        "- Full raw SFT jsonl is intentionally not written by this script.",
    ]
    write_text(out_dir / "reports" / "exp17_r0_dataset_qc_report.md", "\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare redacted Exp17-R0 reason-aware evaluator dataset samples.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--a0-dir", type=Path, default=DEFAULT_A0_DIR)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-redacted-train-samples", type=int, default=80)
    parser.add_argument("--max-redacted-dev-samples", type=int, default=40)
    args = parser.parse_args()
    if not args.train_jsonl.exists() or not args.dev_jsonl.exists():
        raise FileNotFoundError("Train/dev split paths must exist. Test split is intentionally not used.")
    # Read train/dev split only to enforce path existence and document guardrails.
    _ = len(read_jsonl(args.train_jsonl))
    _ = len(read_jsonl(args.dev_jsonl))
    train_examples, train_meta = build_train_examples(args.a0_dir, int(args.max_redacted_train_samples))
    dev_examples = build_dev_examples(args.d1_dir, int(args.max_redacted_dev_samples))
    write_jsonl(args.out_dir / "train_reason_aware_sft_sample_redacted.jsonl", train_examples)
    write_jsonl(args.out_dir / "dev_reason_aware_eval_sample_redacted.jsonl", dev_examples)
    write_qc(args.out_dir, train_examples, dev_examples, train_meta)
    print(json.dumps({"train_redacted": len(train_examples), "dev_redacted": len(dev_examples), "out_dir": str(args.out_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
