"""Build and lock the four Exp42A 2x2 factorial train-only variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp42_rubidist.common import (  # noqa: E402
    PROCESSED_PATH,
    ROOT,
    TRAIN_PATH,
    VARIANTS,
    canonical_model_text,
    ensure_output_dirs,
    human_stats,
    raw_rubric_text,
    read_jsonl,
    sample_id,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_TOKENIZER = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--processed-jsonl", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def variant_target(variant: str, row: dict[str, Any], soft: list[float]) -> list[float]:
    if variant.startswith("v00") or variant.startswith("v10"):
        label = int(row["label_5"])
        return [1.0 if index == label else 0.0 for index in range(1, 6)]
    return list(soft)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, local_files_only=True)
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError(f"Exp42A requires exactly 2654 rows, found {len(rows)}")
    variants: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for position, row in enumerate(rows):
        sid = sample_id(row)
        stats = human_stats(row)
        soft = stats["human_distribution_5"]
        core_hash = stable_hash((sid, row.get("question"), row.get("answer"), row.get("metric_canonical")))
        for variant in VARIANTS:
            with_rubric = variant.startswith("v10") or variant.startswith("v11")
            text, audit = canonical_model_text(tokenizer, row, raw_rubric_text(row) if with_rubric else None)
            variants[variant].append({
                "position": position,
                "variant": variant,
                "sample_id": sid,
                "question_key": str(row["question_key"]),
                "text": text,
                "soft_target_5": variant_target(variant, row, soft),
                "human_distribution_5": soft,
                "gold_label_5": int(row["label_5"]),
                "expected_human_score": stats["expected_human_score"],
                "human_entropy": stats["human_entropy"],
                "human_score_range": stats["human_score_range"],
                "sample_weight": 1.0,
                "language": row.get("language"),
                "metric_group": row.get("metric_group"),
                "subject_canonical": row.get("subject_canonical"),
                "core_input_hash": core_hash,
                **audit,
            })

    reference_ids = [row["sample_id"] for row in variants[VARIANTS[0]]]
    reference_core = [row["core_input_hash"] for row in variants[VARIANTS[0]]]
    if [row["text"] for row in variants["v00_hard_no_rubric"]] != [row["text"] for row in variants["v01_soft_no_rubric"]]:
        raise RuntimeError("V00/V01 input text differs")
    if [row["text"] for row in variants["v10_hard_raw_rubric"]] != [row["text"] for row in variants["v11_soft_raw_rubric"]]:
        raise RuntimeError("V10/V11 input text differs")
    if [row["soft_target_5"] for row in variants["v00_hard_no_rubric"]] != [row["soft_target_5"] for row in variants["v10_hard_raw_rubric"]]:
        raise RuntimeError("Hard targets differ across rubric factor")
    if [row["soft_target_5"] for row in variants["v01_soft_no_rubric"]] != [row["soft_target_5"] for row in variants["v11_soft_raw_rubric"]]:
        raise RuntimeError("Soft targets differ across rubric factor")

    summaries, equivalence, length_audit = [], [], []
    hashes: dict[str, Any] = {}
    for variant, variant_rows in variants.items():
        if len(variant_rows) != 2654 or [row["sample_id"] for row in variant_rows] != reference_ids:
            raise RuntimeError(f"Row/order mismatch for {variant}")
        if [row["core_input_hash"] for row in variant_rows] != reference_core:
            raise RuntimeError(f"Core input mismatch for {variant}")
        if any(abs(sum(row["soft_target_5"]) - 1.0) > 1e-8 or min(row["soft_target_5"]) < 0 for row in variant_rows):
            raise RuntimeError(f"Invalid target for {variant}")
        if any(float(row["sample_weight"]) != 1.0 for row in variant_rows):
            raise RuntimeError(f"Unexpected sample weight for {variant}")
        path = args.out_dir / f"private/data/exp42a_{variant}.jsonl"
        write_jsonl(path, variant_rows)
        lengths = [int(row["input_tokens"]) for row in variant_rows]
        summary = {
            "variant": variant,
            "rows": len(variant_rows),
            "unique_sample_ids": len(set(reference_ids)),
            "unique_question_keys": len({row["question_key"] for row in variant_rows}),
            "target_type": "hard_one_hot" if "hard" in variant else "human_empirical_soft",
            "rubric_input": "raw_rubric" if "raw_rubric" in variant else "none",
            "sample_weight_mean": 1.0,
            "mean_input_tokens": float(np.mean(lengths)),
            "p95_input_tokens": percentile(lengths, 95),
            "max_input_tokens": max(lengths),
            "question_truncation_count": sum(bool(row["question_truncated"]) for row in variant_rows),
            "rubric_truncation_count": sum(bool(row["rubric_truncated"]) for row in variant_rows),
            "answer_truncation_count": sum(bool(row["answer_truncated"]) for row in variant_rows),
        }
        summaries.append(summary)
        length_audit.append(summary)
        equivalence.append({
            "variant": variant,
            "rows_equal": len(variant_rows) == 2654,
            "sample_id_order_equal": [row["sample_id"] for row in variant_rows] == reference_ids,
            "core_hash_equal": [row["core_input_hash"] for row in variant_rows] == reference_core,
            "sample_weights_all_one": all(float(row["sample_weight"]) == 1.0 for row in variant_rows),
            "target_sum_valid": all(abs(sum(row["soft_target_5"]) - 1.0) <= 1e-8 for row in variant_rows),
        })
        hashes[variant] = {
            "private_path": str(path),
            "rows": len(variant_rows),
            "sha256": sha256_file(path),
            "sample_order_sha256": stable_hash(reference_ids),
            "core_identity_sha256": stable_hash(reference_core),
            "target_sha256": stable_hash([row["soft_target_5"] for row in variant_rows]),
            "input_text_sha256": stable_hash([row["text"] for row in variant_rows]),
        }

    no_rubric_budget_equal = [row["answer_token_budget"] for row in variants["v00_hard_no_rubric"]] == [row["answer_token_budget"] for row in variants["v01_soft_no_rubric"]]
    raw_rubric_budget_equal = [row["answer_token_budget"] for row in variants["v10_hard_raw_rubric"]] == [row["answer_token_budget"] for row in variants["v11_soft_raw_rubric"]]
    if not no_rubric_budget_equal or not raw_rubric_budget_equal:
        raise RuntimeError("Answer token budgets differ within an input factor")
    for row in equivalence:
        row["answer_budget_equal_within_input_factor"] = no_rubric_budget_equal if row["variant"].startswith(("v00", "v01")) else raw_rubric_budget_equal

    protocol = {
        "experiment": "Exp42A RubiDist 2x2 Factorial Multiseed Lock",
        "rows": 2654,
        "question_keys": 196,
        "factors": {"target": ["rounded_hard", "three_human_empirical_soft"], "input": ["metric_only", "raw_rubric"]},
        "variants": list(VARIANTS),
        "main_method": "v11_soft_raw_rubric",
        "api_calls": False,
        "teacher_labeling": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    training = {
        "model": "Qwen3-Reranker-0.6B",
        "seeds": [42, 43, 44],
        "folds": 5,
        "epochs": 10,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "micro_batch": 4,
        "eval_batch": 4,
        "gradient_accumulation": 32,
        "effective_batch": 128,
        "max_length": 2048,
        "question_budget": 256,
        "metric_budget": 64,
        "raw_rubric_budget": 256,
        "gradient_clip": 1.0,
        "fixed_final_epoch": 10,
        "checkpoint_selection": "none",
        "loss": "standard_hard_or_soft_cross_entropy",
        "sample_weighting": False,
        "class_weighting": False,
        "oversampling": False,
        "logit_adjustment": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    success_gate = {
        "main_method": "v11_soft_raw_rubric",
        "rubric_gain_vs_v01": {"mae_improvement_min": 0.01, "qwk_improvement_min": 0.02, "crossed_ci_must_exclude_zero": True},
        "soft_noninferiority_vs_v10": {"mae_margin": 0.005, "qwk_margin": 0.01, "exact_margin": 0.005, "brier_or_rps_better": True},
        "guards_vs_v01": {"exact_margin": 0.005, "kendall_margin": 0.01, "abs_bias_margin": 0.01, "low_to_high_noninferior": True, "high_to_low_margin": 0.01, "label5_margin": 0.02},
        "stability": {"favorable_seeds_min": 2, "per_seed_mae_worse_max": 0.02, "per_seed_label5_worse_max": 0.04},
    }
    write_csv(args.out_dir / "tables/exp42a_variant_summary.csv", summaries)
    write_csv(args.out_dir / "tables/exp42a_variant_equivalence.csv", equivalence)
    write_csv(args.out_dir / "tables/exp42a_input_length_audit.csv", length_audit)
    write_json(args.out_dir / "configs/exp42a_factorial_protocol_lock.json", protocol)
    write_json(args.out_dir / "configs/exp42a_training_config.json", training)
    write_json(args.out_dir / "configs/exp42a_success_gate.json", success_gate)
    write_json(args.out_dir / "hashes/exp42a_dataset_hashes.json", {
        "train_path": str(args.train_jsonl),
        "train_sha256": sha256_file(args.train_jsonl),
        "processed_path_audit_only": str(args.processed_jsonl),
        "processed_sha256": sha256_file(args.processed_jsonl),
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "variants": hashes,
        "no_rubric_input_equality": True,
        "raw_rubric_input_equality": True,
        "hard_target_equality": True,
        "soft_target_equality": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    })
    print(json.dumps({"status": "PREPARED", "variants": 4, "rows_per_variant": 2654, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
