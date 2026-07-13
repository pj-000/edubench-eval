"""Prepare two target-blind, order-swapped DeepSeek packets per Exp39A pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    PROMPT_PATH,
    ROOT,
    SCHEMA_PATH,
    ensure_output_dirs,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

FORBIDDEN_KEYS = {
    "assigned_target_score", "target_score", "target_score_range", "source_label_5",
    "source_human_distribution_5", "human_1", "human_2", "human_3", "label_5",
    "acceptance_status", "most_plausible_counterfactual_score", "counterfactual_score_range",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def clean_clause(value: object) -> str:
    return re.sub(r"^\s*[1-5]\s*[:：.\-)]+\s*", "", str(value or "")).strip()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out_dir)
    resolved_path = args.out_dir / "private/resolved_pairs/exp40a_resolved_exp39a_pairs.jsonl"
    if not resolved_path.exists():
        raise FileNotFoundError("Run resolve_exp39a_pair_inputs.py before preparing packets")
    pairs = read_jsonl(resolved_path)
    if len(pairs) != 240:
        raise ValueError("Exp40A packet preparation requires all 240 resolved pairs")

    packets = []
    for pair in pairs:
        shared = {
            "pair_id": pair["pair_id"],
            "sample_id": pair["sample_id"],
            "question_key": pair["question_key"],
            "question": pair["question"],
            "metric": pair["metric"],
            "metric_group": pair["metric_group"],
            "rubric": pair["rubric"],
            "targeted_rubric_clause": clean_clause(pair["targeted_rubric_clause"]),
            "declared_counterfactual_operator": pair["operator"],
            "language": pair.get("language", "unknown"),
        }
        packets.extend(
            [
                {
                    **shared,
                    "order": "original_first",
                    "answer_a": pair["original_answer"],
                    "answer_b": pair["counterfactual_answer"],
                },
                {
                    **shared,
                    "order": "counterfactual_first",
                    "answer_a": pair["counterfactual_answer"],
                    "answer_b": pair["original_answer"],
                },
            ]
        )
    serialized = json.dumps(packets, ensure_ascii=False, sort_keys=True)
    leaked = sorted(key for key in FORBIDDEN_KEYS if f'"{key}"' in serialized)
    if leaked:
        raise RuntimeError(f"Pairwise verification packets leak forbidden target fields: {leaked}")
    if len(packets) != 480 or len({(row["pair_id"], row["order"]) for row in packets}) != 480:
        raise RuntimeError("Expected 480 unique pair/order packets")
    if any(re.match(r"^\s*[1-5]\s*[:：.\-)]+", row["targeted_rubric_clause"]) for row in packets):
        raise RuntimeError("Targeted rubric clause leaked an explicit target score prefix")

    packet_path = args.out_dir / "private/pair_packets/exp40a_pairwise_verification_packets.jsonl"
    write_jsonl(packet_path, packets)
    prompt_copy = args.out_dir / "prompts/exp40a_deepseek_pairwise_judge.md"
    schema_copy = args.out_dir / "schemas/exp40a_pairwise_judgment_schema.json"
    shutil.copy2(PROMPT_PATH, prompt_copy)
    # The frozen API campaign materialized one blank line after the final
    # instruction. Hash that exact request text while keeping Markdown clean.
    request_prompt = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n") + "\n\n"
    shutil.copy2(SCHEMA_PATH, schema_copy)
    lock = {
        "experiment": "Exp40A EduPair-CF",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "temperature": 0,
        "max_tokens": 1024,
        "requests_independent": True,
        "orders": ["original_first", "counterfactual_first"],
        "pair_count": 240,
        "judgment_count": 480,
        "prompt_sha256": hashlib.sha256(request_prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "packet_sha256": sha256_file(packet_path),
        "target_blind": True,
        "forbidden_fields": sorted(FORBIDDEN_KEYS),
        "acceptance_rules_frozen": True,
        "qualification_gate": {
            "pairs_prepared": 240,
            "judgments_completed": 480,
            "schema_success_min": 0.98,
            "accepted_pairs_min": 150,
            "accepted_unique_question_keys_min": 120,
            "targeted_degradation_rate_min": 0.90,
            "non_target_preservation_rate_min": 0.85,
            "position_source_win_rate_difference_max": 0.05,
            "order_conflict_or_tie_rate_max": 0.25,
        },
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    lock_path = args.out_dir / "configs/exp40a_pairwise_protocol_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise RuntimeError(f"Frozen Exp40A pairwise protocol changed: {lock_path}")
    write_json(lock_path, lock)
    training_lock = {
        "experiment": "Exp40A EduPair-CF GroupCV",
        "variants": [
            "v0h_human_soft", "v1_human_real_pairs", "v2_unverified_counterfactual_pairs",
            "v3_edupair_cf", "v4_shuffled_pair_alignment",
        ],
        "original_train_rows": 2654,
        "pair_count_rule": "V1/V2/V3/V4 each match the accepted V3 pair count after qualification GO",
        "loss": "human_empirical_soft_CE + 0.25 * RankNet_pair_loss",
        "lambda_pair": 0.25,
        "epochs": 10,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "micro_batch": 4,
        "gradient_accumulation": 32,
        "effective_batch": 128,
        "max_length": 2048,
        "seed": 42,
        "sample_weighting": False,
        "class_weighting": False,
        "custom_risk_or_ordinal_loss": False,
        "auxiliary_head": False,
        "fixed_final_epoch": 10,
        "heldout_checkpoint_selection": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    training_lock_path = args.out_dir / "configs/exp40a_training_matrix_lock.json"
    if training_lock_path.exists() and json.loads(training_lock_path.read_text(encoding="utf-8")) != training_lock:
        raise RuntimeError(f"Frozen Exp40A training matrix changed: {training_lock_path}")
    write_json(training_lock_path, training_lock)
    write_csv(
        args.out_dir / "tables/exp40a_pairwise_completion.csv",
        [{"stage": "packets_prepared", "pairs": 240, "expected_judgments": 480, "completed_judgments": 0, "rate": 0.0}],
    )
    write_json(
        args.out_dir / "hashes/exp40a_dataset_hashes.json",
        {
            "resolved_pairs_sha256": sha256_file(resolved_path),
            "private_packet_sha256": sha256_file(packet_path),
            "packet_identity_sha256": stable_hash(sorted((row["pair_id"], row["order"]) for row in packets)),
            "training_matrix_sha256": sha256_file(training_lock_path),
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    training_sources = {
        "training_matrix_lock": training_lock_path,
        "prepare_groupcv_folds": Path("thesis_exp/exp40_edupair_cf/prepare_exp40a_groupcv_folds.py"),
        "train_groupcv": Path("thesis_exp/exp40_edupair_cf/train_exp40a_groupcv.py"),
        "collect_groupcv": Path("thesis_exp/exp40_edupair_cf/collect_exp40a_groupcv.py"),
        "bootstrap_groupcv": Path("thesis_exp/exp40_edupair_cf/bootstrap_exp40a_groupcv.py"),
    }
    write_json(
        args.out_dir / "hashes/exp40a_training_config_hashes.json",
        {
            "artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in training_sources.items()
            },
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    print(json.dumps({"status": "PREPARED", "pairs": 240, "judgments": 480, "target_blind": True}, sort_keys=True))


if __name__ == "__main__":
    main()
