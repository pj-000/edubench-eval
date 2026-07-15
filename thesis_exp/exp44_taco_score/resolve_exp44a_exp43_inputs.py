"""Resolve and lock Exp44A train-only inputs against Exp43 E4."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from thesis_exp.exp43_rubimor.common import (
    canonical_metric,
    human_stats,
    sample_id,
    stable_hash,
)
from thesis_exp.exp43_rubimor.prepare_exp43_datasets import format_row
from thesis_exp.exp44_taco_score.common import (
    EXP43_EXPECTED_E4_HASH,
    EXP43_EXPECTED_FOLD_HASH,
    EXP43_FOLD_PATH,
    EXP43_ROOT,
    ROOT,
    TRAIN_PATH,
    assert_train_only,
    ensure_dirs,
    fold_map,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--exp43-root", type=Path, default=EXP43_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_train_only(args.train)
    ensure_dirs(args.out_dir)
    if not args.train.exists():
        raise FileNotFoundError(args.train)
    fold_source = args.exp43_root / "private/data/exp43_groupcv_fold_assignment.csv"
    if not fold_source.exists():
        raise FileNotFoundError(f"Missing locked Exp43 fold assignment: {fold_source}")
    observed_fold_hash = sha256_file(fold_source)
    if observed_fold_hash != EXP43_EXPECTED_FOLD_HASH:
        raise RuntimeError(
            f"Exp43 fold hash mismatch: {observed_fold_hash} != {EXP43_EXPECTED_FOLD_HASH}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    mapping_path = args.exp43_root / "configs/exp43_metric_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["metrics"]
    source = read_jsonl(args.train)
    built = []
    for position, row in enumerate(source):
        stats = human_stats(row)
        text, audit = format_row(tokenizer, row, with_rubric=True)
        metric = canonical_metric(row)
        built.append(
            {
                "position": position,
                "split": "train",
                "variant": "E4",
                "sample_id": sample_id(row),
                "question_key": str(row["question_key"]),
                "text": text,
                "metric": metric,
                "metric_id": int(mapping[metric]),
                "soft_target_5": stats["human_distribution_5"],
                **stats,
                "language": row.get("language") or "unknown",
                "subject": row.get("subject_canonical") or row.get("subject_raw") or "unknown",
                "scenario": row.get("scenario_canonical") or row.get("scenario_raw") or "unknown",
                "education_level": row.get("education_level_canonical") or row.get("education_level_raw") or "unknown",
                "core_input_hash": stable_hash(
                    (sample_id(row), row.get("question"), row.get("answer"), metric)
                ),
                **audit,
            }
        )
    if len(built) != 2654 or len({row["sample_id"] for row in built}) != 2654:
        raise RuntimeError("Exp44A requires exactly 2654 unique train rows")
    folds = fold_map(fold_source)
    if set(folds) != {row["sample_id"] for row in built}:
        raise RuntimeError("Exp43 fold assignment IDs differ from rebuilt E4 IDs")

    train_output = args.out_dir / "private/data/exp44a_train_e4.jsonl"
    fold_output = args.out_dir / "private/data/exp44a_groupcv_fold_assignment.csv"
    write_jsonl(train_output, built)
    # Preserve the locked Exp43 bytes, not merely the same logical assignment.
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fold_source, fold_output)
    train_hash = sha256_file(train_output)
    fold_hash = sha256_file(fold_output)
    if train_hash != EXP43_EXPECTED_E4_HASH:
        raise RuntimeError(
            f"Rebuilt E4 input hash mismatch: {train_hash} != {EXP43_EXPECTED_E4_HASH}"
        )
    if fold_hash != EXP43_EXPECTED_FOLD_HASH:
        raise RuntimeError(
            f"Copied fold hash mismatch: {fold_hash} != {EXP43_EXPECTED_FOLD_HASH}"
        )

    model_config = Path(args.model_name_or_path) / "config.json"
    tokenizer_config = Path(args.model_name_or_path) / "tokenizer_config.json"
    model_hashes = {
        "model_path": str(Path(args.model_name_or_path).resolve()),
        "config_sha256": sha256_file(model_config),
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
    }
    data_lock = {
        "source_commit": "f760132",
        "train_path": str(args.train),
        "train_rows": 2654,
        "question_keys": 196,
        "folds": 5,
        "seed": 42,
        "target": "three-human empirical distribution",
        "hard_label": "half-up rounded human mean; sampling/evaluation only",
        "input": "question + answer + metric + raw rubric + metadata",
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    triplet_lock = {
        "distance": "sum_b=1^4 abs(CDF(H_i)_b-CDF(H_j)_b)",
        "anchor_stream": "all low once + equal without-replacement mid/high per fold and epoch",
        "positive": "same hard label, different question_key, prefer language, min distance",
        "near": "0.5 < D < 2, prefer language, closest to D=1",
        "far": "D >= 2, prefer language, maximum D",
        "partner_cap_per_epoch": 8,
        "epochs": 10,
    }
    model_loss_lock = {
        "model": "Qwen3-Reranker-0.6B full fine-tuning",
        "pooling": "last non-padding token",
        "projection_dim": 128,
        "projection_initialization": "Xavier uniform",
        "base_loss": "L_dist + 0.5 * L_ord",
        "taco_loss_weight": 0.1,
        "temperature": 0.1,
        "distance_margin": "0.1 * D",
        "metric_residual": False,
    }
    gate_lock = {
        "label2_recall_min": 0.05,
        "label2_correct_min": 3,
        "low_to_high_not_worse": True,
        "overall_gain_any": {"MAE": 0.003, "QWK": 0.005, "Kendall_tau": 0.005},
        "protection": {"Exact_drop_max": 0.005, "abs_bias_worsening_max": 0.01, "label5_drop_max": 0.02, "high_to_low_increase_max": 0.01},
        "bootstrap_replicates": 5000,
    }
    write_json(args.out_dir / "configs/exp44a_data_protocol_lock.json", data_lock)
    write_json(args.out_dir / "configs/exp44a_triplet_protocol_lock.json", triplet_lock)
    write_json(args.out_dir / "configs/exp44a_model_loss_lock.json", model_loss_lock)
    write_json(args.out_dir / "configs/exp44a_success_gate.json", gate_lock)
    write_json(args.out_dir / "hashes/exp44a_model_hashes.json", model_hashes)
    write_json(
        args.out_dir / "hashes/exp44a_data_hashes.json",
        {
            "raw_train": {"path": str(args.train), "sha256": sha256_file(args.train)},
            "rebuilt_e4": {"path": str(train_output), "sha256": train_hash},
        },
    )
    write_json(
        args.out_dir / "hashes/exp44a_fold_hashes.json",
        {
            "exp43_expected": EXP43_EXPECTED_FOLD_HASH,
            "exp43_observed": observed_fold_hash,
            "exp44_observed": fold_hash,
            "all_equal": True,
        },
    )
    write_csv(
        args.out_dir / "tables/exp44a_resolved_input_manifest.csv",
        [
            {"artifact": "raw_train", "path": args.train, "rows": 2654, "sha256": sha256_file(args.train)},
            {"artifact": "exp44_e4", "path": train_output, "rows": 2654, "sha256": train_hash},
            {"artifact": "fold_assignment", "path": fold_output, "rows": 2654, "sha256": fold_hash},
            {"artifact": "model_config", "path": model_config, "rows": "", "sha256": model_hashes["config_sha256"]},
            {"artifact": "tokenizer_config", "path": tokenizer_config, "rows": "", "sha256": model_hashes["tokenizer_config_sha256"]},
        ],
    )
    write_csv(
        args.out_dir / "tables/exp44a_fold_hash_audit.csv",
        [{"expected": EXP43_EXPECTED_FOLD_HASH, "exp43": observed_fold_hash, "exp44": fold_hash, "match": True}],
    )
    print(json.dumps({"status": "INPUTS_RESOLVED", "rows": len(built), "fold_hash": fold_hash, "train_hash": train_hash, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
