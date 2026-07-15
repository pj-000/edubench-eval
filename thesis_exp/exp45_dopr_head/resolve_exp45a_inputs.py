"""Resolve and lock Exp45A inputs without reading dev or test."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from thesis_exp.exp43_rubimor.common import canonical_metric, human_stats, sample_id
from thesis_exp.exp43_rubimor.prepare_exp43_datasets import format_row
from thesis_exp.exp45_dopr_head.common import (
    EXPECTED_E4_HASH,
    EXPECTED_FOLD_HASH,
    EXP43_ROOT,
    EXP44_ROOT,
    ROOT,
    TRAIN_PATH,
    assert_train_only,
    ensure_dirs,
    fold_map,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--fold-source", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_train_only(args.train)
    ensure_dirs(args.out_dir)
    fold_source = args.fold_source or EXP43_ROOT / "private/data/exp43_groupcv_fold_assignment.csv"
    if not args.train.exists() or not fold_source.exists():
        raise FileNotFoundError(f"Missing train/fold input: {args.train}, {fold_source}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    mapping = json.loads((EXP43_ROOT / "configs/exp43_metric_mapping.json").read_text(encoding="utf-8"))["metrics"]
    built = []
    for position, row in enumerate(read_jsonl(args.train)):
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
                "core_input_hash": stable_hash((sample_id(row), row.get("question"), row.get("answer"), metric)),
                **audit,
            }
        )
    if len(built) != 2654 or len({row["sample_id"] for row in built}) != 2654:
        raise RuntimeError("Exp45A requires 2654 unique train rows")
    folds = fold_map(fold_source)
    if set(folds) != {row["sample_id"] for row in built}:
        raise RuntimeError("Fold IDs differ from E4 rows")
    assignments = [
        {"sample_id": row["sample_id"], "question_key": row["question_key"], "fold": folds[row["sample_id"]]}
        for row in built
    ]
    semantic_fold_hash = stable_hash(assignments)
    if semantic_fold_hash != EXPECTED_FOLD_HASH:
        raise RuntimeError(f"Fold hash mismatch: {semantic_fold_hash}")

    data_path = args.out_dir / "private/data/exp45a_train_e4.jsonl"
    fold_path = args.out_dir / "private/data/exp45a_groupcv_fold_assignment.csv"
    write_jsonl(data_path, built)
    shutil.copyfile(fold_source, fold_path)
    if sha256_file(data_path) != EXPECTED_E4_HASH:
        raise RuntimeError("Exp45A E4 bytes differ from locked Exp43/44 E4 data")

    model_dir = Path(args.model_name_or_path)
    model_hashes = {
        "model_path": str(model_dir.resolve()),
        "config_sha256": sha256_file(model_dir / "config.json"),
        "tokenizer_config_sha256": sha256_file(model_dir / "tokenizer_config.json"),
    }
    encoder_lock = {
        "model": "Qwen3-Reranker-0.6B", "input": "question+answer+metric+raw rubric+metadata",
        "loss": "human distribution CE + 0.5 ordinal CDF MSE", "pooling": "last non-padding token",
        "full_finetuning": True, "epochs": 10, "learning_rate": 2e-5, "weight_decay": 0.01,
        "warmup_ratio": 0.05, "scheduler": "cosine", "batch_size": 4,
        "gradient_accumulation": 32, "max_length": 2048, "fixed_checkpoint": "epoch10",
        "seed": 42, "folds": 5, "dev_access_count": 0, "test_access_count": 0,
    }
    prototype_lock = {
        "source": "outer-train frozen normalized embeddings only",
        "weight": "three-human empirical distribution", "prediction": "maximum cosine similarity",
        "diagnostic_gate": {"label2_recall_min": 0.05, "label2_correct_min": 3, "class2_predictions_min": 3},
    }
    head_lock = {
        "variants": ["H1_vanilla_cRT", "H2_distributional_ordinal_cRT", "H3_prototype_cRT_no_prior", "H4_DOPR"],
        "batch": "20 per hard class (100 total)", "epochs": 10, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "scheduler": "cosine", "cosine_scale": 16,
        "distributional_loss": "CE + 0.5 ordinal CDF MSE", "prior_exponent": 1.0,
    }
    success_gate = {
        "label2_recall_min": 0.05, "label2_correct_min": 3, "low_to_high_max": 0.7368421053,
        "overall_gain_any": {"MAE": 0.003, "QWK": 0.005, "Kendall_tau": 0.005},
        "protection": {"Exact_drop_max": 0.005, "abs_bias_worsening_max": 0.01, "label5_drop_max": 0.02, "high_to_low_increase_max": 0.01, "mean_score_decline_max": 0.10},
        "bootstrap_replicates": 5000,
    }
    write_json(args.out_dir / "configs/exp45a_encoder_protocol_lock.json", encoder_lock)
    write_json(args.out_dir / "configs/exp45a_prototype_protocol_lock.json", prototype_lock)
    write_json(args.out_dir / "configs/exp45a_head_protocol_lock.json", head_lock)
    write_json(args.out_dir / "configs/exp45a_success_gate.json", success_gate)
    write_json(args.out_dir / "hashes/exp45a_model_hashes.json", model_hashes)
    write_json(args.out_dir / "hashes/exp45a_data_hashes.json", {"raw_train": sha256_file(args.train), "rebuilt_e4": sha256_file(data_path)})
    write_json(args.out_dir / "hashes/exp45a_fold_hashes.json", {"semantic": semantic_fold_hash, "file_sha256": sha256_file(fold_path), "expected": EXPECTED_FOLD_HASH})
    code_paths = sorted(Path("thesis_exp/exp45_dopr_head").glob("*.py")) + sorted(Path("thesis_exp/scripts").glob("run_exp45a_*.sh"))
    write_json(args.out_dir / "hashes/exp45a_code_hashes.json", {str(path): sha256_file(path) for path in code_paths})
    write_json(
        args.out_dir / "hashes/exp45a_training_config_hashes.json",
        {
            "encoder_protocol_sha256": sha256_file(args.out_dir / "configs/exp45a_encoder_protocol_lock.json"),
            "prototype_protocol_sha256": sha256_file(args.out_dir / "configs/exp45a_prototype_protocol_lock.json"),
            "head_protocol_sha256": sha256_file(args.out_dir / "configs/exp45a_head_protocol_lock.json"),
            "success_gate_sha256": sha256_file(args.out_dir / "configs/exp45a_success_gate.json"),
        },
    )
    write_csv(
        args.out_dir / "tables/exp45a_resolved_input_manifest.csv",
        [
            {"artifact": "raw_train", "path": args.train, "rows": 2654, "sha256": sha256_file(args.train)},
            {"artifact": "exp45_e4", "path": data_path, "rows": 2654, "sha256": sha256_file(data_path)},
            {"artifact": "fold_assignment", "path": fold_path, "rows": 2654, "sha256": sha256_file(fold_path)},
            {"artifact": "exp44_metrics", "path": EXP44_ROOT / "tables/exp44a_seed42_metrics.csv", "rows": 4, "sha256": sha256_file(EXP44_ROOT / "tables/exp44a_seed42_metrics.csv")},
        ],
    )
    write_csv(args.out_dir / "tables/exp45a_fold_hash_audit.csv", [{"expected": EXPECTED_FOLD_HASH, "observed": semantic_fold_hash, "match": True, "dev_access_count": 0, "test_access_count": 0}])
    print(json.dumps({"status": "INPUTS_RESOLVED", "rows": 2654, "fold_hash": semantic_fold_hash, "train_hash": sha256_file(data_path), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
