"""Run inference for one frozen Exp27R checkpoint under the final-test gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from thesis_exp.exp17_low_score_evidence.prepare_exp27r_final_test_lock import tree_manifest
from thesis_exp.exp17_low_score_evidence.prepare_exp27r_final_test_lock import file_sha256
from thesis_exp.src.edujudge.exp02.train_ce_baseline import load_model_and_tokenizer
from thesis_exp.src.edujudge.exp27p.common import read_jsonl, write_jsonl
from thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker import (
    TrainConfig, evaluate, make_loader, prepare_dev_rows,
)
from thesis_exp.src.edujudge.exp27r import OUTPUT_DIR, VARIANTS, run_root


def evaluate_frozen(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if os.environ.get("RUN_FINAL_TEST") != "1":
        raise RuntimeError("RUN_FINAL_TEST=1 is required to open test")
    manifest = json.loads((args.out_dir / "configs/exp27r_final_lock_manifest.json").read_text(encoding="utf-8"))
    declared_commit = os.environ.get("EXP27R_SOURCE_COMMIT")
    if declared_commit != manifest["locked_source_commit"]:
        raise RuntimeError(f"Commit lock mismatch: {declared_commit} != {manifest['locked_source_commit']}")
    hashes = list(csv.DictReader((args.out_dir / "tables/exp27r_dataset_and_config_hashes.csv").open(
        "r", encoding="utf-8", newline=""
    )))
    for item in hashes:
        path = Path(item["path"])
        if not path.exists() or file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"Code/data lock mismatch: {path}")
    registry = list(csv.DictReader((args.out_dir / "tables/exp27r_checkpoint_registry.csv").open("r", encoding="utf-8", newline="")))
    matches = [row for row in registry if row["variant"] == args.variant and int(row["seed"]) == args.seed and row["checkpoint_kind"] == args.checkpoint_kind]
    if len(matches) != 1:
        raise ValueError("Frozen checkpoint registry lookup is not unique")
    entry = matches[0]
    checkpoint = Path(entry["checkpoint_path"])
    actual_hash, _ = tree_manifest(checkpoint)
    if actual_hash != entry["checkpoint_tree_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")

    summary = json.loads((run_root(args.variant) / args.variant / f"seed_{args.seed}/run_summary.json").read_text(encoding="utf-8"))
    raw = summary["config"]
    config = TrainConfig(
        variant=args.variant, model_name_or_path=raw["model_name_or_path"],
        train_jsonl=Path(raw["train_jsonl"]), dev_jsonl=args.test_jsonl,
        run_dir=Path(raw["run_dir"]), output_dir=args.out_dir,
        checkpoint_output_dir=Path(raw["checkpoint_output_dir"]),
        high_impact_manifest=Path(raw["high_impact_manifest"]),
        max_length=int(raw["max_length"]), per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=int(raw["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        seed=args.seed, bf16=raw.get("bf16", "auto"), fp16=bool(raw.get("fp16", False)),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        local_files_only=bool(raw.get("local_files_only", True)), checkpoint_dir=checkpoint,
        progress_bar=True,
    )
    model, tokenizer, _ = load_model_and_tokenizer(config)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda")
    model.to(device)
    test_rows = prepare_dev_rows(read_jsonl(args.test_jsonl))
    loader = make_loader(test_rows, tokenizer, config, train=False)
    predictions = evaluate(model, loader, device)
    output = args.out_dir / "predictions_private" / args.checkpoint_kind / args.variant / f"seed_{args.seed}.jsonl"
    write_jsonl(output, predictions)
    return {"variant": args.variant, "seed": args.seed, "checkpoint_kind": args.checkpoint_kind,
            "rows": len(predictions), "output": str(output), "checkpoint_hash_verified": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-kind", choices=["selected", "pure_min_mae"], default="selected")
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate_frozen(parse_args()), sort_keys=True))
