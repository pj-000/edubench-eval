"""Evaluate one frozen Exp49 checkpoint on test after formal authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from thesis_exp.exp49_cphce import OUTPUT_ROOT, VARIANTS, checkpoint_dir
from thesis_exp.exp49_cphce.build_targets import load_split
from thesis_exp.exp49_cphce.freeze_manifest import MANIFEST_PATH
from thesis_exp.exp49_cphce.metric_contract import compute_metrics
from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig, evaluate, load_model_and_tokenizer, make_dataloader
from thesis_exp.src.edujudge.utils.io import write_csv, write_jsonl


def evaluate_frozen(variant: str, seed: int, model_name_or_path: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "TEST_IN_PROGRESS" or int(manifest.get("exp49_test_access_count", -1)) != 1:
        raise PermissionError("Frozen manifest does not authorize a test run")
    key = f"{variant}/seed_{seed}"
    if key not in manifest.get("checkpoints", {}):
        raise KeyError(f"Checkpoint not frozen: {key}")
    checkpoint = checkpoint_dir(variant, seed)
    output_dir = OUTPUT_ROOT / "test" / variant / f"seed_{seed}"
    config = TrainConfig(
        model_name_or_path=model_name_or_path,
        data_dir=Path("EXP49_FROZEN_TEST_IN_MEMORY"),
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint.parent,
        max_length=2048,
        num_train_epochs=10.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=seed,
        bf16="auto",
        fp16=False,
        gradient_checkpointing=False,
        trust_remote_code=False,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=True,
        checkpoint_dir=checkpoint,
        num_workers=0,
        log_steps=20,
        progress_bar=False,
        evaluate_test=True,
    )
    model, tokenizer, model_mode = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = load_split("test", allow_test=True)
    loader = make_dataloader(rows, tokenizer, config, "test", shuffle=False)
    result = evaluate(model, loader, device, "test")
    metrics = {"variant": variant, "seed": seed, "model_mode": model_mode, **compute_metrics(result.predictions)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "test_metrics.csv", [metrics])
    write_jsonl(output_dir / "predictions_test.jsonl", result.predictions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_frozen(args.variant, args.seed, args.model_name_or_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
