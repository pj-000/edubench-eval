"""Generate outcome-independent Consensus-only trajectories for Exp63."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.train import aggregate_text_hash, load_model
from thesis_exp.exp63_same_state_counterfactual import (
    ARTIFACT_ROOT,
    OUTPUT_ROOT,
    SEEDS,
    STAGE_EPOCHS,
)
from thesis_exp.exp63_same_state_counterfactual.runtime import (
    capture_rng,
    hash_strings,
    load_protocol,
    ordered_rows,
    sha256_file,
    write_json,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    TrainConfig,
    make_dataloader,
    set_seed,
)


def parse_args() -> tuple[TrainConfig, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--artifact_dir", type=Path)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", choices=("true",), default="true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or OUTPUT_ROOT / "base" / f"seed_{args.seed}"
    artifact_dir = args.artifact_dir or ARTIFACT_ROOT / f"seed_{args.seed}"
    config = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP63_TRAIN_DEV_ONLY"),
        output_dir=output_dir,
        checkpoint_output_dir=artifact_dir,
        max_length=args.max_length,
        num_train_epochs=float(args.num_train_epochs),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=5,
        progress_bar=not args.no_progress,
        evaluate_test=False,
    )
    return config, args.seed


def save_stage(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: TrainConfig,
    seed: int,
    epoch: int,
    global_step: int,
    device: Any,
    next_window_ids: list[str],
) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "EXP63_COMPLETE_STAGE_STATE_V1",
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng(torch, device),
        "training_config": asdict(config),
        "next_epoch": epoch + 1,
        "next_window_record_ids": next_window_ids,
        "next_window_sha256": hash_strings(next_window_ids),
        "test_access_count": 0,
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch": epoch,
        "global_step": global_step,
        "next_window_sha256": payload["next_window_sha256"],
        "next_window_rows": len(next_window_ids),
    }


def train(config: TrainConfig, seed: int) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    protocol = load_protocol()
    fixed = protocol["fixed_training"]
    expected = {
        "num_train_epochs": float(protocol["base_trajectory"]["epochs"]),
        "learning_rate": fixed["learning_rate"],
        "weight_decay": fixed["weight_decay"],
        "warmup_ratio": fixed["warmup_ratio"],
        "per_device_train_batch_size": fixed["micro_batch_size"],
        "gradient_accumulation_steps": fixed["gradient_accumulation_steps"],
        "max_grad_norm": fixed["max_grad_norm"],
        "max_length": fixed["max_length"],
    }
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise RuntimeError(f"Frozen setting mismatch: {key}")
    if config.evaluate_test:
        raise PermissionError("Exp63 never accesses test")
    set_seed(seed)
    rows = model_rows("train")
    model, tokenizer, model_mode, head_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal Exp63 requires CUDA")
    model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    batches_per_epoch = math.ceil(len(rows) / config.per_device_train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_steps = int(config.num_train_epochs) * updates_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    checkpoints: list[dict[str, Any]] = []
    epoch_audit: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(1, int(config.num_train_epochs) + 1):
        epoch_rows = ordered_rows(rows, seed, epoch)
        loader = make_dataloader(epoch_rows, tokenizer, config, "train", shuffle=False)
        model.train()
        clip_events = 0
        maximum_preclip = 0.0
        for micro_step, batch in enumerate(loader, start=1):
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            targets = torch.tensor(
                [row["soft_target_5"] for row in metadata], dtype=torch.float32, device=device
            )
            scale = 1.0 / config.gradient_accumulation_steps
            outputs = model(
                **inputs,
                labels=labels,
                soft_targets=targets,
                aux_route="consensus_only",
                route_loss_scale=scale,
            )
            objective = cbrd_objective(outputs, labels, targets, variant="consensus_only")
            (objective["optimization_loss"] * scale).backward()
            if micro_step % config.gradient_accumulation_steps == 0 or micro_step == len(loader):
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    .detach()
                    .cpu()
                )
                maximum_preclip = max(maximum_preclip, preclip)
                clip_events += int(preclip > config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        epoch_audit.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "clip_events": clip_events,
                "maximum_preclip_norm": maximum_preclip,
                "learning_rate_after_epoch": scheduler.get_last_lr()[0],
                "record_order_sha256": hash_strings(str(row["record_id"]) for row in epoch_rows),
            }
        )
        write_json(config.output_dir / "epoch_audit.json", epoch_audit)
        print(
            f"[exp63:base] seed={seed} epoch={epoch}/10 step={global_step}/{total_steps} "
            f"clip={clip_events}/{updates_per_epoch} elapsed={time.time()-start:.1f}s",
            flush=True,
        )
        if epoch in STAGE_EPOCHS:
            next_rows = ordered_rows(rows, seed, epoch + 1)
            window_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
            next_ids = [str(row["record_id"]) for row in next_rows[:window_size]]
            checkpoint_path = config.checkpoint_output_dir / f"after_epoch_{epoch}.pt"
            checkpoints.append(
                save_stage(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    seed=seed,
                    epoch=epoch,
                    global_step=global_step,
                    device=device,
                    next_window_ids=next_ids,
                )
            )
            write_json(config.output_dir / "checkpoints.json", checkpoints)
    summary = {
        "status": "COMPLETED",
        "seed": seed,
        "model_mode": model_mode,
        "head_contract": head_contract,
        "train_rows": len(rows),
        "train_text_hash": aggregate_text_hash(rows),
        "updates_per_epoch": updates_per_epoch,
        "total_steps": total_steps,
        "checkpoints": checkpoints,
        "epoch_audit": epoch_audit,
        "test_access_count": 0,
        "elapsed_seconds": time.time() - start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(config.output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    config, seed = parse_args()
    train(config, seed)


if __name__ == "__main__":
    main()

