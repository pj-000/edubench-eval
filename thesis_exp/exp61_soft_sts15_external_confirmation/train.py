"""Formal nine-run, train/dev-only Exp61 trainer.

This entry point is intentionally non-runnable until the final independent
protocol/preflight review flips both protocol and source-lock authorization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    OUTPUT_ROOT,
    SEEDS,
    VARIANTS,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import verify_source_lock
from thesis_exp.exp61_soft_sts15_external_confirmation.data import load_model_rows, rows_contract_sha256
from thesis_exp.exp61_soft_sts15_external_confirmation.geometry import compose_geometry_step
from thesis_exp.exp61_soft_sts15_external_confirmation.mapping import mapping_sha256, mapping_target_lookup
from thesis_exp.exp61_soft_sts15_external_confirmation.metrics import evaluate_hard_head
from thesis_exp.exp61_soft_sts15_external_confirmation.model import ModelConfig, load_model_and_tokenizer
from thesis_exp.exp61_soft_sts15_external_confirmation.runtime import (
    accumulate_residual_vjp,
    capture_rng,
    collate,
    restore_rng,
    routed_objective,
)


@dataclass(frozen=True)
class FormalConfig:
    source_repo: Path
    model_name_or_path: str
    output_dir: Path
    checkpoint_dir: Path
    variant: str
    seed: int
    max_length: int = 256
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    micro_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 32
    max_gradient_norm: float = 1.0
    num_workers: int = 0


def parse_args() -> FormalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--checkpoint_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_ROOT / "runs" / args.variant / f"seed_{args.seed}"
    checkpoint = args.checkpoint_dir or Path("thesis_exp/artifacts/exp61_soft_sts15_external_confirmation") / args.variant / f"seed_{args.seed}" / "epoch10"
    return FormalConfig(args.source_repo, args.model_name_or_path, output, checkpoint, args.variant, args.seed)


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, config: FormalConfig, *, shuffle: bool) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(config.seed)
    batch_size = config.micro_batch_size if shuffle else config.eval_batch_size
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, config.max_length),
    )


def evaluate_dev(model: Any, loader: Any, device: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []
    means: list[float] = []
    labels: list[int] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch_labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**inputs, labels=batch_labels, aux_route="ordinary_hmsa")
            probs = torch.softmax(outputs["hard_logits"].float(), dim=-1).cpu().numpy()
            probabilities.append(probs)
            for row, values in zip(metadata, probs):
                point = float(values @ np.arange(6))
                predictions.append({
                    "record_id": row["record_id"],
                    "component_sha256": row["component_sha256"],
                    "human_mean": row["human_mean"],
                    "hard_label": row["label"],
                    "hard_head_probabilities": values.tolist(),
                    "hard_head_expectation": point,
                    "hard_head_argmax": int(values.argmax()),
                })
                means.append(float(row["human_mean"]))
                labels.append(int(row["label"]))
    metrics = evaluate_hard_head(np.concatenate(probabilities), np.asarray(means), np.asarray(labels))
    return metrics, predictions


def save_epoch10(model: Any, tokenizer: Any, config: FormalConfig, metrics: dict[str, Any]) -> None:
    import torch

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.checkpoint_dir / "dual_head_state_dict.pt")
    tokenizer.save_pretrained(config.checkpoint_dir)
    (config.checkpoint_dir / "exp61_checkpoint.json").write_text(
        json.dumps({"config": {**asdict(config), "source_repo": str(config.source_repo), "output_dir": str(config.output_dir), "checkpoint_dir": str(config.checkpoint_dir)}, "metrics": metrics}, indent=2, sort_keys=True) + "\n"
    )


def train(config: FormalConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    verify_source_lock(require_formal_authorization=True)
    if not torch.cuda.is_available():
        raise RuntimeError("formal Exp61 training requires CUDA")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    train_rows = load_model_rows(config.source_repo, "train")
    dev_rows = load_model_rows(config.source_repo, "dev")
    expected_contracts = {
        "train": "e4abd30af2c17937c03c8d8726a9ac61d88af72490f79b91dcc7206a43596f31",
        "dev": "df15133be9843a5b0ee4853c0dfec82e166ff1246d0b2b2324da3c8a5a0dc871",
    }
    if rows_contract_sha256(train_rows) != expected_contracts["train"] or rows_contract_sha256(dev_rows) != expected_contracts["dev"]:
        raise RuntimeError("Exp61 train/dev dataset contract mismatch")
    mapping_rows = [json.loads(line) for line in MAPPING_PATH.read_text().splitlines() if line]
    mapping_audit = json.loads(MAPPING_AUDIT_PATH.read_text())
    if mapping_sha256(mapping_rows) != mapping_audit["mapping_sha256"] or not all(mapping_audit["checks"].values()):
        raise RuntimeError("Exp61 mapping contract mismatch")
    shuffled = mapping_target_lookup(mapping_rows)
    model, tokenizer, head_contract = load_model_and_tokenizer(ModelConfig(config.model_name_or_path))
    device = torch.device("cuda")
    model.to(device)
    train_loader = make_loader(train_rows, tokenizer, config, shuffle=True)
    dev_loader = make_loader(dev_rows, tokenizer, config, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * config.warmup_ratio), total_updates)
    named_backbone = {name: parameter for name, parameter in model.named_parameters() if name.startswith("backbone.") and parameter.requires_grad}
    aligned_buffers = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named_backbone.items()}
    shuffled_buffers = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named_backbone.items()}
    geometry_audits: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        for micro_step, batch in enumerate(train_loader, start=1):
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            aligned_targets = torch.tensor([row["soft_target"] for row in metadata], dtype=torch.float32, device=device)
            shuffled_targets = torch.tensor([shuffled[row["record_id"]] for row in metadata], dtype=torch.float32, device=device)
            scale = 1.0 / config.gradient_accumulation_steps
            rng_before = capture_rng(torch, device)
            outputs = model(**inputs, labels=labels, soft_targets=aligned_targets, aux_route="routed_hmsa", route_loss_scale=scale)
            objective = routed_objective(outputs, labels, aligned_targets)
            (objective["optimization_loss"] * scale).backward()
            rng_after = capture_rng(torch, device)
            restore_rng(torch, device, rng_before)
            accumulate_residual_vjp(model, inputs, labels, aligned_targets, aligned_buffers, scale)
            restore_rng(torch, device, rng_before)
            accumulate_residual_vjp(model, inputs, labels, shuffled_targets, shuffled_buffers, scale)
            restore_rng(torch, device, rng_after)
            if micro_step % config.gradient_accumulation_steps == 0:
                audit = compose_geometry_step(model, aligned_buffers, shuffled_buffers, variant=config.variant, max_norm=config.max_gradient_norm)
                global_step += 1
                audit.update({"epoch": epoch, "global_step": global_step})
                geometry_audits.append(audit)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                for values in (aligned_buffers, shuffled_buffers):
                    for buffer in values.values():
                        buffer.zero_()
        if len(train_loader) % config.gradient_accumulation_steps != 0:
            raise RuntimeError("frozen Exp61 batch contract unexpectedly produced a partial accumulation window")
    metrics, predictions = evaluate_dev(model, dev_loader, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "EXP61_FIXED_EPOCH10_DEV_COMPLETE",
        "variant": config.variant,
        "seed": config.seed,
        "epoch": 10,
        "global_steps": global_step,
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "head_contract": head_contract,
        "test_access_count": 0,
    }
    (config.output_dir / "epoch10_dev_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (config.output_dir / "epoch10_dev_predictions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions))
    (config.output_dir / "geometry_step_audit.json").write_text(json.dumps(geometry_audits, indent=2, sort_keys=True) + "\n")
    save_epoch10(model, tokenizer, config, metrics)
    return summary


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
