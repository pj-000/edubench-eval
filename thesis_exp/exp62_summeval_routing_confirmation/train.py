"""Frozen train/dev-only trainer for the four-arm Exp62 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp62_summeval_routing_confirmation import (
    CONFIG_ROOT,
    OUTPUT_ROOT,
    REPO_ROOT,
    SEEDS,
    VARIANTS,
)
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    EXPECTED_ANNOTATION_SHA256,
    sha256_file,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import (
    EXPECTED_MANIFEST_SHA256,
    load_model_rows,
    rows_contract_sha256,
)
from thesis_exp.exp62_summeval_routing_confirmation.geometry import compose_geometry_step
from thesis_exp.exp62_summeval_routing_confirmation.metrics import evaluate_hard_head
from thesis_exp.exp62_summeval_routing_confirmation.model import ModelConfig, load_model_and_tokenizer
from thesis_exp.exp62_summeval_routing_confirmation.runtime import (
    accumulate_residual_vjp,
    capture_rng,
    collate,
    objective,
    restore_rng,
)


PROTOCOL_PATH = CONFIG_ROOT / "protocol.json"
EXPECTED_TRAIN_CONTRACT_SHA256 = "9cf52e485e654037332b624cb49240081e77b5bc1f92e9ff65e72589ddeeb04b"
EXPECTED_DEV_CONTRACT_SHA256 = "c723a4eedd29689edb12841ca8f95788af670bb4bd34295784b6bff132e52603"


@dataclass(frozen=True)
class FormalConfig:
    annotations: Path
    model_name_or_path: str
    variant: str
    seed: int
    output_dir: Path
    checkpoint_dir: Path
    max_length: int = 256
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    micro_batch_size: int = 112
    eval_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0


def parse_args() -> FormalConfig:
    parser = argparse.ArgumentParser(description="Formal Exp62 train/dev-only run")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--checkpoint_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_ROOT / "runs" / args.variant / f"seed_{args.seed}"
    checkpoint = args.checkpoint_dir or (
        REPO_ROOT
        / "thesis_exp/artifacts/exp62_summeval_routing_confirmation"
        / args.variant
        / f"seed_{args.seed}"
        / "epoch10"
    )
    return FormalConfig(
        annotations=args.annotations,
        model_name_or_path=args.model_name_or_path,
        variant=args.variant,
        seed=args.seed,
        output_dir=output,
        checkpoint_dir=checkpoint,
    )


def verify_frozen_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checks = {
        "status": protocol.get("status") == "EXP62_PROTOCOL_FROZEN_FORMAL_TRAINING_AUTHORIZED",
        "arms": tuple(protocol.get("arms", ())) == VARIANTS,
        "seeds": tuple(protocol.get("seeds", ())) == SEEDS,
        "fixed_epoch": protocol.get("training", {}).get("epochs") == 10,
        "no_checkpoint_selection": protocol.get("training", {}).get("checkpoint_selection")
        == "fixed epoch 10; no best-checkpoint selection",
        "manifest": protocol.get("data", {}).get("split_manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "test_access_zero": protocol.get("test_access", {}).get("count") == 0,
        "formal_authorized": protocol.get("authorization", {}).get(
            "formal_twenty_run_gpu_training"
        )
        is True,
        "test_not_authorized": protocol.get("authorization", {}).get(
            "one_time_test_evaluation"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp62 frozen-protocol gate failed: {checks}")
    return protocol


def make_loader(
    rows: list[dict[str, Any]], tokenizer: Any, config: FormalConfig, *, shuffle: bool
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        rows,
        batch_size=config.micro_batch_size if shuffle else config.eval_batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, config.max_length),
    )


def evaluate_dev(model: Any, loader: Any, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []
    means: list[float] = []
    labels: list[int] = []
    dimensions: list[str] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch.pop("labels")
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**inputs, aux_route="ordinary_hmsa")
            probs = torch.softmax(outputs["hard_logits"].float(), dim=-1).cpu().numpy()
            probabilities.append(probs)
            for row, values in zip(metadata, probs):
                expectation = float(values @ np.arange(1, 6))
                prediction = int(values.argmax()) + 1
                predictions.append(
                    {
                        "record_id": row["record_id"],
                        "group_id": row["group_id"],
                        "dimension": row["dimension"],
                        "human_mean": row["human_mean"],
                        "hard_label": row["hard_label"],
                        "hard_head_probabilities": values.tolist(),
                        "hard_head_expectation": expectation,
                        "hard_head_argmax": prediction,
                    }
                )
                means.append(float(row["human_mean"]))
                labels.append(int(row["hard_label"]))
                dimensions.append(str(row["dimension"]))
    metrics = evaluate_hard_head(
        np.concatenate(probabilities), np.asarray(means), np.asarray(labels), dimensions
    )
    return metrics, predictions


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_standard(model: Any, max_norm: float) -> dict[str, Any]:
    import torch

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    preclip = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm).detach().cpu())
    postclip = math.sqrt(
        sum(float(parameter.grad.detach().float().square().sum().cpu()) for parameter in parameters)
    )
    if not math.isfinite(postclip) or postclip > max_norm * 1.012:
        raise RuntimeError(f"Exp62 standard clipping violation: {postclip}")
    return {"preclip_norm": preclip, "postclip_norm": postclip}


def train(config: FormalConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup, set_seed

    protocol = verify_frozen_protocol()
    # Import here to avoid a module cycle while contract.py records the
    # trainer's frozen dataset-contract constants.
    from thesis_exp.exp62_summeval_routing_confirmation.contract import verify_source_lock

    source_lock = verify_source_lock(Path(config.model_name_or_path))
    if sha256_file(config.annotations) != EXPECTED_ANNOTATION_SHA256:
        raise RuntimeError("Exp62 annotation source mismatch")
    preflight_path = (
        OUTPUT_ROOT / "preflight" / f"seed_{config.seed}" / "real_model_no_update_preflight.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "EXP62_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS":
        raise RuntimeError(f"Exp62 seed {config.seed} preflight is absent or failed")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Exp62 training requires CUDA")

    set_seed(config.seed)
    train_rows = load_model_rows(config.annotations, "train")
    dev_rows = load_model_rows(config.annotations, "dev")
    contracts = {
        "train": rows_contract_sha256(train_rows),
        "dev": rows_contract_sha256(dev_rows),
    }
    if contracts != {
        "train": EXPECTED_TRAIN_CONTRACT_SHA256,
        "dev": EXPECTED_DEV_CONTRACT_SHA256,
    }:
        raise RuntimeError(f"Exp62 dataset contract mismatch: {contracts}")

    model, tokenizer, initial_head_contract = load_model_and_tokenizer(
        ModelConfig(config.model_name_or_path)
    )
    device = torch.device("cuda")
    model.to(device)
    train_loader = make_loader(train_rows, tokenizer, config, shuffle=True)
    dev_loader = make_loader(dev_rows, tokenizer, config, shuffle=False)
    if len(train_loader) % config.gradient_accumulation_steps:
        raise RuntimeError("Exp62 frozen accumulation window does not divide the train loader")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * config.warmup_ratio), total_updates
    )
    geometry_variant = config.variant in ("orthogonal_only", "parallel_only")
    route = "consensus_only" if config.variant == "direct_residual_blocked" else "routed_hmsa"
    named_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    } if geometry_variant else {}

    config.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    batch_digest = hashlib.sha256()
    step_audits: list[dict[str, Any]] = []
    running = {"hard": 0.0, "aux": 0.0, "total": 0.0}
    micro_steps = global_step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            targets = torch.tensor(
                [row["soft_target"] for row in metadata], dtype=torch.float32, device=device
            )
            batch_digest.update(
                ("\t".join(row["record_id"] for row in metadata) + "\n").encode("utf-8")
            )
            scale = 1.0 / config.gradient_accumulation_steps
            rng_before = capture_rng(torch, device)
            outputs = model(
                **inputs,
                labels=labels,
                soft_targets=targets,
                aux_route=route,
                route_loss_scale=scale,
            )
            losses = objective(outputs, labels, targets, route)
            (losses["optimization_loss"] * scale).backward()
            if geometry_variant:
                rng_after = capture_rng(torch, device)
                restore_rng(torch, device, rng_before)
                accumulate_residual_vjp(
                    model, inputs, labels, targets, residual_buffers, scale
                )
                restore_rng(torch, device, rng_after)
            running["hard"] += float(losses["reported_hard_loss"].detach().cpu())
            running["aux"] += float(losses["reported_aux_soft_ce"].detach().cpu())
            running["total"] += float(losses["optimization_loss"].detach().cpu())
            micro_steps += 1
            if step % config.gradient_accumulation_steps == 0:
                if geometry_variant:
                    audit = compose_geometry_step(
                        model,
                        residual_buffers,
                        variant=config.variant,
                        max_norm=config.max_gradient_norm,
                    )
                else:
                    audit = {"variant": config.variant, **_clip_standard(model, config.max_gradient_norm)}
                global_step += 1
                audit.update({"epoch": epoch, "global_step": global_step})
                step_audits.append(audit)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                for buffer in residual_buffers.values():
                    buffer.zero_()
                if global_step % 10 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[exp62:{config.variant}:seed{config.seed}] "
                        f"epoch={epoch} step={global_step}/{total_updates} "
                        f"loss={running['total']/micro_steps:.5f} elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )

    if global_step != total_updates:
        raise RuntimeError(f"Exp62 update count mismatch: {global_step} != {total_updates}")
    metrics, predictions = evaluate_dev(model, dev_loader, device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = config.checkpoint_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_path)
    checkpoint = {
        "status": "EXP62_FIXED_EPOCH10_DEV_COMPLETE",
        "variant": config.variant,
        "seed": config.seed,
        "epoch": config.epochs,
        "metrics": metrics,
        "state_dict_sha256": sha256_file(state_path),
        "head_contract": initial_head_contract,
        "dataset_contracts": contracts,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_lock_sha256": sha256_file(CONFIG_ROOT / "source_lock.json"),
        "model_manifest_sha256": source_lock["model"]["manifest_sha256"],
        "preflight_sha256": sha256_file(preflight_path),
        "train_batch_order_sha256": batch_digest.hexdigest(),
        "test_access_count": 0,
        "config": {
            **asdict(config),
            "annotations": str(config.annotations),
            "output_dir": str(config.output_dir),
            "checkpoint_dir": str(config.checkpoint_dir),
        },
    }
    _write_json(config.output_dir / "dev_summary.json", checkpoint)
    _write_json(config.output_dir / "dev_predictions.json", predictions)
    _write_json(config.output_dir / "gradient_step_audit.json", step_audits)
    _write_json(config.checkpoint_dir / "checkpoint.json", checkpoint)
    print(json.dumps({"status": checkpoint["status"], "metrics": metrics}, indent=2))
    return checkpoint


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
