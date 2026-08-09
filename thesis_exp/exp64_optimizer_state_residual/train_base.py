"""Train the five independent outcome-blind DRB base trajectories for Exp64.

This module is fail-closed while ``protocol_draft.json`` remains a draft.  It
does not evaluate the development or test splits and it saves complete states
at the three frozen epochs for the later counterfactual audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    EXPECTED_ANNOTATION_SHA256,
    sha256_file,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import (
    EXPECTED_MANIFEST_SHA256,
    load_model_rows,
    rows_contract_sha256,
)
from thesis_exp.exp62_summeval_routing_confirmation.model import (
    ModelConfig,
    load_model_and_tokenizer,
)
from thesis_exp.exp62_summeval_routing_confirmation.runtime import collate, objective
from thesis_exp.exp64_optimizer_state_residual import (
    ARTIFACT_ROOT,
    CONFIG_ROOT,
    OUTPUT_ROOT,
    SEEDS,
    STAGE_EPOCHS,
)


PROTOCOL_PATH = CONFIG_ROOT / "protocol_draft.json"
EXPECTED_TRAIN_CONTRACT_SHA256 = "9cf52e485e654037332b624cb49240081e77b5bc1f92e9ff65e72589ddeeb04b"


@dataclass(frozen=True)
class BaseConfig:
    annotations: Path
    model_name_or_path: str
    seed: int
    output_dir: Path
    artifact_dir: Path
    max_length: int = 256
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    micro_batch_size: int = 112
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0


def parse_args() -> BaseConfig:
    parser = argparse.ArgumentParser(description="Formal outcome-blind Exp64 DRB base run")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--artifact_dir", type=Path)
    args = parser.parse_args()
    return BaseConfig(
        annotations=args.annotations,
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
        output_dir=args.output_dir or OUTPUT_ROOT / "base" / f"seed_{args.seed}",
        artifact_dir=args.artifact_dir or ARTIFACT_ROOT / f"seed_{args.seed}",
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _ordered_rows(rows: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    import torch

    generator = torch.Generator().manual_seed(seed * 1000 + epoch)
    order = torch.randperm(len(rows), generator=generator).tolist()
    return [rows[index] for index in order]


def _capture_complete_rng(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _optimizer_parameter_names(model: Any, optimizer: Any) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise RuntimeError("Exp64 optimizer contains an unnamed parameter")
            group_names.append(names[id(parameter)])
        groups.append(group_names)
    return groups


def _verify_formal_protocol(config: BaseConfig) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checks = {
        "formal_status": protocol.get("status")
        == "EXP64_PROTOCOL_FROZEN_BEFORE_INDEPENDENT_OUTCOMES",
        "seeds": tuple(protocol.get("base_trajectory", {}).get("seeds", ())) == SEEDS,
        "stages": tuple(
            protocol.get("base_trajectory", {}).get("stage_checkpoints_after_epochs", ())
        )
        == STAGE_EPOCHS,
        "manifest": protocol.get("dataset", {}).get("split_manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "annotation": sha256_file(config.annotations) == EXPECTED_ANNOTATION_SHA256,
        "base_authorized": protocol.get("authorization", {}).get(
            "five_base_trajectory_gpu_training"
        )
        is True,
        "counterfactual_authorized": protocol.get("authorization", {}).get(
            "formal_counterfactual_updates"
        )
        is True,
        "test_forbidden": protocol.get("authorization", {}).get("test_evaluation") is False,
    }
    frozen = protocol.get("training", {})
    expected = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_gradient_norm": config.max_gradient_norm,
        "max_length": config.max_length,
    }
    checks["training_settings"] = all(frozen.get(key) == value for key, value in expected.items())
    optimizer = protocol.get("optimizer", {})
    checks["optimizer_settings"] = optimizer == {
        "name": "torch.optim.AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "reason": (
            "freeze the exact non-fused operation order validated by the "
            "finite-difference unit test"
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp64 formal base-training gate failed: {checks}")
    return protocol


def _make_loader(rows: list[dict[str, Any]], tokenizer: Any, config: BaseConfig) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        rows,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, config.max_length),
    )


def _save_stage(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: BaseConfig,
    epoch: int,
    global_step: int,
    device: Any,
    next_window_record_ids: list[str],
) -> dict[str, Any]:
    import torch
    from thesis_exp.exp64_optimizer_state_residual.source_lock import LOCK_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "EXP64_COMPLETE_DRB_STAGE_STATE_V1",
        "seed": config.seed,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_parameter_names": _optimizer_parameter_names(model, optimizer),
        "scheduler": scheduler.state_dict(),
        "rng": _capture_complete_rng(torch, device),
        "training_config": asdict(config),
        "next_epoch": epoch + 1,
        "next_window_record_ids": next_window_record_ids,
        "next_window_sha256": _hash_strings(next_window_record_ids),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_lock_sha256": sha256_file(LOCK_PATH),
        "test_access_count": 0,
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "epoch": epoch,
        "global_step": global_step,
        "next_window_rows": len(next_window_record_ids),
        "next_window_sha256": payload["next_window_sha256"],
    }


def train(config: BaseConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup, set_seed

    protocol = _verify_formal_protocol(config)
    from thesis_exp.exp64_optimizer_state_residual.source_lock import (
        LOCK_PATH,
        verify_source_lock,
    )

    source_lock = verify_source_lock(Path(config.model_name_or_path))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Exp64 base training requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(config.seed)
    train_rows = load_model_rows(config.annotations, "train")
    if rows_contract_sha256(train_rows) != EXPECTED_TRAIN_CONTRACT_SHA256:
        raise RuntimeError("Exp64 frozen train-row contract mismatch")
    model, tokenizer, head_contract = load_model_and_tokenizer(
        ModelConfig(config.model_name_or_path)
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
    )
    batches_per_epoch = math.ceil(len(train_rows) / config.micro_batch_size)
    if batches_per_epoch % config.gradient_accumulation_steps:
        raise RuntimeError("Exp64 accumulation does not divide the epoch")
    updates_per_epoch = batches_per_epoch // config.gradient_accumulation_steps
    total_updates = config.epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * config.warmup_ratio), total_updates
    )
    optimizer.zero_grad(set_to_none=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    checkpoints: list[dict[str, Any]] = []
    epoch_audit: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        epoch_rows = _ordered_rows(train_rows, config.seed, epoch)
        loader = _make_loader(epoch_rows, tokenizer, config)
        model.train()
        clip_events = 0
        maximum_preclip = 0.0
        for batch in loader:
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            targets = torch.tensor(
                [row["soft_target"] for row in metadata], dtype=torch.float32, device=device
            )
            outputs = model(
                **inputs,
                labels=labels,
                soft_targets=targets,
                aux_route="consensus_only",
                route_loss_scale=1.0,
            )
            losses = objective(outputs, labels, targets, "consensus_only")
            losses["optimization_loss"].backward()
            preclip = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                .detach()
                .cpu()
            )
            maximum_preclip = max(maximum_preclip, preclip)
            clip_events += int(preclip > config.max_gradient_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        epoch_ids = [str(row["record_id"]) for row in epoch_rows]
        epoch_audit.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "clip_events": clip_events,
                "maximum_preclip_norm": maximum_preclip,
                "learning_rate_after_epoch": scheduler.get_last_lr()[0],
                "record_order_sha256": _hash_strings(epoch_ids),
            }
        )
        _write_json(config.output_dir / "epoch_audit.json", epoch_audit)
        print(
            f"[exp64:base:seed{config.seed}] epoch={epoch}/{config.epochs} "
            f"step={global_step}/{total_updates} clip={clip_events}/{updates_per_epoch} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True,
        )
        if epoch in STAGE_EPOCHS:
            next_rows = _ordered_rows(train_rows, config.seed, epoch + 1)
            next_ids = [str(row["record_id"]) for row in next_rows[: config.micro_batch_size]]
            checkpoints.append(
                _save_stage(
                    config.artifact_dir / f"after_epoch_{epoch}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    device=device,
                    next_window_record_ids=next_ids,
                )
            )
            _write_json(config.output_dir / "checkpoints.json", checkpoints)

    summary = {
        "status": "EXP64_BASE_TRAJECTORY_COMPLETE_NO_DEV_OUTCOMES_READ",
        "seed": config.seed,
        "head_contract": head_contract,
        "train_rows": len(train_rows),
        "train_contract_sha256": rows_contract_sha256(train_rows),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "checkpoints": checkpoints,
        "epoch_audit": epoch_audit,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_status": protocol["status"],
        "source_lock_sha256": sha256_file(LOCK_PATH),
        "model_manifest_sha256": source_lock["model"]["manifest_sha256"],
        "dev_outcomes_read": 0,
        "test_access_count": 0,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
