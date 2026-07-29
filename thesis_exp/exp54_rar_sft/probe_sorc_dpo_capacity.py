"""Probe one physical pair batch on the fixed train-only P2 smoke group."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
    field_dpo_per_pair,
    field_mean_logps,
    weighted_objective,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_SMOKE_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    SMOKE_ROOT,
    _seed42_checkpoint,
    _set_adapter_trainability,
    load_and_validate_smoke_rows,
    read_json,
    verify_base_model_snapshot,
)


ALLOWED_BATCHES = {1, 2, 4, 8, 16}
OUTPUT_ROOT = SMOKE_ROOT / "private/capacity_probe"
MAX_RESERVED_BYTES = 46 * 1024**3


def _batch_logps(model: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    import torch

    batch_size = int(batch["chosen_input_ids"].shape[0])
    if batch_size < 1:
        raise ValueError("capacity batch cannot be empty")
    input_ids = torch.cat(
        [batch["chosen_input_ids"], batch["rejected_input_ids"]],
        dim=0,
    ).to(model.device)
    attention_mask = torch.cat(
        [
            batch["chosen_attention_mask"],
            batch["rejected_attention_mask"],
        ],
        dim=0,
    ).to(model.device)
    field_mask = torch.cat(
        [batch["chosen_field_mask"], batch["rejected_field_mask"]],
        dim=0,
    ).to(model.device)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    logps = field_mean_logps(logits, input_ids, field_mask)[
        "per_sequence_logp"
    ]
    if logps.shape != (2 * batch_size,):
        raise ValueError("capacity log-probability shape differs")
    return logps[:batch_size], logps[batch_size:]


def _device_identity(torch: Any, expected_uuid: str) -> dict[str, Any]:
    cuda = torch.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("capacity probe requires one visible CUDA device")
    name = str(cuda.get_device_name(0))
    if name != "NVIDIA RTX A6000":
        raise RuntimeError("capacity probe requires NVIDIA RTX A6000")
    uuids = cuda._raw_device_uuid_nvml()
    physical_index = int(cuda._get_nvml_device_index(0))
    if uuids is None or physical_index >= len(uuids):
        raise RuntimeError("capacity probe cannot resolve CUDA UUID")
    uuid = str(uuids[physical_index])
    if uuid != expected_uuid:
        raise RuntimeError("capacity probe CUDA UUID differs")
    free_bytes, total_bytes = cuda.mem_get_info(0)
    if int(free_bytes) < 40 * 1024**3:
        raise RuntimeError("capacity probe A6000 is not sufficiently idle")
    return {
        "name": name,
        "uuid": uuid,
        "free_memory_bytes_before_load": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def run(*, physical_batch: int, cuda_device_uuid: str) -> dict[str, Any]:
    if physical_batch not in ALLOWED_BATCHES:
        raise ValueError("physical batch is outside the preregistered ladder")
    rows, _lock, _plan = load_and_validate_smoke_rows(
        arm="P2_SORC_SCORE",
        smoke_lock_path=DEFAULT_SMOKE_LOCK,
        smoke_plan_path=DEFAULT_SMOKE_PLAN,
    )
    if len(rows) != 32:
        raise ValueError("capacity probe requires the fixed 32-pair P2 group")
    base_path, _base = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    adapter_dir, _hashes = _seed42_checkpoint(DEFAULT_CHECKPOINT_LOCK)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    identity = _device_identity(torch, cuda_device_uuid)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("capacity probe requires BF16")
    config = read_json(DEFAULT_TRAINING_CONFIG)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=True,
    )
    model.load_adapter(
        str(adapter_dir),
        adapter_name="reference",
        is_trainable=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    collator = SORCDPOPairCollator(pad_token_id=151643, cutoff_len=2048)
    chunks = [
        rows[index : index + physical_batch]
        for index in range(0, len(rows), physical_batch)
    ]

    reference_logps = []
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            batch = collator(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected = _batch_logps(model, batch)
            reference_logps.append(
                (chosen.detach().cpu(), rejected.detach().cpu())
            )

    _set_adapter_trainability(model, "default")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimization"]["learning_rate"]),
        betas=tuple(config["optimization"]["betas"]),
        eps=float(config["optimization"]["epsilon"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    total_loss = 0.0
    beta = float(config["loss"]["beta"])
    for index, chunk in enumerate(chunks):
        batch = collator(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy_chosen, policy_rejected = _batch_logps(model, batch)
            per_pair = field_dpo_per_pair(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                reference_chosen_logps=reference_logps[index][0].to(
                    model.device
                ),
                reference_rejected_logps=reference_logps[index][1].to(
                    model.device
                ),
                beta=beta,
                offsets=batch["odpo_offset"].to(model.device),
            )["per_pair_loss"]
            objective = weighted_objective(
                per_pair,
                batch["objective_weight"].to(model.device),
                normalization_pair_count=32,
            )["loss"]
        if not torch.isfinite(objective):
            raise FloatingPointError("capacity-probe loss is non-finite")
        objective.backward()
        total_loss += float(objective.detach().cpu())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=float(config["optimization"]["max_grad_norm"]),
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise FloatingPointError("capacity-probe gradient is invalid")
    optimizer.step()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    result = {
        "schema_version": "exp54-sorc-dpo-capacity-probe-v1",
        "status": "CAPACITY_PROBE_COMPLETE",
        "source_arm": "P2_SORC_SCORE",
        "pair_count": 32,
        "physical_micro_batch_pairs": physical_batch,
        "micro_batches": len(chunks),
        "final_micro_batch_pairs": len(chunks[-1]),
        "normalization_pair_count": 32,
        "optimizer_steps": 1,
        "loss": total_loss,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "within_46_gib_reserved_limit": (
            peak_reserved <= MAX_RESERVED_BYTES
        ),
        "device_identity": identity,
        "semantic_output_inspected": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for value in (
        result["loss"],
        result["gradient_norm_before_clip"],
    ):
        if not math.isfinite(float(value)):
            raise FloatingPointError("capacity-probe result is non-finite")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical-batch",
        type=int,
        choices=sorted(ALLOWED_BATCHES),
        required=True,
    )
    parser.add_argument("--cuda-device-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(
        physical_batch=args.physical_batch,
        cuda_device_uuid=args.cuda_device_uuid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
