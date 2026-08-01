"""Run one frozen train-only Field-DPO/SORC-DPO formal preference job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.probe_sorc_dpo_capacity import (
    _batch_logps,
    _device_identity,
)
from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
    field_dpo_per_pair,
    weighted_objective,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_FORMAL_RUN_ROOT,
    DEFAULT_TRAINING_CONFIG,
    _require_regular_non_symlink,
    _set_adapter_trainability,
    read_json,
    verify_base_model_snapshot,
    verify_p3_qualification_lock,
)


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
MANIFEST_ROOT = RAR_ROOT / "preference_training_candidate/private"
OUTPUT_ROOT = RAR_ROOT / "preference_formal_runs"
QUALIFICATION_ROOT = RAR_ROOT / "preference_pairs/rationale_qualification"
QUALIFICATION_LOCK = QUALIFICATION_ROOT / "final_lock.json"
QUALIFICATION_CANDIDATE_LOCK = QUALIFICATION_ROOT / "candidate_lock.json"
PHYSICAL_BATCH_PAIRS = 4
MINIMUM_FREE_MEMORY_BYTES = 40 * 1024**3
ARMS = (
    "P1_FIELD_DPO",
    "P2_SORC_SCORE",
    "P3_JOINT_SORC",
    "P1_SYN_SEED42",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_path(arm: str) -> Path:
    return MANIFEST_ROOT / f"{arm.lower()}.jsonl"


def _checkpoint_for_seed(
    seed: int,
    *,
    checkpoint_lock_path: Path = DEFAULT_CHECKPOINT_LOCK,
    formal_runs_root: Path = DEFAULT_FORMAL_RUN_ROOT,
) -> tuple[Path, dict[str, str]]:
    _require_regular_non_symlink(
        checkpoint_lock_path,
        label="checkpoint selection lock",
    )
    lock = read_json(checkpoint_lock_path)
    if (
        lock.get("status") != "SFT_DEV_CHECKPOINT_SELECTION_FROZEN"
        or lock.get("test_accessed") is not False
    ):
        raise ValueError("SFT checkpoint selection lock differs")
    selected = [
        row
        for row in lock["checkpoint_bindings"]
        if str(row["arm"]).upper() == "R3" and int(row["seed"]) == seed
    ]
    if len(selected) != 1 or int(selected[0]["selected_epoch"]) != 3:
        raise ValueError(f"seed {seed}: R3 epoch-3 checkpoint differs")
    item = selected[0]
    relative = Path(str(item["checkpoint_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("checkpoint path escapes formal-runs root")
    resolved_root = formal_runs_root.resolve(strict=True)
    adapter_dir = (resolved_root / relative / "adapter").resolve(strict=True)
    if not adapter_dir.is_relative_to(resolved_root):
        raise PermissionError("checkpoint path escapes formal-runs root")
    hashes = {
        name: str(metadata["sha256"])
        for name, metadata in item["adapter_files"].items()
    }
    for name, expected in hashes.items():
        path = adapter_dir / name
        _require_regular_non_symlink(path, label=f"seed {seed} R3 {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"seed {seed}: R3 checkpoint differs: {name}")
    return adapter_dir, hashes


def load_formal_rows(
    *,
    arm: str,
    seed: int,
    config_path: Path = DEFAULT_TRAINING_CONFIG,
    frozen_lock_path: Path = DEFAULT_FROZEN_TRAINING_LOCK,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    for path in (config_path, frozen_lock_path):
        _require_regular_non_symlink(path, label=path.name)
    config = read_json(config_path)
    frozen = read_json(frozen_lock_path)
    if (
        frozen.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or frozen.get("loss_collator_manifests_and_config_frozen") is not True
        or frozen.get("dev_accessed") is not False
        or frozen.get("test_accessed") is not False
    ):
        raise PermissionError("preference training freeze differs")
    if sha256_file(config_path) != str(frozen["source_hashes"]["training_config"]):
        raise ValueError("training configuration hash differs")
    arm_config = config["arms"][arm]
    allowed_seeds = {int(value) for value in arm_config["seeds"]}
    if seed not in allowed_seeds:
        raise ValueError(f"{arm}: seed {seed} is outside the frozen protocol")
    manifest_path = _manifest_path(arm)
    _require_regular_non_symlink(manifest_path, label=f"{arm} manifest")
    expected_hash = str(frozen["private_manifest_hashes"][arm])
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError(f"{arm}: private manifest hash differs")
    rows = read_jsonl(manifest_path)
    expected_count = int(arm_config["pair_count"])
    if len(rows) != expected_count:
        raise ValueError(f"{arm}: manifest pair count differs")
    if any(int(row["cutoff_len"]) != 2048 for row in rows):
        raise ValueError(f"{arm}: cutoff differs")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"{arm}: duplicate pair ID")
    if arm == "P3_JOINT_SORC":
        _require_regular_non_symlink(
            QUALIFICATION_LOCK,
            label="rationale qualification lock",
        )
        verify_p3_qualification_lock(
            qualification_path=QUALIFICATION_LOCK,
            qualification_candidate_lock_path=QUALIFICATION_CANDIDATE_LOCK,
        )
    return rows, config, frozen


def _chunks(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _write_progress(
    *,
    output_dir: Path,
    arm: str,
    seed: int,
    phase: str,
    optimizer_step: int,
    optimizer_steps: int,
    completed_units: int,
    total_units: int,
    started_at: float,
    latest_loss: float | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started_at)
    fraction = completed_units / total_units if total_units else 0.0
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
    value: dict[str, Any] = {
        "schema_version": "exp54-sorc-dpo-formal-progress-v1",
        "status": "RUNNING",
        "arm": arm,
        "seed": seed,
        "phase": phase,
        "optimizer_step": optimizer_step,
        "optimizer_steps": optimizer_steps,
        "optimizer_step_percent": 100.0 * optimizer_step / optimizer_steps,
        "overall_percent": 100.0 * fraction,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": eta,
        "physical_micro_batch_pairs": PHYSICAL_BATCH_PAIRS,
        "dev_accessed": False,
        "test_accessed": False,
    }
    if latest_loss is not None:
        value["latest_optimizer_group_loss"] = latest_loss
    _atomic_json(output_dir / "progress.json", value)


def execute(
    *,
    arm: str,
    seed: int,
    cuda_device_uuid: str,
    output_dir: Path,
    expected_output_root: Path = OUTPUT_ROOT,
    learning_rate_override: float | None = None,
    protocol_overlay_path: Path | None = None,
    result_schema_version: str = "exp54-sorc-dpo-formal-result-v1",
) -> dict[str, Any]:
    rows, config, frozen = load_formal_rows(arm=arm, seed=seed)
    base_path, _base_config = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    adapter_dir, adapter_hashes = _checkpoint_for_seed(seed)
    expected_output = expected_output_root / arm.lower() / f"seed_{seed}"
    if output_dir.resolve(strict=False) != expected_output.resolve(strict=False):
        raise PermissionError("formal output directory differs from fixed path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    identity = _device_identity(torch, cuda_device_uuid)
    if int(identity["free_memory_bytes_before_load"]) < MINIMUM_FREE_MEMORY_BYTES:
        raise RuntimeError("A6000 is not sufficiently idle")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal preference training requires BF16")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

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

    arm_config = config["arms"][arm]
    group_size = int(arm_config["gradient_accumulation_pairs"])
    groups = _chunks(shuffled_rows, group_size)
    optimizer_steps = len(groups)
    if optimizer_steps != 27:
        raise ValueError(f"{arm}: expected exactly 27 optimizer steps")
    reference_chunks = _chunks(shuffled_rows, PHYSICAL_BATCH_PAIRS)
    policy_chunk_count = sum(
        math.ceil(len(group) / PHYSICAL_BATCH_PAIRS) for group in groups
    )
    total_units = len(reference_chunks) + policy_chunk_count
    started_at = time.time()
    _write_progress(
        output_dir=output_dir,
        arm=arm,
        seed=seed,
        phase="reference_logps",
        optimizer_step=0,
        optimizer_steps=optimizer_steps,
        completed_units=0,
        total_units=total_units,
        started_at=started_at,
    )

    reference_by_pair: dict[str, tuple[float, float]] = {}
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for chunk_index, chunk in enumerate(reference_chunks, start=1):
            batch = collator(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected = _batch_logps(model, batch)
            for row, chosen_value, rejected_value in zip(
                chunk,
                chosen.detach().float().cpu().tolist(),
                rejected.detach().float().cpu().tolist(),
            ):
                reference_by_pair[str(row["pair_id"])] = (
                    float(chosen_value),
                    float(rejected_value),
                )
            _write_progress(
                output_dir=output_dir,
                arm=arm,
                seed=seed,
                phase="reference_logps",
                optimizer_step=0,
                optimizer_steps=optimizer_steps,
                completed_units=chunk_index,
                total_units=total_units,
                started_at=started_at,
            )
    if len(reference_by_pair) != len(shuffled_rows):
        raise ValueError("reference-logp inventory differs")

    _set_adapter_trainability(model, "default")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimization = config["optimization"]
    learning_rate = float(optimization["learning_rate"])
    if learning_rate_override is not None:
        learning_rate = float(learning_rate_override)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("formal learning-rate override is invalid")
        if protocol_overlay_path is None:
            raise ValueError("formal learning-rate override lacks protocol overlay")
        _require_regular_non_symlink(
            protocol_overlay_path,
            label="formal protocol overlay",
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=tuple(optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    warmup_steps = math.ceil(
        optimizer_steps * float(optimization["warmup_ratio"])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )
    beta = float(config["loss"]["beta"])
    completed_policy_chunks = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for step_index, group in enumerate(groups, start=1):
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for chunk in _chunks(group, PHYSICAL_BATCH_PAIRS):
            batch = collator(chunk)
            reference_chosen = torch.tensor(
                [reference_by_pair[str(row["pair_id"])][0] for row in chunk],
                dtype=torch.float32,
                device=model.device,
            )
            reference_rejected = torch.tensor(
                [reference_by_pair[str(row["pair_id"])][1] for row in chunk],
                dtype=torch.float32,
                device=model.device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy_chosen, policy_rejected = _batch_logps(model, batch)
                per_pair = field_dpo_per_pair(
                    policy_chosen_logps=policy_chosen,
                    policy_rejected_logps=policy_rejected,
                    reference_chosen_logps=reference_chosen,
                    reference_rejected_logps=reference_rejected,
                    beta=beta,
                    offsets=batch["odpo_offset"].to(model.device),
                )["per_pair_loss"]
                objective = weighted_objective(
                    per_pair,
                    batch["objective_weight"].to(model.device),
                    normalization_pair_count=len(group),
                )["loss"]
            if not torch.isfinite(objective):
                raise FloatingPointError("formal preference loss is non-finite")
            objective.backward()
            group_loss += float(objective.detach().cpu())
            completed_policy_chunks += 1
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=float(optimization["max_grad_norm"]),
        )
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise FloatingPointError("formal gradient norm is invalid")
        optimizer.step()
        scheduler.step()
        losses.append(group_loss)
        grad_norms.append(float(gradient_norm.detach().cpu()))
        _write_progress(
            output_dir=output_dir,
            arm=arm,
            seed=seed,
            phase="preference_training",
            optimizer_step=step_index,
            optimizer_steps=optimizer_steps,
            completed_units=len(reference_chunks) + completed_policy_chunks,
            total_units=total_units,
            started_at=started_at,
            latest_loss=group_loss,
        )

    adapter_output = output_dir / "adapter"
    model.set_adapter("default")
    model.save_pretrained(
        str(adapter_output),
        selected_adapters=["default"],
        safe_serialization=True,
    )
    result = {
        "schema_version": result_schema_version,
        "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
        "arm": arm,
        "seed": seed,
        "pair_count": len(rows),
        "preference_epochs": 1,
        "optimizer_steps": optimizer_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "physical_micro_batch_pairs": PHYSICAL_BATCH_PAIRS,
        "gradient_accumulation_pairs": group_size,
        "final_group_pairs": len(groups[-1]),
        "losses": losses,
        "gradient_norms_before_clip": grad_norms,
        "elapsed_seconds": time.time() - started_at,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_identity": identity,
        "manifest_sha256": sha256_file(_manifest_path(arm)),
        "training_config_sha256": sha256_file(DEFAULT_TRAINING_CONFIG),
        "frozen_training_lock_sha256": sha256_file(
            DEFAULT_FROZEN_TRAINING_LOCK
        ),
        "cold_start_adapter_hashes": adapter_hashes,
        "output_adapter_config_sha256": sha256_file(
            adapter_output / "adapter_config.json"
        ),
        "output_adapter_model_sha256": sha256_file(
            adapter_output / "adapter_model.safetensors"
        ),
        "dev_accessed": False,
        "test_accessed": False,
    }
    if protocol_overlay_path is not None:
        result["protocol_overlay_sha256"] = sha256_file(
            protocol_overlay_path
        )
    _atomic_json(output_dir / "result.json", result)
    completed = {
        "schema_version": "exp54-sorc-dpo-formal-progress-v1",
        "status": "COMPLETE",
        "arm": arm,
        "seed": seed,
        "phase": "complete",
        "optimizer_step": optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "optimizer_step_percent": 100.0,
        "overall_percent": 100.0,
        "elapsed_seconds": result["elapsed_seconds"],
        "estimated_remaining_seconds": 0.0,
        "physical_micro_batch_pairs": PHYSICAL_BATCH_PAIRS,
        "dev_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "progress.json", completed)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--cuda-device-uuid")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, config, frozen = load_formal_rows(arm=args.arm, seed=args.seed)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SORC_DPO_FORMAL_CPU_VALIDATION_PASS",
                    "arm": args.arm,
                    "seed": args.seed,
                    "pairs": len(rows),
                    "optimizer_steps": int(
                        config["arms"][args.arm][
                            "optimizer_steps_per_physical_pass"
                        ]
                    ),
                    "physical_micro_batch_pairs": PHYSICAL_BATCH_PAIRS,
                    "manifest_sha256": frozen["private_manifest_hashes"][
                        args.arm
                    ],
                    "model_loaded": False,
                    "cuda_initialized": False,
                    "dev_accessed": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not args.cuda_device_uuid:
        raise ValueError("--cuda-device-uuid is required for execution")
    output_dir = OUTPUT_ROOT / args.arm.lower() / f"seed_{args.seed}"
    result = execute(
        arm=args.arm,
        seed=args.seed,
        cuda_device_uuid=args.cuda_device_uuid,
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
