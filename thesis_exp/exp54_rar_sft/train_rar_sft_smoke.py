"""Execute one authorized, diagnostic-only Exp54 smoke optimizer step.

The smoke path is intentionally separate from formal training. It consumes
only the frozen eight-event train subset and refuses before model loading,
CUDA initialization, output creation, forward, or backward unless a reviewed
external smoke authorization matches the root-owned trust anchor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    reject_eval_path,
)
from thesis_exp.exp54_rar_sft.block_loss import (
    FixedRARCollator,
    blockwise_causal_loss,
    load_materialized_rows,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    verify_smoke_authorization,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_PRIVATE_SMOKE_DIR,
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    SMOKE_ARMS,
    SMOKE_EVENTS_PER_ARM,
    SMOKE_GRADIENT_ACCUMULATION_STEPS,
    SMOKE_MICRO_BATCH_SIZE,
    SMOKE_OPTIMIZER_STEPS_PER_ARM,
    SMOKE_SOURCE_SEED,
    smoke_manifest_path,
    smoke_prompt_cache_path,
    validate_smoke_plan,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    _read_object,
    _require_runtime,
    _set_determinism,
    require_adapter_free_base_model,
    validate_model_directory_listing,
    validate_training_configuration,
    verify_model_snapshot,
)


def _verify_frozen_smoke_package(
    *,
    smoke_lock_path: Path,
    smoke_plan_path: Path,
    private_smoke_dir: Path,
    arm: str,
) -> tuple[dict[str, Any], Path, Path]:
    for path in (
        smoke_lock_path,
        smoke_plan_path,
        private_smoke_dir,
    ):
        reject_eval_path(path)
    smoke_lock = _read_object(smoke_lock_path)
    smoke_plan = _read_object(smoke_plan_path)
    validate_smoke_plan(smoke_plan)
    if smoke_lock.get("status") != (
        "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("smoke package is not frozen")
    if smoke_lock.get("smoke_plan_sha256") != file_sha256(smoke_plan_path):
        raise ValueError("smoke package does not bind the smoke plan")
    if (
        smoke_lock.get("smoke_subset_frozen") is not True
        or smoke_lock.get("trust_anchor_install_allowed")
        or smoke_lock.get("forward_backward_allowed")
        or smoke_lock.get("smoke_training_allowed")
        or smoke_lock.get("formal_training_allowed")
        or smoke_lock.get("dev_accessed")
        or smoke_lock.get("test_accessed")
        or smoke_lock.get("training_used")
    ):
        raise PermissionError("smoke package crosses its authorization boundary")

    manifest_path = smoke_manifest_path(private_smoke_dir, arm)
    prompt_cache_path = smoke_prompt_cache_path(private_smoke_dir)
    private_hashes = smoke_lock.get("private_artifact_hashes", {})
    if file_sha256(manifest_path) != private_hashes.get(
        "manifests_by_arm", {}
    ).get(arm):
        raise ValueError("private smoke manifest differs from frozen hash")
    if file_sha256(prompt_cache_path) != private_hashes.get(
        "prompt_cache"
    ):
        raise ValueError("private smoke prompt cache differs from frozen hash")
    return smoke_lock, manifest_path, prompt_cache_path


def _adapter_artifact_hashes(adapter_dir: Path) -> dict[str, str]:
    files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    relative_names = [path.relative_to(adapter_dir).as_posix() for path in files]
    forbidden = [
        name
        for name in relative_names
        if name == "model.safetensors"
        or name == "pytorch_model.bin"
        or name.startswith("model-")
        or name.startswith("pytorch_model-")
    ]
    if forbidden:
        raise RuntimeError(f"smoke output contains base weights: {forbidden}")
    if "adapter_config.json" not in relative_names:
        raise RuntimeError("smoke output lacks adapter_config.json")
    if "adapter_model.safetensors" not in relative_names:
        raise RuntimeError("smoke output lacks adapter_model.safetensors")
    return {
        path.relative_to(adapter_dir).as_posix(): file_sha256(path)
        for path in files
    }


def run_smoke_training(
    *,
    arm: str,
    config_path: Path,
    training_configuration_frozen_lock_path: Path,
    smoke_plan_path: Path,
    smoke_package_lock_path: Path,
    smoke_authorization_path: Path | None,
    private_smoke_dir: Path,
    output_dir: Path,
) -> None:
    """Run exactly one optimizer step after every external hard gate."""
    if arm not in SMOKE_ARMS:
        raise ValueError(f"unsupported smoke arm: {arm}")
    config = _read_object(config_path)
    validate_training_configuration(config)
    smoke_lock, manifest_path, prompt_cache_path = (
        _verify_frozen_smoke_package(
            smoke_lock_path=smoke_package_lock_path,
            smoke_plan_path=smoke_plan_path,
            private_smoke_dir=private_smoke_dir,
            arm=arm,
        )
    )
    verify_smoke_authorization(
        authorization_path=smoke_authorization_path,
        config_path=config_path,
        smoke_package_lock_path=smoke_package_lock_path,
        training_configuration_frozen_lock_path=(
            training_configuration_frozen_lock_path
        ),
        arm=arm,
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)

    # No model weights, CUDA state, output, forward, or backward before here.
    verify_model_snapshot(config)
    validate_model_directory_listing(
        Path(config["model"]["local_path"]),
        expected_regular_files=config["model"]["directory_regular_files"],
        expected_listing_sha256=config["model"][
            "directory_listing_sha256"
        ],
    )
    _require_runtime(config)
    _set_determinism(SMOKE_SOURCE_SEED, config)

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    rows = load_materialized_rows(prompt_cache_path, manifest_path)
    if len(rows) != SMOKE_EVENTS_PER_ARM:
        raise ValueError("smoke row count differs")
    observed_active = sum(bool(row["rationale_active"]) for row in rows)
    expected_active = smoke_lock["rationale_active_events_by_arm"][arm]
    if observed_active != expected_active:
        raise ValueError("smoke rationale activity differs from frozen lock")

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["local_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    if model.__class__.__name__ != config["model"]["architecture"]:
        raise RuntimeError("loaded smoke model architecture differs")
    require_adapter_free_base_model(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            task_type=lora["task_type"],
            target_modules=lora["target_modules"],
            modules_to_save=lora["modules_to_save"],
            inference_mode=lora["inference_mode"],
        ),
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in trainable) != lora[
        "expected_trainable_parameters"
    ]:
        raise RuntimeError("smoke LoRA parameter count differs")
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=optimization["learning_rate"],
        betas=tuple(optimization["betas"]),
        eps=optimization["epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optimization["warmup_steps"],
        num_training_steps=optimization["total_optimizer_steps"],
    )
    collator = FixedRARCollator(
        pad_token_id=config["model"]["pad_token_id"],
        cutoff_len=config["data"]["cutoff_len"],
    )
    loader = DataLoader(
        rows,
        batch_size=SMOKE_MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collator,
    )
    if len(loader) != SMOKE_GRADIENT_ACCUMULATION_STEPS:
        raise AssertionError("smoke micro-batch count differs")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_losses: list[float] = []
    score_losses: list[float] = []
    rationale_losses: list[float] = []
    score_token_count = 0
    rationale_token_count = 0
    micro_batches = 0
    for batch in loader:
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            losses = blockwise_causal_loss(
                output.logits,
                batch["input_ids"],
                batch["score_mask"],
                batch["rationale_mask"],
                batch["rationale_active"],
                rationale_weight=1.0,
            )
            scaled_loss = (
                losses["loss"] / SMOKE_GRADIENT_ACCUMULATION_STEPS
            )
        for name in ("loss", "score_loss", "rationale_loss_active_mean"):
            if not torch.isfinite(losses[name]):
                raise FloatingPointError(f"non-finite smoke {name}")
        scaled_loss.backward()
        total_losses.append(float(losses["loss"].detach().cpu()))
        score_losses.append(float(losses["score_loss"].detach().cpu()))
        rationale_losses.append(
            float(losses["rationale_loss_active_mean"].detach().cpu())
        )
        score_token_count += int(
            losses["score_supervised_token_count"].detach().cpu()
        )
        rationale_token_count += int(
            losses["rationale_supervised_token_count"].detach().cpu()
        )
        micro_batches += 1

    if micro_batches != SMOKE_GRADIENT_ACCUMULATION_STEPS:
        raise AssertionError("smoke accumulation group is incomplete")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=optimization["max_grad_norm"],
    )
    gradient_norm_value = float(gradient_norm.detach().cpu())
    if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0:
        raise FloatingPointError("smoke gradient norm is not finite and positive")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 1
    if optimizer_steps != SMOKE_OPTIMIZER_STEPS_PER_ARM:
        raise AssertionError("smoke optimizer-step budget differs")
    if score_token_count < SMOKE_EVENTS_PER_ARM:
        raise AssertionError("smoke score supervision is incomplete")
    if observed_active == 0 and rationale_token_count != 0:
        raise AssertionError("inactive smoke arm has rationale supervision")
    if observed_active > 0 and rationale_token_count <= 0:
        raise AssertionError("active smoke arm lacks rationale supervision")

    output_dir.mkdir(parents=True)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_hashes = _adapter_artifact_hashes(adapter_dir)
    result = {
        "status": "SMOKE_RUN_COMPLETED_DIAGNOSTIC_ONLY",
        "arm": arm,
        "seed": SMOKE_SOURCE_SEED,
        "events": SMOKE_EVENTS_PER_ARM,
        "micro_batches": micro_batches,
        "optimizer_steps": optimizer_steps,
        "rationale_active_events": observed_active,
        "score_supervised_token_count": score_token_count,
        "rationale_supervised_token_count": rationale_token_count,
        "mean_total_loss": sum(total_losses) / len(total_losses),
        "mean_score_loss": sum(score_losses) / len(score_losses),
        "mean_rationale_loss_active": (
            sum(rationale_losses) / len(rationale_losses)
        ),
        "preclip_gradient_norm": gradient_norm_value,
        "adapter_artifact_hashes": adapter_hashes,
        "configuration_sha256": file_sha256(config_path),
        "training_configuration_frozen_lock_sha256": file_sha256(
            training_configuration_frozen_lock_path
        ),
        "smoke_plan_sha256": file_sha256(smoke_plan_path),
        "smoke_package_frozen_lock_sha256": file_sha256(
            smoke_package_lock_path
        ),
        "smoke_manifest_sha256": file_sha256(manifest_path),
        "smoke_prompt_cache_sha256": file_sha256(prompt_cache_path),
        "diagnostic_only": True,
        "hyperparameter_selection_allowed": False,
        "checkpoint_selection_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "smoke_training_executed": True,
        "formal_training_used": False,
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=SMOKE_ARMS)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--training-configuration-frozen-lock",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    )
    parser.add_argument(
        "--smoke-plan",
        type=Path,
        default=DEFAULT_SMOKE_PLAN,
    )
    parser.add_argument(
        "--smoke-package-lock",
        type=Path,
        default=DEFAULT_SMOKE_FROZEN_LOCK,
    )
    parser.add_argument("--smoke-authorization", type=Path)
    parser.add_argument(
        "--private-smoke-dir",
        type=Path,
        default=DEFAULT_PRIVATE_SMOKE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_smoke_training(
        arm=args.arm,
        config_path=args.config,
        training_configuration_frozen_lock_path=(
            args.training_configuration_frozen_lock
        ),
        smoke_plan_path=args.smoke_plan,
        smoke_package_lock_path=args.smoke_package_lock,
        smoke_authorization_path=args.smoke_authorization,
        private_smoke_dir=args.private_smoke_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
