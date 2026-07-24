"""Guarded single-GPU Exp54 LoRA trainer with exact two-block loss.

The checked-in configuration is deliberately not an authorization to train.
This entry point refuses to load model weights unless a later, separately
reviewed authorization lock binds the frozen manifests, configuration, and
training sources.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    reject_eval_path,
)
from thesis_exp.exp54_rar_sft.block_loss import (
    FixedRARCollator,
    blockwise_causal_loss,
    load_materialized_rows,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
DEFAULT_FROZEN_MANIFEST_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
DEFAULT_TRAINING_CONFIG_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_candidate_lock.json"
)
DEFAULT_DATA_DIR = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data"
)
ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
ROWS_PER_LOGICAL_EPOCH = 2_654
LOGICAL_EPOCHS = 3
TRAINING_SOURCE_PATHS = {
    "training_entrypoint": Path(__file__),
    "block_loss_and_collator": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
    ),
    "training_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/training_contract.py"
    ),
    "inference_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/inference_contract.py"
    ),
    "launcher": REPO_ROOT / "thesis_exp/scripts/run_exp54_rar_sft.sh",
}


def _read_object(path: Path) -> dict[str, Any]:
    reject_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def optimizer_group_plan(
    *,
    rows_per_epoch: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> list[int]:
    """Return actual micro-batch divisors for one logical epoch."""
    if min(rows_per_epoch, micro_batch_size, gradient_accumulation_steps) < 1:
        raise ValueError("step-plan inputs must be positive")
    micro_batches = math.ceil(rows_per_epoch / micro_batch_size)
    full, remainder = divmod(micro_batches, gradient_accumulation_steps)
    plan = [gradient_accumulation_steps] * full
    if remainder:
        plan.append(remainder)
    return plan


def validate_training_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Hard-fail changes to the reviewed scientific and execution contract."""
    exact = {
        ("status",): "EXP54_TRAINING_CONFIGURATION_CANDIDATE_NOT_AUTHORIZED",
        ("schema_version",): "exp54-training-configuration-v1",
        ("experiment", "arms"): list(ARMS),
        ("experiment", "seeds"): list(SEEDS),
        ("experiment", "independent_base_reload_per_run"): True,
        ("experiment", "historical_adapter_allowed"): False,
        ("experiment", "dev_or_test_access_during_training"): False,
        ("model", "model_id"): "Qwen/Qwen3-4B-Instruct-2507",
        ("model", "immutable_revision"): (
            "cdbee75f17c01a7cc42f958dc650907174af0554"
        ),
        ("model", "architecture"): "Qwen3ForCausalLM",
        ("model", "local_files_only"): True,
        ("model", "trust_remote_code"): False,
        ("model", "torch_dtype"): "bfloat16",
        ("model", "use_cache"): False,
        ("model", "pad_token_id"): 151643,
        ("model", "eos_token_id"): 151645,
        ("lora", "rank"): 16,
        ("lora", "alpha"): 32,
        ("lora", "dropout"): 0.05,
        ("lora", "bias"): "none",
        ("lora", "task_type"): "CAUSAL_LM",
        ("lora", "target_modules"): [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        ("lora", "expected_trainable_parameters"): 33_030_144,
        ("lora", "expected_base_parameters"): 4_022_468_096,
        ("data", "rows_per_logical_epoch"): ROWS_PER_LOGICAL_EPOCH,
        ("data", "logical_epochs_in_manifest"): LOGICAL_EPOCHS,
        ("data", "events_per_manifest"): (
            ROWS_PER_LOGICAL_EPOCH * LOGICAL_EPOCHS
        ),
        ("data", "physical_manifest_passes"): 1,
        ("data", "shuffle"): False,
        ("data", "dataloader_num_workers"): 0,
        ("data", "drop_last"): False,
        ("data", "packing"): False,
        ("data", "cutoff_len"): 2_048,
        ("data", "padding"): "fixed_max_length",
        ("data", "truncation"): False,
        ("loss", "implementation"): "blockwise_causal_loss",
        ("loss", "score_block_weight"): 1.0,
        ("loss", "rationale_block_weight_when_active"): 1.0,
        ("loss", "inactive_rationale_block_weight"): 0.0,
        ("loss", "per_block_token_reduction"): "mean_per_sample",
        ("loss", "batch_reduction"): "mean_over_samples",
        ("optimization", "micro_batch_size_per_device"): 2,
        ("optimization", "gradient_accumulation_steps"): 4,
        ("optimization", "gradient_accumulation_boundary"): (
            "flush_and_reset_at_each_logical_epoch"
        ),
        ("optimization", "remainder_group_divisor"): (
            "actual_micro_batches_in_group"
        ),
        ("optimization", "optimizer"): "AdamW",
        ("optimization", "learning_rate"): 0.0001,
        ("optimization", "betas"): [0.9, 0.999],
        ("optimization", "epsilon"): 1e-8,
        ("optimization", "weight_decay"): 0.0,
        ("optimization", "max_grad_norm"): 1.0,
        ("optimization", "scheduler"): "cosine",
        ("optimization", "warmup_steps"): 50,
        ("optimization", "optimizer_steps_per_logical_epoch"): 332,
        ("optimization", "total_optimizer_steps"): 996,
        ("precision_and_memory", "autocast_dtype"): "bfloat16",
        ("precision_and_memory", "gradient_checkpointing"): True,
        (
            "precision_and_memory",
            "gradient_checkpointing_use_reentrant",
        ): False,
        ("precision_and_memory", "tf32"): False,
        (
            "precision_and_memory",
            "allow_fp16_reduced_precision_reduction",
        ): False,
        ("determinism", "same_seed_for_data_lora_and_runtime"): True,
        ("determinism", "python_hash_seed_equals_run_seed"): True,
        ("determinism", "cublas_workspace_config"): ":4096:8",
        ("determinism", "torch_deterministic_algorithms"): True,
        ("determinism", "torch_deterministic_warn_only"): False,
        ("determinism", "cudnn_deterministic"): True,
        ("determinism", "cudnn_benchmark"): False,
        ("checkpointing", "save_after_each_logical_epoch"): True,
        ("checkpointing", "resume_allowed"): False,
        ("checkpointing", "overwrite_output_dir"): False,
        ("checkpointing", "evaluation_during_training"): False,
        ("generation", "do_sample"): False,
        ("generation", "num_beams"): 1,
        ("generation", "max_new_tokens"): 256,
        ("generation", "enable_thinking"): False,
        ("authorization", "manifest_frozen_required"): True,
        ("authorization", "configuration_review_required"): True,
        ("authorization", "smoke_training_allowed"): False,
        ("authorization", "formal_training_allowed"): False,
        ("authorization", "dev_accessed"): False,
        ("authorization", "test_accessed"): False,
        ("authorization", "training_used"): False,
    }
    for path, expected in exact.items():
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"training config is missing {'.'.join(path)}")
            value = value[key]
        if value != expected:
            raise ValueError(f"training config differs at {'.'.join(path)}")

    plan = optimizer_group_plan(
        rows_per_epoch=ROWS_PER_LOGICAL_EPOCH,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
    )
    if len(plan) != 332 or plan[-1] != 3 or plan.count(4) != 331:
        raise AssertionError("logical-epoch optimizer plan differs")
    return {
        "micro_batches_per_logical_epoch": sum(plan),
        "optimizer_steps_per_logical_epoch": len(plan),
        "full_accumulation_groups_per_logical_epoch": plan.count(4),
        "remainder_micro_batches_per_logical_epoch": plan[-1],
        "total_optimizer_steps": len(plan) * LOGICAL_EPOCHS,
    }


def verify_model_snapshot(config: dict[str, Any]) -> dict[str, str]:
    model = config["model"]
    model_path = Path(model["local_path"])
    if not model_path.is_absolute():
        raise ValueError("model path must be absolute")
    actual: dict[str, str] = {}
    for name, expected in model["files"].items():
        path = model_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"model file size differs: {name}")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"model file hash differs: {name}")
        actual[name] = digest
    return actual


def verify_frozen_manifest(
    *,
    frozen_lock_path: Path,
    data_dir: Path,
    arm: str,
    seed: int,
) -> tuple[dict[str, Any], Path, Path]:
    frozen = _read_object(frozen_lock_path)
    if (
        frozen.get("status")
        != "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
        or frozen.get("manifest_frozen") is not True
        or frozen.get("smoke_training_allowed")
        or frozen.get("formal_training_allowed")
        or frozen.get("dev_accessed")
        or frozen.get("test_accessed")
        or frozen.get("training_used")
    ):
        raise PermissionError("manifest freeze lock is invalid or over-authorized")
    manifest = data_dir / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
    prompt_cache = data_dir / "shared_prompt_cache.jsonl"
    expected = frozen["private_artifact_hashes"]
    if file_sha256(manifest) != expected["manifests_by_seed"][f"seed{seed}"][arm]:
        raise ValueError("manifest differs from frozen hash")
    if file_sha256(prompt_cache) != expected["shared_prompt_cache"]:
        raise ValueError("prompt cache differs from frozen hash")
    return frozen, manifest, prompt_cache


def require_training_authorization(
    *,
    authorization_lock_path: Path | None,
    config_path: Path,
    frozen_lock_path: Path,
    training_config_lock_path: Path,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    """Require a future external lock; the present repository has none."""
    if authorization_lock_path is None:
        raise PermissionError(
            "training is not authorized: a reviewed authorization lock is required"
        )
    authorization = _read_object(authorization_lock_path)
    if authorization.get("status") != "EXP54_FORMAL_TRAINING_AUTHORIZED":
        raise PermissionError("authorization lock does not permit formal training")
    training_config_lock = _read_object(training_config_lock_path)
    if (
        training_config_lock.get("status")
        != "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED"
        or training_config_lock.get("smoke_training_allowed")
        or training_config_lock.get("formal_training_allowed")
        or training_config_lock.get("dev_accessed")
        or training_config_lock.get("test_accessed")
        or training_config_lock.get("training_used")
    ):
        raise PermissionError("training-configuration audit lock is invalid")
    if authorization.get("config_sha256") != file_sha256(config_path):
        raise PermissionError("authorization does not bind this training config")
    if training_config_lock.get("configuration_sha256") != file_sha256(
        config_path
    ):
        raise PermissionError("configuration audit does not bind this config")
    if authorization.get("frozen_manifest_lock_sha256") != file_sha256(
        frozen_lock_path
    ):
        raise PermissionError("authorization does not bind this manifest freeze")
    if training_config_lock.get("manifest_frozen_lock_sha256") != file_sha256(
        frozen_lock_path
    ):
        raise PermissionError(
            "configuration audit does not bind this manifest freeze"
        )
    if authorization.get("training_configuration_lock_sha256") != file_sha256(
        training_config_lock_path
    ):
        raise PermissionError(
            "authorization does not bind this training-configuration audit"
        )
    actual_sources = {
        name: file_sha256(path) for name, path in TRAINING_SOURCE_PATHS.items()
    }
    if training_config_lock.get("training_source_hashes") != actual_sources:
        raise PermissionError(
            "configuration audit does not bind current training sources"
        )
    if authorization.get("training_source_hashes") != actual_sources:
        raise PermissionError("authorization does not bind current training sources")
    if arm not in authorization.get("allowed_arms", []):
        raise PermissionError(f"authorization does not permit arm {arm}")
    if seed not in authorization.get("allowed_seeds", []):
        raise PermissionError(f"authorization does not permit seed {seed}")
    if (
        authorization.get("dev_accessed")
        or authorization.get("test_accessed")
        or authorization.get("training_used")
    ):
        raise PermissionError("authorization lock violates sealed-data state")
    return authorization


def _set_determinism(seed: int, config: dict[str, Any]) -> None:
    import numpy
    import torch

    required_hash_seed = str(seed)
    if os.environ.get("PYTHONHASHSEED") != required_hash_seed:
        raise RuntimeError(f"PYTHONHASHSEED must equal {required_hash_seed}")
    workspace = config["determinism"]["cublas_workspace_config"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != workspace:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must equal {workspace}")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def _runtime_versions() -> dict[str, str]:
    import platform
    import torch

    packages = (
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "safetensors",
        "tokenizers",
        "numpy",
        "sentencepiece",
        "llamafactory",
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _require_runtime(config: dict[str, Any]) -> None:
    import torch

    expected = config["runtime"]
    actual = _runtime_versions()
    for name, value in actual.items():
        if value != expected[name]:
            raise RuntimeError(f"runtime version differs for {name}")
    if torch.version.cuda != expected["cuda_runtime"]:
        raise RuntimeError("CUDA runtime version differs")
    if torch.backends.cudnn.version() != expected["cudnn"]:
        raise RuntimeError("cuDNN version differs")
    drivers = set(
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    )
    if drivers != {expected["driver"]}:
        raise RuntimeError("NVIDIA driver version differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")
    if torch.cuda.get_device_name(0) != expected["required_gpu_name"]:
        raise RuntimeError("visible GPU model differs")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible GPU does not support bfloat16")


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    logical_epoch_number: int,
    global_step: int,
    arm: str,
    seed: int,
    config_path: Path,
    frozen_lock_path: Path,
    manifest_path: Path,
) -> None:
    import torch

    checkpoint = output_dir / f"checkpoint-logical-epoch-{logical_epoch_number}"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint / "adapter", safe_serialization=True)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
        checkpoint / "rng_state.pt",
    )
    state = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": arm,
        "seed": seed,
        "logical_epoch_number": logical_epoch_number,
        "global_optimizer_step": global_step,
        "config_sha256": file_sha256(config_path),
        "frozen_manifest_lock_sha256": file_sha256(frozen_lock_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dev_accessed": False,
        "test_accessed": False,
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_training(
    *,
    config_path: Path,
    frozen_lock_path: Path,
    training_config_lock_path: Path,
    authorization_lock_path: Path | None,
    data_dir: Path,
    output_dir: Path,
    arm: str,
    seed: int,
) -> None:
    """Run one independently initialized arm/seed after all hard gates."""
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unsupported arm or seed")
    config = _read_object(config_path)
    validate_training_configuration(config)
    _frozen, manifest_path, prompt_cache_path = verify_frozen_manifest(
        frozen_lock_path=frozen_lock_path,
        data_dir=data_dir,
        arm=arm,
        seed=seed,
    )
    require_training_authorization(
        authorization_lock_path=authorization_lock_path,
        config_path=config_path,
        frozen_lock_path=frozen_lock_path,
        training_config_lock_path=training_config_lock_path,
        arm=arm,
        seed=seed,
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)

    # No model weights or CUDA state are touched before authorization succeeds.
    verify_model_snapshot(config)
    _require_runtime(config)
    _set_determinism(seed, config)

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    rows = load_materialized_rows(prompt_cache_path, manifest_path)
    expected_events = ROWS_PER_LOGICAL_EPOCH * LOGICAL_EPOCHS
    if len(rows) != expected_events:
        raise ValueError("frozen manifest event count differs")
    output_dir.mkdir(parents=True)

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["local_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    if model.__class__.__name__ != config["model"]["architecture"]:
        raise RuntimeError("loaded model architecture differs")
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA produced no trainable parameters")
    if sum(parameter.numel() for parameter in trainable) != lora[
        "expected_trainable_parameters"
    ]:
        raise RuntimeError("LoRA trainable-parameter count differs")
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
    group_plan = optimizer_group_plan(
        rows_per_epoch=ROWS_PER_LOGICAL_EPOCH,
        micro_batch_size=optimization["micro_batch_size_per_device"],
        gradient_accumulation_steps=optimization[
            "gradient_accumulation_steps"
        ],
    )
    global_step = 0
    model.train()
    for epoch_index in range(LOGICAL_EPOCHS):
        epoch_rows = rows[
            epoch_index * ROWS_PER_LOGICAL_EPOCH :
            (epoch_index + 1) * ROWS_PER_LOGICAL_EPOCH
        ]
        loader = DataLoader(
            epoch_rows,
            batch_size=optimization["micro_batch_size_per_device"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collator,
        )
        optimizer.zero_grad(set_to_none=True)
        group_index = 0
        within_group = 0
        group_size = group_plan[group_index]
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
                scaled_loss = losses["loss"] / group_size
            scaled_loss.backward()
            within_group += 1
            if within_group == group_size:
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=optimization["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                group_index += 1
                within_group = 0
                if group_index < len(group_plan):
                    group_size = group_plan[group_index]
        if within_group or group_index != len(group_plan):
            raise AssertionError("logical epoch did not flush exactly")
        expected_step = (epoch_index + 1) * len(group_plan)
        if global_step != expected_step:
            raise AssertionError("optimizer step count differs")
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            logical_epoch_number=epoch_index + 1,
            global_step=global_step,
            arm=arm,
            seed=seed,
            config_path=config_path,
            frozen_lock_path=frozen_lock_path,
            manifest_path=manifest_path,
        )
    if global_step != optimization["total_optimizer_steps"]:
        raise AssertionError("formal training ended at an unexpected step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--frozen-manifest-lock",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST_LOCK,
    )
    parser.add_argument(
        "--training-config-lock",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_LOCK,
    )
    parser.add_argument("--authorization-lock", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        config_path=args.config,
        frozen_lock_path=args.frozen_manifest_lock,
        training_config_lock_path=args.training_config_lock,
        authorization_lock_path=args.authorization_lock,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arm=args.arm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
