"""Execute one authenticated, claimed, diagnostic-only Exp54 smoke step."""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.block_loss import (
    FixedRARCollator,
    blockwise_causal_loss,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    AuthenticatedSmokeContext,
    ReadOnceBytes,
    claim_smoke_invocation,
    read_regular_bytes_once,
    reserve_smoke_output_directory,
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
from thesis_exp.exp54_rar_sft.training_contract import (
    require_canonical_positions,
    token_ids_sha256,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    _require_runtime,
    _set_determinism,
    require_adapter_free_base_model,
    validate_model_directory_listing,
    validate_training_configuration,
    verify_model_snapshot,
)


_LORA_KEY_RE = re.compile(r"(?:^|\.)lora_[AB](?:\.|$)")
_DTYPE_TO_SAFETENSORS = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}


@dataclass(frozen=True)
class AuthenticatedPrivateSmokeData:
    rows: list[dict[str, Any]]
    manifest_path: Path
    manifest_sha256: str
    prompt_cache_path: Path
    prompt_cache_sha256: str


def _parse_jsonl_payload(
    material: ReadOnceBytes,
    *,
    label: str,
) -> list[dict[str, Any]]:
    try:
        text = material.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: invalid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label}: invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}: line {line_number} is not an object")
        rows.append(value)
    return rows


def _materialize_rows_from_read_once_payloads(
    prompt_material: ReadOnceBytes,
    manifest_material: ReadOnceBytes,
) -> list[dict[str, Any]]:
    prompt_rows = _parse_jsonl_payload(
        prompt_material,
        label="smoke prompt cache",
    )
    manifest_rows = _parse_jsonl_payload(
        manifest_material,
        label="smoke manifest",
    )
    prompt_by_id = {
        str(row["prompt_cache_id"]): row for row in prompt_rows
    }
    if len(prompt_by_id) != len(prompt_rows):
        raise ValueError("smoke prompt cache contains duplicate IDs")
    output: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for row in manifest_rows:
        event_id = str(row["base_event_id"])
        if event_id in seen_events:
            raise ValueError(f"duplicate smoke event: {event_id}")
        seen_events.add(event_id)
        prompt = prompt_by_id.get(str(row["prompt_cache_id"]))
        if prompt is None:
            raise ValueError(f"{event_id}: smoke prompt is missing")
        if str(prompt["record_id"]) != str(row["record_id"]):
            raise ValueError(f"{event_id}: smoke prompt record differs")
        if prompt["prompt_token_ids_sha256"] != row[
            "prompt_token_ids_sha256"
        ]:
            raise ValueError(f"{event_id}: smoke prompt hash differs")
        input_ids = (
            list(prompt["prompt_token_ids"])
            + list(row["target_token_ids"])
            + list(row["assistant_suffix_token_ids"])
        )
        if len(input_ids) != int(row["sequence_token_count"]):
            raise ValueError(f"{event_id}: smoke sequence length differs")
        if token_ids_sha256(input_ids) != row["full_token_ids_sha256"]:
            raise ValueError(f"{event_id}: smoke sequence hash differs")
        target_ids = list(row["target_token_ids"])
        local_score = require_canonical_positions(
            row["score_token_positions_in_target"],
            field=f"{event_id}: local score positions",
        )
        local_rationale = require_canonical_positions(
            row["rationale_token_positions_in_target"],
            field=f"{event_id}: local rationale positions",
        )
        if row["score_token_ids"] != [
            target_ids[position] for position in local_score
        ]:
            raise ValueError(f"{event_id}: smoke score token IDs differ")
        if row["rationale_token_ids"] != [
            target_ids[position] for position in local_rationale
        ]:
            raise ValueError(f"{event_id}: smoke rationale token IDs differ")
        score_positions = require_canonical_positions(
            row["score_token_positions"],
            field=f"{event_id}: score positions",
        )
        rationale_positions = require_canonical_positions(
            row["rationale_token_positions"],
            field=f"{event_id}: rationale positions",
        )
        if token_ids_sha256(score_positions) != row["score_mask_sha256"]:
            raise ValueError(f"{event_id}: smoke score mask hash differs")
        if token_ids_sha256(rationale_positions) != row[
            "rationale_mask_sha256"
        ]:
            raise ValueError(f"{event_id}: smoke rationale mask hash differs")
        valid_positions = set(range(len(input_ids)))
        if not set(score_positions).issubset(valid_positions):
            raise ValueError(f"{event_id}: smoke score mask is outside sequence")
        if not set(rationale_positions).issubset(valid_positions):
            raise ValueError(
                f"{event_id}: smoke rationale mask is outside sequence"
            )
        if set(score_positions) & set(rationale_positions):
            raise ValueError(f"{event_id}: smoke loss masks overlap")
        output.append(
            {
                "base_event_id": event_id,
                "input_ids": input_ids,
                "score_token_positions": score_positions,
                "rationale_token_positions": rationale_positions,
                "rationale_active": bool(row["rationale_active"]),
                "cutoff_len": int(row["cutoff_len"]),
                "full_token_ids_sha256": str(row["full_token_ids_sha256"]),
            }
        )
    return output


def _verify_frozen_smoke_package(
    *,
    context: AuthenticatedSmokeContext,
    private_smoke_dir: Path,
    arm: str,
) -> AuthenticatedPrivateSmokeData:
    reject_eval_path(private_smoke_dir)
    validate_smoke_plan(context.smoke_plan)
    smoke_lock = context.smoke_lock
    if smoke_lock.get("status") != (
        "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("smoke package is not frozen")
    if smoke_lock.get("smoke_plan_sha256") != context.smoke_plan_sha256:
        raise ValueError("authenticated smoke plan hash differs")
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
    for path in (manifest_path, prompt_cache_path):
        reject_eval_path(path)
    manifest_material = read_regular_bytes_once(
        manifest_path,
        max_bytes=64 * 1024 * 1024,
    )
    prompt_material = read_regular_bytes_once(
        prompt_cache_path,
        max_bytes=64 * 1024 * 1024,
    )
    private_hashes = smoke_lock.get("private_artifact_hashes", {})
    if manifest_material.sha256 != private_hashes.get(
        "manifests_by_arm", {}
    ).get(arm):
        raise ValueError("private smoke manifest differs from frozen hash")
    if prompt_material.sha256 != private_hashes.get("prompt_cache"):
        raise ValueError("private smoke prompt cache differs from frozen hash")
    rows = _materialize_rows_from_read_once_payloads(
        prompt_material,
        manifest_material,
    )
    return AuthenticatedPrivateSmokeData(
        rows=rows,
        manifest_path=manifest_path,
        manifest_sha256=manifest_material.sha256,
        prompt_cache_path=prompt_cache_path,
        prompt_cache_sha256=prompt_material.sha256,
    )


def _adapter_state_spec(
    state: Mapping[str, Any],
    *,
    expected_parameter_count: int,
) -> dict[str, dict[str, Any]]:
    if not state:
        raise RuntimeError("PEFT adapter state is empty")
    spec: dict[str, dict[str, Any]] = {}
    total = 0
    for key, tensor in state.items():
        if not isinstance(key, str) or not _LORA_KEY_RE.search(key):
            raise RuntimeError(f"non-LoRA key in expected adapter state: {key}")
        shape = tuple(int(value) for value in tensor.shape)
        dtype = _DTYPE_TO_SAFETENSORS.get(str(tensor.dtype))
        if dtype is None:
            raise RuntimeError(f"unsupported adapter dtype: {tensor.dtype}")
        numel = int(tensor.numel())
        if numel != math.prod(shape):
            raise RuntimeError(f"adapter tensor numel differs for {key}")
        total += numel
        spec[key] = {
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
        }
    if total != expected_parameter_count:
        raise RuntimeError("expected adapter-state parameter count differs")
    return dict(sorted(spec.items()))


def _adapter_artifact_hashes(
    adapter_dir: Path,
    *,
    expected_tensor_spec: dict[str, dict[str, Any]],
    expected_parameter_count: int,
    safe_open_fn: Callable[..., Any] | None = None,
) -> dict[str, str]:
    try:
        root_metadata = adapter_dir.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError("smoke adapter directory is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
        root_metadata.st_mode
    ):
        raise RuntimeError("smoke adapter root must be a real directory")
    symlinks: list[str] = []
    regular_files: list[Path] = []
    for path in adapter_dir.rglob("*"):
        relative = path.relative_to(adapter_dir).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file():
            regular_files.append(path)
    if symlinks:
        raise RuntimeError(f"smoke adapter output contains symlinks: {symlinks}")
    relative_names = {
        path.relative_to(adapter_dir).as_posix() for path in regular_files
    }
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(relative_names):
        raise RuntimeError("smoke output lacks required adapter artifacts")
    undeclared_weight_files = sorted(
        name
        for name in relative_names
        if (
            name.endswith(".safetensors")
            or name.endswith(".bin")
            or name.endswith(".index.json")
        )
        and name != "adapter_model.safetensors"
    )
    if undeclared_weight_files:
        raise RuntimeError(
            "smoke output contains undeclared weight artifacts: "
            f"{undeclared_weight_files}"
        )
    if safe_open_fn is None:
        from safetensors import safe_open

        safe_open_fn = safe_open
    tensor_path = adapter_dir / "adapter_model.safetensors"
    try:
        with safe_open_fn(
            str(tensor_path),
            framework="pt",
            device="cpu",
        ) as handle:
            actual_keys = set(handle.keys())
            expected_keys = set(expected_tensor_spec)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise RuntimeError(
                    "saved adapter tensor keys differ: "
                    f"missing={missing}, extra={extra}"
                )
            total = 0
            for key in sorted(actual_keys):
                if not _LORA_KEY_RE.search(key):
                    raise RuntimeError(f"saved non-LoRA tensor key: {key}")
                tensor_slice = handle.get_slice(key)
                shape = [int(value) for value in tensor_slice.get_shape()]
                dtype = str(tensor_slice.get_dtype())
                expected = expected_tensor_spec[key]
                if shape != expected["shape"]:
                    raise RuntimeError(f"saved adapter shape differs for {key}")
                if dtype != expected["dtype"]:
                    raise RuntimeError(f"saved adapter dtype differs for {key}")
                total += math.prod(shape)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("adapter_model.safetensors is invalid") from exc
    if total != expected_parameter_count:
        raise RuntimeError("saved adapter parameter count differs")
    return {
        path.relative_to(adapter_dir).as_posix(): (
            read_regular_bytes_once(
                path,
                max_bytes=512 * 1024 * 1024,
            ).sha256
        )
        for path in sorted(regular_files)
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
) -> None:
    """Run exactly one arm-specific optimizer step after a one-use claim."""
    if arm not in SMOKE_ARMS:
        raise ValueError(f"unsupported smoke arm: {arm}")
    context = verify_smoke_authorization(
        authorization_path=smoke_authorization_path,
        config_path=config_path,
        smoke_plan_path=smoke_plan_path,
        smoke_package_lock_path=smoke_package_lock_path,
        training_configuration_frozen_lock_path=(
            training_configuration_frozen_lock_path
        ),
        arm=arm,
    )
    config = context.config
    validate_training_configuration(config)
    private_data = _verify_frozen_smoke_package(
        context=context,
        private_smoke_dir=private_smoke_dir,
        arm=arm,
    )
    rows = private_data.rows
    if len(rows) != SMOKE_EVENTS_PER_ARM:
        raise ValueError("smoke row count differs")
    observed_active = sum(bool(row["rationale_active"]) for row in rows)
    expected_active = context.smoke_lock[
        "rationale_active_events_by_arm"
    ][arm]
    if observed_active != expected_active:
        raise ValueError("smoke rationale activity differs from frozen lock")
    claim = claim_smoke_invocation(context, arm=arm)
    output_dir = reserve_smoke_output_directory(claim)

    # Claim and output reservation precede model, CUDA, forward, and backward.
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
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
    )
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

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
    expected_adapter_spec = _adapter_state_spec(
        get_peft_model_state_dict(model),
        expected_parameter_count=lora["expected_trainable_parameters"],
    )
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

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_hashes = _adapter_artifact_hashes(
        adapter_dir,
        expected_tensor_spec=expected_adapter_spec,
        expected_parameter_count=lora["expected_trainable_parameters"],
    )
    result = {
        "status": "SMOKE_RUN_COMPLETED_DIAGNOSTIC_ONLY",
        "smoke_campaign_id": claim.campaign_id,
        "run_id": claim.run_id,
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
        "adapter_tensor_spec": expected_adapter_spec,
        "adapter_tensor_parameter_count": lora[
            "expected_trainable_parameters"
        ],
        "authorization_sha256": context.authorization_sha256,
        "configuration_sha256": context.config_sha256,
        "training_configuration_frozen_lock_sha256": (
            context.training_configuration_lock_sha256
        ),
        "smoke_plan_sha256": context.smoke_plan_sha256,
        "smoke_package_frozen_lock_sha256": context.smoke_lock_sha256,
        "smoke_manifest_sha256": private_data.manifest_sha256,
        "smoke_prompt_cache_sha256": private_data.prompt_cache_sha256,
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
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
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
    )


if __name__ == "__main__":
    main()
