"""Run one user-authorized, train-only mechanism-control smoke step.

The smoke checks a single optimizer step only. It never reads dev/test and it
does not authorize the nine full mechanism-control training runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.block_loss import (
    FixedRARCollator,
    load_materialized_rows,
)
from thesis_exp.exp54_rar_sft.mechanism_control_losses import (
    FullSequenceDPOPairCollator,
    sequence_sum_logps,
    token_average_causal_loss,
)
from thesis_exp.exp54_rar_sft.probe_sorc_dpo_capacity import _device_identity
from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
    field_dpo_per_pair,
    field_mean_logps,
    weighted_objective,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    _seed42_checkpoint,
    _set_adapter_trainability,
    read_json,
    verify_base_model_snapshot,
)


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_smoke_plan_v1.json"
)
DEFAULT_MECHANISM_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_controls_candidate_v1.json"
)
DEFAULT_SFT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json"
)
DEFAULT_PREFERENCE_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_training_candidate_v1.json"
)
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "mechanism_control_smoke"
PHYSICAL_PREFERENCE_BATCH = 4
PREFERENCE_PAIR_COUNT = 32
SFT_EVENT_COUNT = 8


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("status") != "USER_APPROVED_TRAIN_ONLY_GPU_SMOKE"
        or plan.get("arms") != list(ARMS)
        or int(plan.get("seed", -1)) != 42
        or int(plan.get("optimizer_steps_per_arm", -1)) != 1
        or plan.get("gpu_smoke_allowed") is not True
        or plan.get("full_training_allowed") is not False
        or plan.get("dev_allowed") is not False
        or plan.get("test_allowed") is not False
        or plan.get("model_selection_allowed") is not False
    ):
        raise PermissionError("mechanism-control smoke plan differs")


def _validate_candidate_config(config: dict[str, Any]) -> None:
    if (
        config.get("status")
        != "CPU_IMPLEMENTATION_CANDIDATE_GPU_NOT_AUTHORIZED"
        or int(config.get("run_count", -1)) != 9
        or config["test_boundary"]["test_adaptive_tuning_or_retraining"]
        is not False
        or config["authorization"]["full_gpu_training_allowed"] is not False
        or config["authorization"]["dev_inference_allowed"] is not False
        or config["authorization"]["test_inference_allowed"] is not False
    ):
        raise PermissionError("mechanism-control candidate differs")


def _validate_bindings(
    plan: dict[str, Any],
    *,
    arm: str,
    artifact_root: Path,
    output_root: Path,
    cuda_device_uuid: str | None,
) -> None:
    bindings = plan["implementation_bindings"]
    local_paths = {
        "runner_sha256": Path(__file__),
        "control_loss_sha256": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/mechanism_control_losses.py"
        ),
        "candidate_config_sha256": DEFAULT_MECHANISM_CONFIG,
    }
    for key, path in local_paths.items():
        if sha256_file(path) != str(bindings[key]):
            raise ValueError(f"mechanism-control smoke binding differs: {key}")
    artifact_paths = {
        "candidate_lock_sha256": (
            artifact_root
            / "mechanism_controls_candidate/candidate_lock.json"
        ),
        "candidate_report_sha256": (
            artifact_root
            / "mechanism_controls_candidate/candidate_report.json"
        ),
        "candidate_audit_sha256": (
            artifact_root
            / "mechanism_controls_candidate/audit_report.json"
        ),
    }
    for key, path in artifact_paths.items():
        reject_eval_path(path)
        if not path.is_file() or sha256_file(path) != str(bindings[key]):
            raise ValueError(f"mechanism-control artifact binding differs: {key}")
    if str(artifact_root) != str(plan["server_paths"]["artifact_root"]):
        raise ValueError("mechanism-control artifact root differs")
    if str(output_root) != str(plan["server_paths"]["output_root"]):
        raise ValueError("mechanism-control smoke output root differs")
    assignment = plan["gpu_assignments"][arm]
    if (
        assignment.get("name") != "NVIDIA RTX A6000"
        or not str(assignment.get("uuid") or "").startswith("GPU-")
    ):
        raise ValueError("mechanism-control GPU assignment differs")
    if cuda_device_uuid is not None and cuda_device_uuid != assignment["uuid"]:
        raise ValueError("requested CUDA UUID differs from smoke plan")


def _load_r3_rows(artifact_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    lock_path = artifact_root / "protocol/smoke_training_package_frozen_lock.json"
    prompt_path = (
        artifact_root
        / "data/smoke_v1_stratified/smoke_prompt_cache_seed42.jsonl"
    )
    manifest_path = (
        artifact_root
        / "data/smoke_v1_stratified/smoke_manifest_r3_seed42.jsonl"
    )
    for path in (lock_path, prompt_path, manifest_path):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = read_json(lock_path)
    if (
        lock.get("status")
        != "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
        or int(lock.get("events_per_arm", -1)) != SFT_EVENT_COUNT
        or lock.get("test_accessed") is not False
        or sha256_file(prompt_path)
        != lock["private_artifact_hashes"]["prompt_cache"]
        or sha256_file(manifest_path)
        != lock["private_artifact_hashes"]["manifests_by_arm"]["R3"]
    ):
        raise ValueError("frozen R3 smoke subset differs")
    rows = load_materialized_rows(prompt_path, manifest_path)
    if len(rows) != SFT_EVENT_COUNT:
        raise ValueError("R3-TOKENAVG smoke must contain eight events")
    if sum(bool(row["rationale_active"]) for row in rows) != 4:
        raise ValueError("R3-TOKENAVG smoke activity vector differs")
    return rows, {
        "prompt_cache_sha256": sha256_file(prompt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "smoke_lock_sha256": sha256_file(lock_path),
    }


def _preference_smoke_paths(artifact_root: Path) -> dict[str, Path]:
    root = artifact_root / "preference_smoke/private"
    return {
        "P1_FIELD_DPO": root / "p1_field_dpo.jsonl",
        "P1_SYN_LR5E6": root / "p1_syn_seed42.jsonl",
    }


def _load_preference_rows(
    arm: str,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    smoke_lock_path = artifact_root / "preference_smoke/smoke_package_lock.json"
    smoke_paths = _preference_smoke_paths(artifact_root)
    for path in (smoke_lock_path, *smoke_paths.values()):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    smoke_lock = read_json(smoke_lock_path)
    if (
        smoke_lock.get("source_split") != "train"
        or smoke_lock.get("test_accessed") is not False
        or sha256_file(smoke_paths["P1_FIELD_DPO"])
        != smoke_lock["private_subset_hashes"]["P1_FIELD_DPO"]
        or sha256_file(smoke_paths["P1_SYN_LR5E6"])
        != smoke_lock["private_subset_hashes"]["P1_SYN_SEED42"]
    ):
        raise ValueError("frozen preference smoke subsets differ")

    if arm == "P1_SYN_LR5E6":
        rows = read_jsonl(smoke_paths[arm])
        source_hash = sha256_file(smoke_paths[arm])
    elif arm == "P1_FULLSEQ":
        source_smoke = read_jsonl(smoke_paths["P1_FIELD_DPO"])
        source_ids = [str(row["pair_id"]) for row in source_smoke]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("P1 smoke subset contains duplicate pair IDs")
        full_path = (
            artifact_root
            / "mechanism_controls_candidate/private/p1_fullseq.jsonl"
        )
        audit_path = (
            artifact_root
            / "mechanism_controls_candidate/audit_report.json"
        )
        for path in (full_path, audit_path):
            reject_eval_path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
        audit = read_json(audit_path)
        if (
            audit.get("status")
            != "MECHANISM_CONTROL_CPU_AUDIT_PASS_GPU_NOT_AUTHORIZED"
            or audit.get("test_accessed") is not False
            or audit.get("p1_fullseq_manifest_sha256")
            != sha256_file(full_path)
        ):
            raise ValueError("P1-FULLSEQ candidate audit differs")
        full_rows = read_jsonl(full_path)
        full_by_id = {str(row["pair_id"]): row for row in full_rows}
        if len(full_by_id) != len(full_rows):
            raise ValueError("P1-FULLSEQ manifest contains duplicate pair IDs")
        try:
            rows = [full_by_id[pair_id] for pair_id in source_ids]
        except KeyError as exc:
            raise ValueError("P1-FULLSEQ smoke pair is missing") from exc
        for source, control in zip(source_smoke, rows, strict=True):
            for key, value in source.items():
                if control.get(key) != value:
                    raise ValueError(
                        f"P1-FULLSEQ changed frozen smoke field: {key}"
                    )
        source_hash = sha256_file(full_path)
    else:
        raise ValueError(f"unknown preference smoke arm: {arm}")

    if len(rows) != PREFERENCE_PAIR_COUNT:
        raise ValueError("preference smoke must contain exactly 32 pairs")
    if {str(row["pair_type"]) for row in rows} != {
        "adjacent_score",
        "severe_l2h",
        "h2l_guard",
    }:
        raise ValueError("preference smoke lacks one of the three score blocks")
    return rows, {
        "source_manifest_sha256": source_hash,
        "smoke_lock_sha256": sha256_file(smoke_lock_path),
    }


def _parameter_snapshot(trainable: list[Any]) -> list[Any]:
    return [parameter.detach().float().cpu().clone() for parameter in trainable]


def _parameter_delta_l2(trainable: list[Any], before: list[Any]) -> float:
    import torch

    squared = 0.0
    for parameter, original in zip(trainable, before, strict=True):
        delta = parameter.detach().float().cpu() - original
        squared += float(torch.sum(delta * delta))
    return math.sqrt(squared)


def _preference_logps(
    model: Any,
    batch: dict[str, Any],
    *,
    mode: str,
) -> tuple[Any, Any]:
    import torch

    batch_size = int(batch["chosen_input_ids"].shape[0])
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
    if mode == "fullseq":
        active_mask = torch.cat(
            [
                batch["chosen_completion_mask"],
                batch["rejected_completion_mask"],
            ],
            dim=0,
        ).to(model.device)
    elif mode == "field":
        active_mask = torch.cat(
            [
                batch["chosen_field_mask"],
                batch["rejected_field_mask"],
            ],
            dim=0,
        ).to(model.device)
    else:
        raise ValueError(f"unknown preference log-probability mode: {mode}")
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    if mode == "fullseq":
        logps = sequence_sum_logps(logits, input_ids, active_mask)[
            "per_sequence_logp"
        ]
    else:
        logps = field_mean_logps(logits, input_ids, active_mask)[
            "per_sequence_logp"
        ]
    if logps.shape != (2 * batch_size,):
        raise ValueError("preference log-probability shape differs")
    return logps[:batch_size], logps[batch_size:]


def _train_r3_tokenavg(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    config = read_json(DEFAULT_SFT_CONFIG)
    base_path, verified = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            task_type=str(lora["task_type"]),
            target_modules=list(lora["target_modules"]),
            modules_to_save=lora["modules_to_save"],
            inference_mode=bool(lora["inference_mode"]),
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in trainable) != int(
        lora["expected_trainable_parameters"]
    ):
        raise ValueError("R3-TOKENAVG LoRA parameter count differs")
    before = _parameter_snapshot(trainable)
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        betas=tuple(optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    collator = FixedRARCollator(
        pad_token_id=int(config["model"]["pad_token_id"]),
        cutoff_len=int(config["data"]["cutoff_len"]),
    )
    loader = DataLoader(
        rows,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collator,
    )
    if len(loader) != 4:
        raise ValueError("R3-TOKENAVG smoke micro-batch count differs")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
            loss_result = token_average_causal_loss(
                logits,
                batch["input_ids"],
                batch["score_mask"],
                batch["rationale_mask"],
                batch["rationale_active"],
                active_sample_scale=2.0,
            )
            objective = loss_result["loss"] / len(loader)
        if not torch.isfinite(objective):
            raise FloatingPointError("R3-TOKENAVG smoke loss is non-finite")
        objective.backward()
        losses.append(float(loss_result["loss"].detach().cpu()))
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=float(optimization["max_grad_norm"]),
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise FloatingPointError("R3-TOKENAVG gradient is invalid")
    optimizer.step()
    delta_l2 = _parameter_delta_l2(trainable, before)
    if not math.isfinite(delta_l2) or delta_l2 <= 0:
        raise FloatingPointError("R3-TOKENAVG parameter update is zero")
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    return {
        "losses": losses,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "parameter_update_l2": delta_l2,
        "optimizer_steps": 1,
        "micro_batches": len(loader),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "base_model_directory_listing_sha256": verified["model"][
            "directory_listing_sha256"
        ],
        "adapter_config_sha256": sha256_file(
            adapter_dir / "adapter_config.json"
        ),
        "adapter_model_sha256": sha256_file(
            adapter_dir / "adapter_model.safetensors"
        ),
        "device_identity": device_identity,
    }


def _train_preference(
    *,
    arm: str,
    rows: list[dict[str, Any]],
    artifact_root: Path,
    output_dir: Path,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    config = read_json(DEFAULT_PREFERENCE_CONFIG)
    base_path, _verified = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    checkpoint_lock = (
        artifact_root / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
    )
    adapter_dir, adapter_hashes = _seed42_checkpoint(
        checkpoint_lock,
        formal_runs_root=artifact_root / "formal_runs",
    )
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
    if arm == "P1_FULLSEQ":
        collator: Any = FullSequenceDPOPairCollator(
            pad_token_id=151643,
            cutoff_len=2048,
        )
        mode = "fullseq"
    elif arm == "P1_SYN_LR5E6":
        collator = SORCDPOPairCollator(
            pad_token_id=151643,
            cutoff_len=2048,
        )
        mode = "field"
    else:
        raise ValueError(f"unknown preference arm: {arm}")
    chunks = [
        rows[index : index + PHYSICAL_PREFERENCE_BATCH]
        for index in range(0, len(rows), PHYSICAL_PREFERENCE_BATCH)
    ]
    if len(chunks) != 8 or any(len(chunk) != 4 for chunk in chunks):
        raise ValueError("preference smoke physical batches differ")

    reference_logps = []
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            batch = collator(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected = _preference_logps(
                    model,
                    batch,
                    mode=mode,
                )
            reference_logps.append(
                (chosen.detach().cpu(), rejected.detach().cpu())
            )

    _set_adapter_trainability(model, "default")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before = _parameter_snapshot(trainable)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=5e-6,
        betas=tuple(config["optimization"]["betas"]),
        eps=float(config["optimization"]["epsilon"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    beta = float(config["loss"]["beta"])
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for index, chunk in enumerate(chunks):
        batch = collator(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy_chosen, policy_rejected = _preference_logps(
                model,
                batch,
                mode=mode,
            )
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
                normalization_pair_count=PREFERENCE_PAIR_COUNT,
            )["loss"]
        if not torch.isfinite(objective):
            raise FloatingPointError(f"{arm} smoke loss is non-finite")
        objective.backward()
        losses.append(float(objective.detach().cpu()))
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=float(config["optimization"]["max_grad_norm"]),
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise FloatingPointError(f"{arm} smoke gradient is invalid")
    optimizer.step()
    delta_l2 = _parameter_delta_l2(trainable, before)
    if not math.isfinite(delta_l2) or delta_l2 <= 0:
        raise FloatingPointError(f"{arm} smoke parameter update is zero")
    output_adapter = output_dir / "adapter"
    model.set_adapter("default")
    model.save_pretrained(
        str(output_adapter),
        selected_adapters=["default"],
        safe_serialization=True,
    )
    return {
        "losses": losses,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "parameter_update_l2": delta_l2,
        "optimizer_steps": 1,
        "micro_batches": len(chunks),
        "physical_micro_batch_pairs": PHYSICAL_PREFERENCE_BATCH,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cold_start_adapter_hashes": adapter_hashes,
        "adapter_config_sha256": sha256_file(
            output_adapter / "adapter_config.json"
        ),
        "adapter_model_sha256": sha256_file(
            output_adapter / "adapter_model.safetensors"
        ),
        "device_identity": device_identity,
    }


def run(
    *,
    arm: str,
    cuda_device_uuid: str,
    artifact_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown mechanism-control smoke arm: {arm}")
    for path in (
        DEFAULT_PLAN,
        DEFAULT_MECHANISM_CONFIG,
        DEFAULT_SFT_CONFIG,
        DEFAULT_PREFERENCE_CONFIG,
    ):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = read_json(DEFAULT_PLAN)
    _validate_plan(plan)
    _validate_candidate_config(read_json(DEFAULT_MECHANISM_CONFIG))
    _validate_bindings(
        plan,
        arm=arm,
        artifact_root=artifact_root,
        output_root=output_root,
        cuda_device_uuid=cuda_device_uuid,
    )
    reject_eval_path(artifact_root)
    reject_eval_path(output_root)
    output_dir = output_root / arm.lower()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    if arm == "R3_TOKENAVG":
        rows, source_hashes = _load_r3_rows(artifact_root)
    else:
        rows, source_hashes = _load_preference_rows(arm, artifact_root)

    import torch

    identity = _device_identity(torch, cuda_device_uuid)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("mechanism-control smoke requires BF16")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = time.time()
    if arm == "R3_TOKENAVG":
        training = _train_r3_tokenavg(
            rows=rows,
            output_dir=output_dir,
            device_identity=identity,
        )
    else:
        training = _train_preference(
            arm=arm,
            rows=rows,
            artifact_root=artifact_root,
            output_dir=output_dir,
            device_identity=identity,
        )
    result = {
        "schema_version": "exp54-mechanism-control-smoke-result-v1",
        "status": "MECHANISM_CONTROL_TRAIN_ONLY_SMOKE_COMPLETE",
        "arm": arm,
        "seed": 42,
        "source_row_count": len(rows),
        "source_hashes": source_hashes,
        "training": training,
        "elapsed_seconds": time.time() - started_at,
        "semantic_output_inspected": False,
        "dev_accessed": False,
        "test_accessed": False,
        "full_training_started": False,
    }
    _atomic_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cuda-device-uuid")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = read_json(DEFAULT_PLAN)
    _validate_plan(plan)
    _validate_candidate_config(read_json(DEFAULT_MECHANISM_CONFIG))
    _validate_bindings(
        plan,
        arm=args.arm,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        cuda_device_uuid=(
            args.cuda_device_uuid if args.execute else None
        ),
    )
    if args.arm == "R3_TOKENAVG":
        rows, hashes = _load_r3_rows(args.artifact_root)
    else:
        rows, hashes = _load_preference_rows(args.arm, args.artifact_root)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "MECHANISM_CONTROL_SMOKE_CPU_VALIDATION_PASS",
                    "arm": args.arm,
                    "rows": len(rows),
                    "source_hashes": hashes,
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
        raise ValueError("--execute requires --cuda-device-uuid")
    result = run(
        arm=args.arm,
        cuda_device_uuid=args.cuda_device_uuid,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
