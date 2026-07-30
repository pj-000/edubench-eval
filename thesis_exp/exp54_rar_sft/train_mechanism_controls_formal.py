"""Run the nine train-only Exp54 mechanism-control training jobs.

This runner is intentionally separate from the frozen RAR-SFT and SORC-DPO
entry points.  It changes exactly one mechanism per control and refuses to
read dev/test data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    reject_eval_path,
)
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
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    LOGICAL_EPOCHS,
    ROWS_PER_LOGICAL_EPOCH,
    _require_runtime,
    _save_checkpoint,
    _set_determinism,
    optimizer_group_plan,
    require_adapter_free_base_model,
    validate_training_configuration,
    verify_frozen_manifest,
    verify_model_snapshot,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_formal import (
    _checkpoint_for_seed,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    _set_adapter_trainability,
    read_json,
    verify_base_model_snapshot,
)


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
SEEDS = (42, 43, 44)
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_formal_plan_v1.json"
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
DEFAULT_SFT_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
DEFAULT_PREFERENCE_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_training_candidate_v1.json"
)
DEFAULT_LR_OVERLAY = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_lr5e6_followup_v1.json"
)
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "mechanism_control_formal"
PHYSICAL_PREFERENCE_BATCH = 4
PREFERENCE_PAIR_COUNT = 838
PREFERENCE_GROUP_SIZE = 32
PREFERENCE_STEPS = 27
MINIMUM_FREE_MEMORY_BYTES = 40 * 1024**3


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _chunks(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _validate_plan(
    plan: dict[str, Any],
    *,
    artifact_root: Path,
    output_root: Path,
    seed: int,
    cuda_device_uuid: str | None,
) -> None:
    if (
        plan.get("status")
        != "USER_APPROVED_TRAIN_ONLY_FULL_MECHANISM_CONTROLS"
        or plan.get("arms") != list(ARMS)
        or plan.get("seeds") != list(SEEDS)
        or plan.get("per_seed_serial_order") != list(ARMS)
        or int(plan.get("run_count", -1)) != 9
        or plan.get("full_gpu_training_allowed") is not True
        or plan.get("dev_allowed") is not False
        or plan.get("test_allowed") is not False
        or plan.get("model_selection_allowed") is not False
    ):
        raise PermissionError("mechanism-control formal plan differs")
    if str(artifact_root) != str(plan["server_paths"]["artifact_root"]):
        raise ValueError("formal artifact root differs")
    if str(output_root) != str(plan["server_paths"]["output_root"]):
        raise ValueError("formal output root differs")
    bindings = plan["implementation_bindings"]
    repository_paths = {
        "runner_sha256": Path(__file__),
        "control_loss_sha256": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/mechanism_control_losses.py"
        ),
        "candidate_config_sha256": DEFAULT_MECHANISM_CONFIG,
        "sft_config_sha256": DEFAULT_SFT_CONFIG,
        "preference_config_sha256": DEFAULT_PREFERENCE_CONFIG,
        "lr_overlay_sha256": DEFAULT_LR_OVERLAY,
        "train_rar_sft_sha256": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/train_rar_sft.py"
        ),
        "train_sorc_dpo_formal_sha256": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/train_sorc_dpo_formal.py"
        ),
        "train_sorc_dpo_smoke_sha256": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/train_sorc_dpo_smoke.py"
        ),
        "sorc_dpo_loss_sha256": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/sorc_dpo_loss.py"
        ),
        "block_loss_sha256": (
            REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
        ),
        "device_probe_sha256": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/probe_sorc_dpo_capacity.py"
        ),
    }
    artifact_paths = {
        "smoke_audit_sha256": (
            artifact_root / "mechanism_control_smoke/audit_report.json"
        ),
        "candidate_lock_sha256": (
            artifact_root / "mechanism_controls_candidate/candidate_lock.json"
        ),
        "candidate_report_sha256": (
            artifact_root
            / "mechanism_controls_candidate/candidate_report.json"
        ),
        "candidate_audit_sha256": (
            artifact_root / "mechanism_controls_candidate/audit_report.json"
        ),
        "materialized_manifest_lock_sha256": (
            artifact_root
            / "protocol/materialized_manifest_frozen_lock.json"
        ),
        "checkpoint_selection_lock_sha256": (
            artifact_root
            / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
        ),
    }
    for key, path in {**repository_paths, **artifact_paths}.items():
        reject_eval_path(path)
        if not path.is_file() or sha256_file(path) != str(bindings[key]):
            raise ValueError(f"formal implementation binding differs: {key}")
    if (
        plan.get("candidate_artifacts_promoted_to_formal_inputs") is not True
        or bindings["p1_fullseq_manifest_sha256"]
        != "b6d2289c96ef8bad92e219ef9c9e452c2d46b2acd90cb4db702cca8413cacac7"
        or bindings["p1_syn_manifest_sha256"]
        != "1c65f892f01e558797fcae204d71b3b6e4609ccbe8d34d44b3e60250a679cddc"
    ):
        raise ValueError("formal private-manifest promotion differs")
    assignment = plan["gpu_assignments_by_seed"][str(seed)]
    if (
        assignment.get("name") != "NVIDIA RTX A6000"
        or not str(assignment.get("uuid") or "").startswith("GPU-")
    ):
        raise ValueError("formal GPU assignment differs")
    if cuda_device_uuid is not None and cuda_device_uuid != assignment["uuid"]:
        raise ValueError("requested CUDA UUID differs from formal plan")


def _load_r3_rows(
    artifact_root: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], Path, dict[str, str]]:
    frozen_path = artifact_root / "protocol/materialized_manifest_frozen_lock.json"
    _frozen, manifest_path, prompt_path = verify_frozen_manifest(
        frozen_lock_path=frozen_path,
        data_dir=artifact_root / "data",
        arm="R3",
        seed=seed,
    )
    rows = load_materialized_rows(prompt_path, manifest_path)
    expected = ROWS_PER_LOGICAL_EPOCH * LOGICAL_EPOCHS
    if len(rows) != expected:
        raise ValueError("R3-TOKENAVG event count differs")
    return rows, manifest_path, {
        "manifest_sha256": file_sha256(manifest_path),
        "prompt_cache_sha256": file_sha256(prompt_path),
        "frozen_manifest_lock_sha256": file_sha256(frozen_path),
    }


def _preference_paths(artifact_root: Path) -> dict[str, Path]:
    candidate = artifact_root / "mechanism_controls_candidate"
    training = artifact_root / "preference_training_candidate/private"
    return {
        "candidate_lock": candidate / "candidate_lock.json",
        "candidate_report": candidate / "candidate_report.json",
        "candidate_audit": candidate / "audit_report.json",
        "P1_FULLSEQ": candidate / "private/p1_fullseq.jsonl",
        "P1_SYN_LR5E6": training / "p1_syn_seed42.jsonl",
    }


def _load_preference_rows(
    arm: str,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], Path, dict[str, str]]:
    paths = _preference_paths(artifact_root)
    for path in paths.values():
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = read_json(paths["candidate_lock"])
    report = read_json(paths["candidate_report"])
    audit = read_json(paths["candidate_audit"])
    if (
        lock.get("status") != "MECHANISM_CONTROL_CPU_CANDIDATE_NOT_FROZEN"
        or report.get("status")
        != "MECHANISM_CONTROL_MANIFEST_CANDIDATE_CPU_ONLY"
        or audit.get("status")
        != "MECHANISM_CONTROL_CPU_AUDIT_PASS_GPU_NOT_AUTHORIZED"
        or lock.get("dev_accessed") is not False
        or lock.get("test_accessed") is not False
        or report.get("dev_accessed") is not False
        or report.get("test_accessed") is not False
        or audit.get("dev_accessed") is not False
        or audit.get("test_accessed") is not False
        or lock["candidate_hashes"]["candidate_report"]
        != sha256_file(paths["candidate_report"])
        or audit["candidate_lock_sha256"]
        != sha256_file(paths["candidate_lock"])
        or audit["candidate_report_sha256"]
        != sha256_file(paths["candidate_report"])
    ):
        raise ValueError("mechanism-control candidate hash chain differs")
    manifest_path = paths[arm]
    expected_hash = (
        "b6d2289c96ef8bad92e219ef9c9e452c2d46b2acd90cb4db702cca8413cacac7"
        if arm == "P1_FULLSEQ"
        else "1c65f892f01e558797fcae204d71b3b6e4609ccbe8d34d44b3e60250a679cddc"
    )
    if sha256_file(manifest_path) != expected_hash:
        raise ValueError(f"{arm}: source manifest differs")
    rows = read_jsonl(manifest_path)
    if len(rows) != PREFERENCE_PAIR_COUNT:
        raise ValueError(f"{arm}: expected 838 pairs")
    if len({str(row["pair_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{arm}: duplicate pair ID")
    if {str(row["pair_type"]) for row in rows} != {
        "adjacent_score",
        "severe_l2h",
        "h2l_guard",
    }:
        raise ValueError(f"{arm}: score-block inventory differs")
    return rows, manifest_path, {
        "source_manifest_sha256": sha256_file(manifest_path),
        "candidate_lock_sha256": sha256_file(paths["candidate_lock"]),
        "candidate_report_sha256": sha256_file(paths["candidate_report"]),
        "candidate_audit_sha256": sha256_file(paths["candidate_audit"]),
    }


def _write_progress(
    *,
    output_dir: Path,
    arm: str,
    seed: int,
    phase: str,
    completed_units: int,
    total_units: int,
    optimizer_step: int,
    optimizer_steps: int,
    started_at: float,
    latest_loss: float | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started_at)
    fraction = completed_units / total_units if total_units else 0.0
    progress: dict[str, Any] = {
        "schema_version": "exp54-mechanism-control-formal-progress-v1",
        "status": "RUNNING",
        "arm": arm,
        "seed": seed,
        "phase": phase,
        "completed_units": completed_units,
        "total_units": total_units,
        "completion_percent": 100.0 * fraction,
        "optimizer_step": optimizer_step,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": (
            elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
        ),
        "dev_accessed": False,
        "test_accessed": False,
    }
    if latest_loss is not None:
        progress["latest_loss"] = latest_loss
    _atomic_json(output_dir / "progress.json", progress)


def _preference_logps(
    model: Any,
    batch: dict[str, Any],
    *,
    mode: str,
) -> tuple[Any, Any]:
    import torch

    size = int(batch["chosen_input_ids"].shape[0])
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
        mask = torch.cat(
            [
                batch["chosen_completion_mask"],
                batch["rejected_completion_mask"],
            ],
            dim=0,
        ).to(model.device)
    elif mode == "field":
        mask = torch.cat(
            [batch["chosen_field_mask"], batch["rejected_field_mask"]],
            dim=0,
        ).to(model.device)
    else:
        raise ValueError(f"unknown preference mode: {mode}")
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    values = (
        sequence_sum_logps(logits, input_ids, mask)["per_sequence_logp"]
        if mode == "fullseq"
        else field_mean_logps(logits, input_ids, mask)["per_sequence_logp"]
    )
    if values.shape != (2 * size,):
        raise ValueError("preference log-probability shape differs")
    return values[:size], values[size:]


def _execute_r3(
    *,
    seed: int,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    source_hashes: dict[str, str],
    artifact_root: Path,
    output_dir: Path,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    config = read_json(DEFAULT_SFT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
    _require_runtime(config)
    _set_determinism(seed, config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["local_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
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
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in trainable) != int(
        lora["expected_trainable_parameters"]
    ):
        raise ValueError("R3-TOKENAVG LoRA parameter count differs")
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        betas=tuple(optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimization["warmup_steps"]),
        num_training_steps=int(optimization["total_optimizer_steps"]),
    )
    collator = FixedRARCollator(
        pad_token_id=int(config["model"]["pad_token_id"]),
        cutoff_len=int(config["data"]["cutoff_len"]),
    )
    group_plan = optimizer_group_plan(
        rows_per_epoch=ROWS_PER_LOGICAL_EPOCH,
        micro_batch_size=int(
            optimization["micro_batch_size_per_device"]
        ),
        gradient_accumulation_steps=int(
            optimization["gradient_accumulation_steps"]
        ),
    )
    total_steps = int(optimization["total_optimizer_steps"])
    started_at = time.time()
    global_step = 0
    losses: list[float] = []
    gradient_norms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch_index in range(LOGICAL_EPOCHS):
        epoch_rows = rows[
            epoch_index * ROWS_PER_LOGICAL_EPOCH :
            (epoch_index + 1) * ROWS_PER_LOGICAL_EPOCH
        ]
        loader = DataLoader(
            epoch_rows,
            batch_size=int(
                optimization["micro_batch_size_per_device"]
            ),
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collator,
        )
        optimizer.zero_grad(set_to_none=True)
        group_index = 0
        within_group = 0
        group_size = group_plan[group_index]
        group_loss = 0.0
        for batch in loader:
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                raw_loss = token_average_causal_loss(
                    logits,
                    batch["input_ids"],
                    batch["score_mask"],
                    batch["rationale_mask"],
                    batch["rationale_active"],
                    active_sample_scale=2.0,
                )["loss"]
                objective = raw_loss / group_size
            if not torch.isfinite(objective):
                raise FloatingPointError("R3-TOKENAVG loss is non-finite")
            objective.backward()
            group_loss += float(raw_loss.detach().cpu()) / group_size
            within_group += 1
            if within_group != group_size:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=float(optimization["max_grad_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("R3-TOKENAVG gradient is non-finite")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            losses.append(group_loss)
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            _write_progress(
                output_dir=output_dir,
                arm="R3_TOKENAVG",
                seed=seed,
                phase=f"logical_epoch_{epoch_index + 1}",
                completed_units=global_step,
                total_units=total_steps,
                optimizer_step=global_step,
                optimizer_steps=total_steps,
                started_at=started_at,
                latest_loss=group_loss,
            )
            group_index += 1
            within_group = 0
            group_loss = 0.0
            if group_index < len(group_plan):
                group_size = group_plan[group_index]
        if group_index != len(group_plan) or within_group != 0:
            raise RuntimeError("R3-TOKENAVG accumulation boundary differs")
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            logical_epoch_number=epoch_index + 1,
            global_step=global_step,
            arm="R3_TOKENAVG",
            seed=seed,
            config_path=DEFAULT_SFT_CONFIG,
            frozen_lock_path=(
                artifact_root
                / "protocol/materialized_manifest_frozen_lock.json"
            ),
            manifest_path=manifest_path,
        )
    if global_step != total_steps:
        raise RuntimeError("R3-TOKENAVG optimizer-step count differs")
    adapter = output_dir / "checkpoint-logical-epoch-3/adapter"
    return {
        "schema_version": "exp54-r3-tokenavg-formal-result-v1",
        "status": "MECHANISM_CONTROL_FORMAL_TRAINING_COMPLETE",
        "arm": "R3_TOKENAVG",
        "seed": seed,
        "event_count": len(rows),
        "logical_epochs": LOGICAL_EPOCHS,
        "optimizer_steps": global_step,
        "learning_rate": float(optimization["learning_rate"]),
        "loss_reduction": "token_average_active_scaled_by_2",
        "losses": losses,
        "gradient_norms_before_clip": gradient_norms,
        "elapsed_seconds": time.time() - started_at,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_identity": device_identity,
        "source_hashes": source_hashes,
        "output_adapter_config_sha256": sha256_file(
            adapter / "adapter_config.json"
        ),
        "output_adapter_model_sha256": sha256_file(
            adapter / "adapter_model.safetensors"
        ),
        "dev_accessed": False,
        "test_accessed": False,
    }


def _execute_preference(
    *,
    arm: str,
    seed: int,
    rows: list[dict[str, Any]],
    manifest_path: Path,
    source_hashes: dict[str, str],
    artifact_root: Path,
    output_dir: Path,
    device_identity: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )

    config = read_json(DEFAULT_PREFERENCE_CONFIG)
    overlay = read_json(DEFAULT_LR_OVERLAY)
    if (
        overlay.get("only_override", {}).get("field")
        != "optimization.learning_rate"
        or float(overlay["only_override"]["to"]) != 5e-6
    ):
        raise ValueError("preference LR overlay differs")
    base_path, _verified = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    adapter_dir, adapter_hashes = _checkpoint_for_seed(
        seed,
        checkpoint_lock_path=(
            artifact_root
            / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
        ),
        formal_runs_root=artifact_root / "formal_runs",
    )
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
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
    groups = _chunks(shuffled, PREFERENCE_GROUP_SIZE)
    if len(groups) != PREFERENCE_STEPS or len(groups[-1]) != 6:
        raise ValueError("preference optimizer grouping differs")
    reference_chunks = _chunks(shuffled, PHYSICAL_PREFERENCE_BATCH)
    policy_chunk_count = sum(
        math.ceil(len(group) / PHYSICAL_PREFERENCE_BATCH)
        for group in groups
    )
    total_units = len(reference_chunks) + policy_chunk_count
    started_at = time.time()
    _write_progress(
        output_dir=output_dir,
        arm=arm,
        seed=seed,
        phase="reference_logps",
        completed_units=0,
        total_units=total_units,
        optimizer_step=0,
        optimizer_steps=PREFERENCE_STEPS,
        started_at=started_at,
    )
    reference_by_pair: dict[str, tuple[float, float]] = {}
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for chunk_index, chunk in enumerate(reference_chunks, start=1):
            batch = collator(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected = _preference_logps(
                    model,
                    batch,
                    mode=mode,
                )
            for row, chosen_value, rejected_value in zip(
                chunk,
                chosen.detach().float().cpu().tolist(),
                rejected.detach().float().cpu().tolist(),
                strict=True,
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
                completed_units=chunk_index,
                total_units=total_units,
                optimizer_step=0,
                optimizer_steps=PREFERENCE_STEPS,
                started_at=started_at,
            )
    if len(reference_by_pair) != len(shuffled):
        raise ValueError("preference reference-logp inventory differs")
    _set_adapter_trainability(model, "default")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=5e-6,
        betas=tuple(optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    warmup_steps = math.ceil(
        PREFERENCE_STEPS * float(optimization["warmup_ratio"])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=PREFERENCE_STEPS,
    )
    beta = float(config["loss"]["beta"])
    completed_policy_chunks = 0
    losses: list[float] = []
    gradient_norms: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for step_index, group in enumerate(groups, start=1):
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for chunk in _chunks(group, PHYSICAL_PREFERENCE_BATCH):
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
                policy_chosen, policy_rejected = _preference_logps(
                    model,
                    batch,
                    mode=mode,
                )
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
                raise FloatingPointError(f"{arm} loss is non-finite")
            objective.backward()
            group_loss += float(objective.detach().cpu())
            completed_policy_chunks += 1
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=float(optimization["max_grad_norm"]),
        )
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise FloatingPointError(f"{arm} gradient is invalid")
        optimizer.step()
        scheduler.step()
        losses.append(group_loss)
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        _write_progress(
            output_dir=output_dir,
            arm=arm,
            seed=seed,
            phase="preference_training",
            completed_units=len(reference_chunks) + completed_policy_chunks,
            total_units=total_units,
            optimizer_step=step_index,
            optimizer_steps=PREFERENCE_STEPS,
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
    return {
        "schema_version": "exp54-preference-mechanism-formal-result-v1",
        "status": "MECHANISM_CONTROL_FORMAL_TRAINING_COMPLETE",
        "arm": arm,
        "seed": seed,
        "pair_count": len(rows),
        "preference_epochs": 1,
        "optimizer_steps": PREFERENCE_STEPS,
        "learning_rate": 5e-6,
        "beta": beta,
        "log_probability_mode": mode,
        "physical_micro_batch_pairs": PHYSICAL_PREFERENCE_BATCH,
        "gradient_accumulation_pairs": PREFERENCE_GROUP_SIZE,
        "final_group_pairs": len(groups[-1]),
        "losses": losses,
        "gradient_norms_before_clip": gradient_norms,
        "elapsed_seconds": time.time() - started_at,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_identity": device_identity,
        "source_hashes": source_hashes,
        "source_manifest_sha256": sha256_file(manifest_path),
        "cold_start_adapter_hashes": adapter_hashes,
        "training_config_sha256": sha256_file(DEFAULT_PREFERENCE_CONFIG),
        "lr_overlay_sha256": sha256_file(DEFAULT_LR_OVERLAY),
        "output_adapter_config_sha256": sha256_file(
            adapter_output / "adapter_config.json"
        ),
        "output_adapter_model_sha256": sha256_file(
            adapter_output / "adapter_model.safetensors"
        ),
        "dev_accessed": False,
        "test_accessed": False,
    }


def validate(
    *,
    arm: str,
    seed: int,
    artifact_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    plan = read_json(DEFAULT_PLAN)
    _validate_plan(
        plan,
        artifact_root=artifact_root,
        output_root=output_root,
        seed=seed,
        cuda_device_uuid=None,
    )
    if arm == "R3_TOKENAVG":
        rows, _manifest, hashes = _load_r3_rows(artifact_root, seed)
        steps = 996
    else:
        rows, _manifest, hashes = _load_preference_rows(arm, artifact_root)
        steps = 27
    return {
        "status": "MECHANISM_CONTROL_FORMAL_CPU_VALIDATION_PASS",
        "arm": arm,
        "seed": seed,
        "source_rows": len(rows),
        "optimizer_steps": steps,
        "source_hashes": hashes,
        "model_loaded": False,
        "cuda_initialized": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def execute(
    *,
    arm: str,
    seed: int,
    cuda_device_uuid: str,
    artifact_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    plan = read_json(DEFAULT_PLAN)
    _validate_plan(
        plan,
        artifact_root=artifact_root,
        output_root=output_root,
        seed=seed,
        cuda_device_uuid=cuda_device_uuid,
    )
    output_dir = output_root / arm.lower() / f"seed_{seed}"
    reject_eval_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if arm == "R3_TOKENAVG":
        rows, manifest_path, hashes = _load_r3_rows(artifact_root, seed)
    else:
        rows, manifest_path, hashes = _load_preference_rows(
            arm,
            artifact_root,
        )
    import torch

    identity = _device_identity(torch, cuda_device_uuid)
    if int(identity["free_memory_bytes_before_load"]) < MINIMUM_FREE_MEMORY_BYTES:
        raise RuntimeError("assigned A6000 is not sufficiently idle")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal mechanism-control training requires BF16")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if arm == "R3_TOKENAVG":
            result = _execute_r3(
                seed=seed,
                rows=rows,
                manifest_path=manifest_path,
                source_hashes=hashes,
                artifact_root=artifact_root,
                output_dir=output_dir,
                device_identity=identity,
            )
        else:
            result = _execute_preference(
                arm=arm,
                seed=seed,
                rows=rows,
                manifest_path=manifest_path,
                source_hashes=hashes,
                artifact_root=artifact_root,
                output_dir=output_dir,
                device_identity=identity,
            )
    except BaseException as exc:
        _atomic_json(
            output_dir / "failure.json",
            {
                "status": "MECHANISM_CONTROL_FORMAL_TRAINING_FAILED",
                "arm": arm,
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "dev_accessed": False,
                "test_accessed": False,
            },
        )
        raise
    _atomic_json(output_dir / "result.json", result)
    _atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": "exp54-mechanism-control-formal-progress-v1",
            "status": "COMPLETE",
            "arm": arm,
            "seed": seed,
            "completion_percent": 100.0,
            "optimizer_step": result["optimizer_steps"],
            "optimizer_steps": result["optimizer_steps"],
            "elapsed_seconds": result["elapsed_seconds"],
            "estimated_remaining_seconds": 0.0,
            "dev_accessed": False,
            "test_accessed": False,
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
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
    if args.validate_only:
        result = validate(
            arm=args.arm,
            seed=args.seed,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
        )
    else:
        if not args.cuda_device_uuid:
            raise ValueError("--cuda-device-uuid is required for execution")
        result = execute(
            arm=args.arm,
            seed=args.seed,
            cuda_device_uuid=args.cuda_device_uuid,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
