"""Frozen train/dev-only trainer for the Exp60 residual-geometry ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.train import (
    aggregate_text_hash,
    evaluate,
    git_head,
    limit,
    load_model,
    save_checkpoint,
)
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REPO_ROOT,
    SOURCE_LOCK_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.mapping import (
    mapping_sha256,
    mapping_target_lookup,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    StepProgressBar,
    TrainConfig,
    format_duration,
    make_dataloader,
    save_predictions,
    set_seed,
    write_json,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


VARIANTS = (
    "consensus_only",
    "aligned_orthogonal_only",
    "matched_shuffled_orthogonal_only",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> tuple[TrainConfig, str]:
    parser = argparse.ArgumentParser(description="Frozen Exp60 residual-geometry trainer")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--checkpoint_output_dir", type=Path)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True, choices=(47, 48, 49))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--bf16", choices=("auto", "true", "false"), default="true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    args = parser.parse_args()
    output_dir = args.output_dir or OUTPUT_ROOT / "runs" / args.variant / f"seed_{args.seed}"
    checkpoint_dir = args.checkpoint_output_dir or (
        REPO_ROOT / "thesis_exp" / "artifacts" / "exp60_geometry_matched_shuffle" / args.variant / f"seed_{args.seed}"
    )
    config = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP60_IN_MEMORY_TRAIN_DEV_ONLY"),
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
        evaluate_test=False,
    )
    return config, args.variant


def verify_contract(*, require_source_lock: bool) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP60_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS":
        raise RuntimeError("Exp60 protocol is not frozen")
    if protocol.get("allowed_splits") != ["train", "dev"] or protocol.get("test_access_count") != 0:
        raise RuntimeError("Invalid Exp60 split contract")
    lock = None
    if SOURCE_LOCK_PATH.is_file():
        lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
        for relative, expected in lock["files"].items():
            actual = sha256_file(REPO_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"Exp60 source-lock mismatch: {relative}: {actual} != {expected}")
    elif require_source_lock:
        raise FileNotFoundError(SOURCE_LOCK_PATH)
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_lock_sha256": sha256_file(SOURCE_LOCK_PATH) if lock else None,
    }


def _capture_rng(torch: Any, device: Any) -> tuple[Any, Any]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(device)


def _restore_rng(torch: Any, device: Any, state: tuple[Any, Any]) -> None:
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


def _install_residual_capture_hooks(
    model: Any,
    residual_buffers: dict[str, Any],
) -> list[Any]:
    handles = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        def capture(gradient: Any, parameter_name: str = name) -> Any:
            if parameter_name in residual_buffers:
                residual_buffers[parameter_name].add_(gradient.detach().float())
            return gradient.new_zeros(gradient.shape)

        handles.append(parameter.register_hook(capture))
    return handles


def _accumulate_residual_vjp(
    model: Any,
    inputs: dict[str, Any],
    labels: Any,
    targets: Any,
    residual_buffers: dict[str, Any],
    loss_scale: float,
) -> float:
    handles = _install_residual_capture_hooks(model, residual_buffers)
    try:
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="residual_only",
            route_loss_scale=loss_scale,
        )
        objective = cbrd_objective(outputs, labels, targets, variant="residual_only")
        auxiliary_loss = objective["reported_aux_soft_ce"]
        (auxiliary_loss * loss_scale).backward()
        return float(auxiliary_loss.detach().cpu())
    finally:
        for handle in handles:
            handle.remove()


def _sum_squares(torch: Any, values: list[Any], device: Any) -> Any:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for value in values:
        total = total + value.detach().float().square().sum()
    return total


def post_cast_relative_tolerance(dtype_name: str) -> float:
    """Return the frozen storage-only norm tolerance for a gradient dtype."""

    if dtype_name == "torch.bfloat16":
        return 1.5 * (2.0 ** -7)
    return 1e-6


def compose_geometry_step(
    model: Any,
    aligned_residual_buffers: dict[str, Any],
    shuffled_residual_buffers: dict[str, Any],
    *,
    variant: str,
    max_norm: float,
) -> dict[str, Any]:
    """Stream the two projections, match norms, select one arm, and clip once.

    Only the two diagnostic VJP buffers persist.  Common and component tensors
    are recomputed one parameter at a time so a 0.6B model never materializes
    additional full-model gradient dictionaries.
    """

    import torch

    if variant not in VARIANTS:
        raise ValueError(f"Unknown Exp60 variant: {variant}")
    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not named:
        raise RuntimeError("No accumulated gradients")
    device = next(iter(named.values())).grad.device
    common_sq = torch.zeros((), device=device, dtype=torch.float64)
    aligned_sq = torch.zeros((), device=device, dtype=torch.float64)
    shuffled_sq = torch.zeros((), device=device, dtype=torch.float64)
    common_aligned_dot = torch.zeros((), device=device, dtype=torch.float64)
    common_shuffled_dot = torch.zeros((), device=device, dtype=torch.float64)
    for name, parameter in named.items():
        if not name.startswith("backbone."):
            continue
        routed = parameter.grad.detach().float()
        aligned_residual = aligned_residual_buffers.get(name)
        if aligned_residual is None:
            aligned_residual = torch.zeros_like(routed)
        shuffled_residual = shuffled_residual_buffers.get(name)
        if shuffled_residual is None:
            shuffled_residual = torch.zeros_like(routed)
        common = routed - aligned_residual
        common_sq += common.square().sum(dtype=torch.float64)
        aligned_sq += aligned_residual.square().sum(dtype=torch.float64)
        shuffled_sq += shuffled_residual.square().sum(dtype=torch.float64)
        common_aligned_dot += (common * aligned_residual).sum(dtype=torch.float64)
        common_shuffled_dot += (common * shuffled_residual).sum(dtype=torch.float64)
    common_sq_value = float(common_sq.detach().cpu())
    aligned_sq_value = float(aligned_sq.detach().cpu())
    shuffled_sq_value = float(shuffled_sq.detach().cpu())
    aligned_dot_value = float(common_aligned_dot.detach().cpu())
    shuffled_dot_value = float(common_shuffled_dot.detach().cpu())
    aligned_coefficient = (
        aligned_dot_value / common_sq_value if common_sq_value > 0.0 else 0.0
    )
    shuffled_coefficient = (
        shuffled_dot_value / common_sq_value if common_sq_value > 0.0 else 0.0
    )

    aligned_orthogonal_sq = torch.zeros((), device=device, dtype=torch.float64)
    shuffled_orthogonal_sq = torch.zeros((), device=device, dtype=torch.float64)
    for name, parameter in named.items():
        if not name.startswith("backbone."):
            continue
        routed = parameter.grad.detach().float()
        aligned_residual = aligned_residual_buffers[name]
        shuffled_residual = shuffled_residual_buffers[name]
        common = routed - aligned_residual
        aligned_orthogonal = aligned_residual - aligned_coefficient * common
        shuffled_orthogonal = shuffled_residual - shuffled_coefficient * common
        aligned_orthogonal_sq += aligned_orthogonal.square().sum(dtype=torch.float64)
        shuffled_orthogonal_sq += shuffled_orthogonal.square().sum(dtype=torch.float64)
    aligned_orthogonal_norm = math.sqrt(
        max(0.0, float(aligned_orthogonal_sq.detach().cpu()))
    )
    shuffled_orthogonal_norm = math.sqrt(
        max(0.0, float(shuffled_orthogonal_sq.detach().cpu()))
    )
    if aligned_orthogonal_norm > 0.0 and shuffled_orthogonal_norm <= 0.0:
        raise RuntimeError(
            "EXP60_STOP_SHUFFLED_ORTHOGONAL_ZERO_WHILE_ALIGNED_NONZERO"
        )
    shuffled_scale = (
        aligned_orthogonal_norm / shuffled_orthogonal_norm
        if shuffled_orthogonal_norm > 0.0
        else 0.0
    )

    actual_aligned_sq = torch.zeros((), device=device, dtype=torch.float64)
    actual_shuffled_sq = torch.zeros((), device=device, dtype=torch.float64)
    aligned_common_dot = torch.zeros((), device=device, dtype=torch.float64)
    shuffled_common_dot = torch.zeros((), device=device, dtype=torch.float64)
    aligned_total_sq = torch.zeros((), device=device, dtype=torch.float64)
    shuffled_total_sq = torch.zeros((), device=device, dtype=torch.float64)
    for name, parameter in named.items():
        routed = parameter.grad.detach().float()
        aligned_residual = aligned_residual_buffers.get(name)
        shuffled_residual = shuffled_residual_buffers.get(name)
        if aligned_residual is None:
            aligned_residual = torch.zeros_like(routed)
        if shuffled_residual is None:
            shuffled_residual = torch.zeros_like(routed)
        common = routed - aligned_residual
        if name.startswith("backbone."):
            aligned_component = aligned_residual - aligned_coefficient * common
            shuffled_component = shuffled_scale * (
                shuffled_residual - shuffled_coefficient * common
            )
            actual_aligned_sq += aligned_component.square().sum(dtype=torch.float64)
            actual_shuffled_sq += shuffled_component.square().sum(dtype=torch.float64)
            aligned_common_dot += (common * aligned_component).sum(dtype=torch.float64)
            shuffled_common_dot += (common * shuffled_component).sum(dtype=torch.float64)
        else:
            aligned_component = torch.zeros_like(common)
            shuffled_component = torch.zeros_like(common)
        aligned_preclip = common + aligned_component
        shuffled_preclip = common + shuffled_component
        aligned_total_sq += aligned_preclip.square().sum(dtype=torch.float64)
        shuffled_total_sq += shuffled_preclip.square().sum(dtype=torch.float64)
        if variant == "consensus_only":
            selected = common
        elif variant == "aligned_orthogonal_only":
            selected = aligned_preclip
        else:
            selected = shuffled_preclip
        parameter.grad.copy_(selected.to(dtype=parameter.grad.dtype))

    actual_aligned_norm = math.sqrt(max(0.0, float(actual_aligned_sq.detach().cpu())))
    actual_shuffled_norm = math.sqrt(max(0.0, float(actual_shuffled_sq.detach().cpu())))
    common_norm = math.sqrt(max(0.0, common_sq_value))
    aligned_orthogonality_error = abs(float(aligned_common_dot.detach().cpu())) / (
        common_norm * actual_aligned_norm + 1e-12
    )
    shuffled_orthogonality_error = abs(float(shuffled_common_dot.detach().cpu())) / (
        common_norm * actual_shuffled_norm + 1e-12
    )
    aligned_total_norm = math.sqrt(max(0.0, float(aligned_total_sq.detach().cpu())))
    shuffled_total_norm = math.sqrt(max(0.0, float(shuffled_total_sq.detach().cpu())))
    component_norm_relative_error = abs(actual_aligned_norm - actual_shuffled_norm) / max(
        actual_aligned_norm, actual_shuffled_norm, 1e-12
    )
    preclip_total_norm_relative_error = abs(aligned_total_norm - shuffled_total_norm) / max(
        aligned_total_norm, shuffled_total_norm, 1e-12
    )
    clip_aligned = min(1.0, max_norm / (aligned_total_norm + 1e-6))
    clip_shuffled = min(1.0, max_norm / (shuffled_total_norm + 1e-6))
    clip_coefficient_relative_error = abs(clip_aligned - clip_shuffled) / max(
        clip_aligned, clip_shuffled, 1e-12
    )
    if aligned_orthogonality_error > 1e-6:
        raise RuntimeError("Aligned orthogonality gate failure")
    if shuffled_orthogonality_error > 1e-6:
        raise RuntimeError("Shuffled orthogonality gate failure")
    if component_norm_relative_error > 1e-4:
        raise RuntimeError("Component norm matching gate failure")
    if preclip_total_norm_relative_error > 1e-4:
        raise RuntimeError("Preclip total norm matching gate failure")
    if clip_coefficient_relative_error > 1e-4:
        raise RuntimeError("Clip coefficient matching gate failure")
    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in named.values()],
            max_norm=max_norm,
        ).detach().cpu()
    )
    postclip_sq = _sum_squares(
        torch,
        [parameter.grad for parameter in named.values()],
        device,
    )
    postclip_norm = math.sqrt(max(0.0, float(postclip_sq.detach().cpu())))
    gradient_dtype = str(next(iter(named.values())).grad.dtype)
    storage_relative_tolerance = post_cast_relative_tolerance(gradient_dtype)
    post_cast_limit = max_norm * (1.0 + storage_relative_tolerance)
    if not math.isfinite(postclip_norm) or postclip_norm > post_cast_limit:
        raise RuntimeError(f"Standard-clipped gradient norm violation: {postclip_norm}")
    return {
        "variant": variant,
        "common_backbone_norm": common_norm,
        "aligned_residual_norm": math.sqrt(max(0.0, aligned_sq_value)),
        "shuffled_residual_norm": math.sqrt(max(0.0, shuffled_sq_value)),
        "aligned_projection_coefficient": aligned_coefficient,
        "shuffled_projection_coefficient": shuffled_coefficient,
        "aligned_orthogonal_norm": actual_aligned_norm,
        "shuffled_orthogonal_norm_before_match": shuffled_orthogonal_norm,
        "shuffled_orthogonal_norm_after_match": actual_shuffled_norm,
        "shuffled_scale": shuffled_scale,
        "component_norm_relative_error": component_norm_relative_error,
        "preclip_total_norm_aligned": aligned_total_norm,
        "preclip_total_norm_shuffled": shuffled_total_norm,
        "preclip_total_norm_relative_error": preclip_total_norm_relative_error,
        "clip_coefficient_aligned": clip_aligned,
        "clip_coefficient_shuffled": clip_shuffled,
        "clip_coefficient_relative_error": clip_coefficient_relative_error,
        "aligned_normalized_orthogonality_error": aligned_orthogonality_error,
        "shuffled_normalized_orthogonality_error": shuffled_orthogonality_error,
        "preclip_norm": preclip_norm,
        "postclip_norm": postclip_norm,
        "gradient_storage_dtype": gradient_dtype,
        "post_cast_relative_tolerance": storage_relative_tolerance,
        "post_cast_norm_limit": post_cast_limit,
    }


def train(config: TrainConfig, variant: str) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if variant not in VARIANTS:
        raise ValueError(f"Missing or invalid Exp60 variant: {variant}")
    if config.evaluate_test:
        raise PermissionError("Exp60 must never evaluate test")
    if config.gradient_accumulation_steps != 32 or config.max_grad_norm != 1.0:
        raise RuntimeError("Exp60 accumulation and clipping must match the frozen protocol")
    contract_files = verify_contract(
        require_source_lock=os.environ.get("EXP60_REQUIRE_SOURCE_LOCK") == "1"
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit(model_rows("train"), config.max_train_samples)
    dev_rows = limit(model_rows("dev"), config.max_eval_samples)
    mapping_rows = [
        json.loads(line)
        for line in MAPPING_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping_audit = json.loads(MAPPING_AUDIT_PATH.read_text(encoding="utf-8"))
    if mapping_sha256(mapping_rows) != mapping_audit["mapping_sha256"]:
        raise RuntimeError("Exp60 mapping hash mismatch")
    shuffled_targets_by_id = mapping_target_lookup(mapping_rows)
    if config.max_train_samples is None and len(shuffled_targets_by_id) != len(train_rows):
        raise RuntimeError("Exp60 mapping does not cover the complete training set")
    model, tokenizer, model_mode, initial_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal Exp60 training requires CUDA")
    model.to(device)
    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * updates_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_dir = config.checkpoint_output_dir / "best"
    epoch10_dir = config.checkpoint_output_dir / "epoch10"
    best_exact = -1.0
    best_epoch: int | None = None
    best_metrics: dict[str, Any] | None = None
    epoch10_result: Any = None
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    step_audit: list[dict[str, Any]] = []
    named_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    aligned_residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    }
    shuffled_residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    }
    global_step = micro_step = 0
    start = time.time()
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp60 {variant}")
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_zero in range(int(math.ceil(config.num_train_epochs))):
            if epoch_zero >= config.num_train_epochs:
                break
            epoch = epoch_zero + 1
            model.train()
            hard_running = aux_running = total_running = 0.0
            for step, batch in enumerate(train_loader, start=1):
                metadata = batch.pop("metadata")
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                aligned_targets = torch.tensor(
                    [row["soft_target_5"] for row in metadata],
                    dtype=torch.float32,
                    device=device,
                )
                shuffled_targets = torch.tensor(
                    [shuffled_targets_by_id[str(row["record_id"])] for row in metadata],
                    dtype=torch.float32,
                    device=device,
                )
                loss_scale = 1.0 / float(config.gradient_accumulation_steps)
                rng_before = _capture_rng(torch, device)
                outputs = model(
                    **inputs,
                    labels=labels,
                    soft_targets=aligned_targets,
                    aux_route="routed_hmsa",
                    route_loss_scale=loss_scale,
                )
                objective = cbrd_objective(
                    outputs,
                    labels,
                    aligned_targets,
                    variant="routed_hmsa",
                )
                loss = objective["optimization_loss"]
                (loss * loss_scale).backward()
                rng_after = _capture_rng(torch, device)
                _restore_rng(torch, device, rng_before)
                aligned_diagnostic_aux = _accumulate_residual_vjp(
                    model,
                    inputs,
                    labels,
                    aligned_targets,
                    aligned_residual_buffers,
                    loss_scale,
                )
                _restore_rng(torch, device, rng_before)
                shuffled_diagnostic_aux = _accumulate_residual_vjp(
                    model,
                    inputs,
                    labels,
                    shuffled_targets,
                    shuffled_residual_buffers,
                    loss_scale,
                )
                _restore_rng(torch, device, rng_after)
                hard_value = float(objective["reported_hard_loss"].detach().cpu())
                aux_value = float(objective["reported_aux_soft_ce"].detach().cpu())
                total_value = float(loss.detach().cpu())
                hard_running += hard_value
                aux_running += aux_value
                total_running += total_value
                micro_step += 1
                if len(trace) < 64:
                    trace.append(
                        {
                            "micro_step": micro_step,
                            "record_ids": [row["record_id"] for row in metadata],
                            "hard_loss": hard_value,
                            "reported_original_soft_ce": aux_value,
                            "aligned_diagnostic_aux_soft_ce": aligned_diagnostic_aux,
                            "shuffled_diagnostic_aux_soft_ce": shuffled_diagnostic_aux,
                            "optimization_loss": total_value,
                            "route_loss_scale": loss_scale,
                        }
                    )
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    audit = compose_geometry_step(
                        model,
                        aligned_residual_buffers,
                        shuffled_residual_buffers,
                        variant=variant,
                        max_norm=config.max_grad_norm,
                    )
                    global_step += 1
                    audit.update(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "microbatches_in_window": (
                                config.gradient_accumulation_steps
                                if step % config.gradient_accumulation_steps == 0
                                else step % config.gradient_accumulation_steps
                            ),
                        }
                    )
                    step_audit.append(audit)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    for buffer in aligned_residual_buffers.values():
                        buffer.zero_()
                    for buffer in shuffled_residual_buffers.values():
                        buffer.zero_()
                    if global_step % config.log_steps == 0 or global_step == total_steps:
                        elapsed = time.time() - start
                        progress.render(
                            global_step,
                            epoch,
                            total_running / max(1, step),
                            scheduler.get_last_lr()[0],
                            elapsed,
                            max(0, total_steps - global_step) * elapsed / max(1, global_step),
                        )
            progress.newline()
            result = evaluate(model, dev_loader, device, "dev")
            metrics = {
                **result.metrics,
                "epoch": epoch,
                "global_step": global_step,
                "train_hard_loss": hard_running / len(train_loader),
                "train_reported_aux_soft_ce": aux_running / len(train_loader),
                "train_optimization_loss": total_running / len(train_loader),
                "maximum_component_norm_relative_error_so_far": max(
                    float(row["component_norm_relative_error"]) for row in step_audit
                ),
                "maximum_clip_coefficient_relative_error_so_far": max(
                    float(row["clip_coefficient_relative_error"]) for row in step_audit
                ),
            }
            history.append(metrics)
            write_json(config.output_dir / "dev_metrics_history.json", history)
            write_csv(config.output_dir / "tables" / "dev_metrics_history.csv", history)
            write_json(config.output_dir / "geometry_step_audit.json", step_audit)
            print(
                f"[exp60:{variant}] epoch={epoch} Exact={metrics['Exact_rounded']:.6f} "
                f"MAE={metrics['MAE_human_mean']:.6f} Kendall={metrics['Kendall_human_mean']:.6f} "
                f"max_norm_match_error={metrics['maximum_component_norm_relative_error_so_far']:.3e}",
                flush=True,
            )
            candidate_exact = float(metrics["Exact_rounded"])
            if candidate_exact > best_exact:
                best_exact = candidate_exact
                best_epoch = epoch
                best_metrics = dict(metrics)
                save_checkpoint(
                    model,
                    tokenizer,
                    best_dir,
                    config,
                    metrics,
                    initial_contract,
                    variant,
                )
                save_predictions(config.output_dir, result, suffix="dev_best")
            if epoch == 10:
                epoch10_result = result
                save_checkpoint(
                    model,
                    tokenizer,
                    epoch10_dir,
                    config,
                    metrics,
                    initial_contract,
                    variant,
                )
                save_predictions(config.output_dir, result, suffix="dev_epoch10")
    finally:
        progress.close()
    if best_epoch is None or best_metrics is None or epoch10_result is None:
        raise RuntimeError("Exp60 requires both a secondary best checkpoint and fixed epoch 10")
    final = epoch10_result
    selected = {
        **final.metrics,
        "epoch": 10,
        "global_step": int(history[-1]["global_step"]),
    }
    save_predictions(config.output_dir, final)
    write_json(config.output_dir / "selected_dev_metrics.json", selected)
    write_json(config.output_dir / "secondary_best_dev_exact_metrics.json", best_metrics)
    write_csv(
        config.output_dir / "tables" / "selected_dev_metrics.csv",
        [{"variant": variant, "seed": config.seed, **selected}],
    )
    write_json(config.output_dir / "training_trace_first64.json", {"trace": trace})
    write_json(config.output_dir / "geometry_step_audit.json", step_audit)
    elapsed = time.time() - start
    summary = {
        "status": "COMPLETED",
        "variant": variant,
        "seed": config.seed,
        "runtime_head": git_head(),
        "model_mode": model_mode,
        "model_name_or_path": config.model_name_or_path,
        "initial_head_contract": initial_contract,
        "selected_epoch": 10,
        "selected_metrics": selected,
        "checkpoint_path": str(epoch10_dir),
        "checkpoint_rule": "fixed epoch 10 primary",
        "secondary_best_dev_exact_epoch": best_epoch,
        "secondary_best_dev_exact_metrics": best_metrics,
        "inference": "hard_head_raw_logit_argmax",
        "preclip_gradient": {
            "consensus_only": "G_C",
            "aligned_orthogonal_only": "G_C + O_A",
            "matched_shuffled_orthogonal_only": "G_C + O_pi_tilde",
        }[variant],
        "clipping": "standard global clip_grad_norm_ with max_norm=1.0",
        "optimizer_steps": global_step,
        "maximum_aligned_normalized_orthogonality_error": max(
            float(row["aligned_normalized_orthogonality_error"]) for row in step_audit
        ),
        "maximum_shuffled_normalized_orthogonality_error": max(
            float(row["shuffled_normalized_orthogonality_error"]) for row in step_audit
        ),
        "maximum_component_norm_relative_error": max(
            float(row["component_norm_relative_error"]) for row in step_audit
        ),
        "maximum_preclip_total_norm_relative_error": max(
            float(row["preclip_total_norm_relative_error"]) for row in step_audit
        ),
        "maximum_clip_coefficient_relative_error": max(
            float(row["clip_coefficient_relative_error"]) for row in step_audit
        ),
        "mapping_sha256": mapping_audit["mapping_sha256"],
        "mapping_effective_change_rate": mapping_audit["effective_change_rate"],
        "train_text_hash": aggregate_text_hash(train_rows),
        "dev_text_hash": aggregate_text_hash(dev_rows),
        "frozen_contract_files": contract_files,
        "test_access_count": 0,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(config.output_dir / "run_summary.json", summary)
    write_text(
        config.output_dir / "run_summary.md",
        f"# Exp60 {variant} seed {config.seed}\n\n"
        "- Primary checkpoint: fixed epoch 10\n"
        f"- Secondary best-Exact epoch: {best_epoch}\n"
        f"- Exact: {selected['Exact_rounded']:.6f}\n"
        f"- MAE: {selected['MAE_human_mean']:.6f}\n"
        f"- Kendall: {selected['Kendall_human_mean']:.6f}\n"
        f"- Maximum aligned orthogonality error: {summary['maximum_aligned_normalized_orthogonality_error']:.6e}\n"
        f"- Maximum shuffled orthogonality error: {summary['maximum_shuffled_normalized_orthogonality_error']:.6e}\n"
        f"- Maximum component-norm match error: {summary['maximum_component_norm_relative_error']:.6e}\n"
        f"- Maximum clip-coefficient match error: {summary['maximum_clip_coefficient_relative_error']:.6e}\n"
        f"- Runtime: {format_duration(elapsed)}\n"
        "- Test accessed: no\n",
    )
    return summary


def main() -> None:
    config, variant = parse_args()
    result = train(config, variant)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
