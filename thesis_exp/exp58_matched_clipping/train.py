"""Frozen train/dev-only trainer for the Exp58 common-scale-matched control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

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
from thesis_exp.exp58_matched_clipping import (
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REPO_ROOT,
    SOURCE_LOCK_PATH,
)
from thesis_exp.exp58_matched_clipping.matched_update import (
    clip_coefficient,
    largest_safe_beta,
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


VARIANT = "common_scale_matched_routed"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Frozen Exp58 matched-clipping trainer")
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
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44, 45, 46))
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
    output_dir = args.output_dir or OUTPUT_ROOT / "runs" / VARIANT / f"seed_{args.seed}"
    checkpoint_dir = args.checkpoint_output_dir or (
        REPO_ROOT / "thesis_exp" / "artifacts" / "exp58_matched_clipping" / VARIANT / f"seed_{args.seed}"
    )
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP58_IN_MEMORY_TRAIN_DEV_ONLY"),
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


def verify_contract(*, require_source_lock: bool) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP58_MATCHED_CLIPPING_PROTOCOL_V3_FROZEN_AFTER_SEED42_IMPLEMENTATION_STOP":
        raise RuntimeError("Exp58 protocol is not frozen")
    if protocol.get("allowed_splits") != ["train", "dev"] or protocol.get("test_access_count") != 0:
        raise RuntimeError("Invalid Exp58 split contract")
    lock = None
    if SOURCE_LOCK_PATH.is_file():
        lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
        for relative, expected in lock["files"].items():
            actual = sha256_file(REPO_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"Exp58 source-lock mismatch: {relative}: {actual} != {expected}")
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


def compose_matched_step(
    model: Any,
    residual_buffers: dict[str, Any],
    *,
    max_norm: float,
) -> dict[str, Any]:
    """Replace accumulated Routed gradients with the frozen matched update."""

    import torch

    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not named:
        raise RuntimeError("No accumulated gradients")
    device = next(iter(named.values())).grad.device
    common_values: list[Any] = []
    routed_values: list[Any] = []
    residual_values: list[Any] = []
    common_residual_dot = torch.zeros((), device=device, dtype=torch.float32)
    common_cache: dict[str, Any] = {}
    for name, parameter in named.items():
        routed = parameter.grad.detach().float()
        residual = residual_buffers.get(name)
        if residual is None:
            residual = torch.zeros_like(routed)
        common = routed - residual
        common_cache[name] = common
        routed_values.append(routed)
        residual_values.append(residual)
        common_values.append(common)
        common_residual_dot = common_residual_dot + (common * residual).sum()
    common_sq = float(_sum_squares(torch, common_values, device).detach().cpu())
    routed_sq = float(_sum_squares(torch, routed_values, device).detach().cpu())
    residual_sq = float(_sum_squares(torch, residual_values, device).detach().cpu())
    dot = float(common_residual_dot.detach().cpu())
    common_norm = math.sqrt(max(0.0, common_sq))
    routed_norm = math.sqrt(max(0.0, routed_sq))
    residual_norm = math.sqrt(max(0.0, residual_sq))
    alpha_common = clip_coefficient(common_norm, max_norm=max_norm)
    alpha_routed = clip_coefficient(routed_norm, max_norm=max_norm)
    beta, cap_active = largest_safe_beta(
        common_sq=common_sq,
        residual_sq=residual_sq,
        common_residual_dot=dot,
        alpha_common=alpha_common,
        alpha_routed=alpha_routed,
        max_norm=max_norm,
    )
    expected_matched_sq = torch.zeros((), device=device, dtype=torch.float32)
    for name, parameter in named.items():
        residual = residual_buffers.get(name)
        if residual is None:
            residual = torch.zeros_like(common_cache[name])
        matched = alpha_common * common_cache[name] + beta * residual
        expected_matched_sq = expected_matched_sq + matched.square().sum()
        parameter.grad.copy_(matched.to(dtype=parameter.grad.dtype))
    expected_matched_norm = math.sqrt(
        max(0.0, float(expected_matched_sq.detach().cpu()))
    )
    if (
        not math.isfinite(expected_matched_norm)
        or expected_matched_norm > max_norm + 1e-6
    ):
        raise RuntimeError(
            f"Constructed FP32 matched update norm violation: {expected_matched_norm}"
        )
    actual_sq = _sum_squares(
        torch,
        [parameter.grad for parameter in named.values()],
        device,
    )
    actual_norm = math.sqrt(max(0.0, float(actual_sq.detach().cpu())))
    gradient_dtype = str(next(iter(named.values())).grad.dtype)
    storage_relative_tolerance = post_cast_relative_tolerance(gradient_dtype)
    post_cast_limit = max_norm * (1.0 + storage_relative_tolerance)
    if not math.isfinite(actual_norm) or actual_norm > post_cast_limit:
        raise RuntimeError(f"BF16-stored matched update norm violation: {actual_norm}")
    return {
        "common_norm": common_norm,
        "routed_norm": routed_norm,
        "residual_norm": residual_norm,
        "common_residual_dot": dot,
        "common_residual_cosine": (
            dot / (common_norm * residual_norm)
            if common_norm > 0.0 and residual_norm > 0.0
            else None
        ),
        "alpha_C": alpha_common,
        "alpha_R": alpha_routed,
        "beta": beta,
        "beta_over_alpha_R": beta / alpha_routed if alpha_routed > 0.0 else None,
        "beta_cap_active": cap_active,
        "matched_update_expected_fp32_norm": expected_matched_norm,
        "matched_update_norm": actual_norm,
        "matched_update_post_cast_norm_delta": actual_norm - expected_matched_norm,
        "gradient_storage_dtype": gradient_dtype,
        "post_cast_relative_tolerance": storage_relative_tolerance,
        "post_cast_norm_limit": post_cast_limit,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if config.evaluate_test:
        raise PermissionError("Exp58 must never evaluate test")
    if config.gradient_accumulation_steps != 32 or config.max_grad_norm != 1.0:
        raise RuntimeError("Exp58 accumulation and clipping must match the frozen protocol")
    contract_files = verify_contract(
        require_source_lock=os.environ.get("EXP58_REQUIRE_SOURCE_LOCK") == "1"
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit(model_rows("train"), config.max_train_samples)
    dev_rows = limit(model_rows("dev"), config.max_eval_samples)
    model, tokenizer, model_mode, initial_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal Exp58 training requires CUDA")
    model.to(device)
    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * updates_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_dir = config.checkpoint_output_dir / "best"
    best_exact = -1.0
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    step_audit: list[dict[str, Any]] = []
    named_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    }
    global_step = micro_step = cap_events = 0
    start = time.time()
    progress = StepProgressBar(total_steps, config.progress_bar, desc="Exp58 matched")
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
                targets = torch.tensor(
                    [row["soft_target_5"] for row in metadata],
                    dtype=torch.float32,
                    device=device,
                )
                loss_scale = 1.0 / float(config.gradient_accumulation_steps)
                rng_before = _capture_rng(torch, device)
                outputs = model(
                    **inputs,
                    labels=labels,
                    soft_targets=targets,
                    aux_route="routed_hmsa",
                    route_loss_scale=loss_scale,
                )
                objective = cbrd_objective(
                    outputs,
                    labels,
                    targets,
                    variant="routed_hmsa",
                )
                loss = objective["optimization_loss"]
                (loss * loss_scale).backward()
                rng_after = _capture_rng(torch, device)
                _restore_rng(torch, device, rng_before)
                diagnostic_aux = _accumulate_residual_vjp(
                    model,
                    inputs,
                    labels,
                    targets,
                    residual_buffers,
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
                            "diagnostic_aux_soft_ce": diagnostic_aux,
                            "optimization_loss": total_value,
                            "route_loss_scale": loss_scale,
                        }
                    )
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    audit = compose_matched_step(
                        model,
                        residual_buffers,
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
                    cap_events += int(audit["beta_cap_active"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    for buffer in residual_buffers.values():
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
                "matched_beta_cap_fraction_so_far": cap_events / max(1, global_step),
            }
            history.append(metrics)
            write_json(config.output_dir / "dev_metrics_history.json", history)
            write_csv(config.output_dir / "tables" / "dev_metrics_history.csv", history)
            write_json(config.output_dir / "matched_step_audit.json", step_audit)
            print(
                f"[exp58:matched] epoch={epoch} Exact={metrics['Exact_rounded']:.6f} "
                f"MAE={metrics['MAE_human_mean']:.6f} Kendall={metrics['Kendall_human_mean']:.6f} "
                f"cap_fraction={metrics['matched_beta_cap_fraction_so_far']:.6f}",
                flush=True,
            )
            candidate_exact = float(metrics["Exact_rounded"])
            if candidate_exact > best_exact:
                best_exact = candidate_exact
                best_epoch = epoch
                save_checkpoint(
                    model,
                    tokenizer,
                    best_dir,
                    config,
                    metrics,
                    initial_contract,
                    VARIANT,
                )
                save_predictions(config.output_dir, result, suffix="dev_best")
    finally:
        progress.close()
    if best_epoch is None:
        raise RuntimeError("No Exp58 checkpoint selected")
    model.load_state_dict(
        torch.load(best_dir / "state_dict.pt", map_location=device, weights_only=True)
    )
    final = evaluate(model, dev_loader, device, "dev")
    selected = {
        **final.metrics,
        "epoch": best_epoch,
        "global_step": next(
            int(row["global_step"]) for row in history if int(row["epoch"]) == best_epoch
        ),
    }
    save_predictions(config.output_dir, final)
    write_json(config.output_dir / "selected_dev_metrics.json", selected)
    write_csv(
        config.output_dir / "tables" / "selected_dev_metrics.csv",
        [{"variant": VARIANT, "seed": config.seed, **selected}],
    )
    write_json(config.output_dir / "training_trace_first64.json", {"trace": trace})
    write_json(config.output_dir / "matched_step_audit.json", step_audit)
    elapsed = time.time() - start
    cap_fraction = cap_events / max(1, global_step)
    summary = {
        "status": "COMPLETED",
        "variant": VARIANT,
        "seed": config.seed,
        "runtime_head": git_head(),
        "model_mode": model_mode,
        "model_name_or_path": config.model_name_or_path,
        "initial_head_contract": initial_contract,
        "selected_epoch": best_epoch,
        "selected_metrics": selected,
        "checkpoint_path": str(best_dir),
        "checkpoint_rule": "highest hard-head dev Exact_rounded; ties keep earlier epoch",
        "inference": "hard_head_raw_logit_argmax",
        "matched_update": "alpha_C * G_C + beta * R",
        "beta_cap_events": cap_events,
        "optimizer_steps": global_step,
        "beta_cap_fraction": cap_fraction,
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
        f"# Exp58 matched seed {config.seed}\n\n"
        f"- Selected epoch: {best_epoch}\n"
        f"- Exact: {selected['Exact_rounded']:.6f}\n"
        f"- MAE: {selected['MAE_human_mean']:.6f}\n"
        f"- Kendall: {selected['Kendall_human_mean']:.6f}\n"
        f"- Beta cap fraction: {cap_fraction:.6f}\n"
        f"- Runtime: {format_duration(elapsed)}\n"
        "- Test accessed: no\n",
    )
    return summary


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
