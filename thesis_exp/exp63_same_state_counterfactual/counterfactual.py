"""Apply four one-step counterfactual updates from each frozen Exp63 state."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.train import evaluate, load_model
from thesis_exp.exp59_residual_geometry.train import (
    _accumulate_residual_vjp,
    _capture_rng,
    _restore_rng,
)
from thesis_exp.exp63_same_state_counterfactual import (
    ARMS,
    ARTIFACT_ROOT,
    OUTPUT_ROOT,
    SEEDS,
    STAGE_EPOCHS,
)
from thesis_exp.exp63_same_state_counterfactual.runtime import (
    hash_strings,
    load_protocol,
    ordered_rows,
    sha256_file,
    write_json,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig, make_dataloader, set_seed


def parse_args() -> tuple[int, str, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--artifact_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir or ARTIFACT_ROOT / f"seed_{args.seed}"
    output_dir = args.output_dir or OUTPUT_ROOT / "counterfactual" / f"seed_{args.seed}"
    return args.seed, args.model_name_or_path, artifact_dir, output_dir


def config_for(seed: int, model_name_or_path: str, output_dir: Path, artifact_dir: Path) -> TrainConfig:
    return TrainConfig(
        model_name_or_path=model_name_or_path,
        data_dir=Path("EXP63_TRAIN_DEV_ONLY"),
        output_dir=output_dir,
        checkpoint_output_dir=artifact_dir,
        max_length=2048,
        num_train_epochs=10.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=seed,
        bf16="true",
        fp16=False,
        gradient_checkpointing=True,
        trust_remote_code=False,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=5,
        progress_bar=False,
        evaluate_test=False,
    )


def scalar_geometry(
    common: dict[str, Any], residual: dict[str, Any]
) -> dict[str, float]:
    common_backbone_sq = residual_sq = dot = 0.0
    for name, common_value in common.items():
        if not name.startswith("backbone."):
            continue
        common_double = common_value.double()
        residual_double = residual[name].double()
        common_backbone_sq += float(common_double.square().sum())
        residual_sq += float(residual_double.square().sum())
        dot += float((common_double * residual_double).sum())
    coefficient = dot / common_backbone_sq if common_backbone_sq > 0.0 else 0.0
    parallel_sq = orthogonal_sq = orthogonal_dot = reconstruction_sq = 0.0
    for name, common_value in common.items():
        if not name.startswith("backbone."):
            continue
        common_double = common_value.double()
        residual_double = residual[name].double()
        parallel = coefficient * common_double
        orthogonal = residual_double - parallel
        parallel_sq += float(parallel.square().sum())
        orthogonal_sq += float(orthogonal.square().sum())
        orthogonal_dot += float((orthogonal * common_double).sum())
        reconstruction_sq += float((residual_double - parallel - orthogonal).square().sum())
    common_norm = math.sqrt(max(0.0, common_backbone_sq))
    residual_norm = math.sqrt(max(0.0, residual_sq))
    orthogonal_norm = math.sqrt(max(0.0, orthogonal_sq))
    return {
        "projection_coefficient": coefficient,
        "common_backbone_norm": common_norm,
        "residual_norm": residual_norm,
        "parallel_norm": math.sqrt(max(0.0, parallel_sq)),
        "orthogonal_norm": orthogonal_norm,
        "residual_common_cosine": dot / (common_norm * residual_norm + 1e-30),
        "reconstruction_relative_error": math.sqrt(max(0.0, reconstruction_sq))
        / max(residual_norm, 1e-30),
        "normalized_orthogonality_error": abs(orthogonal_dot)
        / (orthogonal_norm * common_norm + 1e-30),
    }


def candidate_tensor(
    name: str,
    arm: str,
    common_value: Any,
    residual_value: Any | None,
    coefficient: float,
) -> Any:
    if residual_value is None or not name.startswith("backbone.") or arm == "blocked":
        return common_value
    if arm == "full_residual":
        return common_value + residual_value
    if arm == "parallel_only":
        return common_value + coefficient * common_value
    if arm == "orthogonal_only":
        return common_value + residual_value - coefficient * common_value
    raise ValueError(arm)


def candidate_raw_norm(
    common: dict[str, Any], residual: dict[str, Any], arm: str, coefficient: float
) -> float:
    total = 0.0
    for name, common_value in common.items():
        value = candidate_tensor(name, arm, common_value, residual.get(name), coefficient)
        total += float(value.double().square().sum())
    return math.sqrt(max(0.0, total))


def collect_components(
    model: Any,
    window_loader: Any,
    device: Any,
    accumulation_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    named_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    }
    model.train()
    model.zero_grad(set_to_none=True)
    record_ids: list[str] = []
    common_losses: list[float] = []
    residual_losses: list[float] = []
    for batch in window_loader:
        metadata = batch.pop("metadata")
        record_ids.extend(str(row["record_id"]) for row in metadata)
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        targets = torch.tensor(
            [row["soft_target_5"] for row in metadata], dtype=torch.float32, device=device
        )
        scale = 1.0 / accumulation_steps
        before = _capture_rng(torch, device)
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="consensus_only",
            route_loss_scale=scale,
        )
        objective = cbrd_objective(outputs, labels, targets, variant="consensus_only")
        (objective["optimization_loss"] * scale).backward()
        common_losses.append(float(objective["optimization_loss"].detach().cpu()))
        after = _capture_rng(torch, device)
        _restore_rng(torch, device, before)
        residual_losses.append(
            _accumulate_residual_vjp(
                model, inputs, labels, targets, residual_buffers, scale
            )
        )
        _restore_rng(torch, device, after)
    common = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    residual = {name: value.detach().cpu().clone() for name, value in residual_buffers.items()}
    model.zero_grad(set_to_none=True)
    del residual_buffers
    torch.cuda.empty_cache()
    return common, residual, {
        "record_ids": record_ids,
        "record_ids_sha256": hash_strings(record_ids),
        "mean_common_objective": sum(common_losses) / len(common_losses),
        "mean_residual_aux_ce": sum(residual_losses) / len(residual_losses),
    }


def assign_matched_gradient(
    model: Any,
    common: dict[str, Any],
    residual: dict[str, Any],
    arm: str,
    coefficient: float,
    target_norm: float,
    clip_threshold: float,
) -> dict[str, float]:
    import torch

    raw_norm = candidate_raw_norm(common, residual, arm, coefficient)
    if raw_norm < target_norm:
        raise RuntimeError(f"Raw norm {raw_norm} is below frozen target {target_norm}")
    scale = target_norm / raw_norm
    model.zero_grad(set_to_none=True)
    parameters = []
    for name, parameter in model.named_parameters():
        if name not in common:
            continue
        value = candidate_tensor(name, arm, common[name], residual.get(name), coefficient)
        parameter.grad = (value * scale).to(device=parameter.device, dtype=parameter.dtype)
        parameters.append(parameter)
    # A scalar recalibration after BF16 storage makes the four realized norms
    # equal without changing any candidate direction.
    realized = float(torch.nn.utils.clip_grad_norm_(parameters, float("inf")).detach().cpu())
    correction = target_norm / realized
    for parameter in parameters:
        parameter.grad.mul_(correction)
    realized = float(torch.nn.utils.clip_grad_norm_(parameters, float("inf")).detach().cpu())
    preclip = float(
        torch.nn.utils.clip_grad_norm_(parameters, clip_threshold).detach().cpu()
    )
    postclip = float(torch.nn.utils.clip_grad_norm_(parameters, float("inf")).detach().cpu())
    return {
        "raw_fp64_norm": raw_norm,
        "initial_scale": scale,
        "bf16_recalibration": correction,
        "matched_preclip_norm": preclip,
        "postclip_norm": postclip,
        "effective_clip_coefficient": min(1.0, clip_threshold / max(preclip, 1e-30)),
        "realized_before_clip": realized,
    }


def relevant_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "eval_hard_ce_loss",
        "MAE_human_mean",
        "Exact_rounded",
        "Kendall_human_mean",
        "QWK_rounded",
        "Macro_F1",
    )
    return {key: metrics.get(key) for key in keys}


def run_stage(
    *,
    seed: int,
    stage_epoch: int,
    checkpoint_path: Path,
    config: TrainConfig,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    protocol = load_protocol()
    target_norm = float(protocol["fairness_controls"]["full_parameter_target_gradient_norm"])
    clip_threshold = float(protocol["fairness_controls"]["common_global_clip_threshold"])
    checkpoint_sha = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["seed"] != seed or payload["epoch"] != stage_epoch:
        raise RuntimeError("Checkpoint identity mismatch")
    next_rows = ordered_rows(train_rows, seed, stage_epoch + 1)
    window_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    window_rows = next_rows[:window_size]
    window_ids = [str(row["record_id"]) for row in window_rows]
    if window_ids != payload["next_window_record_ids"]:
        raise RuntimeError("Counterfactual window differs from the frozen checkpoint contract")
    set_seed(seed)
    model, tokenizer, _, _ = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.gradient_checkpointing_enable()
    model.load_state_dict(payload["model"], strict=True)
    window_loader = make_dataloader(window_rows, tokenizer, config, "train", shuffle=False)
    common, residual, window_audit = collect_components(
        model, window_loader, device, config.gradient_accumulation_steps
    )
    geometry = scalar_geometry(common, residual)
    gates = protocol["implementation_gates"]
    if geometry["reconstruction_relative_error"] > gates["residual_reconstruction_relative_error_at_most"]:
        raise RuntimeError("Residual reconstruction gate failed")
    if geometry["normalized_orthogonality_error"] > gates["normalized_orthogonality_error_at_most"]:
        raise RuntimeError("Residual orthogonality gate failed")
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    model.load_state_dict(payload["model"], strict=True)
    before = relevant_metrics(evaluate(model, dev_loader, device, "dev").metrics)
    arm_results: dict[str, Any] = {}
    total_steps = 10 * math.ceil(
        math.ceil(len(train_rows) / config.per_device_train_batch_size)
        / config.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    for arm in ARMS:
        model.load_state_dict(payload["model"], strict=True)
        optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        # Construct the scheduler before restoring either state.  LambdaLR's
        # constructor writes its step-zero learning rate into the optimizer;
        # constructing it after optimizer.load_state_dict would silently turn
        # the counterfactual step into a zero-LR no-op.
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
        scheduler.load_state_dict(copy.deepcopy(payload["scheduler"]))
        state_lr = float(optimizer.param_groups[0]["lr"])
        checkpoint_lr = float(payload["optimizer"]["param_groups"][0]["lr"])
        scheduler_lr = float(scheduler.get_last_lr()[0])
        if checkpoint_lr <= 0.0:
            raise RuntimeError(f"Checkpoint learning rate is not positive: {checkpoint_lr}")
        if abs(state_lr - checkpoint_lr) > 1e-15:
            raise RuntimeError(
                f"Optimizer LR restore failure: {state_lr} != {checkpoint_lr}"
            )
        if abs(scheduler_lr - checkpoint_lr) > 1e-15:
            raise RuntimeError(
                f"Scheduler LR restore failure: {scheduler_lr} != {checkpoint_lr}"
            )
        norm_audit = assign_matched_gradient(
            model,
            common,
            residual,
            arm,
            geometry["projection_coefficient"],
            target_norm,
            clip_threshold,
        )
        if abs(norm_audit["matched_preclip_norm"] - target_norm) > gates["matched_norm_absolute_error_at_most"]:
            raise RuntimeError(f"Matched-norm gate failed for {arm}: {norm_audit}")
        if norm_audit["postclip_norm"] > gates["post_clip_norm_at_most"]:
            raise RuntimeError(f"No-clipping gate failed for {arm}: {norm_audit}")
        optimizer.step()
        scheduler.step()
        model.zero_grad(set_to_none=True)
        post = relevant_metrics(evaluate(model, dev_loader, device, "dev").metrics)
        arm_results[arm] = {
            "learning_rate": state_lr,
            "checkpoint_learning_rate": checkpoint_lr,
            "scheduler_learning_rate": scheduler_lr,
            "learning_rate_restore_pass": True,
            "norm_audit": norm_audit,
            "post": post,
            "delta_from_pre": {
                key: (post[key] - before[key])
                for key in post
                if post[key] is not None and before[key] is not None
            },
        }
        print(
            f"[exp63:cf] seed={seed} stage={stage_epoch} arm={arm} "
            f"CE={post['eval_hard_ce_loss']:.9f} norm={norm_audit['postclip_norm']:.6f}",
            flush=True,
        )
        del optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    for arm in ARMS:
        arm_results[arm]["delta_CE_vs_blocked"] = (
            arm_results[arm]["post"]["eval_hard_ce_loss"]
            - arm_results["blocked"]["post"]["eval_hard_ce_loss"]
        )
    result = {
        "status": "COMPLETED",
        "seed": seed,
        "stage_epoch": stage_epoch,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_global_step": payload["global_step"],
        "next_window_epoch": stage_epoch + 1,
        "window": window_audit,
        "geometry": geometry,
        "pre_update_probe": before,
        "arms": arm_results,
        "state_contract": {
            "same_checkpoint_for_all_arms": True,
            "same_optimizer_state_for_all_arms": True,
            "same_scheduler_state_for_all_arms": True,
            "same_window_for_all_arms": True,
            "same_full_parameter_target_norm": target_norm,
            "same_clip_threshold": clip_threshold,
        },
        "test_access_count": 0,
    }
    write_json(output_dir / f"after_epoch_{stage_epoch}.json", result)
    del payload, model, common, residual
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    seed, model_name, artifact_dir, output_dir = parse_args()
    start = time.time()
    config = config_for(seed, model_name, output_dir, artifact_dir)
    train_rows = model_rows("train")
    dev_rows = model_rows("dev")
    results = []
    for stage_epoch in STAGE_EPOCHS:
        checkpoint = artifact_dir / f"after_epoch_{stage_epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        results.append(
            run_stage(
                seed=seed,
                stage_epoch=stage_epoch,
                checkpoint_path=checkpoint,
                config=config,
                train_rows=train_rows,
                dev_rows=dev_rows,
                output_dir=output_dir,
            )
        )
    summary = {
        "status": "COMPLETED",
        "seed": seed,
        "stage_results": results,
        "elapsed_seconds": time.time() - start,
        "test_access_count": 0,
    }
    write_json(output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
