"""Real-model, no-update preflight for the Exp58 matched gradient control."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows, write_json
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.preflight import config
from thesis_exp.exp57_cbrd.train import load_model
from thesis_exp.exp58_matched_clipping import OUTPUT_ROOT
from thesis_exp.exp58_matched_clipping.matched_update import (
    clip_coefficient,
    largest_safe_beta,
)
from thesis_exp.exp58_matched_clipping.train import (
    _accumulate_residual_vjp,
    _capture_rng,
    _restore_rng,
    compose_matched_step,
)


TRACE_PATH = (
    OUTPUT_ROOT.parent
    / "exp57_cbrd"
    / "runs"
    / "consensus_only"
    / "seed_42"
    / "training_trace_first64.json"
)


def parameter_group(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("hard_head."):
        return "hard_head"
    if name.startswith("soft_head."):
        return "soft_head"
    return "other"


def squared_norm(values: dict[str, Any], names: set[str] | None = None) -> float:
    selected = values.keys() if names is None else names
    return sum(float(values[name].double().square().sum()) for name in selected)


def relative_error(left: dict[str, Any], right: dict[str, Any], names: set[str]) -> float:
    numerator = sum(
        float((left[name].double() - right[name].double()).square().sum())
        for name in names
    )
    denominator = sum(float(right[name].double().square().sum()) for name in names)
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12)


def run(
    model_path: str,
    *,
    microbatches: int = 32,
    precision: str = "bf16",
) -> dict[str, Any]:
    import torch

    from thesis_exp.src.edujudge.exp02.train_ce_baseline import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if microbatches != 32:
        raise ValueError("Formal preflight requires a complete 32-microbatch window")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))["trace"][:microbatches]
    if len(trace) != microbatches:
        raise RuntimeError("Incomplete frozen trace")
    lookup = {str(row["record_id"]): row for row in model_rows("train")}
    raw_batches = [
        [lookup[str(record_id)] for record_id in row["record_ids"]]
        for row in trace
    ]
    set_seed(42)
    cfg = config(model_path, bf16="true" if precision == "bf16" else "false")
    cfg.gradient_checkpointing = True
    model, tokenizer, model_mode, head_contract = load_model(cfg)
    device = torch.device("cuda")
    model.to(device)
    model.gradient_checkpointing_enable()
    model.train()
    batches = []
    for batch in raw_batches:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        targets = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        batches.append((encoded, labels, targets))
    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    names = set(named)

    def capture(route: str) -> dict[str, Any]:
        set_seed(42)
        model.zero_grad(set_to_none=True)
        for encoded_cpu, labels_cpu, targets_cpu in batches:
            encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
            labels = labels_cpu.to(device)
            targets = targets_cpu.to(device)
            outputs = model(
                **encoded,
                labels=labels,
                soft_targets=targets,
                aux_route=route,
                route_loss_scale=1.0 / 32.0,
            )
            objective = cbrd_objective(outputs, labels, targets, variant=route)
            loss = (
                objective["reported_aux_soft_ce"]
                if route == "residual_only"
                else objective["optimization_loss"]
            )
            (loss / 32.0).backward()
        return {
            name: (
                parameter.grad.detach().float().cpu().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float32)
            )
            for name, parameter in named.items()
        }

    common = capture("consensus_only")
    routed = capture("routed_hmsa")
    residual_raw = capture("residual_only")
    residual = {
        name: (
            residual_raw[name]
            if parameter_group(name) == "backbone"
            else torch.zeros_like(residual_raw[name])
        )
        for name in names
    }
    difference = {name: routed[name] - common[name] for name in names}
    groups = {
        group: {name for name in names if parameter_group(name) == group}
        for group in ("backbone", "hard_head", "soft_head", "other")
    }
    groups["all"] = names
    decomposition_errors = {
        group: relative_error(difference, residual, group_names)
        for group, group_names in groups.items()
        if group_names
    }
    difference_group_norms = {
        group: math.sqrt(squared_norm(difference, group_names))
        for group, group_names in groups.items()
        if group_names
    }
    difference_residual_dot = sum(
        float((difference[name].double() * residual[name].double()).sum())
        for name in names
    )
    common_sq = squared_norm(common)
    routed_sq = squared_norm(routed)
    residual_sq = squared_norm(residual)
    dot = sum(float((common[name].double() * residual[name].double()).sum()) for name in names)
    alpha_common = clip_coefficient(math.sqrt(common_sq))
    alpha_routed = clip_coefficient(math.sqrt(routed_sq))
    beta, beta_cap_active = largest_safe_beta(
        common_sq=common_sq,
        residual_sq=residual_sq,
        common_residual_dot=dot,
        alpha_common=alpha_common,
        alpha_routed=alpha_routed,
    )
    standard_direct = {name: alpha_routed * routed[name] for name in names}
    standard_reconstructed = {
        name: alpha_routed * (common[name] + residual[name]) for name in names
    }
    standard_reconstruction_error = relative_error(
        standard_reconstructed, standard_direct, names
    )
    expected_matched = {
        name: alpha_common * common[name] + beta * residual[name]
        for name in names
    }
    expected_matched_norm = math.sqrt(squared_norm(expected_matched))

    set_seed(42)
    model.zero_grad(set_to_none=True)
    residual_buffers = {
        name: torch.zeros_like(parameter, device=device, dtype=torch.float32)
        for name, parameter in named.items()
        if name.startswith("backbone.")
    }
    for encoded_cpu, labels_cpu, targets_cpu in batches:
        inputs = {key: value.to(device) for key, value in encoded_cpu.items()}
        labels = labels_cpu.to(device)
        targets = targets_cpu.to(device)
        rng_before = _capture_rng(torch, device)
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="routed_hmsa",
            route_loss_scale=1.0 / 32.0,
        )
        objective = cbrd_objective(outputs, labels, targets, variant="routed_hmsa")
        (objective["optimization_loss"] / 32.0).backward()
        rng_after = _capture_rng(torch, device)
        _restore_rng(torch, device, rng_before)
        _accumulate_residual_vjp(
            model,
            inputs,
            labels,
            targets,
            residual_buffers,
            1.0 / 32.0,
        )
        _restore_rng(torch, device, rng_after)
    paired_audit = compose_matched_step(model, residual_buffers, max_norm=1.0)
    actual_matched = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in named.items()
    }
    paired_matched_relative_error = relative_error(
        actual_matched, expected_matched, names
    )
    difference_norm = difference_group_norms["all"]
    residual_norm = math.sqrt(residual_sq)
    residual_direction_cosine = (
        difference_residual_dot / (difference_norm * residual_norm)
        if difference_norm > 0.0 and residual_norm > 0.0
        else 1.0
    )

    def norm_relative_error(left: float, right: float) -> float:
        return abs(left - right) / max(abs(right), 1e-12)

    component_norm_relative_errors = {
        "common": norm_relative_error(paired_audit["common_norm"], math.sqrt(common_sq)),
        "routed": norm_relative_error(paired_audit["routed_norm"], math.sqrt(routed_sq)),
        "residual": norm_relative_error(paired_audit["residual_norm"], residual_norm),
    }
    paired_post_cast_limit = 1.01171875 if precision == "bf16" else 1.000001
    invariant_checks = {
        "hard_head_difference_norm_at_most_1e_12": difference_group_norms["hard_head"] <= 1e-12,
        "soft_head_difference_norm_at_most_1e_12": difference_group_norms["soft_head"] <= 1e-12,
        "expected_matched_norm_at_most_1_000001": expected_matched_norm <= 1.000001,
        "paired_expected_fp32_norm_at_most_1_000001": paired_audit["matched_update_expected_fp32_norm"] <= 1.000001,
        "paired_actual_norm_within_precision_storage_limit": paired_audit["matched_update_norm"] <= paired_post_cast_limit,
        "cublas_workspace_config_recorded": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "no_optimizer_step": True,
        "no_test_access": True,
    }
    if precision == "fp32":
        precision_checks = {
            "fp32_global_decomposition_relative_error_at_most_0_02": decomposition_errors["all"] <= 0.02,
            "fp32_standard_post_clip_reconstruction_error_at_most_1e_4": standard_reconstruction_error <= 1e-4,
            "fp32_paired_trainer_matches_independent_construction_at_most_1e_4": paired_matched_relative_error <= 1e-4,
        }
    else:
        bf16_tolerance = 1.5 * (2.0 ** -7)
        precision_checks = {
            "bf16_residual_direction_cosine_at_least_0_9": residual_direction_cosine >= 0.9,
            "bf16_component_norm_relative_errors_at_most_5e_4": max(component_norm_relative_errors.values()) <= 5e-4,
            "bf16_standard_post_clip_reconstruction_error_at_most_1_5_unit_roundoff": standard_reconstruction_error <= bf16_tolerance,
            "bf16_paired_trainer_matches_independent_construction_at_most_1_5_unit_roundoff": paired_matched_relative_error <= bf16_tolerance,
        }
    checks = {**invariant_checks, **precision_checks}
    report = {
        "status": "EXP58_PREFLIGHT_PASS" if all(checks.values()) else "EXP58_PREFLIGHT_FAIL",
        "checks": checks,
        "model_mode": model_mode,
        "head_contract": head_contract,
        "precision": precision,
        "microbatches": microbatches,
        "examples": microbatches * 4,
        "decomposition_relative_errors": decomposition_errors,
        "residual_direction_cosine": residual_direction_cosine,
        "component_norm_relative_errors": component_norm_relative_errors,
        "difference_group_norms": difference_group_norms,
        "common_norm": math.sqrt(common_sq),
        "routed_norm": math.sqrt(routed_sq),
        "residual_norm": residual_norm,
        "alpha_C": alpha_common,
        "alpha_R": alpha_routed,
        "beta": beta,
        "beta_cap_active": beta_cap_active,
        "standard_post_clip_reconstruction_relative_error": standard_reconstruction_error,
        "expected_matched_norm": expected_matched_norm,
        "paired_matched_relative_error": paired_matched_relative_error,
        "paired_audit": paired_audit,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    output = (
        OUTPUT_ROOT
        / "audit"
        / f"preflight_real_qwen3_full_accumulation_{precision}_v3.json"
    )
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--microbatches", type=int, default=32)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    result = run(
        args.model_name_or_path,
        microbatches=args.microbatches,
        precision=args.precision,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "EXP58_PREFLIGHT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
