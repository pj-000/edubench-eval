"""Real-model, no-update preflight for Exp59 residual geometry."""

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
from thesis_exp.exp59_residual_geometry import OUTPUT_ROOT
from thesis_exp.exp59_residual_geometry.geometry import (
    compose_preclip_gradient,
    decompose_residual,
)
from thesis_exp.exp59_residual_geometry.train import (
    _accumulate_residual_vjp,
    _capture_rng,
    _restore_rng,
    compose_geometry_step,
    verify_contract,
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

    contract = verify_contract(require_source_lock=True)

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
    endpoint_residual_errors = {
        group: relative_error(difference, residual, group_names)
        for group, group_names in groups.items()
        if group_names
    }
    parallel, orthogonal, geometry = decompose_residual(common, residual)
    reconstructed_residual = {
        name: parallel[name] + orthogonal[name] for name in names
    }
    reconstructed_full = compose_preclip_gradient(
        compose_preclip_gradient(common, parallel), orthogonal
    )
    full_route_reconstruction_error = relative_error(
        reconstructed_full, routed, names
    )
    component_head_norms = {
        f"{component_name}_{head}": math.sqrt(
            squared_norm(component, groups[head])
        )
        for component_name, component in (
            ("parallel", parallel),
            ("orthogonal", orthogonal),
        )
        for head in ("hard_head", "soft_head")
    }

    def clipped(values: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
        norm = math.sqrt(squared_norm(values))
        alpha = min(1.0, 1.0 / (norm + 1e-6))
        return {name: alpha * values[name] for name in names}, norm, alpha

    expected = {}
    for variant, component in (
        ("parallel_only", parallel),
        ("orthogonal_only", orthogonal),
    ):
        expected[variant] = clipped(compose_preclip_gradient(common, component))

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
    paired_routed = {
        name: parameter.grad.detach().clone() for name, parameter in named.items()
    }
    paired_audits = {}
    paired_relative_errors = {}
    for variant in ("parallel_only", "orthogonal_only"):
        for name, parameter in named.items():
            parameter.grad = paired_routed[name].clone()
        paired_audits[variant] = compose_geometry_step(
            model,
            residual_buffers,
            variant=variant,
            max_norm=1.0,
        )
        actual = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in named.items()
        }
        paired_relative_errors[variant] = relative_error(
            actual, expected[variant][0], names
        )
    representation_tolerance = 1.5 * (2.0 ** -7)
    invariant_checks = {
        "parallel_hard_head_norm_at_most_1e_12": component_head_norms["parallel_hard_head"] <= 1e-12,
        "parallel_soft_head_norm_at_most_1e_12": component_head_norms["parallel_soft_head"] <= 1e-12,
        "orthogonal_hard_head_norm_at_most_1e_12": component_head_norms["orthogonal_hard_head"] <= 1e-12,
        "orthogonal_soft_head_norm_at_most_1e_12": component_head_norms["orthogonal_soft_head"] <= 1e-12,
        "parallel_standard_clip_norm_safe": paired_audits["parallel_only"]["postclip_norm"] <= 1.0 + representation_tolerance,
        "orthogonal_standard_clip_norm_safe": paired_audits["orthogonal_only"]["postclip_norm"] <= 1.0 + representation_tolerance,
        "cublas_workspace_config_recorded": os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "no_optimizer_step": True,
        "no_test_access": True,
    }
    if precision == "fp32":
        precision_checks = {
            "fp32_endpoint_residual_relative_error_at_most_1e_4": endpoint_residual_errors["all"] <= 1e-4,
            "fp32_residual_reconstruction_error_at_most_1e_6": geometry.reconstruction_relative_error <= 1e-6,
            "fp32_orthogonality_error_at_most_1e_6": geometry.normalized_orthogonality_error <= 1e-6,
            "fp32_full_route_reconstruction_error_at_most_1e_4": full_route_reconstruction_error <= 1e-4,
            "fp32_parallel_trainer_matches_independent_at_most_1e_4": paired_relative_errors["parallel_only"] <= 1e-4,
            "fp32_orthogonal_trainer_matches_independent_at_most_1e_4": paired_relative_errors["orthogonal_only"] <= 1e-4,
        }
    else:
        precision_checks = {
            "bf16_residual_reconstruction_within_representation_tolerance": geometry.reconstruction_relative_error <= representation_tolerance,
            "bf16_orthogonality_within_representation_tolerance": geometry.normalized_orthogonality_error <= representation_tolerance,
            "bf16_full_route_reconstruction_within_representation_tolerance": full_route_reconstruction_error <= representation_tolerance,
            "bf16_parallel_trainer_matches_independent_within_tolerance": paired_relative_errors["parallel_only"] <= representation_tolerance,
            "bf16_orthogonal_trainer_matches_independent_within_tolerance": paired_relative_errors["orthogonal_only"] <= representation_tolerance,
        }
    checks = {**invariant_checks, **precision_checks}
    report = {
        "status": "EXP59_PREFLIGHT_PASS" if all(checks.values()) else "EXP59_PREFLIGHT_FAIL",
        "checks": checks,
        "contract": contract,
        "model_mode": model_mode,
        "head_contract": head_contract,
        "precision": precision,
        "microbatches": microbatches,
        "examples": microbatches * 4,
        "endpoint_residual_relative_errors": endpoint_residual_errors,
        "geometry": geometry.as_dict(),
        "component_head_norms": component_head_norms,
        "full_route_reconstruction_relative_error": full_route_reconstruction_error,
        "independent_preclip_norms": {
            variant: values[1] for variant, values in expected.items()
        },
        "independent_clip_coefficients": {
            variant: values[2] for variant, values in expected.items()
        },
        "paired_trainer_relative_errors": paired_relative_errors,
        "paired_audits": paired_audits,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    output = (
        OUTPUT_ROOT
        / "audit"
        / f"preflight_real_qwen3_full_accumulation_{precision}_v2.json"
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
    if result["status"] != "EXP59_PREFLIGHT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
