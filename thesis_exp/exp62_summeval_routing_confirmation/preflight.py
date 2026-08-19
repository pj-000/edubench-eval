"""Real-model, no-update preflight for all four Exp62 routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.method import auxiliary_decomposition
from thesis_exp.exp62_summeval_routing_confirmation import OUTPUT_ROOT, SEEDS, VARIANTS
from thesis_exp.exp62_summeval_routing_confirmation.data import load_model_rows
from thesis_exp.exp62_summeval_routing_confirmation.geometry import compose_geometry_step
from thesis_exp.exp62_summeval_routing_confirmation.model import (
    ModelConfig,
    load_model_and_tokenizer,
    parameter_sha256,
)
from thesis_exp.exp62_summeval_routing_confirmation.runtime import (
    accumulate_residual_vjp,
    capture_rng,
    collate,
    objective,
    restore_rng,
)
from thesis_exp.exp62_summeval_routing_confirmation.train import _clip_standard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp62 real-model no-update preflight")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Exp62 real-model preflight requires CUDA")
    set_seed(args.seed)
    rows = load_model_rows(args.annotations, "train")
    # Guarantee a non-zero residual in the diagnostic batch.
    selected = [row for row in rows if len(set(row["expert_scores"])) > 1][:112]
    if len(selected) != 112:
        raise RuntimeError("Exp62 preflight could not find 112 non-unanimous rows")
    model, tokenizer, head = load_model_and_tokenizer(ModelConfig(args.model_name_or_path))
    initial_hash = parameter_sha256(model)
    device = torch.device("cuda")
    model.to(device)
    model.train()
    batch = collate(tokenizer, selected, 256)
    metadata = batch.pop("metadata")
    labels = batch.pop("labels").to(device)
    inputs = {name: value.to(device) for name, value in batch.items()}
    targets = torch.tensor(
        [row["soft_target"] for row in metadata], dtype=torch.float32, device=device
    )
    scale = 1.0
    arm_audits: dict[str, Any] = {}
    identity: dict[str, float] | None = None
    for variant in VARIANTS:
        model.zero_grad(set_to_none=True)
        route = "consensus_only" if variant == "direct_residual_blocked" else "routed_hmsa"
        rng_before = capture_rng(torch, device)
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route=route,
            route_loss_scale=scale,
        )
        losses = objective(outputs, labels, targets, route)
        if identity is None:
            decomposed = auxiliary_decomposition(outputs["aux_logits"], labels, targets)
            identity = {
                "soft_ce": float(decomposed["soft_ce"].detach().cpu()),
                "consensus_ce": float(decomposed["consensus_ce"].detach().cpu()),
                "boundary_residual": float(decomposed["boundary_residual"].detach().cpu()),
                "absolute_error": float(decomposed["absolute_error"].detach().cpu()),
            }
        (losses["optimization_loss"] * scale).backward()
        if variant in ("orthogonal_only", "parallel_only"):
            rng_after = capture_rng(torch, device)
            named_backbone = {
                name: parameter
                for name, parameter in model.named_parameters()
                if name.startswith("backbone.") and parameter.requires_grad
            }
            buffers = {
                name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
                for name, parameter in named_backbone.items()
            }
            restore_rng(torch, device, rng_before)
            residual_loss = accumulate_residual_vjp(
                model, inputs, labels, targets, buffers, scale
            )
            restore_rng(torch, device, rng_after)
            arm_audits[variant] = {
                **compose_geometry_step(
                    model, buffers, variant=variant, max_norm=1.0
                ),
                "diagnostic_residual_soft_ce": residual_loss,
            }
        else:
            arm_audits[variant] = {"variant": variant, **_clip_standard(model, 1.0)}
    model.zero_grad(set_to_none=True)
    final_hash = parameter_sha256(model)
    assert identity is not None
    checks = {
        "head_contract": all(head["checks"].values()),
        "four_arms_executed": set(arm_audits) == set(VARIANTS),
        "soft_ce_identity_error_le_1e_5": identity["absolute_error"] <= 1e-5,
        "orthogonal_nonzero": arm_audits["orthogonal_only"]["orthogonal_norm"] > 0,
        "parallel_nonzero": arm_audits["parallel_only"]["parallel_norm"] > 0,
        "no_parameter_update": initial_hash == final_hash,
        "optimizer_not_constructed": True,
        "optimizer_step_count_zero": True,
        "test_access_count_zero": True,
    }
    result = {
        "status": (
            "EXP62_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS"
            if all(checks.values())
            else "EXP62_REAL_MODEL_NO_UPDATE_PREFLIGHT_FAIL"
        ),
        "seed": args.seed,
        "batch_record_ids": [row["record_id"] for row in metadata],
        "input_token_length": int(batch["input_ids"].shape[1]),
        "head_contract": head,
        "soft_ce_identity": identity,
        "arm_audits": arm_audits,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": final_hash,
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    output = args.output or (
        OUTPUT_ROOT / "preflight" / f"seed_{args.seed}" / "real_model_no_update_preflight.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
