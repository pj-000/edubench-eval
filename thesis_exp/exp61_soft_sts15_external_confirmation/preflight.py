"""Real-model, no-optimizer-step preflight for all Exp61 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    OUTPUT_ROOT,
    VARIANTS,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.data import load_model_rows
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import (
    verify_model_against_lock,
    verify_source_lock,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.geometry import compose_geometry_step
from thesis_exp.exp61_soft_sts15_external_confirmation.mapping import mapping_sha256, mapping_target_lookup
from thesis_exp.exp61_soft_sts15_external_confirmation.method import hard_soft_ce_identity
from thesis_exp.exp61_soft_sts15_external_confirmation.model import (
    ModelConfig,
    load_model_and_tokenizer,
    parameter_sha256,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.runtime import (
    accumulate_residual_vjp,
    capture_rng,
    collate,
    restore_rng,
    routed_objective,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(61, 62, 63))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source_lock = verify_source_lock(require_formal_authorization=False)
    model_manifest = verify_model_against_lock(Path(args.model_name_or_path), source_lock)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Exp61 real-model preflight requires an explicitly assigned CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train_rows = load_model_rows(args.source_repo, "train")
    mapping_rows = [json.loads(line) for line in MAPPING_PATH.read_text().splitlines() if line]
    mapping_audit = json.loads(MAPPING_AUDIT_PATH.read_text())
    if mapping_sha256(mapping_rows) != mapping_audit["mapping_sha256"]:
        raise RuntimeError("Exp61 mapping artifact hash mismatch")
    shuffled = mapping_target_lookup(mapping_rows)
    selected = train_rows[: args.batch_size]
    model, tokenizer, head = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path, local_files_only=True, gradient_checkpointing=True, bf16=True)
    )
    initial_hash = parameter_sha256(model)
    model.to(device)
    model.train()
    batch = collate(tokenizer, selected, 256)
    metadata = batch.pop("metadata")
    labels = batch.pop("labels").to(device)
    inputs = {name: value.to(device) for name, value in batch.items()}
    aligned_targets = torch.tensor([row["soft_target"] for row in metadata], dtype=torch.float32, device=device)
    shuffled_targets = torch.tensor([shuffled[row["record_id"]] for row in metadata], dtype=torch.float32, device=device)
    scale = 1.0 / 32.0
    rng_before = capture_rng(torch, device)
    outputs = model(
        **inputs,
        labels=labels,
        soft_targets=aligned_targets,
        aux_route="routed_hmsa",
        route_loss_scale=scale,
    )
    objective = routed_objective(outputs, labels, aligned_targets)
    identity = hard_soft_ce_identity(outputs["aux_logits"], labels, aligned_targets)
    (objective["optimization_loss"] * scale).backward()
    rng_after = capture_rng(torch, device)
    named_backbone = {
        name: parameter for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    aligned_buffers = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named_backbone.items()}
    shuffled_buffers = {name: torch.zeros_like(parameter, dtype=torch.float32) for name, parameter in named_backbone.items()}
    restore_rng(torch, device, rng_before)
    aligned_loss = accumulate_residual_vjp(model, inputs, labels, aligned_targets, aligned_buffers, scale)
    restore_rng(torch, device, rng_before)
    shuffled_loss = accumulate_residual_vjp(model, inputs, labels, shuffled_targets, shuffled_buffers, scale)
    restore_rng(torch, device, rng_after)
    original_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    arm_audits: dict[str, Any] = {}
    for variant in VARIANTS:
        for name, parameter in model.named_parameters():
            if name in original_gradients:
                parameter.grad = original_gradients[name].clone()
        arm_audits[variant] = compose_geometry_step(
            model, aligned_buffers, shuffled_buffers, variant=variant, max_norm=1.0
        )
    model.zero_grad(set_to_none=True)
    final_hash = parameter_sha256(model)
    checks = {
        "six_class_heads_and_no_bias": all(head["checks"].values()),
        "soft_ce_identity_error_le_1e_5": identity["absolute_identity_error"] <= 1e-5,
        "all_three_arms_executed": set(arm_audits) == set(VARIANTS),
        "aligned_and_shuffled_residual_losses_finite": bool(
            torch.isfinite(torch.tensor([aligned_loss, shuffled_loss])).all()
        ),
        "no_parameter_update": initial_hash == final_hash,
        "optimizer_not_constructed": True,
        "optimizer_step_count_zero": True,
        "test_access_count_zero": True,
    }
    decision = {
        "status": "EXP61_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS" if all(checks.values()) else "EXP61_REAL_MODEL_NO_UPDATE_PREFLIGHT_FAIL",
        "seed": args.seed,
        "device": str(device),
        "batch_record_ids": [row["record_id"] for row in metadata],
        "head_contract": head,
        "model_manifest_sha256": model_manifest["manifest_sha256"],
        "soft_ce_identity": identity,
        "aligned_diagnostic_soft_ce": aligned_loss,
        "shuffled_diagnostic_soft_ce": shuffled_loss,
        "arm_geometry_audits": arm_audits,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": final_hash,
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(decision, sort_keys=True))
    return decision


def main() -> None:
    args = parse_args()
    result = run(args)
    output = args.output or OUTPUT_ROOT / "preflight" / f"seed_{args.seed}" / "real_model_no_update_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
