"""Real-model Stage 0 parity audit for the CBRD routing implementation.

Run this only in the server's existing PyTorch/Qwen3 environment.  It does not
train, writes only Exp57 audit artifacts, loads two development rows, and
never accesses the historical test split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.model import head_contract, make_cbrd_dual_head_classifier, module_hash


EXPECTED_SEED42_HEAD_HASH = "d7c922a1956118af437c52189ffc993465277a72698cef38290e47404247f9e2"


def _torch() -> tuple[Any, Any]:
    import torch
    from torch.nn import functional as functional

    return torch, functional


def config(model_path: str, *, bf16: str) -> Any:
    from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig

    return TrainConfig(
        model_name_or_path=model_path,
        data_dir=Path("EXP57_PREFLIGHT_TRAIN_DEV_ONLY"),
        output_dir=OUTPUT_ROOT / "audit" / "preflight_runtime",
        checkpoint_output_dir=Path("UNUSED"),
        max_length=2048,
        num_train_epochs=10.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=42,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=False,
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


def _select_backbone_parameters(model: Any) -> list[tuple[str, Any]]:
    candidates = [
        (name, parameter)
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "layers." in name and parameter.numel() <= 5_000_000
    ]
    if not candidates:
        candidates = [
            (name, parameter)
            for name, parameter in model.backbone.named_parameters()
            if parameter.requires_grad and parameter.numel() <= 5_000_000
        ]
    if len(candidates) < 2:
        raise RuntimeError("Could not select lower and upper compact backbone parameters")
    return [candidates[0], candidates[-1]]


def _clone_gradient(parameter: Any) -> Any:
    if parameter.grad is None:
        raise AssertionError("Expected a gradient but found None")
    return parameter.grad.detach().float().cpu().clone()


def _max_abs(left: Any, right: Any) -> float:
    return float((left - right).abs().max())


def _max_relative(left: Any, right: Any) -> float:
    numerator = _max_abs(left, right)
    denominator = max(float(left.abs().max()), 1e-12)
    return numerator / denominator


def run(model_path: str, *, bf16: str = "false") -> dict[str, Any]:
    torch, _ = _torch()
    from thesis_exp.src.edujudge.exp02.train_ce_baseline import load_model_and_tokenizer, set_seed

    set_seed(42)
    if torch.cuda.is_available():
        # Make the audit compare routing rather than fused-attention atomic
        # noise.  This changes only the two-row preflight kernel selection,
        # never the planned training configuration.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = config(model_path, bf16=bf16)
    base, tokenizer, model_mode = load_model_and_tokenizer(cfg)
    if model_mode != "sequence_classification":
        raise RuntimeError(f"Expected sequence classification, got {model_mode}")
    base_head_hash = module_hash(base.score)
    # Exp51 initialized the copied classifier in BF16.  A freshly initialized
    # FP32 head is numerically related but must not be expected to have the
    # same byte hash after float32 serialization.  Enforce the historical hash
    # only in the matched BF16 audit; FP32 is the strict routing check.
    historical_head_hash_required = bf16 == "true"
    if historical_head_hash_required and base_head_hash != EXPECTED_SEED42_HEAD_HASH:
        raise AssertionError(f"Seed42 base-head hash mismatch: {base_head_hash}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    model = make_cbrd_dual_head_classifier(base).to(device).eval()
    contract = head_contract(model)
    if contract["hard_head_hash"] != contract["soft_head_hash"] or not contract["storage_independent"]:
        raise AssertionError(f"Copied-head initialization failed: {contract}")
    rows = model_rows("dev")[:2]
    encoded = tokenizer([row["text"] for row in rows], truncation=True, max_length=256, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    targets = torch.tensor([row["soft_target_5"] for row in rows], dtype=torch.float32, device=device)
    selected_backbone = _select_backbone_parameters(model)

    with torch.no_grad():
        base_logits = base(**encoded).logits.detach().float().cpu()
        wrapped_logits = model(**encoded)["hard_logits"].detach().float().cpu()
    hard_path_max_difference = _max_abs(base_logits, wrapped_logits)

    def capture(variant: str) -> dict[str, Any]:
        set_seed(42)
        model.zero_grad(set_to_none=True)
        outputs = model(
            **encoded,
            labels=labels,
            soft_targets=targets,
            aux_route=variant,
            retain_backbone_hidden=True,
        )
        objective = cbrd_objective(outputs, labels, targets, variant=variant)
        objective["optimization_loss"].backward()
        return {
            "hard_head": {name: _clone_gradient(parameter) for name, parameter in model.hard_head.named_parameters()},
            "soft_head": {name: _clone_gradient(parameter) for name, parameter in model.soft_head.named_parameters()},
            "backbone": {name: _clone_gradient(parameter) for name, parameter in selected_backbone},
            "hidden": _clone_gradient(outputs["backbone_hidden"]),
            "values": {
                "hard_aux_logit_max_difference": float((outputs["aux_logits"] - outputs["aux_head_logits"]).abs().max().detach().cpu()),
                "route_aux_logit_max_difference": float((outputs["aux_logits"] - outputs["aux_route_logits"]).abs().max().detach().cpu()),
                "reported_aux_soft_ce": float(objective["reported_aux_soft_ce"].detach().cpu()),
                "route_reconstructed_soft_ce": float(objective["route_reconstructed_soft_ce"].detach().cpu()),
            },
        }

    ordinary = capture("ordinary_hmsa")
    ordinary_repeat = capture("ordinary_hmsa")
    routed = capture("routed_hmsa")
    differences: dict[str, float] = {}
    relative_differences: dict[str, float] = {}
    repeat_differences: dict[str, float] = {}
    repeat_relative_differences: dict[str, float] = {}
    for group in ("hard_head", "soft_head", "backbone"):
        for name, ordinary_gradient in ordinary[group].items():
            differences[f"{group}.{name}"] = _max_abs(ordinary_gradient, routed[group][name])
            relative_differences[f"{group}.{name}"] = _max_relative(ordinary_gradient, routed[group][name])
            repeat_differences[f"{group}.{name}"] = _max_abs(
                ordinary_gradient,
                ordinary_repeat[group][name],
            )
            repeat_relative_differences[f"{group}.{name}"] = _max_relative(
                ordinary_gradient,
                ordinary_repeat[group][name],
            )
    differences["backbone_hidden"] = _max_abs(ordinary["hidden"], routed["hidden"])
    relative_differences["backbone_hidden"] = _max_relative(ordinary["hidden"], routed["hidden"])
    repeat_differences["backbone_hidden"] = _max_abs(
        ordinary["hidden"],
        ordinary_repeat["hidden"],
    )
    repeat_relative_differences["backbone_hidden"] = _max_relative(
        ordinary["hidden"],
        ordinary_repeat["hidden"],
    )
    logit_difference = max(
        ordinary["values"]["hard_aux_logit_max_difference"],
        ordinary["values"]["route_aux_logit_max_difference"],
    )
    reconstruction_difference = abs(
        ordinary["values"]["reported_aux_soft_ce"] - ordinary["values"]["route_reconstructed_soft_ce"]
    )
    amp_mode = bf16 == "true"
    tolerance = 1e-5 if amp_mode else 1e-6
    relative_tolerance = 1e-5 if amp_mode else 1e-6
    route_max = max(differences.values())
    route_relative_max = max(relative_differences.values())
    repeat_max = max(repeat_differences.values())
    repeat_relative_max = max(repeat_relative_differences.values())
    # Strict equality is preferred.  If the installed CUDA stack still emits
    # nondeterministic early-layer gradients, require the routed comparison to
    # stay within ordinary-vs-ordinary repeat noise (plus the fixed tolerance).
    backbone_noise_gate = (
        route_max <= max(tolerance, repeat_max + tolerance)
        and route_relative_max
        <= max(relative_tolerance, repeat_relative_max + relative_tolerance)
    )
    report = {
        "status": "PASS" if max(logit_difference, reconstruction_difference, hard_path_max_difference, differences["hard_head.weight"], differences["soft_head.weight"], differences["backbone_hidden"]) <= tolerance and backbone_noise_gate else "FAIL",
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "model_mode": model_mode,
        "base_head_hash": base_head_hash,
        "historical_seed42_head_hash": EXPECTED_SEED42_HEAD_HASH,
        "historical_head_hash_required": historical_head_hash_required,
        "historical_head_hash_matches": base_head_hash == EXPECTED_SEED42_HEAD_HASH,
        "head_contract": contract,
        "rows": [row["record_id"] for row in rows],
        "selected_backbone_parameters": [name for name, _ in selected_backbone],
        "per_tensor_max_gradient_difference": differences,
        "per_tensor_relative_gradient_difference": relative_differences,
        "max_gradient_difference": route_max,
        "max_relative_gradient_difference": route_relative_max,
        "ordinary_repeat_per_tensor_max_gradient_difference": repeat_differences,
        "ordinary_repeat_per_tensor_relative_gradient_difference": repeat_relative_differences,
        "ordinary_repeat_max_gradient_difference": repeat_max,
        "ordinary_repeat_max_relative_gradient_difference": repeat_relative_max,
        "backbone_noise_gate": backbone_noise_gate,
        "auxiliary_logit_max_difference": logit_difference,
        "ce_reconstruction_difference": reconstruction_difference,
        "hard_path_max_difference": hard_path_max_difference,
        "tolerance": tolerance,
        "relative_tolerance": relative_tolerance,
        "bf16": bf16,
        "implementation": "single_soft_ce_hidden_gradient_hook",
        "deterministic_audit_kernels": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "test_access_count": 0,
    }
    output = OUTPUT_ROOT / "audit" / f"stage0_real_qwen3_routing_bf16_{bf16}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--bf16", choices=("true", "false"), default="false")
    args = parser.parse_args()
    report = run(args.model_name_or_path, bf16=args.bf16)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
