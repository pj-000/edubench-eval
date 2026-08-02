"""Audit how the aligned residual interacts with global gradient clipping.

The audit replays one frozen 32-microbatch accumulation window at a fixed
model state.  It never performs an optimizer step and never reads test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.model import make_cbrd_dual_head_classifier
from thesis_exp.exp57_cbrd.preflight import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("hard_head."):
        return "hard_head"
    if name.startswith("soft_head."):
        return "soft_head"
    return "other"


def squared_norms(gradients: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {"all": 0.0}
    for name, gradient in gradients.items():
        group = parameter_group(name)
        value = float(gradient.double().square().sum())
        values[group] = values.get(group, 0.0) + value
        values["all"] += value
    return values


def rooted(values: dict[str, float]) -> dict[str, float]:
    return {key: math.sqrt(max(0.0, value)) for key, value in values.items()}


def clip_coefficient(total_norm: float, max_norm: float = 1.0) -> float:
    return min(1.0, max_norm / (total_norm + 1e-6))


def scaled(values: dict[str, float], coefficient: float) -> dict[str, float]:
    return {key: value * coefficient for key, value in values.items()}


def cosine(dot: float, left_sq: float, right_sq: float) -> float | None:
    denominator = math.sqrt(max(0.0, left_sq) * max(0.0, right_sq))
    return None if denominator == 0.0 else dot / denominator


def run(
    *,
    model_path: str,
    trace_path: Path,
    output_path: Path,
    seed: int,
    checkpoint_path: Path | None,
    microbatches: int = 32,
) -> dict[str, Any]:
    import torch

    from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
        load_model_and_tokenizer,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if microbatches != 32:
        raise ValueError("The frozen training accumulation window contains exactly 32 microbatches")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))["trace"][:microbatches]
    if len(trace) != microbatches or any(len(row["record_ids"]) != 4 for row in trace):
        raise RuntimeError("Trace does not contain one complete 32 x 4 accumulation window")
    train_lookup = {str(row["record_id"]): row for row in model_rows("train")}
    batches = [
        [train_lookup[str(record_id)] for record_id in trace_row["record_ids"]]
        for trace_row in trace
    ]

    set_seed(seed)
    cfg = config(model_path, bf16="true")
    cfg.gradient_checkpointing = True
    base, tokenizer, model_mode = load_model_and_tokenizer(cfg)
    if model_mode != "sequence_classification":
        raise RuntimeError(model_mode)
    device = torch.device("cuda")
    model = make_cbrd_dual_head_classifier(base).to(device)
    model.gradient_checkpointing_enable()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
    model.train()

    encoded_batches: list[tuple[dict[str, Any], Any, Any]] = []
    for batch in batches:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        targets = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded_batches.append((encoded, labels, targets))

    def capture(mode: str) -> dict[str, Any]:
        set_seed(seed)
        model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for encoded_cpu, labels_cpu, targets_cpu in encoded_batches:
            encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
            labels = labels_cpu.to(device)
            targets = targets_cpu.to(device)
            route = "detached_soft" if mode == "hard" else mode
            outputs = model(
                **encoded,
                labels=labels,
                soft_targets=targets,
                aux_route=route,
                route_loss_scale=1.0 / 32.0,
            )
            objective = cbrd_objective(outputs, labels, targets, variant=route)
            loss = (
                objective["reported_hard_loss"]
                if mode == "hard"
                else objective["optimization_loss"]
            )
            (loss / 32.0).backward()
            loss_sum += float(loss.detach().cpu())
        gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return {"mean_unscaled_loss": loss_sum / microbatches, "gradients": gradients}

    common = capture("consensus_only")
    common_sq = squared_norms(common["gradients"])
    common_norms = rooted(common_sq)
    common_clip = clip_coefficient(common_norms["all"])

    routed = capture("routed_hmsa")
    routed_sq = squared_norms(routed["gradients"])
    routed_norms = rooted(routed_sq)
    routed_clip = clip_coefficient(routed_norms["all"])

    residual_gradients: dict[str, Any] = {}
    common_residual_dot: dict[str, float] = {"all": 0.0}
    residual_sq: dict[str, float] = {"all": 0.0}
    for name in sorted(set(common["gradients"]) | set(routed["gradients"])):
        if name not in common["gradients"] or name not in routed["gradients"]:
            raise RuntimeError(f"Gradient support differs for {name}")
        common_gradient = common["gradients"][name]
        residual = routed["gradients"][name] - common_gradient
        group = parameter_group(name)
        dot_value = float((common_gradient.double() * residual.double()).sum())
        residual_value = float(residual.double().square().sum())
        common_residual_dot[group] = common_residual_dot.get(group, 0.0) + dot_value
        common_residual_dot["all"] += dot_value
        residual_sq[group] = residual_sq.get(group, 0.0) + residual_value
        residual_sq["all"] += residual_value
        if residual_value > 0.0:
            residual_gradients[name] = residual
    del common["gradients"]
    del routed["gradients"]

    hard = capture("hard")
    hard_sq = squared_norms(hard["gradients"])
    hard_residual_dot: dict[str, float] = {"all": 0.0}
    for name, residual in residual_gradients.items():
        if name not in hard["gradients"]:
            continue
        group = parameter_group(name)
        value = float((hard["gradients"][name].double() * residual.double()).sum())
        hard_residual_dot[group] = hard_residual_dot.get(group, 0.0) + value
        hard_residual_dot["all"] += value

    groups = sorted(set(common_sq) | set(routed_sq) | set(residual_sq) | set(hard_sq))
    common_residual_cosine = {
        group: cosine(
            common_residual_dot.get(group, 0.0),
            common_sq.get(group, 0.0),
            residual_sq.get(group, 0.0),
        )
        for group in groups
    }
    hard_residual_cosine = {
        group: cosine(
            hard_residual_dot.get(group, 0.0),
            hard_sq.get(group, 0.0),
            residual_sq.get(group, 0.0),
        )
        for group in groups
    }
    residual_norms = rooted(residual_sq)
    hard_norms = rooted(hard_sq)
    checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path else None
    report = {
        "status": "CLIP_GRADIENT_AUDIT_COMPLETE",
        "model_state": "selected_checkpoint" if checkpoint_path else "initial",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "trace_path": str(trace_path),
        "microbatches": microbatches,
        "examples": microbatches * 4,
        "gradient_accumulation_steps": 32,
        "max_grad_norm": 1.0,
        "deterministic_kernels": True,
        "pre_clip_norms": {
            "hard_loss_only": hard_norms,
            "consensus_route_total": common_norms,
            "aligned_route_total": routed_norms,
            "aligned_residual_difference": residual_norms,
        },
        "clip_coefficients": {
            "consensus_route": common_clip,
            "aligned_route": routed_clip,
            "aligned_over_consensus": routed_clip / common_clip,
        },
        "post_clip_total_norms": {
            "consensus_route": scaled(common_norms, common_clip),
            "aligned_route": scaled(routed_norms, routed_clip),
        },
        "cosines": {
            "consensus_route_total_vs_residual": common_residual_cosine,
            "hard_loss_gradient_vs_residual": hard_residual_cosine,
        },
        "mean_unscaled_losses": {
            "hard": hard["mean_unscaled_loss"],
            "consensus_route": common["mean_unscaled_loss"],
            "aligned_route": routed["mean_unscaled_loss"],
        },
        "interpretation_scope": (
            "No optimizer step. The residual is the paired deterministic difference "
            "between aligned and consensus routes at one fixed model state and one "
            "complete frozen accumulation window."
        ),
        "test_access_count": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--microbatches", type=int, default=32)
    args = parser.parse_args()
    result = run(
        model_path=args.model_name_or_path,
        trace_path=args.trace_path,
        output_path=args.output_path,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        microbatches=args.microbatches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
