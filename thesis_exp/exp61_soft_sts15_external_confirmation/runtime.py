"""Shared no-test runtime primitives for Exp61 preflight and training."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation.method import soft_cross_entropy


def collate(tokenizer: Any, rows: list[dict[str, Any]], max_length: int = 256) -> dict[str, Any]:
    import torch

    encoded = tokenizer(
        [row["text"] for row in rows],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    encoded["metadata"] = rows
    return encoded


def routed_objective(outputs: dict[str, Any], labels: Any, targets: Any) -> dict[str, Any]:
    import torch

    hard = torch.nn.functional.cross_entropy(outputs["hard_logits"].float(), labels.long())
    auxiliary = soft_cross_entropy(outputs["aux_logits"], targets)
    return {"hard_ce": hard, "auxiliary_soft_ce": auxiliary, "optimization_loss": hard + auxiliary}


def install_residual_capture_hooks(
    model: Any, residual_buffers: dict[str, Any]
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


def accumulate_residual_vjp(
    model: Any,
    inputs: dict[str, Any],
    labels: Any,
    targets: Any,
    residual_buffers: dict[str, Any],
    loss_scale: float,
) -> float:
    handles = install_residual_capture_hooks(model, residual_buffers)
    try:
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="residual_only",
            route_loss_scale=loss_scale,
        )
        loss = soft_cross_entropy(outputs["aux_logits"], targets)
        (loss * loss_scale).backward()
        return float(loss.detach().cpu())
    finally:
        for handle in handles:
            handle.remove()


def capture_rng(torch: Any, device: Any) -> tuple[Any, Any]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def restore_rng(torch: Any, device: Any, state: tuple[Any, Any]) -> None:
    torch.set_rng_state(state[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state[1], device)
