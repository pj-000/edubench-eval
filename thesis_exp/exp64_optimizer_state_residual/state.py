"""Checkpoint/RNG contracts shared by Exp64 preflight and formal audit."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def capture_rng(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(torch: Any, device: Any, state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError(f"Exp64 incomplete RNG state: {sorted(set(state))}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise RuntimeError("Exp64 CUDA checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def optimizer_contract(
    model: Any, optimizer_payload: dict[str, Any], parameter_names: list[list[str]] | None
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Map serialized optimizer state and group settings back to model names."""

    runtime_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    serialized_groups = optimizer_payload.get("param_groups", [])
    if not serialized_groups:
        raise RuntimeError("Exp64 checkpoint has no optimizer parameter groups")
    if parameter_names is None:
        # Legacy Exp63 checkpoints used one AdamW over ``model.parameters()``.
        lengths = [len(group["params"]) for group in serialized_groups]
        if sum(lengths) != len(runtime_names):
            raise RuntimeError("Cannot infer legacy optimizer parameter-name order")
        parameter_names = []
        offset = 0
        for length in lengths:
            parameter_names.append(runtime_names[offset : offset + length])
            offset += length
    if len(parameter_names) != len(serialized_groups):
        raise RuntimeError("Exp64 optimizer group-name count mismatch")

    states: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for serialized_group, names in zip(serialized_groups, parameter_names):
        identifiers = serialized_group["params"]
        if len(identifiers) != len(names):
            raise RuntimeError("Exp64 optimizer parameter-name alignment mismatch")
        frozen_group = {
            key: value for key, value in serialized_group.items() if key != "params"
        }
        for identifier, name in zip(identifiers, names):
            if name in states:
                raise RuntimeError(f"duplicate Exp64 optimizer parameter: {name}")
            if identifier not in optimizer_payload["state"]:
                raise RuntimeError(f"missing Exp64 optimizer state for {name}")
            states[name] = optimizer_payload["state"][identifier]
            groups[name] = frozen_group
    if set(states) != set(runtime_names):
        raise RuntimeError("Exp64 optimizer/model parameter set mismatch")
    return states, groups


__all__ = ["capture_rng", "optimizer_contract", "restore_rng"]
