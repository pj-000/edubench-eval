"""Shared train/dev-only runtime primitives for Exp62."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp59_residual_geometry.train import (
    _accumulate_residual_vjp as accumulate_residual_vjp,
)
from thesis_exp.exp59_residual_geometry.train import _capture_rng as capture_rng
from thesis_exp.exp59_residual_geometry.train import _restore_rng as restore_rng


def collate(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    import torch

    encoded = tokenizer(
        [row["text"] for row in rows],
        truncation=False,
        padding=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > max_length:
        raise RuntimeError(
            f"Exp62 input length {encoded['input_ids'].shape[1]} exceeds frozen {max_length}"
        )
    encoded["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    encoded["metadata"] = rows
    return encoded


def objective(outputs: dict[str, Any], labels: Any, targets: Any, route: str) -> dict[str, Any]:
    return cbrd_objective(outputs, labels, targets, variant=route)


__all__ = [
    "accumulate_residual_vjp",
    "capture_rng",
    "collate",
    "objective",
    "restore_rng",
]
