"""Losses for Exp16A boundary linking."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def ordinal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Mapping[int, float] | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if class_weights:
        weights = torch.ones_like(labels, dtype=loss.dtype, device=loss.device)
        for label, weight in class_weights.items():
            weights = torch.where(labels == int(label), torch.full_like(weights, float(weight)), weights)
        loss = loss * weights.unsqueeze(-1)
    return loss.mean()


def parse_class_weights(spec: str | None) -> dict[int, float] | None:
    if not spec:
        return None
    out: dict[int, float] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        label, weight = item.split(":", 1)
        out[int(label.strip())] = float(weight.strip())
    invalid = sorted(label for label in out if label not in [1, 2, 3, 4, 5])
    if invalid:
        raise ValueError(f"class_weights labels must be 1..5, got {invalid}")
    return out
