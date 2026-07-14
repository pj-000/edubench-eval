"""Locked Exp43 distributional, ordinal, and pair-ranking losses."""

from __future__ import annotations

from typing import Any


def distribution_ce(logits: Any, targets: Any) -> Any:
    import torch
    return -(targets.float() * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def ordinal_cdf_mse(logits: Any, targets: Any) -> Any:
    import torch
    probabilities = torch.softmax(logits.float(), dim=-1)
    return torch.mean((torch.cumsum(probabilities, dim=-1)[:, :4] - torch.cumsum(targets.float(), dim=-1)[:, :4]) ** 2)


def expected_score(logits: Any) -> Any:
    import torch
    values = torch.arange(1, 6, dtype=torch.float32, device=logits.device)
    return torch.softmax(logits.float(), dim=-1).matmul(values)


def pair_rank_loss(positive_logits: Any, negative_logits: Any) -> Any:
    import torch.nn.functional as functional
    return functional.softplus(-(expected_score(positive_logits) - expected_score(negative_logits))).mean()


def total_point_loss(variant: str, logits: Any, targets: Any) -> tuple[Any, dict[str, Any]]:
    dist = distribution_ce(logits, targets)
    ordinal = ordinal_cdf_mse(logits, targets) if variant in {"E4", "E5", "E6", "E6N"} else dist.new_zeros(())
    total = dist + 0.5 * ordinal
    return total, {"distribution": dist, "ordinal": ordinal, "weighted_ordinal": 0.5 * ordinal}

