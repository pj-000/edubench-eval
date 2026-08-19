"""MeanAux expected-score projection and fixed auxiliary loss."""

from __future__ import annotations

from typing import Any


def expected_score(aux_logits: Any) -> Any:
    import torch

    if aux_logits.ndim != 2 or aux_logits.shape[-1] != 5:
        raise ValueError(f"Expected [batch, 5] logits; got {tuple(aux_logits.shape)}")
    scores = torch.arange(
        1,
        6,
        dtype=torch.float32,
        device=aux_logits.device,
    )
    return (torch.softmax(aux_logits.float(), dim=-1) * scores).sum(dim=-1)


def mean_aux_smooth_l1(
    aux_logits: Any,
    human_mean: Any,
    *,
    beta: float = 1.0,
) -> Any:
    from torch.nn import functional as F

    prediction = expected_score(aux_logits)
    target = human_mean.float().reshape_as(prediction)
    return F.smooth_l1_loss(prediction, target, beta=beta)
