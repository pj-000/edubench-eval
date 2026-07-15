"""Locked pointwise and contrastive objectives for Exp44A."""

from __future__ import annotations

from typing import Any


def distribution_loss(logits: Any, targets: Any) -> Any:
    import torch

    return -(targets * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def ordinal_cdf_loss(logits: Any, targets: Any) -> Any:
    import torch

    probabilities = torch.softmax(logits.float(), dim=-1)
    predicted_cdf = torch.cumsum(probabilities, dim=-1)[..., :-1]
    target_cdf = torch.cumsum(targets.float(), dim=-1)[..., :-1]
    return torch.mean((predicted_cdf - target_cdf) ** 2)


def base_loss(logits: Any, targets: Any) -> tuple[Any, dict[str, Any]]:
    dist = distribution_loss(logits, targets)
    ordinal = ordinal_cdf_loss(logits, targets)
    return dist + 0.5 * ordinal, {"distribution": dist, "ordinal": ordinal}


def taco_loss(
    projections: Any,
    near_distance: Any,
    far_distance: Any,
    *,
    use_margin: bool,
    temperature: float = 0.1,
) -> tuple[Any, dict[str, Any]]:
    """Projection order is anchor, positive, near, far for each triplet."""
    import torch

    if projections.ndim != 3 or projections.shape[1] != 4:
        raise ValueError(f"Expected [N,4,D] projection tensor, got {tuple(projections.shape)}")
    anchor, positive, near, far = (projections[:, index] for index in range(4))
    similarity_positive = (anchor * positive).sum(dim=-1)
    similarity_near = (anchor * near).sum(dim=-1)
    similarity_far = (anchor * far).sum(dim=-1)
    if use_margin:
        near_margin = 0.1 * near_distance.float()
        far_margin = 0.1 * far_distance.float()
    else:
        near_margin = torch.zeros_like(similarity_near)
        far_margin = torch.zeros_like(similarity_far)
    near_term = (similarity_near - similarity_positive + near_margin) / temperature
    far_term = (similarity_far - similarity_positive + far_margin) / temperature
    zeros = torch.zeros_like(near_term)
    loss = torch.logsumexp(torch.stack((zeros, near_term, far_term), dim=-1), dim=-1).mean()
    diagnostics = {
        "triplet_accuracy": ((similarity_positive > similarity_near) & (similarity_positive > similarity_far)).float().mean(),
        "near_margin_violation": (similarity_positive < similarity_near + near_margin).float().mean(),
        "far_margin_violation": (similarity_positive < similarity_far + far_margin).float().mean(),
        "mean_positive_similarity": similarity_positive.mean(),
        "mean_near_similarity": similarity_near.mean(),
        "mean_far_similarity": similarity_far.mean(),
    }
    return loss, diagnostics

