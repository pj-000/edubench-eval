"""Weighted ordinal loss for Exp5 L1."""

from __future__ import annotations

from typing import Any


def make_ordinal_targets(label_5: Any) -> Any:
    import torch

    labels = torch.as_tensor(label_5)
    if labels.numel() and ((labels < 1).any() or (labels > 5).any()):
        raise ValueError("label_5 must be in 1..5 for ordinal targets.")
    thresholds = torch.arange(1, 5, device=labels.device).view(1, 4)
    return (labels.long().view(-1, 1) > thresholds).to(dtype=torch.float32)


def ordinal_bce_loss_per_sample(logits: Any, ordinal_targets: Any) -> Any:
    import torch

    if logits.shape[-1] != 4:
        raise ValueError(f"Expected ordinal logits shape [batch,4], got {tuple(logits.shape)}")
    if ordinal_targets.shape != logits.shape:
        raise ValueError(
            "ordinal_targets shape must match logits shape; "
            f"got targets={tuple(ordinal_targets.shape)} logits={tuple(logits.shape)}"
        )
    loss_by_threshold = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(),
        ordinal_targets.float(),
        reduction="none",
    )
    return loss_by_threshold.mean(dim=1)


def weighted_ordinal_loss(logits: Any, label_5: Any, class_weights: Any) -> tuple[Any, dict[str, float]]:
    import torch

    labels = torch.as_tensor(label_5, device=logits.device).long()
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.float().dtype)
    if weights.ndim != 1 or weights.shape[0] <= 5:
        raise ValueError("class_weights must be a 1D tensor/list indexable by label_5 values 1..5.")
    ordinal_targets = make_ordinal_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
    per_sample_loss = ordinal_bce_loss_per_sample(logits, ordinal_targets)
    sample_weights = weights[labels]
    denominator = sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).eps)
    loss = (sample_weights * per_sample_loss).sum() / denominator
    if torch.isnan(loss):
        raise FloatingPointError("weighted ordinal loss became NaN.")
    debug = {
        "mean_base_loss": float(per_sample_loss.detach().mean().cpu()),
        "mean_weighted_loss": float((sample_weights * per_sample_loss).detach().mean().cpu()),
        "mean_sample_weight": float(sample_weights.detach().mean().cpu()),
        "min_sample_weight": float(sample_weights.detach().min().cpu()),
        "max_sample_weight": float(sample_weights.detach().max().cpu()),
    }
    return loss, debug
