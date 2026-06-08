"""Ordinal loss variants for Exp5."""

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


def expected_score_from_ordinal_logits(logits: Any) -> Any:
    import torch

    if logits.shape[-1] != 4:
        raise ValueError(f"Expected ordinal logits shape [batch,4], got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits.float())
    return 1.0 + probs.sum(dim=-1)


def asymmetric_low_score_penalty(logits: Any, label_5: Any, margin: float) -> tuple[Any, dict[str, float]]:
    import torch

    labels = torch.as_tensor(label_5, device=logits.device).long()
    if labels.numel() and ((labels < 1).any() or (labels > 5).any()):
        raise ValueError("label_5 must be in 1..5 for asymmetric low-score penalty.")
    if margin < 0:
        raise ValueError("margin must be >= 0.")
    s_hat = expected_score_from_ordinal_logits(logits)
    low_mask = (labels <= 2).to(dtype=logits.float().dtype)
    over = torch.clamp(s_hat - labels.float() - float(margin), min=0.0)
    penalty = low_mask * (over / 4.0).pow(2)
    low_count = int(low_mask.detach().sum().cpu())
    active_penalty_count = int(((penalty > 0) & (labels <= 2)).detach().sum().cpu())
    if low_count:
        low_s_hat = s_hat[labels <= 2]
        low_over = over[labels <= 2]
        low_penalty = penalty[labels <= 2]
        mean_s_hat_low = float(low_s_hat.detach().mean().cpu())
        mean_over_low = float(low_over.detach().mean().cpu())
        mean_penalty_low_only = float(low_penalty.detach().mean().cpu())
    else:
        mean_s_hat_low = 0.0
        mean_over_low = 0.0
        mean_penalty_low_only = 0.0
    debug = {
        "mean_penalty": float(penalty.detach().mean().cpu()),
        "mean_penalty_low_only": mean_penalty_low_only,
        "active_low_count": float(low_count),
        "active_penalty_count": float(active_penalty_count),
        "mean_s_hat_low": mean_s_hat_low,
        "mean_over_low": mean_over_low,
    }
    return penalty, debug


def asymmetric_ordinal_loss(
    logits: Any,
    label_5: Any,
    lambda_low: float,
    margin: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    if lambda_low <= 0:
        raise ValueError("lambda_low must be > 0.")
    labels = torch.as_tensor(label_5, device=logits.device).long()
    if labels.numel() and ((labels < 1).any() or (labels > 5).any()):
        raise ValueError("label_5 must be in 1..5 for asymmetric ordinal loss.")
    ordinal_targets = make_ordinal_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
    base_loss_per_sample = ordinal_bce_loss_per_sample(logits, ordinal_targets)
    penalty_per_sample, penalty_debug = asymmetric_low_score_penalty(logits, labels, margin)
    loss_per_sample = base_loss_per_sample + float(lambda_low) * penalty_per_sample
    loss = loss_per_sample.mean()
    if torch.isnan(loss):
        raise FloatingPointError("asymmetric ordinal loss became NaN.")
    debug = {
        "mean_base_loss": float(base_loss_per_sample.detach().mean().cpu()),
        "mean_penalty": penalty_debug["mean_penalty"],
        "lambda_low": float(lambda_low),
        "margin": float(margin),
        "mean_total_loss": float(loss_per_sample.detach().mean().cpu()),
        "active_low_count": penalty_debug["active_low_count"],
        "active_penalty_count": penalty_debug["active_penalty_count"],
        "mean_s_hat_low": penalty_debug["mean_s_hat_low"],
        "mean_over_low": penalty_debug["mean_over_low"],
    }
    return loss, debug


def threshold_low_score_penalty(logits: Any, label_5: Any) -> tuple[Any, dict[str, float]]:
    import torch

    if logits.shape[-1] != 4:
        raise ValueError(f"Expected ordinal logits shape [batch,4], got {tuple(logits.shape)}")
    labels = torch.as_tensor(label_5, device=logits.device).long()
    if labels.numel() and ((labels < 1).any() or (labels > 5).any()):
        raise ValueError("label_5 must be in 1..5 for threshold low-score penalty.")
    probs = torch.sigmoid(logits.float())
    p_gt_3 = probs[:, 2]
    p_gt_4 = probs[:, 3]
    low_mask = (labels <= 2).to(dtype=logits.float().dtype)
    penalty = low_mask * (p_gt_3.pow(2) + p_gt_4.pow(2)) / 2.0
    if torch.isnan(penalty).any():
        raise FloatingPointError("threshold low-score penalty became NaN.")
    low_count = int(low_mask.detach().sum().cpu())
    if low_count:
        low_selector = labels <= 2
        low_penalty = penalty[low_selector]
        low_p_gt_3 = p_gt_3[low_selector]
        low_p_gt_4 = p_gt_4[low_selector]
        mean_threshold_penalty_low_only = float(low_penalty.detach().mean().cpu())
        mean_p_gt_3_low = float(low_p_gt_3.detach().mean().cpu())
        mean_p_gt_4_low = float(low_p_gt_4.detach().mean().cpu())
        max_p_gt_3_low = float(low_p_gt_3.detach().max().cpu())
        max_p_gt_4_low = float(low_p_gt_4.detach().max().cpu())
    else:
        mean_threshold_penalty_low_only = 0.0
        mean_p_gt_3_low = 0.0
        mean_p_gt_4_low = 0.0
        max_p_gt_3_low = 0.0
        max_p_gt_4_low = 0.0
    debug = {
        "mean_threshold_penalty": float(penalty.detach().mean().cpu()),
        "mean_threshold_penalty_low_only": mean_threshold_penalty_low_only,
        "active_low_count": float(low_count),
        "mean_p_gt_3_low": mean_p_gt_3_low,
        "mean_p_gt_4_low": mean_p_gt_4_low,
        "max_p_gt_3_low": max_p_gt_3_low,
        "max_p_gt_4_low": max_p_gt_4_low,
    }
    return penalty, debug


def weighted_threshold_ordinal_loss(
    logits: Any,
    label_5: Any,
    class_weights: Any,
    mu_thr: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    if mu_thr <= 0:
        raise ValueError("mu_thr must be > 0.")
    labels = torch.as_tensor(label_5, device=logits.device).long()
    if labels.numel() and ((labels < 1).any() or (labels > 5).any()):
        raise ValueError("label_5 must be in 1..5 for weighted threshold ordinal loss.")
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.float().dtype)
    if weights.ndim != 1 or weights.shape[0] <= 5:
        raise ValueError("class_weights must be a 1D tensor/list indexable by label_5 values 1..5.")
    ordinal_targets = make_ordinal_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
    base_loss_per_sample = ordinal_bce_loss_per_sample(logits, ordinal_targets)
    sample_weights = weights[labels]
    denominator = sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).eps)
    weighted_base_loss = (sample_weights * base_loss_per_sample).sum() / denominator
    penalty_per_sample, penalty_debug = threshold_low_score_penalty(logits, labels)
    threshold_penalty = penalty_per_sample.mean()
    loss = weighted_base_loss + float(mu_thr) * threshold_penalty
    if torch.isnan(loss):
        raise FloatingPointError("weighted threshold ordinal loss became NaN.")
    debug = {
        "weighted_base_loss": float(weighted_base_loss.detach().cpu()),
        "threshold_penalty": float(threshold_penalty.detach().cpu()),
        "mu_thr": float(mu_thr),
        "mean_total_loss": float(loss.detach().cpu()),
        "mean_sample_weight": float(sample_weights.detach().mean().cpu()),
        "min_sample_weight": float(sample_weights.detach().min().cpu()),
        "max_sample_weight": float(sample_weights.detach().max().cpu()),
        "active_low_count": penalty_debug["active_low_count"],
        "mean_p_gt_3_low": penalty_debug["mean_p_gt_3_low"],
        "mean_p_gt_4_low": penalty_debug["mean_p_gt_4_low"],
        "max_p_gt_3_low": penalty_debug["max_p_gt_3_low"],
        "max_p_gt_4_low": penalty_debug["max_p_gt_4_low"],
    }
    return loss, debug
