"""Loss helpers for Exp9 risk-aware pairwise ordinal training."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LAMBDA_PAIR,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    ORDINAL_THRESHOLDS,
)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def sigmoid(values: Any) -> Any:
    if _is_torch_tensor(values):
        import torch

        return torch.sigmoid(values)
    arr = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))


def scalar_score_from_logits(logits: Any) -> Any:
    if _is_torch_tensor(logits):
        return 1.0 + sigmoid(logits.float()).sum(dim=-1)
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected ordinal logits shape [batch,4], got {arr.shape}")
    return 1.0 + sigmoid(arr).sum(axis=1)


def make_ordinal_targets(label_5: Any) -> Any:
    if _is_torch_tensor(label_5):
        import torch

        labels = label_5.reshape(-1).long()
        thresholds = torch.tensor(ORDINAL_THRESHOLDS, device=labels.device, dtype=torch.long).reshape(1, 4)
        return (labels.reshape(-1, 1) > thresholds).to(dtype=torch.float32)
    labels = np.asarray(label_5, dtype=np.int64).reshape(-1, 1)
    thresholds = np.asarray(ORDINAL_THRESHOLDS, dtype=np.int64).reshape(1, 4)
    return (labels > thresholds).astype(np.float64)


def pair_margin(
    label_gap: Any,
    low_high: Any,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN,
) -> Any:
    if _is_torch_tensor(label_gap):
        return float(margin_scale) * (label_gap.float() / 4.0) + float(low_high_margin) * low_high.float()
    gap = np.asarray(label_gap, dtype=np.float64)
    risk = np.asarray(low_high, dtype=np.float64)
    return float(margin_scale) * (gap / 4.0) + float(low_high_margin) * risk


def pair_weight(
    label_gap: Any,
    low_high: Any,
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
) -> Any:
    if _is_torch_tensor(label_gap):
        return 1.0 + float(low_high_weight) * low_high.float() + float(gap_weight) * ((label_gap.float() - 1.0) / 3.0)
    gap = np.asarray(label_gap, dtype=np.float64)
    risk = np.asarray(low_high, dtype=np.float64)
    return 1.0 + float(low_high_weight) * risk + float(gap_weight) * ((gap - 1.0) / 3.0)


def _stable_softplus(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0) + np.log1p(np.exp(-np.abs(values)))


def _stable_bce_with_logits(logit: float, target: float) -> float:
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def weighted_ordinal_bce(logits: Any, label_5: Any, class_weights: Any) -> tuple[Any, dict[str, float]]:
    """QD-B1-style weighted ordinal BCE with torch and numpy fallback."""
    if _is_torch_tensor(logits):
        import torch
        from torch.nn import functional as F

        labels = label_5.reshape(-1).long().to(device=logits.device)
        weights = torch.as_tensor(class_weights, dtype=logits.float().dtype, device=logits.device)
        targets = make_ordinal_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
        per_sample = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(dim=1)
        sample_weights = weights[labels]
        loss = (sample_weights * per_sample).sum() / sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).eps)
        debug = {
            "mean_point_base_loss": float(per_sample.detach().mean().cpu()),
            "mean_point_sample_weight": float(sample_weights.detach().mean().cpu()),
            "min_point_sample_weight": float(sample_weights.detach().min().cpu()),
            "max_point_sample_weight": float(sample_weights.detach().max().cpu()),
        }
        return loss, debug
    logits_arr = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(label_5, dtype=np.int64).reshape(-1)
    targets = make_ordinal_targets(labels)
    weights = np.asarray(class_weights, dtype=np.float64).reshape(-1)
    per_sample = np.array(
        [np.mean([_stable_bce_with_logits(float(z), float(t)) for z, t in zip(row, target)]) for row, target in zip(logits_arr, targets)]
    )
    sample_weights = weights[labels]
    loss = float(np.sum(sample_weights * per_sample) / max(np.sum(sample_weights), 1e-12))
    return loss, {
        "mean_point_base_loss": float(np.mean(per_sample)),
        "mean_point_sample_weight": float(np.mean(sample_weights)),
        "min_point_sample_weight": float(np.min(sample_weights)),
        "max_point_sample_weight": float(np.max(sample_weights)),
    }


def pairwise_ordinal_loss(
    win_logits: Any,
    lose_logits: Any,
    label_gap: Any,
    low_high: Any,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN,
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(win_logits):
        import torch
        from torch.nn import functional as F

        gap = label_gap.to(device=win_logits.device).float().reshape(-1)
        risk = low_high.to(device=win_logits.device).float().reshape(-1)
        margins = pair_margin(gap, risk, margin_scale, low_high_margin)
        weights = pair_weight(gap, risk, low_high_weight, gap_weight)
        score_gap = scalar_score_from_logits(win_logits) - scalar_score_from_logits(lose_logits)
        losses = weights * F.softplus(margins - score_gap)
        loss = losses.mean()
        low_high_mask = risk > 0.5
        adjacent_mask = gap == 1
        debug = {
            "L_pair": float(loss.detach().cpu()),
            "weighted_L_pair": float(loss.detach().cpu()),
            "mean_pair_weight": float(weights.detach().mean().cpu()),
            "mean_pair_margin": float(margins.detach().mean().cpu()),
            "mean_score_gap": float(score_gap.detach().mean().cpu()),
            "low_high_pair_loss": float(losses[low_high_mask].detach().mean().cpu()) if bool(low_high_mask.any().detach().cpu()) else 0.0,
            "adjacent_pair_loss": float(losses[adjacent_mask].detach().mean().cpu()) if bool(adjacent_mask.any().detach().cpu()) else 0.0,
        }
        return loss, debug
    gap_arr = np.asarray(label_gap, dtype=np.float64).reshape(-1)
    risk_arr = np.asarray(low_high, dtype=np.float64).reshape(-1)
    margins = pair_margin(gap_arr, risk_arr, margin_scale, low_high_margin)
    weights = pair_weight(gap_arr, risk_arr, low_high_weight, gap_weight)
    score_gap = scalar_score_from_logits(win_logits) - scalar_score_from_logits(lose_logits)
    losses = weights * _stable_softplus(margins - score_gap)
    loss = float(np.mean(losses))
    return loss, {
        "L_pair": loss,
        "weighted_L_pair": loss,
        "mean_pair_weight": float(np.mean(weights)),
        "mean_pair_margin": float(np.mean(margins)),
        "mean_score_gap": float(np.mean(score_gap)),
        "low_high_pair_loss": float(np.mean(losses[risk_arr > 0.5])) if np.any(risk_arr > 0.5) else 0.0,
        "adjacent_pair_loss": float(np.mean(losses[gap_arr == 1])) if np.any(gap_arr == 1) else 0.0,
    }


def total_pairwise_training_loss(
    win_logits: Any,
    lose_logits: Any,
    win_labels: Any,
    lose_labels: Any,
    label_gap: Any,
    low_high: Any,
    class_weights: Any,
    lambda_pair: float = DEFAULT_LAMBDA_PAIR,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN,
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(win_logits):
        import torch

        point_logits = torch.cat([win_logits, lose_logits], dim=0)
        point_labels = torch.cat([win_labels.reshape(-1), lose_labels.reshape(-1)], dim=0)
        point_loss, point_debug = weighted_ordinal_bce(point_logits, point_labels, class_weights)
        pair_loss, pair_debug = pairwise_ordinal_loss(
            win_logits,
            lose_logits,
            label_gap,
            low_high,
            margin_scale,
            low_high_margin,
            low_high_weight,
            gap_weight,
        )
        total = point_loss + float(lambda_pair) * pair_loss
        debug = {
            "L_total": float(total.detach().cpu()),
            "L_point": float(point_loss.detach().cpu()),
            **pair_debug,
            **point_debug,
        }
        return total, debug
    point_logits = np.concatenate([np.asarray(win_logits), np.asarray(lose_logits)], axis=0)
    point_labels = np.concatenate([np.asarray(win_labels).reshape(-1), np.asarray(lose_labels).reshape(-1)], axis=0)
    point_loss, point_debug = weighted_ordinal_bce(point_logits, point_labels, class_weights)
    pair_loss, pair_debug = pairwise_ordinal_loss(
        win_logits,
        lose_logits,
        label_gap,
        low_high,
        margin_scale,
        low_high_margin,
        low_high_weight,
        gap_weight,
    )
    total = float(point_loss + float(lambda_pair) * pair_loss)
    return total, {"L_total": total, "L_point": float(point_loss), **pair_debug, **point_debug}
