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
    return scalar_score_from_logits_with_projection(logits, use_projection=False)


def scalar_score_from_logits_with_projection(logits: Any, use_projection: bool = False) -> Any:
    if _is_torch_tensor(logits):
        probs = sigmoid(logits.float())
        if use_projection:
            from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import (
                project_nonincreasing_probs_torch,
            )

            probs = project_nonincreasing_probs_torch(probs)
        return 1.0 + probs.sum(dim=-1)
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected ordinal logits shape [batch,4], got {arr.shape}")
    probs = sigmoid(arr)
    if use_projection:
        from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import project_nonincreasing_probs

        probs = project_nonincreasing_probs(probs)
    return 1.0 + probs.sum(axis=1)


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


def weighted_ordinal_bce_from_probs(probs: Any, label_5: Any, class_weights: Any) -> tuple[Any, dict[str, float]]:
    """Weighted ordinal BCE on already-projected probabilities."""
    if _is_torch_tensor(probs):
        import torch
        from torch.nn import functional as F

        labels = label_5.reshape(-1).long().to(device=probs.device)
        weights = torch.as_tensor(class_weights, dtype=probs.float().dtype, device=probs.device)
        targets = make_ordinal_targets(labels).to(device=probs.device, dtype=probs.float().dtype)
        clipped = probs.float().clamp(1e-6, 1.0 - 1e-6)
        per_sample = F.binary_cross_entropy(clipped, targets, reduction="none").mean(dim=1)
        sample_weights = weights[labels]
        loss = (sample_weights * per_sample).sum() / sample_weights.sum().clamp_min(torch.finfo(sample_weights.dtype).eps)
        debug = {
            "mean_point_base_loss": float(per_sample.detach().mean().cpu()),
            "mean_point_sample_weight": float(sample_weights.detach().mean().cpu()),
            "min_point_sample_weight": float(sample_weights.detach().min().cpu()),
            "max_point_sample_weight": float(sample_weights.detach().max().cpu()),
            "point_loss_uses_projected_probs": 1.0,
        }
        return loss, debug
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(label_5, dtype=np.int64).reshape(-1)
    targets = make_ordinal_targets(labels)
    weights = np.asarray(class_weights, dtype=np.float64).reshape(-1)
    clipped = np.clip(probs_arr, 1e-6, 1.0 - 1e-6)
    per_sample = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped), axis=1)
    sample_weights = weights[labels]
    loss = float(np.sum(sample_weights * per_sample) / max(np.sum(sample_weights), 1e-12))
    return loss, {
        "mean_point_base_loss": float(np.mean(per_sample)),
        "mean_point_sample_weight": float(np.mean(sample_weights)),
        "min_point_sample_weight": float(np.min(sample_weights)),
        "max_point_sample_weight": float(np.max(sample_weights)),
        "point_loss_uses_projected_probs": 1.0,
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
    use_projected_score: bool = False,
) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(win_logits):
        import torch
        from torch.nn import functional as F

        gap = label_gap.to(device=win_logits.device).float().reshape(-1)
        risk = low_high.to(device=win_logits.device).float().reshape(-1)
        margins = pair_margin(gap, risk, margin_scale, low_high_margin)
        weights = pair_weight(gap, risk, low_high_weight, gap_weight)
        score_gap = scalar_score_from_logits_with_projection(win_logits, use_projected_score) - scalar_score_from_logits_with_projection(
            lose_logits, use_projected_score
        )
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
            "pair_score_uses_projection": 1.0 if use_projected_score else 0.0,
        }
        return loss, debug
    gap_arr = np.asarray(label_gap, dtype=np.float64).reshape(-1)
    risk_arr = np.asarray(low_high, dtype=np.float64).reshape(-1)
    margins = pair_margin(gap_arr, risk_arr, margin_scale, low_high_margin)
    weights = pair_weight(gap_arr, risk_arr, low_high_weight, gap_weight)
    score_gap = scalar_score_from_logits_with_projection(win_logits, use_projected_score) - scalar_score_from_logits_with_projection(
        lose_logits, use_projected_score
    )
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
        "pair_score_uses_projection": 1.0 if use_projected_score else 0.0,
    }


def anchor_bce_with_logits(logits: Any, ref_logits: Any) -> tuple[Any, dict[str, float]]:
    """Anchor current logits to QD-B1 reference probabilities."""
    if _is_torch_tensor(logits):
        from torch.nn import functional as F

        target_probs = sigmoid(ref_logits.float()).detach()
        loss = F.binary_cross_entropy_with_logits(logits.float(), target_probs, reduction="mean")
        return loss, {
            "L_anchor": float(loss.detach().cpu()),
            "mean_anchor_target_prob": float(target_probs.detach().mean().cpu()),
        }
    logits_arr = np.asarray(logits, dtype=np.float64)
    ref_arr = np.asarray(ref_logits, dtype=np.float64)
    target_probs = sigmoid(ref_arr)
    losses = np.array(
        [
            _stable_bce_with_logits(float(logit), float(target))
            for row, target_row in zip(logits_arr, target_probs)
            for logit, target in zip(row, target_row)
        ],
        dtype=np.float64,
    )
    loss = float(np.mean(losses)) if losses.size else 0.0
    return loss, {"L_anchor": loss, "mean_anchor_target_prob": float(np.mean(target_probs)) if target_probs.size else 0.0}


def monotonic_regularization(logits: Any) -> tuple[Any, dict[str, float]]:
    """Penalize cumulative probability increases p_{t+1} > p_t."""
    if _is_torch_tensor(logits):
        import torch

        probs = sigmoid(logits.float())
        violations = torch.relu(probs[:, 1:] - probs[:, :-1])
        loss = violations.mean()
        violation_rate = (violations > 0).float().mean()
        return loss, {
            "L_mono": float(loss.detach().cpu()),
            "mono_pair_violation_rate": float(violation_rate.detach().cpu()),
        }
    probs_arr = sigmoid(np.asarray(logits, dtype=np.float64))
    violations_arr = np.maximum(0.0, probs_arr[:, 1:] - probs_arr[:, :-1])
    return float(np.mean(violations_arr)), {
        "L_mono": float(np.mean(violations_arr)),
        "mono_pair_violation_rate": float(np.mean(violations_arr > 0.0)),
    }


def projected_probs_from_logits(logits: Any) -> Any:
    if _is_torch_tensor(logits):
        from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import (
            project_nonincreasing_probs_torch,
        )

        return project_nonincreasing_probs_torch(sigmoid(logits.float()))
    from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import project_nonincreasing_probs

    return project_nonincreasing_probs(sigmoid(np.asarray(logits, dtype=np.float64)))


def anchor_projected_mse(logits: Any, ref_logits: Any, use_projected_anchor: bool) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(logits):
        from torch.nn import functional as F

        current_probs = projected_probs_from_logits(logits)
        ref_probs = projected_probs_from_logits(ref_logits) if use_projected_anchor else sigmoid(ref_logits.float())
        loss = F.mse_loss(current_probs.float(), ref_probs.detach().float(), reduction="mean")
        return loss, {
            "L_anchor": float(loss.detach().cpu()),
            "mean_anchor_target_prob": float(ref_probs.detach().mean().cpu()),
            "anchor_uses_projected_current": 1.0,
            "anchor_uses_projected_reference": 1.0 if use_projected_anchor else 0.0,
        }
    current_probs = projected_probs_from_logits(logits)
    ref_probs = projected_probs_from_logits(ref_logits) if use_projected_anchor else sigmoid(np.asarray(ref_logits, dtype=np.float64))
    loss = float(np.mean((current_probs - ref_probs) ** 2)) if current_probs.size else 0.0
    return loss, {
        "L_anchor": loss,
        "mean_anchor_target_prob": float(np.mean(ref_probs)) if np.asarray(ref_probs).size else 0.0,
        "anchor_uses_projected_current": 1.0,
        "anchor_uses_projected_reference": 1.0 if use_projected_anchor else 0.0,
    }


def raw_projection_consistency(logits: Any, eta_proj: float = 0.1) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(logits):
        import torch

        probs = sigmoid(logits.float())
        projected = projected_probs_from_logits(logits)
        order_loss = torch.relu(probs[:, 1:] - probs[:, :-1]).pow(2).mean()
        delta_loss = (probs - projected).pow(2).mean()
        loss = order_loss + float(eta_proj) * delta_loss
        violation_rate = (probs[:, 1:] < probs[:, :-1]).float().mean()
        actual_violation_rate = (probs[:, 1:] > probs[:, :-1]).float().mean()
        return loss, {
            "L_mono": float(loss.detach().cpu()),
            "L_raw_order": float(order_loss.detach().cpu()),
            "L_raw_proj_delta": float(delta_loss.detach().cpu()),
            "eta_proj": float(eta_proj),
            "mono_pair_violation_rate": float(actual_violation_rate.detach().cpu()),
            "mono_pair_nonviolation_rate": float(violation_rate.detach().cpu()),
        }
    probs_arr = sigmoid(np.asarray(logits, dtype=np.float64))
    projected_arr = projected_probs_from_logits(logits)
    order_loss = np.maximum(0.0, probs_arr[:, 1:] - probs_arr[:, :-1]) ** 2
    delta_loss = (probs_arr - projected_arr) ** 2
    loss = float(np.mean(order_loss) + float(eta_proj) * np.mean(delta_loss))
    return loss, {
        "L_mono": loss,
        "L_raw_order": float(np.mean(order_loss)),
        "L_raw_proj_delta": float(np.mean(delta_loss)),
        "eta_proj": float(eta_proj),
        "mono_pair_violation_rate": float(np.mean(probs_arr[:, 1:] > probs_arr[:, :-1])),
        "mono_pair_nonviolation_rate": float(np.mean(probs_arr[:, 1:] < probs_arr[:, :-1])),
    }


def total_anchored_pairwise_training_loss(
    win_logits: Any,
    lose_logits: Any,
    ref_win_logits: Any | None,
    ref_lose_logits: Any | None,
    win_labels: Any,
    lose_labels: Any,
    label_gap: Any,
    low_high: Any,
    class_weights: Any,
    lambda_point: float = 1.0,
    lambda_pair: float = DEFAULT_LAMBDA_PAIR,
    lambda_anchor: float = 0.0,
    lambda_mono: float = 0.0,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    low_high_margin: float = DEFAULT_LOW_HIGH_MARGIN,
    low_high_weight: float = DEFAULT_LOW_HIGH_WEIGHT,
    gap_weight: float = DEFAULT_GAP_WEIGHT,
    projection_in_pair_score: bool = False,
    projection_in_point_loss: bool = False,
    projection_in_anchor: bool = False,
    use_projected_anchor: bool = False,
    use_raw_projection_consistency: bool = False,
    eta_proj: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    if _is_torch_tensor(win_logits):
        import torch

        point_logits = torch.cat([win_logits, lose_logits], dim=0)
        point_labels = torch.cat([win_labels.reshape(-1), lose_labels.reshape(-1)], dim=0)
        if projection_in_point_loss:
            point_loss, point_debug = weighted_ordinal_bce_from_probs(
                projected_probs_from_logits(point_logits),
                point_labels,
                class_weights,
            )
        else:
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
            use_projected_score=projection_in_pair_score,
        )
        if float(lambda_anchor) != 0.0:
            if ref_win_logits is None or ref_lose_logits is None:
                raise ValueError("lambda_anchor is non-zero but reference logits are missing.")
            ref_logits = torch.cat([ref_win_logits, ref_lose_logits], dim=0)
            if projection_in_anchor:
                anchor_loss, anchor_debug = anchor_projected_mse(point_logits, ref_logits, use_projected_anchor)
            else:
                anchor_loss, anchor_debug = anchor_bce_with_logits(point_logits, ref_logits)
        else:
            anchor_loss = point_logits.float().sum() * 0.0
            anchor_debug = {"L_anchor": 0.0, "mean_anchor_target_prob": 0.0}
        if use_raw_projection_consistency:
            mono_loss, mono_debug = raw_projection_consistency(point_logits, eta_proj=eta_proj)
        else:
            mono_loss, mono_debug = monotonic_regularization(point_logits)
        total = (
            float(lambda_point) * point_loss
            + float(lambda_pair) * pair_loss
            + float(lambda_anchor) * anchor_loss
            + float(lambda_mono) * mono_loss
        )
        point_value = float(point_loss.detach().cpu())
        pair_value = float(pair_loss.detach().cpu())
        anchor_value = float(anchor_loss.detach().cpu())
        mono_value = float(mono_loss.detach().cpu())
        total_value = float(total.detach().cpu())
        debug = {
            "L_total": total_value,
            "loss_total": total_value,
            "L_point": point_value,
            "loss_point": point_value,
            "weighted_loss_point": float(lambda_point) * point_value,
            **pair_debug,
            "loss_pair_raw": pair_value,
            "loss_pair": pair_value,
            "weighted_loss_pair": float(lambda_pair) * pair_value,
            **anchor_debug,
            "loss_anchor": anchor_value,
            "weighted_loss_anchor": float(lambda_anchor) * anchor_value,
            **mono_debug,
            "loss_mono": mono_value,
            "weighted_loss_mono": float(lambda_mono) * mono_value,
            **point_debug,
            "projection_in_pair_score": 1.0 if projection_in_pair_score else 0.0,
            "projection_in_point_loss": 1.0 if projection_in_point_loss else 0.0,
            "projection_in_anchor": 1.0 if projection_in_anchor else 0.0,
            "use_projected_anchor": 1.0 if use_projected_anchor else 0.0,
            "use_raw_projection_consistency": 1.0 if use_raw_projection_consistency else 0.0,
        }
        return total, debug
    point_logits_arr = np.concatenate([np.asarray(win_logits), np.asarray(lose_logits)], axis=0)
    point_labels_arr = np.concatenate([np.asarray(win_labels).reshape(-1), np.asarray(lose_labels).reshape(-1)], axis=0)
    if projection_in_point_loss:
        point_loss, point_debug = weighted_ordinal_bce_from_probs(
            projected_probs_from_logits(point_logits_arr),
            point_labels_arr,
            class_weights,
        )
    else:
        point_loss, point_debug = weighted_ordinal_bce(point_logits_arr, point_labels_arr, class_weights)
    pair_loss, pair_debug = pairwise_ordinal_loss(
        win_logits,
        lose_logits,
        label_gap,
        low_high,
        margin_scale,
        low_high_margin,
        low_high_weight,
        gap_weight,
        use_projected_score=projection_in_pair_score,
    )
    if float(lambda_anchor) != 0.0:
        if ref_win_logits is None or ref_lose_logits is None:
            raise ValueError("lambda_anchor is non-zero but reference logits are missing.")
        ref_logits_arr = np.concatenate([np.asarray(ref_win_logits), np.asarray(ref_lose_logits)], axis=0)
        if projection_in_anchor:
            anchor_loss, anchor_debug = anchor_projected_mse(point_logits_arr, ref_logits_arr, use_projected_anchor)
        else:
            anchor_loss, anchor_debug = anchor_bce_with_logits(point_logits_arr, ref_logits_arr)
    else:
        anchor_loss = 0.0
        anchor_debug = {"L_anchor": 0.0, "mean_anchor_target_prob": 0.0}
    if use_raw_projection_consistency:
        mono_loss, mono_debug = raw_projection_consistency(point_logits_arr, eta_proj=eta_proj)
    else:
        mono_loss, mono_debug = monotonic_regularization(point_logits_arr)
    total = float(
        float(lambda_point) * point_loss
        + float(lambda_pair) * pair_loss
        + float(lambda_anchor) * anchor_loss
        + float(lambda_mono) * mono_loss
    )
    point_value = float(point_loss)
    pair_value = float(pair_loss)
    anchor_value = float(anchor_loss)
    mono_value = float(mono_loss)
    return total, {
        "L_total": total,
        "loss_total": total,
        "L_point": point_value,
        "loss_point": point_value,
        "weighted_loss_point": float(lambda_point) * point_value,
        **pair_debug,
        "loss_pair_raw": pair_value,
        "loss_pair": pair_value,
        "weighted_loss_pair": float(lambda_pair) * pair_value,
        **anchor_debug,
        "loss_anchor": anchor_value,
        "weighted_loss_anchor": float(lambda_anchor) * anchor_value,
        **mono_debug,
        "loss_mono": mono_value,
        "weighted_loss_mono": float(lambda_mono) * mono_value,
        **point_debug,
        "projection_in_pair_score": 1.0 if projection_in_pair_score else 0.0,
        "projection_in_point_loss": 1.0 if projection_in_point_loss else 0.0,
        "projection_in_anchor": 1.0 if projection_in_anchor else 0.0,
        "use_projected_anchor": 1.0 if use_projected_anchor else 0.0,
        "use_raw_projection_consistency": 1.0 if use_raw_projection_consistency else 0.0,
    }


def total_anchored_pointwise_training_loss(
    logits: Any,
    ref_logits: Any | None,
    labels: Any,
    class_weights: Any,
    lambda_point: float = 1.0,
    lambda_anchor: float = 0.0,
    lambda_mono: float = 0.0,
    projection_in_point_loss: bool = False,
    projection_in_anchor: bool = False,
    use_projected_anchor: bool = False,
    use_raw_projection_consistency: bool = False,
    eta_proj: float = 0.1,
) -> tuple[Any, dict[str, float]]:
    """Pointwise-only anchored objective for lambda_pair=0 ablations."""
    if projection_in_point_loss:
        point_loss, point_debug = weighted_ordinal_bce_from_probs(projected_probs_from_logits(logits), labels, class_weights)
    else:
        point_loss, point_debug = weighted_ordinal_bce(logits, labels, class_weights)
    if float(lambda_anchor) != 0.0:
        if ref_logits is None:
            raise ValueError("lambda_anchor is non-zero but reference logits are missing.")
        if projection_in_anchor:
            anchor_loss, anchor_debug = anchor_projected_mse(logits, ref_logits, use_projected_anchor)
        else:
            anchor_loss, anchor_debug = anchor_bce_with_logits(logits, ref_logits)
    elif _is_torch_tensor(logits):
        anchor_loss = logits.float().sum() * 0.0
        anchor_debug = {"L_anchor": 0.0, "mean_anchor_target_prob": 0.0}
    else:
        anchor_loss = 0.0
        anchor_debug = {"L_anchor": 0.0, "mean_anchor_target_prob": 0.0}
    if use_raw_projection_consistency:
        mono_loss, mono_debug = raw_projection_consistency(logits, eta_proj=eta_proj)
    else:
        mono_loss, mono_debug = monotonic_regularization(logits)
    total = float(lambda_point) * point_loss + float(lambda_anchor) * anchor_loss + float(lambda_mono) * mono_loss
    if _is_torch_tensor(logits):
        point_value = float(point_loss.detach().cpu())
        anchor_value = float(anchor_loss.detach().cpu())
        mono_value = float(mono_loss.detach().cpu())
        total_value = float(total.detach().cpu())
        debug = {
            "L_total": total_value,
            "loss_total": total_value,
            "L_point": point_value,
            "loss_point": point_value,
            "weighted_loss_point": float(lambda_point) * point_value,
            "L_pair": 0.0,
            "weighted_L_pair": 0.0,
            "loss_pair_raw": 0.0,
            "loss_pair": 0.0,
            "weighted_loss_pair": 0.0,
            "mean_pair_weight": 0.0,
            "mean_pair_margin": 0.0,
            "mean_score_gap": 0.0,
            "low_high_pair_loss": 0.0,
            "adjacent_pair_loss": 0.0,
            **anchor_debug,
            "loss_anchor": anchor_value,
            "weighted_loss_anchor": float(lambda_anchor) * anchor_value,
            **mono_debug,
            "loss_mono": mono_value,
            "weighted_loss_mono": float(lambda_mono) * mono_value,
            **point_debug,
            "projection_in_point_loss": 1.0 if projection_in_point_loss else 0.0,
            "projection_in_anchor": 1.0 if projection_in_anchor else 0.0,
            "use_projected_anchor": 1.0 if use_projected_anchor else 0.0,
            "use_raw_projection_consistency": 1.0 if use_raw_projection_consistency else 0.0,
        }
        return total, debug
    point_value = float(point_loss)
    anchor_value = float(anchor_loss)
    mono_value = float(mono_loss)
    return float(total), {
        "L_total": float(total),
        "loss_total": float(total),
        "L_point": point_value,
        "loss_point": point_value,
        "weighted_loss_point": float(lambda_point) * point_value,
        "L_pair": 0.0,
        "weighted_L_pair": 0.0,
        "loss_pair_raw": 0.0,
        "loss_pair": 0.0,
        "weighted_loss_pair": 0.0,
        "mean_pair_weight": 0.0,
        "mean_pair_margin": 0.0,
        "mean_score_gap": 0.0,
        "low_high_pair_loss": 0.0,
        "adjacent_pair_loss": 0.0,
        **anchor_debug,
        "loss_anchor": anchor_value,
        "weighted_loss_anchor": float(lambda_anchor) * anchor_value,
        **mono_debug,
        "loss_mono": mono_value,
        "weighted_loss_mono": float(lambda_mono) * mono_value,
        **point_debug,
        "projection_in_point_loss": 1.0 if projection_in_point_loss else 0.0,
        "projection_in_anchor": 1.0 if projection_in_anchor else 0.0,
        "use_projected_anchor": 1.0 if use_projected_anchor else 0.0,
        "use_raw_projection_consistency": 1.0 if use_raw_projection_consistency else 0.0,
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
