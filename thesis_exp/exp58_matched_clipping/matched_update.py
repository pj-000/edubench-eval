"""Pure component-wise gradient composition for the Exp58 matched control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MatchedScalars:
    common_norm: float
    routed_norm: float
    residual_norm: float
    common_residual_dot: float
    alpha_common: float
    alpha_routed: float
    beta: float
    beta_cap_active: bool
    matched_norm: float


def clip_coefficient(norm: float, *, max_norm: float = 1.0, epsilon: float = 1e-6) -> float:
    if norm < 0.0 or max_norm <= 0.0 or epsilon < 0.0:
        raise ValueError("Invalid clipping inputs")
    return min(1.0, max_norm / (norm + epsilon))


def largest_safe_beta(
    *,
    common_sq: float,
    residual_sq: float,
    common_residual_dot: float,
    alpha_common: float,
    alpha_routed: float,
    max_norm: float = 1.0,
    tolerance: float = 1e-12,
) -> tuple[float, bool]:
    """Return the largest beta <= alpha_routed satisfying the norm bound."""

    values = (
        common_sq,
        residual_sq,
        common_residual_dot,
        alpha_common,
        alpha_routed,
        max_norm,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite matched-clipping scalar")
    if common_sq < 0.0 or residual_sq < 0.0 or alpha_common < 0.0 or alpha_routed < 0.0:
        raise ValueError("Squared norms and coefficients must be non-negative")
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    if residual_sq <= tolerance:
        return 0.0, False
    preferred_sq = (
        alpha_common * alpha_common * common_sq
        + 2.0 * alpha_common * alpha_routed * common_residual_dot
        + alpha_routed * alpha_routed * residual_sq
    )
    if preferred_sq <= max_norm * max_norm + tolerance:
        return alpha_routed, False
    a = residual_sq
    b = 2.0 * alpha_common * common_residual_dot
    c = alpha_common * alpha_common * common_sq - max_norm * max_norm
    discriminant = max(0.0, b * b - 4.0 * a * c)
    positive_root = (-b + math.sqrt(discriminant)) / (2.0 * a)
    beta = min(alpha_routed, max(0.0, positive_root))
    return beta, beta < alpha_routed - tolerance


def summarize_components(
    common: Mapping[str, Any],
    residual: Mapping[str, Any],
    *,
    max_norm: float = 1.0,
    epsilon: float = 1e-6,
) -> MatchedScalars:
    """Compute frozen scalar coefficients without modifying gradients."""

    import numpy as np

    if common.keys() != residual.keys():
        raise ValueError("Common and residual parameter supports differ")
    common_sq = 0.0
    residual_sq = 0.0
    common_residual_dot = 0.0
    for name in common:
        left_value = common[name]
        right_value = residual[name]
        left = (
            left_value.detach().double().cpu().numpy()
            if hasattr(left_value, "detach")
            else np.asarray(left_value, dtype=np.float64)
        )
        right = (
            right_value.detach().double().cpu().numpy()
            if hasattr(right_value, "detach")
            else np.asarray(right_value, dtype=np.float64)
        )
        if left.shape != right.shape:
            raise ValueError(f"Gradient shape mismatch for {name}")
        common_sq += float(np.square(left).sum())
        residual_sq += float(np.square(right).sum())
        common_residual_dot += float((left * right).sum())
    routed_sq = common_sq + 2.0 * common_residual_dot + residual_sq
    routed_sq = max(0.0, routed_sq)
    common_norm = math.sqrt(common_sq)
    routed_norm = math.sqrt(routed_sq)
    residual_norm = math.sqrt(residual_sq)
    alpha_common = clip_coefficient(common_norm, max_norm=max_norm, epsilon=epsilon)
    alpha_routed = clip_coefficient(routed_norm, max_norm=max_norm, epsilon=epsilon)
    beta, cap_active = largest_safe_beta(
        common_sq=common_sq,
        residual_sq=residual_sq,
        common_residual_dot=common_residual_dot,
        alpha_common=alpha_common,
        alpha_routed=alpha_routed,
        max_norm=max_norm,
    )
    matched_sq = (
        alpha_common * alpha_common * common_sq
        + 2.0 * alpha_common * beta * common_residual_dot
        + beta * beta * residual_sq
    )
    return MatchedScalars(
        common_norm=common_norm,
        routed_norm=routed_norm,
        residual_norm=residual_norm,
        common_residual_dot=common_residual_dot,
        alpha_common=alpha_common,
        alpha_routed=alpha_routed,
        beta=beta,
        beta_cap_active=cap_active,
        matched_norm=math.sqrt(max(0.0, matched_sq)),
    )


def compose_matched_gradients(
    common: Mapping[str, Any],
    residual: Mapping[str, Any],
    scalars: MatchedScalars,
) -> dict[str, Any]:
    if common.keys() != residual.keys():
        raise ValueError("Common and residual parameter supports differ")
    return {
        name: scalars.alpha_common * common[name] + scalars.beta * residual[name]
        for name in common
    }
