"""Geometry-matched aligned and shuffled residual interventions for Exp60."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from thesis_exp.exp59_residual_geometry.geometry import (
    compose_preclip_gradient,
    decompose_residual,
    parameter_group,
)


@dataclass(frozen=True)
class MatchAudit:
    aligned_orthogonal_norm: float
    shuffled_orthogonal_norm_before_match: float
    shuffled_orthogonal_norm_after_match: float
    shuffled_scale: float
    component_norm_relative_error: float
    preclip_total_norm_aligned: float
    preclip_total_norm_shuffled: float
    preclip_total_norm_relative_error: float
    clip_coefficient_aligned: float
    clip_coefficient_shuffled: float
    clip_coefficient_relative_error: float
    aligned_normalized_orthogonality_error: float
    shuffled_normalized_orthogonality_error: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _double(value: Any) -> Any:
    if hasattr(value, "double"):
        return value.double()
    import numpy as np

    return np.asarray(value, dtype=np.float64)


def _sq(values: dict[str, Any], names: set[str] | None = None) -> float:
    selected = set(values) if names is None else names
    return sum(float((_double(values[name]) ** 2).sum()) for name in selected)


def _dot(left: dict[str, Any], right: dict[str, Any], names: set[str]) -> float:
    return sum(float((_double(left[name]) * _double(right[name])).sum()) for name in names)


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _clip_coefficient(norm: float, max_norm: float) -> float:
    return min(1.0, max_norm / (norm + 1e-6))


def match_shuffled_orthogonal(
    common: dict[str, Any],
    aligned_residual: dict[str, Any],
    shuffled_residual: dict[str, Any],
    *,
    max_norm: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any], MatchAudit]:
    """Return aligned and norm-matched shuffled components.

    Both residuals are independently projected against the same current common
    backbone gradient.  The shuffled orthogonal component is then scaled once,
    without search, to the aligned orthogonal norm.
    """

    if common.keys() != aligned_residual.keys() or common.keys() != shuffled_residual.keys():
        raise ValueError("Common and residual parameter sets differ")
    _, aligned, aligned_audit = decompose_residual(common, aligned_residual)
    _, shuffled, shuffled_audit = decompose_residual(common, shuffled_residual)
    aligned_norm = aligned_audit.orthogonal_norm
    shuffled_norm = shuffled_audit.orthogonal_norm
    if aligned_norm > 0.0 and shuffled_norm <= 0.0:
        raise RuntimeError("EXP60_STOP_SHUFFLED_ORTHOGONAL_ZERO_WHILE_ALIGNED_NONZERO")
    scale = aligned_norm / shuffled_norm if shuffled_norm > 0.0 else 0.0
    matched = {
        name: (scale * value if parameter_group(name) == "backbone" else 0.0 * value)
        for name, value in shuffled.items()
    }
    backbone = {name for name in common if parameter_group(name) == "backbone"}
    matched_norm = math.sqrt(max(0.0, _sq(matched, backbone)))
    common_norm = math.sqrt(max(0.0, _sq(common, backbone)))
    matched_dot = _dot(common, matched, backbone)
    matched_orthogonality = abs(matched_dot) / (common_norm * matched_norm + 1e-12)
    aligned_total = compose_preclip_gradient(common, aligned)
    shuffled_total = compose_preclip_gradient(common, matched)
    aligned_total_norm = math.sqrt(max(0.0, _sq(aligned_total)))
    shuffled_total_norm = math.sqrt(max(0.0, _sq(shuffled_total)))
    clip_aligned = _clip_coefficient(aligned_total_norm, max_norm)
    clip_shuffled = _clip_coefficient(shuffled_total_norm, max_norm)
    audit = MatchAudit(
        aligned_orthogonal_norm=aligned_norm,
        shuffled_orthogonal_norm_before_match=shuffled_norm,
        shuffled_orthogonal_norm_after_match=matched_norm,
        shuffled_scale=scale,
        component_norm_relative_error=_relative(aligned_norm, matched_norm),
        preclip_total_norm_aligned=aligned_total_norm,
        preclip_total_norm_shuffled=shuffled_total_norm,
        preclip_total_norm_relative_error=_relative(aligned_total_norm, shuffled_total_norm),
        clip_coefficient_aligned=clip_aligned,
        clip_coefficient_shuffled=clip_shuffled,
        clip_coefficient_relative_error=_relative(clip_aligned, clip_shuffled),
        aligned_normalized_orthogonality_error=aligned_audit.normalized_orthogonality_error,
        shuffled_normalized_orthogonality_error=matched_orthogonality,
    )
    return aligned, matched, audit


def select_component(
    variant: str,
    common: dict[str, Any],
    aligned: dict[str, Any],
    matched_shuffled: dict[str, Any],
) -> dict[str, Any]:
    if variant == "consensus_only":
        return {name: 0.0 * value for name, value in common.items()}
    if variant == "aligned_orthogonal_only":
        return aligned
    if variant == "matched_shuffled_orthogonal_only":
        return matched_shuffled
    raise ValueError(f"Unknown Exp60 variant: {variant}")
