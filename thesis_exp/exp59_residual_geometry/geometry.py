"""Accumulation-window residual geometry for Exp59."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryAudit:
    common_backbone_norm: float
    residual_norm: float
    parallel_norm: float
    orthogonal_norm: float
    residual_common_cosine: float | None
    projection_coefficient: float
    reconstruction_relative_error: float
    normalized_orthogonality_error: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parameter_group(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("hard_head."):
        return "hard_head"
    if name.startswith("soft_head."):
        return "soft_head"
    return "other"


def _dot(left: dict[str, Any], right: dict[str, Any], names: set[str]) -> float:
    return sum(float((_double(left[name]) * _double(right[name])).sum()) for name in names)


def _double(value: Any) -> Any:
    if hasattr(value, "double"):
        return value.double()
    import numpy as np

    return np.asarray(value, dtype=np.float64)


def _zeros_like(value: Any) -> Any:
    if hasattr(value, "new_zeros"):
        return value.new_zeros(value.shape)
    import numpy as np

    return np.zeros_like(value)


def _clone(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return value.copy()


def _sq(values: dict[str, Any], names: set[str]) -> float:
    return _dot(values, values, names)


def _relative_error(left: dict[str, Any], right: dict[str, Any], names: set[str]) -> float:
    numerator = sum(
        float(((_double(left[name]) - _double(right[name])) ** 2).sum())
        for name in names
    )
    denominator = _sq(right, names)
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12)


def decompose_residual(
    common: dict[str, Any],
    residual: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], GeometryAudit]:
    """Project a backbone-only residual onto the common backbone gradient."""

    if common.keys() != residual.keys():
        raise ValueError("common and residual parameter sets differ")
    names = set(common)
    backbone = {name for name in names if parameter_group(name) == "backbone"}
    if not backbone:
        raise ValueError("no trainable backbone parameters")
    common_sq = _sq(common, backbone)
    residual_sq = _sq(residual, backbone)
    common_residual_dot = _dot(common, residual, backbone)
    coefficient = common_residual_dot / common_sq if common_sq > 0.0 else 0.0
    parallel = {}
    orthogonal = {}
    for name in names:
        if name in backbone and common_sq > 0.0:
            parallel[name] = coefficient * common[name]
            orthogonal[name] = residual[name] - parallel[name]
        elif name in backbone:
            parallel[name] = _zeros_like(common[name])
            orthogonal[name] = _clone(residual[name])
        else:
            parallel[name] = _zeros_like(common[name])
            orthogonal[name] = _zeros_like(common[name])
    reconstructed = {name: parallel[name] + orthogonal[name] for name in names}
    parallel_sq = _sq(parallel, backbone)
    orthogonal_sq = _sq(orthogonal, backbone)
    orthogonal_common_dot = _dot(orthogonal, common, backbone)
    common_norm = math.sqrt(max(0.0, common_sq))
    residual_norm = math.sqrt(max(0.0, residual_sq))
    orthogonal_norm = math.sqrt(max(0.0, orthogonal_sq))
    audit = GeometryAudit(
        common_backbone_norm=common_norm,
        residual_norm=residual_norm,
        parallel_norm=math.sqrt(max(0.0, parallel_sq)),
        orthogonal_norm=orthogonal_norm,
        residual_common_cosine=(
            common_residual_dot / (common_norm * residual_norm)
            if common_norm > 0.0 and residual_norm > 0.0
            else None
        ),
        projection_coefficient=coefficient,
        reconstruction_relative_error=_relative_error(reconstructed, residual, names),
        normalized_orthogonality_error=(
            abs(orthogonal_common_dot)
            / (orthogonal_norm * common_norm + 1e-12)
        ),
    )
    return parallel, orthogonal, audit


def compose_preclip_gradient(
    common: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    if common.keys() != component.keys():
        raise ValueError("common and component parameter sets differ")
    return {name: common[name] + component[name] for name in common}
