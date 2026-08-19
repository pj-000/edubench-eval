"""Frozen geometry selection surface for Exp61.

The projection/matching kernel is the already FP64-audited Exp60 kernel.  This
module deliberately narrows its public variants to the three Exp61 arms and
renames the common-only arm according to the Soft-STS-15 target semantics.
"""

from __future__ import annotations

from typing import Any

from thesis_exp.exp60_geometry_matched_shuffle.geometry import (
    MatchAudit,
    match_shuffled_orthogonal,
)
from thesis_exp.exp60_geometry_matched_shuffle.train import (
    compose_geometry_step as _compose_exp60_geometry_step,
)


def select_component(
    variant: str,
    common: dict[str, Any],
    aligned: dict[str, Any],
    matched_shuffled: dict[str, Any],
) -> dict[str, Any]:
    if variant == "quantized_mean_only":
        return {name: 0.0 * value for name, value in common.items()}
    if variant == "aligned_orthogonal_only":
        return aligned
    if variant == "matched_shuffled_orthogonal_only":
        return matched_shuffled
    raise ValueError(f"unknown Exp61 variant: {variant}")


def compose_geometry_step(
    model: Any,
    aligned_residual_buffers: dict[str, Any],
    shuffled_residual_buffers: dict[str, Any],
    *,
    variant: str,
    max_norm: float,
) -> dict[str, Any]:
    alias = "consensus_only" if variant == "quantized_mean_only" else variant
    audit = _compose_exp60_geometry_step(
        model,
        aligned_residual_buffers,
        shuffled_residual_buffers,
        variant=alias,
        max_norm=max_norm,
    )
    audit["exp61_variant"] = variant
    audit["geometry_kernel_variant"] = alias
    return audit


__all__ = [
    "MatchAudit",
    "match_shuffled_orthogonal",
    "select_component",
    "compose_geometry_step",
]
