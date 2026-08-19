"""Loss reporting for the single-soft-CE CBRD routing implementation."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp57_cbrd.method import boundary_residual, consensus_cross_entropy, soft_cross_entropy


VARIANTS = (
    "ordinary_hmsa",
    "routed_hmsa",
    "dual_hard",
    "consensus_only",
    "residual_only",
    "sign_flipped",
    "shuffled_residual",
    "detached_soft",
)


def cbrd_objective(
    outputs: dict[str, Any],
    labels: Any,
    targets: Any,
    *,
    variant: str,
    residual_targets: Any | None = None,
    aux_weight: float = 1.0,
) -> dict[str, Any]:
    """Build a hard-main CBRD objective with exactly one update per group."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown CBRD variant: {variant}")
    hard = consensus_cross_entropy(outputs["hard_logits"], labels)
    ordinary_aux = soft_cross_entropy(outputs["aux_logits"], targets)
    route_consensus = consensus_cross_entropy(outputs["aux_logits"], labels)
    route_residual = boundary_residual(
        outputs["aux_logits"],
        labels,
        targets if residual_targets is None else residual_targets,
    )
    head_hard = consensus_cross_entropy(outputs["aux_logits"], labels)
    if outputs.get("aux_route") != variant:
        raise ValueError(
            f"Forward route {outputs.get('aux_route')} does not match objective {variant}"
        )
    # Gradient routing happens at the auxiliary hidden branch.  Every soft
    # variant uses one identical CE scalar, so the auxiliary head update and
    # learning-curve value are held fixed.  DualHard is the explicit generic
    # second-hard-task control.
    auxiliary_objective = head_hard if variant == "dual_hard" else ordinary_aux
    optimization = hard + aux_weight * auxiliary_objective

    return {
        "optimization_loss": optimization,
        "reported_hard_loss": hard,
        "reported_aux_soft_ce": ordinary_aux,
        "auxiliary_head_soft_ce": ordinary_aux,
        "route_consensus_ce": route_consensus,
        "route_boundary_residual": route_residual,
        "route_reconstructed_soft_ce": route_consensus + route_residual,
    }
