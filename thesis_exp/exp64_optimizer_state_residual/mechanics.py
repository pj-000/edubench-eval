"""Frozen mathematical primitives for Exp64.

The functions in this module do not load data, evaluate outcomes or choose a
checkpoint.  They exist so that the AdamW finite-difference and shared-scaling
contracts can be tested independently of the real model.
"""

from __future__ import annotations

import math
from typing import Any


def squared_l2(tensors: dict[str, Any]) -> float:
    """Return an FP64-reduced squared L2 norm for named tensors."""

    return sum(float(value.detach().double().square().sum()) for value in tensors.values())


def l2_norm(tensors: dict[str, Any]) -> float:
    return math.sqrt(max(0.0, squared_l2(tensors)))


def shared_scale(
    candidates: dict[str, dict[str, Any]], *, target_norm: float = 0.95
) -> tuple[float, dict[str, float]]:
    """Compute the one scale shared by every complete candidate gradient."""

    if not candidates:
        raise ValueError("Exp64 shared scaling requires at least one candidate")
    if not 0.0 < target_norm < 1.0:
        raise ValueError("Exp64 target norm must lie strictly between zero and one")
    norms = {name: l2_norm(values) for name, values in candidates.items()}
    return shared_scale_from_norms(norms, target_norm=target_norm), norms


def shared_scale_from_norms(
    norms: dict[str, float], *, target_norm: float = 0.95
) -> float:
    """Compute the frozen shared scale from precomputed FP64 norms."""

    if not norms:
        raise ValueError("Exp64 shared scaling requires at least one candidate")
    if not 0.0 < target_norm < 1.0:
        raise ValueError("Exp64 target norm must lie strictly between zero and one")
    if not all(math.isfinite(value) for value in norms.values()):
        raise RuntimeError(f"non-finite Exp64 candidate norm: {norms}")
    maximum = max(norms.values())
    if maximum <= 0.0:
        raise RuntimeError("all Exp64 candidate gradients are zero")
    return min(1.0, target_norm / maximum)


def _step_value(step: Any) -> float:
    if hasattr(step, "item"):
        return float(step.item())
    return float(step)


def adamw_parameter_after_step(
    parameter: Any,
    gradient: Any,
    state: dict[str, Any],
    group: dict[str, Any],
) -> Any:
    """Reproduce one non-fused, non-AMSGrad PyTorch AdamW parameter step.

    Operations deliberately follow PyTorch's order and preserve the parameter
    and optimizer-state dtypes.  The base state is never mutated.
    """

    import torch

    if bool(group.get("amsgrad", False)):
        raise NotImplementedError("Exp64 freezes amsgrad=False")
    if bool(group.get("maximize", False)):
        gradient = -gradient
    beta1, beta2 = (float(value) for value in group["betas"])
    learning_rate = float(group["lr"])
    weight_decay = float(group["weight_decay"])
    epsilon = float(group["eps"])
    next_step = _step_value(state["step"]) + 1.0

    value = parameter.detach().clone()
    grad = gradient.detach().to(device=value.device, dtype=value.dtype)
    exp_avg = state["exp_avg"].detach().to(value.device).clone()
    exp_avg_sq = state["exp_avg_sq"].detach().to(value.device).clone()

    value.mul_(1.0 - learning_rate * weight_decay)
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step
    denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(epsilon)
    value.addcdiv_(exp_avg, denominator, value=-(learning_rate / bias_correction1))
    if not torch.isfinite(value).all():
        raise RuntimeError("non-finite exact AdamW parameter value")
    return value


def exact_adamw_displacement(
    parameters: dict[str, Any],
    gradients: dict[str, Any],
    states: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact one-step displacement for every named parameter."""

    if set(parameters) != set(gradients) or set(parameters) != set(states):
        raise RuntimeError("Exp64 parameter/gradient/optimizer-state keys differ")
    missing_groups = set(parameters) - set(groups)
    if missing_groups:
        raise RuntimeError(f"Exp64 optimizer groups missing parameters: {sorted(missing_groups)}")
    return {
        name: adamw_parameter_after_step(
            parameter, gradients[name], states[name], groups[name]
        ).sub(parameter.detach())
        for name, parameter in parameters.items()
    }


def subtract_displacements(
    candidate: dict[str, Any], blocked: dict[str, Any]
) -> dict[str, Any]:
    if set(candidate) != set(blocked):
        raise RuntimeError("Exp64 displacement keys differ")
    return {name: candidate[name] - blocked[name] for name in candidate}


def fixed_denominator_attributable_displacement(
    blocked_gradients: dict[str, Any],
    candidate_gradients: dict[str, Any],
    states: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Frozen-denominator Adam approximation to candidate-minus-blocked.

    The candidate changes the first moment, while the denominator is the one
    produced by the Blocked gradient.  Weight decay cancels from this
    attributable difference.
    """

    if set(blocked_gradients) != set(candidate_gradients):
        raise RuntimeError("Exp64 fixed-denominator gradient keys differ")
    result: dict[str, Any] = {}
    for name, blocked in blocked_gradients.items():
        group = groups[name]
        state = states[name]
        beta1, beta2 = (float(value) for value in group["betas"])
        next_step = _step_value(state["step"]) + 1.0
        bias_correction1 = 1.0 - beta1**next_step
        bias_correction2 = 1.0 - beta2**next_step
        blocked_value = blocked.detach().to(
            device=state["exp_avg_sq"].device, dtype=state["exp_avg_sq"].dtype
        )
        denominator = state["exp_avg_sq"].detach().clone()
        denominator.mul_(beta2).addcmul_(
            blocked_value, blocked_value, value=1.0 - beta2
        )
        denominator.sqrt_().div_(math.sqrt(bias_correction2)).add_(float(group["eps"]))
        moment_difference = (candidate_gradients[name] - blocked).detach().to(
            device=denominator.device, dtype=denominator.dtype
        )
        moment_difference.mul_(1.0 - beta1)
        result[name] = moment_difference.div_(denominator).mul_(
            -(float(group["lr"]) / bias_correction1)
        )
    return result


def dot(left: dict[str, Any], right: dict[str, Any]) -> float:
    """FP64-reduced dot product over common named keys."""

    if set(left) != set(right):
        raise RuntimeError("Exp64 dot-product keys differ")
    return sum(
        float((left[name].detach().double() * right[name].detach().double()).sum())
        for name in left
    )


def kendall_tau_b(
    predicted: list[float], observed: list[float], *, outcome_tie_tolerance: float = 1e-8
) -> float:
    """No-dependency Kendall tau-b with a frozen observed-outcome tie rule."""

    if len(predicted) != len(observed) or len(predicted) < 2:
        raise RuntimeError("Exp64 Kendall inputs are invalid")
    y = [
        0.0 if abs(value) <= outcome_tie_tolerance else float(value)
        for value in observed
    ]
    concordant = discordant = tied_x = tied_y = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            dx = predicted[first] - predicted[second]
            dy = y[first] - y[second]
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                tied_x += 1
            elif dy == 0.0:
                tied_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


__all__ = [
    "adamw_parameter_after_step",
    "dot",
    "exact_adamw_displacement",
    "fixed_denominator_attributable_displacement",
    "kendall_tau_b",
    "l2_norm",
    "shared_scale",
    "shared_scale_from_norms",
    "subtract_displacements",
]
