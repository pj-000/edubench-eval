"""Mathematical primitives for CBRD.

For a hard consensus label ``y`` and the empirical three-rater target ``d``,
the auxiliary soft cross entropy has the exact decomposition

    CE(d, z) = CE(e_y, z) + (e_y - d)^T z.

The second term is called the *signed boundary residual*.  In CBRD it is a
gradient-routing term, not a separately normalized probability target.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


STATE_ZERO = "zero"
STATE_DOWN = "down"
STATE_UP = "up"
VALID_STATES = (STATE_ZERO, STATE_DOWN, STATE_UP)


def thirds(target: Sequence[float]) -> tuple[int, int, int, int, int]:
    """Return a three-rater distribution as exact integer thirds."""

    if len(target) != 5:
        raise ValueError(f"Expected five classes, got {len(target)}")
    values = tuple(int(round(float(value) * 3.0)) for value in target)
    if sum(values) != 3 or any(value < 0 for value in values):
        raise ValueError(f"Not a valid three-rater target: {target}")
    if any(abs(float(value) / 3.0 - float(raw)) > 1e-8 for value, raw in zip(values, target)):
        raise ValueError(f"Target is not on the three-rater grid: {target}")
    return values  # type: ignore[return-value]


def soft_target_from_scores(scores: Sequence[int]) -> tuple[float, float, float, float, float]:
    """Build the empirical five-class distribution from exactly three scores."""

    if len(scores) != 3 or any(int(score) not in (1, 2, 3, 4, 5) for score in scores):
        raise ValueError(f"Expected three scores in 1..5, got {scores}")
    return tuple(scores.count(label) / 3.0 for label in range(1, 6))  # type: ignore[return-value]


def one_hot_thirds(label: int) -> tuple[int, int, int, int, int]:
    if label not in (1, 2, 3, 4, 5):
        raise ValueError(f"label must be in 1..5, got {label}")
    return tuple(3 if index == label - 1 else 0 for index in range(5))  # type: ignore[return-value]


def residual_thirds(label: int, target: Sequence[float]) -> tuple[int, int, int, int, int]:
    """Return ``3 * (e_y - d)`` exactly."""

    hard = one_hot_thirds(label)
    target_values = thirds(target)
    return tuple(left - right for left, right in zip(hard, target_values))  # type: ignore[return-value]


def target_state(label: int, target: Sequence[float]) -> str:
    """Classify an observed target as unanimous, lower-boundary, or upper-boundary.

    This contract intentionally accepts only the patterns that exist in the
    frozen three-rater corpus: ``[k,k,k]``, ``[k-1,k,k]``, and
    ``[k,k,k+1]``.  There are three relation *types*, while the full residual
    vector still has nine possible values (one zero plus four directions on
    each side).
    """

    values = thirds(target)
    if max(range(5), key=values.__getitem__) + 1 != label:
        raise ValueError(f"Target mode does not match hard label {label}: {values}")
    if values[label - 1] == 3:
        return STATE_ZERO
    nonzero = [index for index, value in enumerate(values) if value]
    if len(nonzero) != 2 or values[label - 1] != 2:
        raise ValueError(f"Target is outside the observed consensus/boundary support: {values}")
    neighbor = next(index + 1 for index in nonzero if index != label - 1)
    if abs(neighbor - label) != 1 or values[neighbor - 1] != 1:
        raise ValueError(f"Target is not an adjacent 2:1 split: {values}")
    return STATE_DOWN if neighbor < label else STATE_UP


def human_mean(target: Sequence[float]) -> float:
    return sum((index + 1) * float(value) for index, value in enumerate(target))


def describe_target(label: int, target: Sequence[float]) -> dict[str, Any]:
    state = target_state(label, target)
    values = thirds(target)
    residual = residual_thirds(label, target)
    return {
        "hard_label": label,
        "target_thirds": list(values),
        "state": state,
        "human_mean": human_mean(target),
        "mean_minus_hard_label": human_mean(target) - float(label),
        "residual_thirds": list(residual),
    }


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch.nn import functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - machine dependent
        raise RuntimeError("CBRD tensor audits require a Python environment with PyTorch") from error
    return torch, functional


def soft_cross_entropy(logits: Any, targets: Any) -> Any:
    """The unweighted, temperature-one empirical-distribution CE used by HMSA."""

    _, functional = _torch()
    return -(targets.float() * functional.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def consensus_cross_entropy(logits: Any, labels: Any) -> Any:
    """CE against the hard consensus label, with the same reduction as soft CE."""

    _, functional = _torch()
    return functional.cross_entropy(logits.float(), labels.long())


def boundary_residual(logits: Any, labels: Any, targets: Any) -> Any:
    """Return the scalar residual route ``mean((e_y-d)^T z)``.

    The target is detached: it is an observed annotation, never a learned
    tensor.  This is deliberately not a probability loss by itself.
    """

    torch, functional = _torch()
    hard = functional.one_hot(labels.long(), num_classes=logits.shape[-1]).to(dtype=logits.float().dtype)
    residual = (hard - targets.float()).detach()
    return (residual * logits.float()).sum(dim=-1).mean()


def auxiliary_decomposition(logits: Any, labels: Any, targets: Any) -> dict[str, Any]:
    """Compute both sides of the CE identity and its numerical residual."""

    soft = soft_cross_entropy(logits, targets)
    consensus = consensus_cross_entropy(logits, labels)
    residual = boundary_residual(logits, labels, targets)
    return {
        "soft_ce": soft,
        "consensus_ce": consensus,
        "boundary_residual": residual,
        "reconstructed_soft_ce": consensus + residual,
        "absolute_error": (soft - consensus - residual).abs(),
    }

