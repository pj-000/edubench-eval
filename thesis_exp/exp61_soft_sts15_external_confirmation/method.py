"""Exact six-class/five-rating target algebra used by Exp61."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    LABELS,
    NUM_CLASSES,
    NUM_PUBLISHED_RATINGS,
)


def validate_scores(values: Iterable[int]) -> tuple[int, ...]:
    scores = tuple(int(value) for value in values)
    if len(scores) != NUM_PUBLISHED_RATINGS:
        raise ValueError(f"Exp61 requires exactly {NUM_PUBLISHED_RATINGS} ratings")
    if any(value not in LABELS for value in scores):
        raise ValueError("Exp61 ratings must be integers in [0, 5]")
    return scores


def target_fifths(values: Iterable[int]) -> tuple[int, ...]:
    scores = validate_scores(values)
    counts = Counter(scores)
    return tuple(counts[label] for label in LABELS)


def empirical_target(values: Iterable[int]) -> tuple[float, ...]:
    return tuple(value / NUM_PUBLISHED_RATINGS for value in target_fifths(values))


def human_mean(values: Iterable[int]) -> float:
    scores = validate_scores(values)
    return sum(scores) / NUM_PUBLISHED_RATINGS


def quantized_mean_label(values: Iterable[int]) -> int:
    return math.floor(human_mean(values) + 0.5)


def one_hot(label: int) -> tuple[float, ...]:
    if int(label) not in LABELS:
        raise ValueError("hard label must be in [0, 5]")
    return tuple(float(index == int(label)) for index in LABELS)


def residual_target(label: int, target: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in target)
    if len(values) != NUM_CLASSES:
        raise ValueError("target must have six entries")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
        raise ValueError("target must sum to one")
    hard = one_hot(label)
    return tuple(hard_value - soft_value for hard_value, soft_value in zip(hard, values))


def soft_cross_entropy(logits: Any, target: Any) -> Any:
    """Mean soft cross entropy, keeping numerical work in FP32."""

    import torch
    from torch.nn import functional as functional

    if logits.ndim != 2 or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("Exp61 logits must have shape [batch, 6]")
    if target.shape != logits.shape:
        raise ValueError("soft target shape must equal logits shape")
    return -(target.float() * functional.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def hard_soft_ce_identity(logits: Any, labels: Any, targets: Any) -> dict[str, float]:
    """Numerically audit CE(d)=CE(e_y)+<e_y-d,z> for mean losses."""

    import torch
    from torch.nn import functional as functional

    hard = functional.cross_entropy(logits.float(), labels.long())
    soft = soft_cross_entropy(logits, targets)
    one_hot_values = functional.one_hot(labels.long(), NUM_CLASSES).float()
    residual = ((one_hot_values - targets.float()) * logits.float()).sum(-1).mean()
    error = float((soft - hard - residual).abs().detach().cpu())
    return {
        "hard_ce": float(hard.detach().cpu()),
        "soft_ce": float(soft.detach().cpu()),
        "residual_linear_term": float(residual.detach().cpu()),
        "absolute_identity_error": error,
    }
