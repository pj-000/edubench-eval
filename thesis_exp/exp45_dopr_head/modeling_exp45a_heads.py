"""Cosine heads, locked losses, and balanced batches for Exp45A."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def normalize_rows(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def balanced_batch_indices(
    labels: np.ndarray,
    *,
    fold: int,
    epoch: int,
    seed: int = 42,
    rows_per_class: int = 20,
) -> Iterator[np.ndarray]:
    """Yield deterministic 20-per-class batches with replacement only as needed."""

    labels = np.asarray(labels, dtype=np.int64)
    pools = [np.flatnonzero(labels == label) for label in range(1, 6)]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError(f"Exp45A balanced sampler found an empty class: {[len(pool) for pool in pools]}")
    batches = max(math.ceil(len(labels) / (rows_per_class * 5)), 1)
    rng = np.random.default_rng(seed + 10_000 * fold + 100 * epoch)
    for _ in range(batches):
        pieces = [
            rng.choice(pool, size=rows_per_class, replace=len(pool) < rows_per_class)
            for pool in pools
        ]
        indices = np.concatenate(pieces)
        rng.shuffle(indices)
        yield indices


def build_head(dimension: int, prototypes: np.ndarray | None = None, scale: float = 16.0):
    import torch
    from torch import nn

    class CosineHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(5, dimension))
            self.scale = float(scale)
            if prototypes is None:
                nn.init.xavier_uniform_(self.weight)
            else:
                value = torch.as_tensor(normalize_rows(prototypes), dtype=torch.float32)
                if tuple(value.shape) != (5, dimension):
                    raise ValueError(f"Prototype shape mismatch: {tuple(value.shape)}")
                self.weight.data.copy_(value)

        def forward(self, embeddings):
            normalized_h = nn.functional.normalize(embeddings, dim=-1)
            normalized_w = nn.functional.normalize(self.weight, dim=-1)
            return self.scale * normalized_h @ normalized_w.t()

    return CosineHead()


def hard_ce_loss(logits, labels):
    import torch.nn.functional as functional

    return functional.cross_entropy(logits, labels - 1)


def distributional_ordinal_loss(logits, targets):
    import torch
    import torch.nn.functional as functional

    log_probs = functional.log_softmax(logits, dim=-1)
    distribution = -(targets * log_probs).sum(dim=-1).mean()
    probabilities = torch.softmax(logits, dim=-1)
    ordinal = ((probabilities.cumsum(-1)[:, :4] - targets.cumsum(-1)[:, :4]) ** 2).mean()
    return distribution + 0.5 * ordinal, distribution, ordinal


def restored_logits(evidence_logits, priors):
    import torch

    prior = torch.as_tensor(priors, dtype=evidence_logits.dtype, device=evidence_logits.device)
    if tuple(prior.shape) != (5,) or not torch.isfinite(prior).all() or torch.any(prior <= 0):
        raise ValueError("Exp45A priors must be five finite positive values")
    return evidence_logits + torch.log(prior + 1e-12)


__all__ = [
    "balanced_batch_indices",
    "build_head",
    "distributional_ordinal_loss",
    "hard_ce_loss",
    "normalize_rows",
    "restored_logits",
]
