"""CORAL-style rank-consistent ordinal scoring head."""

from __future__ import annotations

import math
from typing import Any

try:  # Keep module importable on local machines without torch.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised by setup fallback.
    torch = None
    nn = None
    F = None


_BaseModule = nn.Module if nn is not None else object


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("Softplus inverse requires a positive value.")
    return math.log(math.expm1(value))


class CoralOrdinalHead(_BaseModule):
    """Map hidden states to four ordered ordinal logits.

    The head learns one shared latent quality score ``s = w^T h`` and four
    trainable thresholds ``tau_1 <= tau_2 <= tau_3 <= tau_4``. Logits are
    ``s - tau_t``, so they are monotonic by construction.
    """

    def __init__(self, hidden_size: int, initial_threshold: float = -1.5, initial_gap: float = 1.0) -> None:
        if torch is None or nn is None or F is None:
            raise ModuleNotFoundError("torch is required to instantiate CoralOrdinalHead.")
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_outputs = 4
        self.quality = nn.Linear(self.hidden_size, 1)
        self.threshold_base = nn.Parameter(torch.tensor(float(initial_threshold), dtype=torch.float32))
        self.threshold_deltas = nn.Parameter(
            torch.full((3,), _inverse_softplus(float(initial_gap)), dtype=torch.float32)
        )

    def thresholds(self) -> Any:
        gaps = F.softplus(self.threshold_deltas)
        return torch.cat([self.threshold_base.reshape(1), self.threshold_base + torch.cumsum(gaps, dim=0)])

    def forward(self, hidden: Any) -> Any:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden shape [batch,{self.hidden_size}], got {tuple(hidden.shape)}")
        if hidden.dtype != self.quality.weight.dtype:
            hidden = hidden.to(dtype=self.quality.weight.dtype)
        score = self.quality(hidden)
        logits = score - self.thresholds().to(device=hidden.device, dtype=score.dtype).reshape(1, 4)
        return logits


def assert_rank_consistent(logits: Any, tolerance: float = 1e-6) -> None:
    if torch is None:
        rows = logits
        for row in rows:
            if any(float(row[idx]) + tolerance < float(row[idx + 1]) for idx in range(3)):
                raise AssertionError(f"CORAL logits are not monotonic: {row}")
        return
    if logits.shape[-1] != 4:
        raise AssertionError(f"Expected logits shape [batch,4], got {tuple(logits.shape)}")
    diffs = logits[..., :-1] - logits[..., 1:]
    if bool((diffs < -tolerance).any().detach().cpu()):
        raise AssertionError("CORAL logits are not monotonic within tolerance.")
