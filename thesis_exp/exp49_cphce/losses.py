"""The only two allowed Exp49 losses."""

from __future__ import annotations

from typing import Any


def hard_cross_entropy(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    return F.cross_entropy(logits.float(), labels)


def soft_cross_entropy(logits: Any, targets: Any) -> Any:
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(targets.float() * log_probs).sum(dim=-1).mean()
