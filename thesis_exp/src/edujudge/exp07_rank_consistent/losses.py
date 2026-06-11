"""Loss and target helpers for Exp7 CORAL ordinal training."""

from __future__ import annotations

import math
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import ORDINAL_THRESHOLDS


def _labels_to_list(label_5: Any) -> list[int]:
    if hasattr(label_5, "detach"):
        return [int(value) for value in label_5.detach().cpu().reshape(-1).tolist()]
    if isinstance(label_5, (int, float)):
        return [int(label_5)]
    return [int(value) for value in label_5]


def _validate_labels(labels: list[int]) -> None:
    bad = [value for value in labels if value < 1 or value > 5]
    if bad:
        raise ValueError(f"label_5 must be in 1..5 for CORAL ordinal loss, got {bad[:5]}")


def make_ordinal_targets(label_5: Any) -> Any:
    labels_list = _labels_to_list(label_5)
    _validate_labels(labels_list)
    try:
        import torch
    except ModuleNotFoundError:
        return [[1.0 if label > threshold else 0.0 for threshold in ORDINAL_THRESHOLDS] for label in labels_list]

    labels = torch.as_tensor(label_5, dtype=torch.long)
    if hasattr(label_5, "device"):
        labels = labels.to(device=label_5.device)
    thresholds = torch.tensor(ORDINAL_THRESHOLDS, dtype=torch.long, device=labels.device)
    return (labels.reshape(-1, 1) > thresholds.reshape(1, -1)).to(dtype=torch.float32)


def _stable_bce_with_logits(logit: float, target: float) -> float:
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def _fallback_coral_loss(logits: Any, label_5: Any) -> float:
    logit_rows = [[float(value) for value in row] for row in logits]
    if any(len(row) != 4 for row in logit_rows):
        raise ValueError("Expected logits shape [batch,4] for CORAL ordinal loss.")
    targets = make_ordinal_targets(label_5)
    if len(targets) != len(logit_rows):
        raise ValueError("logits batch size and label_5 length do not match.")
    per_sample = []
    for logit_row, target_row in zip(logit_rows, targets):
        per_sample.append(sum(_stable_bce_with_logits(z, h) for z, h in zip(logit_row, target_row)) / 4.0)
    loss = sum(per_sample) / len(per_sample) if per_sample else float("nan")
    if not math.isfinite(loss):
        raise FloatingPointError("CORAL ordinal loss became non-finite.")
    return loss


def coral_ordinal_loss(logits: Any, label_5: Any) -> Any:
    try:
        import torch
        from torch.nn import functional as F
    except ModuleNotFoundError:
        return _fallback_coral_loss(logits, label_5)

    if logits.shape[-1] != 4:
        raise ValueError(f"Expected logits shape [batch,4] for CORAL ordinal loss, got {tuple(logits.shape)}")
    labels = torch.as_tensor(label_5, device=logits.device, dtype=torch.long)
    labels_list = [int(value) for value in labels.detach().cpu().reshape(-1).tolist()]
    _validate_labels(labels_list)
    targets = make_ordinal_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
    loss_per_threshold = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    loss = loss_per_threshold.mean(dim=1).mean()
    if bool(torch.isnan(loss).detach().cpu()):
        raise FloatingPointError("CORAL ordinal loss became NaN.")
    return loss
