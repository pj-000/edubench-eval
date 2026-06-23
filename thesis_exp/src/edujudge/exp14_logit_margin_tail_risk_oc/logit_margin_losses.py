"""Logit-margin tail-risk losses for Exp14."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _to_float(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _quantile(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _threshold_index(threshold_t: int) -> int:
    threshold = int(threshold_t)
    if threshold < 1 or threshold > 4:
        raise ValueError(f"logit_margin_risk_threshold_t must be in [1,4], got {threshold_t}")
    return threshold - 1


def l2h_logit_margin_tail_loss_from_logits(
    logits: Any,
    labels: Any,
    risk_threshold_t: int = 3,
    loss_type: str = "squared_hinge",
    margin_prob_label1: float = 0.35,
    margin_prob_label2: float = 0.45,
    margin_logit_label1: float | None = None,
    margin_logit_label2: float | None = None,
    weight_label1: float = 1.0,
    weight_label2: float = 2.0,
    tail_fraction: float = 1.0,
    min_topk: int = 1,
) -> tuple[Any, dict[str, float]]:
    """Penalize the high-risk top tail of low-score samples on raw threshold logits."""

    if str(loss_type) != "squared_hinge":
        raise ValueError(f"Unsupported logit_margin_loss_type: {loss_type}")
    idx = _threshold_index(risk_threshold_t)
    margin1 = _logit(margin_prob_label1) if margin_logit_label1 is None else float(margin_logit_label1)
    margin2 = _logit(margin_prob_label2) if margin_logit_label2 is None else float(margin_logit_label2)
    fraction = min(max(float(tail_fraction), 0.0), 1.0)
    min_k = max(1, int(min_topk))

    if _is_torch_tensor(logits):
        import torch

        labels_t = labels.reshape(-1).long().to(device=logits.device)
        z3 = logits.float()[:, idx]
        low_mask = labels_t <= 2
        if not bool(low_mask.any().detach().cpu()):
            zero = logits.float().sum() * 0.0
            return zero, {
                "L_l2h_logit_margin": 0.0,
                "l2h_logit_margin_loss": 0.0,
                "low_logit_margin_active_count": 0.0,
                "low_logit_margin_tail_count": 0.0,
                "logit_margin_risk_threshold_t": float(risk_threshold_t),
                "logit_margin_tail_fraction": fraction,
                "margin_logit_label1": margin1,
                "margin_logit_label2": margin2,
                "low_z3_mean": float("nan"),
                "low_z3_q90": float("nan"),
                "low_z3_q95": float("nan"),
                "label1_z3_mean": float("nan"),
                "label2_z3_mean": float("nan"),
            }
        margins = torch.where(
            labels_t == 1,
            torch.full_like(z3, margin1),
            torch.full_like(z3, margin2),
        )
        weights = torch.where(
            labels_t == 1,
            torch.full_like(z3, float(weight_label1)),
            torch.full_like(z3, float(weight_label2)),
        )
        per_sample = weights * torch.relu(z3 - margins).pow(2)
        low_values = per_sample[low_mask]
        low_count = int(low_values.numel())
        topk_count = min(low_count, max(min_k, int(math.ceil(low_count * fraction)))) if fraction > 0.0 else min(low_count, min_k)
        tail_values = torch.topk(low_values, k=topk_count, largest=True).values
        loss = tail_values.mean()
        low_z3 = z3[low_mask]
        label1_z3 = z3[labels_t == 1]
        label2_z3 = z3[labels_t == 2]
        return loss, {
            "L_l2h_logit_margin": _to_float(loss),
            "l2h_logit_margin_loss": _to_float(loss),
            "low_logit_margin_active_count": float(low_count),
            "low_logit_margin_tail_count": float(topk_count),
            "logit_margin_risk_threshold_t": float(risk_threshold_t),
            "logit_margin_tail_fraction": fraction,
            "logit_margin_weight_label1": float(weight_label1),
            "logit_margin_weight_label2": float(weight_label2),
            "margin_prob_label1": float(margin_prob_label1),
            "margin_prob_label2": float(margin_prob_label2),
            "margin_logit_label1": margin1,
            "margin_logit_label2": margin2,
            "low_z3_mean": _to_float(low_z3.mean()),
            "low_z3_q90": _to_float(torch.quantile(low_z3.float(), 0.90)),
            "low_z3_q95": _to_float(torch.quantile(low_z3.float(), 0.95)),
            "label1_z3_mean": _to_float(label1_z3.mean()) if label1_z3.numel() else float("nan"),
            "label2_z3_mean": _to_float(label2_z3.mean()) if label2_z3.numel() else float("nan"),
        }

    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    z3_arr = np.asarray(logits, dtype=np.float64)[:, idx]
    low_mask_arr = labels_arr <= 2
    if not np.any(low_mask_arr):
        return 0.0, {
            "L_l2h_logit_margin": 0.0,
            "l2h_logit_margin_loss": 0.0,
            "low_logit_margin_active_count": 0.0,
            "low_logit_margin_tail_count": 0.0,
            "logit_margin_risk_threshold_t": float(risk_threshold_t),
            "logit_margin_tail_fraction": fraction,
            "margin_logit_label1": margin1,
            "margin_logit_label2": margin2,
            "low_z3_mean": float("nan"),
            "low_z3_q90": float("nan"),
            "low_z3_q95": float("nan"),
            "label1_z3_mean": float("nan"),
            "label2_z3_mean": float("nan"),
        }
    margins_arr = np.where(labels_arr == 1, margin1, margin2)
    weights_arr = np.where(labels_arr == 1, float(weight_label1), float(weight_label2))
    per_sample_arr = weights_arr * np.maximum(0.0, z3_arr - margins_arr) ** 2
    low_values_arr = per_sample_arr[low_mask_arr]
    low_count_arr = int(low_values_arr.size)
    topk_count_arr = min(low_count_arr, max(min_k, int(math.ceil(low_count_arr * fraction)))) if fraction > 0.0 else min(low_count_arr, min_k)
    tail_values_arr = np.sort(low_values_arr)[-topk_count_arr:]
    loss_value = float(np.mean(tail_values_arr))
    low_z3_arr = z3_arr[low_mask_arr]
    return loss_value, {
        "L_l2h_logit_margin": loss_value,
        "l2h_logit_margin_loss": loss_value,
        "low_logit_margin_active_count": float(low_count_arr),
        "low_logit_margin_tail_count": float(topk_count_arr),
        "logit_margin_risk_threshold_t": float(risk_threshold_t),
        "logit_margin_tail_fraction": fraction,
        "logit_margin_weight_label1": float(weight_label1),
        "logit_margin_weight_label2": float(weight_label2),
        "margin_prob_label1": float(margin_prob_label1),
        "margin_prob_label2": float(margin_prob_label2),
        "margin_logit_label1": margin1,
        "margin_logit_label2": margin2,
        "low_z3_mean": _mean(low_z3_arr),
        "low_z3_q90": _quantile(low_z3_arr, 0.90),
        "low_z3_q95": _quantile(low_z3_arr, 0.95),
        "label1_z3_mean": _mean(z3_arr[labels_arr == 1]),
        "label2_z3_mean": _mean(z3_arr[labels_arr == 2]),
    }
