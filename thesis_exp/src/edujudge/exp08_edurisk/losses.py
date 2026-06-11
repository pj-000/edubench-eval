"""EduRisk ordinal loss components for Exp8."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp08_edurisk import (
    DEFAULT_ALPHA_RISK,
    DEFAULT_BETA_BCE,
    DEFAULT_CLASS_BALANCE_BETA,
    DEFAULT_LAMBDA_HL,
    DEFAULT_LAMBDA_LH,
    DEFAULT_TAU,
    LABELS,
    ORDINAL_THRESHOLDS,
)
from thesis_exp.src.edujudge.exp08_edurisk.coral_distribution import (
    _is_torch_tensor,
    assert_q_raw_sane,
    q_from_logits,
    q_raw_sanity,
)


EPS = 1e-8


def _cfg(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _labels_to_list(label_5: Any) -> list[int]:
    if _is_torch_tensor(label_5):
        return [int(value) for value in label_5.detach().cpu().reshape(-1).tolist()]
    if isinstance(label_5, (int, float)):
        return [int(label_5)]
    arr = np.asarray(label_5).reshape(-1)
    return [int(value) for value in arr.tolist()]


def _validate_labels(labels: list[int]) -> None:
    bad = [value for value in labels if value not in LABELS]
    if bad:
        raise ValueError(f"label_5 must be in 1..5 for EduRisk loss, got {bad[:5]}")


def make_soft_ordinal_targets(label_5: Any, tau: float = DEFAULT_TAU) -> Any:
    """Return soft ordinal targets s_y(k) proportional to exp(-abs(k-y)/tau)."""
    if tau <= 0:
        raise ValueError("tau must be positive for soft ordinal targets.")
    labels = _labels_to_list(label_5)
    _validate_labels(labels)
    if _is_torch_tensor(label_5):
        import torch

        y = label_5.to(dtype=torch.float32).reshape(-1, 1)
        classes = torch.arange(1, 6, dtype=torch.float32, device=label_5.device).reshape(1, 5)
        scores = torch.exp(-torch.abs(classes - y) / float(tau))
        return scores / scores.sum(dim=1, keepdim=True).clamp(min=EPS)
    arr_y = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    classes = np.arange(1, 6, dtype=np.float64).reshape(1, 5)
    scores = np.exp(-np.abs(classes - arr_y) / float(tau))
    return scores / np.clip(scores.sum(axis=1, keepdims=True), EPS, None)


def make_cumulative_targets(label_5: Any) -> Any:
    """Return CORAL cumulative targets I(label > threshold)."""
    labels = _labels_to_list(label_5)
    _validate_labels(labels)
    if _is_torch_tensor(label_5):
        import torch

        thresholds = torch.tensor(ORDINAL_THRESHOLDS, dtype=torch.long, device=label_5.device)
        return (label_5.reshape(-1, 1).long() > thresholds.reshape(1, -1)).to(dtype=torch.float32)
    arr = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
    thresholds = np.asarray(ORDINAL_THRESHOLDS, dtype=np.int64).reshape(1, 4)
    return (arr > thresholds).astype(np.float64)


def make_edurisk_cost_matrix(
    label_5: Any,
    lambda_lh: float = DEFAULT_LAMBDA_LH,
    lambda_hl: float = DEFAULT_LAMBDA_HL,
    normalized_cost: bool = True,
) -> Any:
    """Return per-sample 5-class risk costs C(y,k)."""
    labels = _labels_to_list(label_5)
    _validate_labels(labels)
    if _is_torch_tensor(label_5):
        import torch

        y = label_5.to(dtype=torch.float32).reshape(-1, 1)
        k = torch.arange(1, 6, dtype=torch.float32, device=label_5.device).reshape(1, 5)
        scale = 4.0 if normalized_cost else 1.0
        base = torch.abs(y - k) / scale
        lh = ((y <= 2.0) & (k >= 4.0)).to(dtype=torch.float32) * torch.square((k - y) / scale)
        hl = ((y == 5.0) & (k <= 3.0)).to(dtype=torch.float32) * torch.square((5.0 - k) / scale)
        return base + float(lambda_lh) * lh + float(lambda_hl) * hl
    arr_y = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    k = np.arange(1, 6, dtype=np.float64).reshape(1, 5)
    scale = 4.0 if normalized_cost else 1.0
    base = np.abs(arr_y - k) / scale
    lh = ((arr_y <= 2) & (k >= 4)).astype(np.float64) * np.square((k - arr_y) / scale)
    hl = ((arr_y == 5) & (k <= 3)).astype(np.float64) * np.square((5 - k) / scale)
    return base + float(lambda_lh) * lh + float(lambda_hl) * hl


def effective_number_weights_from_counts(
    counts: dict[int, int],
    beta: float = DEFAULT_CLASS_BALANCE_BETA,
) -> list[dict[str, float | int]]:
    """Return normalized effective-number weights with mean weight equal to one."""
    if beta <= 0 or beta >= 1:
        raise ValueError("class_balance_beta must be in (0,1).")
    raw_values: list[float] = []
    rows: list[dict[str, float | int]] = []
    for label in LABELS:
        n = int(counts.get(label, 0))
        if n <= 0:
            effective_number = 0.0
            raw_weight = 0.0
        else:
            effective_number = (1.0 - math.pow(beta, n)) / (1.0 - beta)
            raw_weight = 1.0 / effective_number
        raw_values.append(raw_weight)
        rows.append(
            {
                "label_5": label,
                "count": n,
                "effective_number": effective_number,
                "raw_weight": raw_weight,
                "normalized_weight": 0.0,
            }
        )
    positive = [value for value in raw_values if value > 0]
    normalizer = (len(LABELS) / sum(positive)) if positive else 0.0
    for row, raw_weight in zip(rows, raw_values):
        row["normalized_weight"] = raw_weight * normalizer if raw_weight > 0 else 0.0
    return rows


def weight_vector_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    weights = np.ones(5, dtype=np.float64)
    for row in rows:
        label = int(row["label_5"])
        weights[label - 1] = float(row["normalized_weight"])
    return weights


def _stable_bce_with_logits(logit: float, target: float) -> float:
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def _numpy_loss(logits: Any, label_5: Any, config: Any, class_weights: Any) -> tuple[float, dict[str, float]]:
    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim != 2 or logits_arr.shape[1] != 4:
        raise ValueError(f"Expected logits shape [batch,4], got {logits_arr.shape}")
    labels = np.asarray(_labels_to_list(label_5), dtype=np.int64)
    _validate_labels(labels.tolist())
    if labels.shape[0] != logits_arr.shape[0]:
        raise ValueError("logits batch size and label_5 length do not match.")
    tau = float(_cfg(config, "tau", DEFAULT_TAU))
    alpha = float(_cfg(config, "alpha_risk", DEFAULT_ALPHA_RISK))
    beta_bce = float(_cfg(config, "beta_bce", DEFAULT_BETA_BCE))
    lambda_lh = float(_cfg(config, "lambda_lh", DEFAULT_LAMBDA_LH))
    lambda_hl = float(_cfg(config, "lambda_hl", DEFAULT_LAMBDA_HL))
    normalized_cost = bool(_cfg(config, "normalized_cost", True))

    _, q_raw, q_safe = q_from_logits(logits_arr)
    assert_q_raw_sane(q_raw)
    soft = np.asarray(make_soft_ordinal_targets(labels, tau=tau), dtype=np.float64)
    costs = np.asarray(
        make_edurisk_cost_matrix(labels, lambda_lh=lambda_lh, lambda_hl=lambda_hl, normalized_cost=normalized_cost),
        dtype=np.float64,
    )
    targets = np.asarray(make_cumulative_targets(labels), dtype=np.float64)
    weights = np.asarray(class_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != 5:
        raise ValueError("class_weights must contain 5 values.")
    sample_weights = weights[labels - 1]

    soft_ce = -np.sum(soft * np.log(np.clip(q_safe, EPS, 1.0)), axis=1)
    risk = np.sum(q_safe * costs, axis=1)
    bce = np.asarray(
        [
            np.mean([_stable_bce_with_logits(float(z), float(t)) for z, t in zip(logit_row, target_row)])
            for logit_row, target_row in zip(logits_arr, targets)
        ],
        dtype=np.float64,
    )
    total_per_sample = sample_weights * (soft_ce + alpha * risk + beta_bce * bce)
    total = float(np.mean(total_per_sample)) if total_per_sample.size else float("nan")
    if not math.isfinite(total):
        raise FloatingPointError("EduRisk loss became non-finite.")
    sanity = q_raw_sanity(q_raw)
    debug = {
        "L_total": total,
        "L_softCE": float(np.mean(soft_ce)),
        "L_risk": float(np.mean(risk)),
        "L_cumBCE": float(np.mean(bce)),
        "mean_weight": float(np.mean(sample_weights)),
        "min_weight": float(np.min(sample_weights)),
        "max_weight": float(np.max(sample_weights)),
        "weighted_L_softCE": float(np.mean(sample_weights * soft_ce)),
        "weighted_L_risk": float(np.mean(sample_weights * alpha * risk)),
        "weighted_L_cumBCE": float(np.mean(sample_weights * beta_bce * bce)),
        "mean_q_raw_min": float(np.mean(np.min(q_raw, axis=1))),
        "mean_q_raw_sum": float(sanity["mean_q_raw_sum"]),
        "max_q_raw_sum_error": float(sanity["max_q_raw_sum_error"]),
    }
    return total, debug


def edurisk_loss(logits: Any, label_5: Any, config: Any, class_weights: Any) -> tuple[Any, dict[str, float]]:
    """Compute EduRisk loss and a scalar debug dictionary.

    The torch branch returns a differentiable scalar loss. The fallback branch
    returns a Python float for local sanity checks without torch installed.
    """
    if not _is_torch_tensor(logits):
        return _numpy_loss(logits, label_5, config, class_weights)

    import torch
    from torch.nn import functional as F

    if logits.shape[-1] != 4:
        raise ValueError(f"Expected logits shape [batch,4], got {tuple(logits.shape)}")
    labels = label_5.to(device=logits.device, dtype=torch.long).reshape(-1)
    _validate_labels([int(value) for value in labels.detach().cpu().tolist()])
    tau = float(_cfg(config, "tau", DEFAULT_TAU))
    alpha = float(_cfg(config, "alpha_risk", DEFAULT_ALPHA_RISK))
    beta_bce = float(_cfg(config, "beta_bce", DEFAULT_BETA_BCE))
    lambda_lh = float(_cfg(config, "lambda_lh", DEFAULT_LAMBDA_LH))
    lambda_hl = float(_cfg(config, "lambda_hl", DEFAULT_LAMBDA_HL))
    normalized_cost = bool(_cfg(config, "normalized_cost", True))

    _, q_raw, q_safe = q_from_logits(logits.float())
    assert_q_raw_sane(q_raw)
    soft = make_soft_ordinal_targets(labels, tau=tau).to(device=logits.device, dtype=logits.float().dtype)
    costs = make_edurisk_cost_matrix(
        labels,
        lambda_lh=lambda_lh,
        lambda_hl=lambda_hl,
        normalized_cost=normalized_cost,
    ).to(device=logits.device, dtype=logits.float().dtype)
    targets = make_cumulative_targets(labels).to(device=logits.device, dtype=logits.float().dtype)
    weights = torch.as_tensor(class_weights, dtype=logits.float().dtype, device=logits.device).reshape(5)
    sample_weights = weights[labels - 1]

    soft_ce = -(soft * torch.log(q_safe.float().clamp(min=EPS))).sum(dim=1)
    risk = (q_safe.float() * costs).sum(dim=1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(dim=1)
    loss = (sample_weights * (soft_ce + alpha * risk + beta_bce * bce)).mean()
    if bool(torch.isnan(loss).detach().cpu()):
        raise FloatingPointError("EduRisk loss became NaN.")
    sanity = q_raw_sanity(q_raw)
    row_mins = torch.min(q_raw.float(), dim=1).values
    debug = {
        "L_total": float(loss.detach().cpu()),
        "L_softCE": float(soft_ce.mean().detach().cpu()),
        "L_risk": float(risk.mean().detach().cpu()),
        "L_cumBCE": float(bce.mean().detach().cpu()),
        "mean_weight": float(sample_weights.mean().detach().cpu()),
        "min_weight": float(sample_weights.min().detach().cpu()),
        "max_weight": float(sample_weights.max().detach().cpu()),
        "weighted_L_softCE": float((sample_weights * soft_ce).mean().detach().cpu()),
        "weighted_L_risk": float((sample_weights * alpha * risk).mean().detach().cpu()),
        "weighted_L_cumBCE": float((sample_weights * beta_bce * bce).mean().detach().cpu()),
        "mean_q_raw_min": float(row_mins.mean().detach().cpu()),
        "mean_q_raw_sum": float(sanity["mean_q_raw_sum"]),
        "max_q_raw_sum_error": float(sanity["max_q_raw_sum_error"]),
    }
    return loss, debug
