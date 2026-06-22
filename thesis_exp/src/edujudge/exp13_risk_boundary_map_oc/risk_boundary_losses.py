"""Risk-boundary losses for Exp13 MAP-OC variants."""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _to_float(value: Any) -> float:
    if _is_torch_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _nanmean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _risk_probs_from_logits(logits: Any, score_source: str) -> Any:
    source = str(score_source).lower()
    if _is_torch_tensor(logits):
        import torch

        probs = torch.sigmoid(logits.float())
        if source == "raw":
            return probs
        if source == "projected":
            from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import (
                project_nonincreasing_probs_torch,
            )

            return project_nonincreasing_probs_torch(probs)
        raise ValueError(f"Unsupported risk score source: {score_source}")
    probs_arr = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    if source == "raw":
        return probs_arr
    if source == "projected":
        from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import project_nonincreasing_probs

        return project_nonincreasing_probs(probs_arr)
    raise ValueError(f"Unsupported risk score source: {score_source}")


def _threshold_index(threshold_t: int) -> int:
    threshold = int(threshold_t)
    if threshold < 1 or threshold > 4:
        raise ValueError(f"risk_threshold_t must be in [1,4], got {threshold_t}")
    return threshold - 1


def l2h_squared_hinge_risk_loss_from_logits(
    logits: Any,
    labels: Any,
    score_source: str = "projected",
    risk_threshold_t: int = 3,
    risk_loss_type: str = "squared_hinge",
    tau_label1: float = 0.35,
    tau_label2: float = 0.45,
    weight_label1: float = 1.0,
    weight_label2: float = 1.0,
    normalize_by_low_count: bool = True,
) -> tuple[Any, dict[str, float]]:
    """Penalize low-score samples whose P(y > risk_threshold_t) crosses a boundary."""

    if str(risk_loss_type) != "squared_hinge":
        raise ValueError(f"Unsupported risk_loss_type: {risk_loss_type}")
    probs = _risk_probs_from_logits(logits, score_source)
    idx = _threshold_index(risk_threshold_t)
    if _is_torch_tensor(probs):
        import torch

        labels_t = labels.reshape(-1).long().to(device=probs.device)
        scores = probs[:, idx].float()
        low_mask = labels_t <= 2
        if not bool(low_mask.any().detach().cpu()):
            zero = probs.sum() * 0.0
            return zero, {
                "L_l2h_risk": 0.0,
                "l2h_risk_loss": 0.0,
                "risk_active_low_count": 0.0,
                "risk_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
                "risk_threshold_t": float(risk_threshold_t),
                "p_gt_3_low_mean": float("nan"),
                "p_gt_3_label1_mean": float("nan"),
                "p_gt_3_label2_mean": float("nan"),
            }
        tau = torch.where(
            labels_t == 1,
            torch.full_like(scores, float(tau_label1)),
            torch.full_like(scores, float(tau_label2)),
        )
        weights = torch.where(
            labels_t == 1,
            torch.full_like(scores, float(weight_label1)),
            torch.full_like(scores, float(weight_label2)),
        )
        per_sample = weights * torch.relu(scores - tau).pow(2)
        masked = per_sample[low_mask]
        if normalize_by_low_count:
            loss = masked.mean()
        else:
            loss = per_sample.sum() / scores.numel()
        label1_mask = labels_t == 1
        label2_mask = labels_t == 2
        label1_scores = scores[label1_mask]
        label2_scores = scores[label2_mask]
        low_scores = scores[low_mask]
        return loss, {
            "L_l2h_risk": _to_float(loss),
            "l2h_risk_loss": _to_float(loss),
            "risk_active_low_count": float(int(low_mask.sum().detach().cpu())),
            "risk_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
            "risk_threshold_t": float(risk_threshold_t),
            "risk_tau_label1": float(tau_label1),
            "risk_tau_label2": float(tau_label2),
            "risk_weight_label1": float(weight_label1),
            "risk_weight_label2": float(weight_label2),
            "p_gt_3_low_mean": _to_float(low_scores.mean()),
            "p_gt_3_label1_mean": _to_float(label1_scores.mean()) if label1_scores.numel() else float("nan"),
            "p_gt_3_label2_mean": _to_float(label2_scores.mean()) if label2_scores.numel() else float("nan"),
        }
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.asarray(probs, dtype=np.float64)[:, idx]
    low_mask_arr = labels_arr <= 2
    if not np.any(low_mask_arr):
        return 0.0, {
            "L_l2h_risk": 0.0,
            "l2h_risk_loss": 0.0,
            "risk_active_low_count": 0.0,
            "risk_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
            "risk_threshold_t": float(risk_threshold_t),
            "p_gt_3_low_mean": float("nan"),
            "p_gt_3_label1_mean": float("nan"),
            "p_gt_3_label2_mean": float("nan"),
        }
    tau_arr = np.where(labels_arr == 1, float(tau_label1), float(tau_label2))
    weight_arr = np.where(labels_arr == 1, float(weight_label1), float(weight_label2))
    per_sample_arr = weight_arr * np.maximum(0.0, scores_arr - tau_arr) ** 2
    loss_value = float(np.mean(per_sample_arr[low_mask_arr]) if normalize_by_low_count else np.sum(per_sample_arr) / len(scores_arr))
    return loss_value, {
        "L_l2h_risk": loss_value,
        "l2h_risk_loss": loss_value,
        "risk_active_low_count": float(np.sum(low_mask_arr)),
        "risk_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
        "risk_threshold_t": float(risk_threshold_t),
        "risk_tau_label1": float(tau_label1),
        "risk_tau_label2": float(tau_label2),
        "risk_weight_label1": float(weight_label1),
        "risk_weight_label2": float(weight_label2),
        "p_gt_3_low_mean": _nanmean(scores_arr[low_mask_arr]),
        "p_gt_3_label1_mean": _nanmean(scores_arr[labels_arr == 1]),
        "p_gt_3_label2_mean": _nanmean(scores_arr[labels_arr == 2]),
    }


def t3_calibration_loss_from_logits(
    logits: Any,
    labels: Any,
    score_source: str = "projected",
    loss_type: str = "brier",
    low_negative_weight: float = 2.0,
    threshold_t: int = 3,
) -> tuple[Any, dict[str, float]]:
    """Optional calibration loss for threshold P(y > 3)."""

    probs = _risk_probs_from_logits(logits, score_source)
    idx = _threshold_index(threshold_t)
    if _is_torch_tensor(probs):
        import torch
        from torch.nn import functional as F

        labels_t = labels.reshape(-1).long().to(device=probs.device)
        scores = probs[:, idx].float().clamp(1e-6, 1.0 - 1e-6)
        targets = (labels_t > threshold_t).float()
        weights = torch.where(labels_t <= 2, torch.full_like(scores, float(low_negative_weight)), torch.ones_like(scores))
        if str(loss_type) == "brier":
            per_sample = (scores - targets).pow(2)
        elif str(loss_type) == "bce":
            per_sample = F.binary_cross_entropy(scores, targets, reduction="none")
        else:
            raise ValueError(f"Unsupported t3_calibration_loss_type: {loss_type}")
        loss = (weights * per_sample).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
        low_mask = labels_t <= 2
        label1_mask = labels_t == 1
        label2_mask = labels_t == 2
        return loss, {
            "L_t3_calibration": _to_float(loss),
            "t3_calibration_loss": _to_float(loss),
            "t3_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
            "t3_low_negative_weight": float(low_negative_weight),
            "p_gt_3_low_mean": _to_float(scores[low_mask].mean()) if bool(low_mask.any().detach().cpu()) else float("nan"),
            "p_gt_3_label1_mean": _to_float(scores[label1_mask].mean()) if bool(label1_mask.any().detach().cpu()) else float("nan"),
            "p_gt_3_label2_mean": _to_float(scores[label2_mask].mean()) if bool(label2_mask.any().detach().cpu()) else float("nan"),
        }
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores_arr = np.clip(np.asarray(probs, dtype=np.float64)[:, idx], 1e-6, 1.0 - 1e-6)
    targets_arr = (labels_arr > threshold_t).astype(np.float64)
    weights_arr = np.where(labels_arr <= 2, float(low_negative_weight), 1.0)
    if str(loss_type) == "brier":
        per_sample_arr = (scores_arr - targets_arr) ** 2
    elif str(loss_type) == "bce":
        per_sample_arr = -(targets_arr * np.log(scores_arr) + (1.0 - targets_arr) * np.log(1.0 - scores_arr))
    else:
        raise ValueError(f"Unsupported t3_calibration_loss_type: {loss_type}")
    loss_value = float(np.sum(weights_arr * per_sample_arr) / max(np.sum(weights_arr), 1e-12))
    return loss_value, {
        "L_t3_calibration": loss_value,
        "t3_calibration_loss": loss_value,
        "t3_score_source_projected": 1.0 if str(score_source) == "projected" else 0.0,
        "t3_low_negative_weight": float(low_negative_weight),
        "p_gt_3_low_mean": _nanmean(scores_arr[labels_arr <= 2]),
        "p_gt_3_label1_mean": _nanmean(scores_arr[labels_arr == 1]),
        "p_gt_3_label2_mean": _nanmean(scores_arr[labels_arr == 2]),
    }
