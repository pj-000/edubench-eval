"""CORAL cumulative probability to class-distribution helpers for Exp8."""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def sigmoid(values: Any) -> Any:
    if _is_torch_tensor(values):
        import torch

        return torch.sigmoid(values)
    arr = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))


def probs_from_logits(logits: Any) -> Any:
    if _is_torch_tensor(logits) and logits.shape[-1] != 4:
        raise ValueError(f"Expected CORAL logits shape [batch,4], got {tuple(logits.shape)}")
    arr = np.asarray(logits) if not _is_torch_tensor(logits) else None
    if arr is not None and (arr.ndim != 2 or arr.shape[1] != 4):
        raise ValueError(f"Expected CORAL logits shape [batch,4], got {arr.shape}")
    return sigmoid(logits)


def q_from_probs(probs: Any, eps: float = 1e-8) -> tuple[Any, Any]:
    """Return `(q_raw, q_safe)` from rank-consistent cumulative probabilities."""
    if _is_torch_tensor(probs):
        import torch

        if probs.shape[-1] != 4:
            raise ValueError(f"Expected probs shape [batch,4], got {tuple(probs.shape)}")
        q_raw = torch.stack(
            [
                1.0 - probs[:, 0],
                probs[:, 0] - probs[:, 1],
                probs[:, 1] - probs[:, 2],
                probs[:, 2] - probs[:, 3],
                probs[:, 3],
            ],
            dim=1,
        )
        q_safe = torch.clamp(q_raw, min=eps, max=1.0)
        q_safe = q_safe / q_safe.sum(dim=1, keepdim=True).clamp(min=eps)
        return q_raw, q_safe
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected probs shape [batch,4], got {arr.shape}")
    q_raw = np.column_stack(
        [
            1.0 - arr[:, 0],
            arr[:, 0] - arr[:, 1],
            arr[:, 1] - arr[:, 2],
            arr[:, 2] - arr[:, 3],
            arr[:, 3],
        ]
    )
    q_safe = np.clip(q_raw, eps, 1.0)
    q_safe = q_safe / np.clip(q_safe.sum(axis=1, keepdims=True), eps, None)
    return q_raw, q_safe


def q_from_logits(logits: Any, eps: float = 1e-8) -> tuple[Any, Any, Any]:
    probs = probs_from_logits(logits)
    q_raw, q_safe = q_from_probs(probs, eps=eps)
    return probs, q_raw, q_safe


def q_raw_sanity(q_raw: Any, tolerance: float = 1e-6) -> dict[str, Any]:
    if _is_torch_tensor(q_raw):
        import torch

        row_sums = q_raw.float().sum(dim=1)
        min_value = float(torch.min(q_raw.float()).detach().cpu()) if q_raw.numel() else float("nan")
        max_sum_error = float(torch.max(torch.abs(row_sums - 1.0)).detach().cpu()) if row_sums.numel() else float("nan")
        finite = bool(torch.isfinite(q_raw.float()).all().detach().cpu())
        ok = finite and min_value >= -tolerance and max_sum_error <= tolerance
        return {
            "ok": ok,
            "finite": finite,
            "min_q_raw": min_value,
            "mean_q_raw_sum": float(torch.mean(row_sums).detach().cpu()) if row_sums.numel() else float("nan"),
            "max_q_raw_sum_error": max_sum_error,
        }
    arr = np.asarray(q_raw, dtype=np.float64)
    row_sums = arr.sum(axis=1) if arr.size else np.array([], dtype=np.float64)
    min_value = float(np.min(arr)) if arr.size else float("nan")
    max_sum_error = float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else float("nan")
    finite = bool(np.isfinite(arr).all())
    ok = finite and min_value >= -tolerance and max_sum_error <= tolerance
    return {
        "ok": ok,
        "finite": finite,
        "min_q_raw": min_value,
        "mean_q_raw_sum": float(np.mean(row_sums)) if row_sums.size else float("nan"),
        "max_q_raw_sum_error": max_sum_error,
    }


def assert_q_raw_sane(q_raw: Any, tolerance: float = 1e-6) -> None:
    sanity = q_raw_sanity(q_raw, tolerance=tolerance)
    if not sanity["ok"]:
        raise AssertionError(f"q_raw sanity failed: {sanity}")


def monotonicity_metrics(probs: Any, split: str | None = None, tolerance: float = 1e-7) -> dict[str, Any]:
    arr = np.asarray(probs.detach().cpu() if _is_torch_tensor(probs) else probs, dtype=np.float64)
    if arr.size == 0:
        arr = arr.reshape(0, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected probs shape [n,4], got {arr.shape}")
    if arr.shape[0] == 0:
        row: dict[str, Any] = {
            "n": 0,
            "monotonic_violation_rate": float("nan"),
            "mean_violation_magnitude": float("nan"),
            "p1_ge_p2_rate": float("nan"),
            "p2_ge_p3_rate": float("nan"),
            "p3_ge_p4_rate": float("nan"),
        }
    else:
        pair_ok = arr[:, :-1] + tolerance >= arr[:, 1:]
        violation_magnitude = np.maximum(arr[:, 1:] - arr[:, :-1], 0.0).sum(axis=1)
        row = {
            "n": int(arr.shape[0]),
            "monotonic_violation_rate": float(np.mean(np.any(~pair_ok, axis=1))),
            "mean_violation_magnitude": float(np.mean(violation_magnitude)),
            "p1_ge_p2_rate": float(np.mean(pair_ok[:, 0])),
            "p2_ge_p3_rate": float(np.mean(pair_ok[:, 1])),
            "p3_ge_p4_rate": float(np.mean(pair_ok[:, 2])),
        }
    if split is not None:
        row = {"split": split, **row}
    for idx, threshold in enumerate([1, 2, 3, 4]):
        row[f"mean_prob_gt_{threshold}"] = float(np.mean(arr[:, idx])) if arr.shape[0] else float("nan")
        row[f"prob_gt_{threshold}_positive_rate"] = (
            float(np.mean(arr[:, idx] > 0.5)) if arr.shape[0] else float("nan")
        )
    return row


def monotonic_violation_for_probs(probs: Any, tolerance: float = 1e-7) -> bool:
    values = [float(value) for value in probs]
    return any(values[idx] + tolerance < values[idx + 1] for idx in range(3))


def pred_label_cumulative(probs: Any) -> Any:
    if _is_torch_tensor(probs):
        return 1 + (probs > 0.5).sum(dim=1).long()
    arr = np.asarray(probs, dtype=np.float64)
    return (1 + (arr > 0.5).sum(axis=1)).astype(int)


def pred_label_argmax(q_raw: Any) -> Any:
    if _is_torch_tensor(q_raw):
        return 1 + q_raw.argmax(dim=1).long()
    arr = np.asarray(q_raw, dtype=np.float64)
    return (1 + np.argmax(arr, axis=1)).astype(int)


def expected_score_from_q(q_raw: Any) -> Any:
    if _is_torch_tensor(q_raw):
        import torch

        labels = torch.arange(1, 6, device=q_raw.device, dtype=q_raw.float().dtype)
        return (q_raw.float() * labels.reshape(1, 5)).sum(dim=1)
    arr = np.asarray(q_raw, dtype=np.float64)
    labels = np.arange(1, 6, dtype=np.float64).reshape(1, 5)
    return (arr * labels).sum(axis=1)
