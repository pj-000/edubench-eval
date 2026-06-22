"""Exact monotonic projection utilities for ordinal threshold probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pava_nonincreasing_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    blocks: list[dict[str, Any]] = []
    for idx, value in enumerate(arr):
        blocks.append({"start": idx, "end": idx + 1, "sum": float(value), "count": 1})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["sum"] / left["count"]
            right_mean = right["sum"] / right["count"]
            if left_mean >= right_mean:
                break
            merged = {
                "start": left["start"],
                "end": right["end"],
                "sum": left["sum"] + right["sum"],
                "count": left["count"] + right["count"],
            }
            blocks[-2:] = [merged]
    out = np.empty_like(arr, dtype=np.float64)
    for block in blocks:
        out[block["start"] : block["end"]] = block["sum"] / block["count"]
    return np.clip(out, 0.0, 1.0)


def project_nonincreasing_probs(p: Any) -> np.ndarray:
    """Project [4] or [batch, 4] probabilities onto q1 >= q2 >= q3 >= q4.

    This is exact Euclidean isotonic projection via PAVA, followed by [0, 1]
    clipping. Clipping preserves the non-increasing order and gives the bounded
    projection for the probability box used here.
    """

    arr = np.asarray(p, dtype=np.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError(f"Expected shape [4], got {original_shape}")
        return _pava_nonincreasing_1d(arr)
    if arr.ndim == 2:
        if arr.shape[1] != 4:
            raise ValueError(f"Expected shape [batch,4], got {original_shape}")
        return np.stack([_pava_nonincreasing_1d(row) for row in arr], axis=0)
    raise ValueError(f"Expected shape [4] or [batch,4], got {original_shape}")


def project_nonincreasing_probs_torch(p: Any) -> Any:
    """Torch PAVA projection with piecewise gradients through the chosen blocks.

    The merge decisions are made on detached scalar values, so gradients flow
    through block averages inside the active PAVA partition. This is exact for
    the forward pass and intentionally avoids sorting, which would change the
    semantic positions of ordinal thresholds.
    """

    import torch

    if p.ndim == 1:
        if p.shape[0] != 4:
            raise ValueError(f"Expected shape [4], got {tuple(p.shape)}")
        flat = p.reshape(1, 4)
        squeeze = True
    elif p.ndim == 2:
        if p.shape[1] != 4:
            raise ValueError(f"Expected shape [batch,4], got {tuple(p.shape)}")
        flat = p
        squeeze = False
    else:
        raise ValueError(f"Expected shape [4] or [batch,4], got {tuple(p.shape)}")
    rows = []
    for row in flat:
        blocks: list[tuple[Any, int]] = []
        for idx in range(4):
            blocks.append((row[idx], 1))
            while len(blocks) >= 2:
                left_sum, left_count = blocks[-2]
                right_sum, right_count = blocks[-1]
                left_mean = left_sum / float(left_count)
                right_mean = right_sum / float(right_count)
                if bool((left_mean.detach() >= right_mean.detach()).cpu().item()):
                    break
                blocks[-2:] = [(left_sum + right_sum, left_count + right_count)]
        values = []
        for block_sum, block_count in blocks:
            values.extend([block_sum / float(block_count)] * block_count)
        rows.append(torch.stack(values).clamp(0.0, 1.0))
    projected = torch.stack(rows, dim=0).to(dtype=p.dtype, device=p.device)
    return projected.reshape(4) if squeeze else projected


def monotonic_violation_rate(probs: Any) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return float(np.mean(np.any(arr[:, :-1] < arr[:, 1:], axis=1)))


def projection_delta_summary(raw: Any, projected: Any) -> dict[str, float]:
    raw_arr = np.asarray(raw, dtype=np.float64)
    proj_arr = np.asarray(projected, dtype=np.float64)
    if raw_arr.ndim == 1:
        raw_arr = raw_arr.reshape(1, -1)
        proj_arr = proj_arr.reshape(1, -1)
    delta = proj_arr - raw_arr
    l2 = np.sqrt(np.sum(delta * delta, axis=1))
    linf = np.max(np.abs(delta), axis=1)
    return {
        "mean_projection_l2_delta": float(np.mean(l2)) if l2.size else float("nan"),
        "mean_projection_linf_delta": float(np.mean(linf)) if linf.size else float("nan"),
    }
