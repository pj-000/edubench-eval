"""Monotonicity metrics and prediction helpers for Exp7."""

from __future__ import annotations

from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent import ORDINAL_THRESHOLDS


def sigmoid(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))


def prediction_from_probs(probs: Any) -> tuple[int, float]:
    values = [float(value) for value in probs]
    pred_label = 1 + sum(1 for value in values if value > 0.5)
    pred_label = int(min(5, max(1, pred_label)))
    expected_score = float(min(5.0, max(1.0, 1.0 + sum(values))))
    return pred_label, expected_score


def monotonic_violation_for_probs(probs: Any, tolerance: float = 1e-7) -> bool:
    values = [float(value) for value in probs]
    return any(values[idx] + tolerance < values[idx + 1] for idx in range(3))


def monotonicity_metrics(probs: Any, split: str | None = None, tolerance: float = 1e-7) -> dict[str, Any]:
    arr = np.asarray(probs, dtype=np.float64)
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
    for idx, threshold in enumerate(ORDINAL_THRESHOLDS):
        if arr.shape[0] == 0:
            row[f"mean_prob_gt_{threshold}"] = float("nan")
            row[f"prob_gt_{threshold}_positive_rate"] = float("nan")
        else:
            row[f"mean_prob_gt_{threshold}"] = float(np.mean(arr[:, idx]))
            row[f"prob_gt_{threshold}_positive_rate"] = float(np.mean(arr[:, idx] > 0.5))
    return row
