"""Temperature scaling for Exp7-C ordinal probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp07_rank_consistent.calibrate_thresholds import compute_metrics, row_with_metrics
from thesis_exp.src.edujudge.exp07_rank_consistent.calibration_data import TEMPERATURE_GRID, SplitData, sigmoid


def ordinal_targets(labels: np.ndarray) -> np.ndarray:
    return np.stack([(labels > threshold).astype(float) for threshold in range(1, 5)], axis=1)


def ordinal_bce(probs: np.ndarray, labels: np.ndarray) -> float:
    targets = ordinal_targets(labels)
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    losses = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def temperature_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(logits / float(temperature))


def evaluate_temperature_grid(base_model: str, split_data: SplitData, *, selected_t: float | None = None) -> list[dict[str, Any]]:
    if split_data.logits is None:
        return []
    rows: list[dict[str, Any]] = []
    for temperature in TEMPERATURE_GRID:
        probs = temperature_probs(split_data.logits, temperature)
        pred = 1 + (probs > 0.5).sum(axis=1)
        metrics = compute_metrics(split_data.labels, split_data.human_mean, pred.astype(int), probs)
        prefix = {
            "base_model": base_model,
            "split": split_data.split,
            "temperature": temperature,
            "ordinal_bce": ordinal_bce(probs, split_data.labels),
            "selected": selected_t is not None and abs(temperature - selected_t) < 1e-12,
            "selection_objective": "dev_ordinal_bce",
            "selection_split": "dev",
        }
        rows.append(row_with_metrics(prefix, metrics))
    return rows


def select_temperature(base_model: str, dev_data: SplitData) -> dict[str, Any] | None:
    rows = evaluate_temperature_grid(base_model, dev_data)
    if not rows:
        return None
    return min(rows, key=lambda row: (float(row["ordinal_bce"]), float(row["MAE_label"])))
