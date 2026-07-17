"""Single metric contract for Exp49 and recomputable historical predictions."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import kendall_tau_b, qwk


LABELS = (1, 2, 3, 4, 5)
PAPER_BIN_STATUS = "REPRODUCED_SPLIT_TOLERANT"


def score_bin(values: np.ndarray) -> np.ndarray:
    """Paper-compatible low/mid/high mapping recovered against all five judges."""
    return np.where(values <= 2, 0, np.where(values == 3, 1, 2))


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1) + 1
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float((prediction[mask] == labels[mask]).mean()) - float(confidence[mask].mean()))
    return value


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    gold = np.asarray([int(row["label_5"]) for row in rows], dtype=int)
    human_mean = np.asarray([float(row["human_mean_5"]) for row in rows], dtype=float)
    pred = np.asarray([int(row["pred_label_5"]) for row in rows], dtype=int)
    expected = np.asarray([float(row.get("pred_score_expected", row["pred_label_5"])) for row in rows], dtype=float)
    has_probs = all(all(f"prob_{label}" in row for label in LABELS) for row in rows)
    probs = (
        np.asarray([[float(row[f"prob_{label}"]) for label in LABELS] for row in rows], dtype=float)
        if has_probs
        else np.eye(5, dtype=float)[pred - 1]
    )
    low = gold <= 2
    high = gold >= 4
    diff = pred.astype(float) - human_mean
    expected_diff = expected - human_mean
    counts = Counter(pred.tolist())
    output: dict[str, Any] = {
        "n": len(rows),
        "Exact_rounded": float(np.mean(pred == gold)),
        "MAE_human_mean": float(np.mean(np.abs(diff))),
        "Bias_human_mean": float(np.mean(diff)),
        "Kendall_human_mean": kendall_tau_b(human_mean.tolist(), pred.astype(float).tolist()),
        "BinAgreement_paper_3way": float(np.mean(score_bin(gold) == score_bin(pred))),
        "paper_bin_status": PAPER_BIN_STATUS,
        "L2H_count": int(np.sum(low & (pred >= 4))),
        "L2H_rounded": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "L2H_n": int(low.sum()),
        "H2L_count": int(np.sum(high & (pred <= 2))),
        "H2L_rounded": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
        "H2L_n": int(high.sum()),
        "MAE_expected_human_mean": float(np.mean(np.abs(expected_diff))),
        "QWK_rounded": qwk(gold.tolist(), pred.tolist()),
    }
    clipped = np.clip(probs, 1e-12, 1.0)
    output["NLL_rounded"] = float(-np.log(clipped[np.arange(len(gold)), gold - 1]).mean())
    one_hot = np.eye(5, dtype=float)[gold - 1]
    output["Brier_rounded"] = float(np.square(probs - one_hot).sum(axis=1).mean())
    output["ECE_rounded"] = _ece(probs, gold)
    for label in LABELS:
        mask = gold == label
        correct = int(np.sum(mask & (pred == label)))
        total = int(mask.sum())
        output[f"Recall_{label}_rounded"] = correct / total if total else float("nan")
        output[f"Recall_{label}_correct"] = correct
        output[f"Recall_{label}_total"] = total
        output[f"Predicted_{label}_count"] = counts[label]
    return output


def finite_primary_metrics(metrics: dict[str, Any]) -> bool:
    keys = ("Exact_rounded", "MAE_human_mean", "Bias_human_mean", "Kendall_human_mean")
    return all(math.isfinite(float(metrics[key])) for key in keys)
