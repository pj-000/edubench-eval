"""Frozen dev metrics and the registered hard-head boundary diagnostic."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd.method import STATE_ZERO, target_state
from thesis_exp.src.edujudge.exp02.train_ce_baseline import kendall_tau_b, qwk


LABELS = (1, 2, 3, 4, 5)


def score_bin(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 2, 0, np.where(values == 3, 1, 2))


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Match the locked Exp49 primary metric definitions."""

    if not rows:
        return {"n": 0}
    gold = np.asarray([int(row["label_5"]) for row in rows], dtype=int)
    human_mean = np.asarray([float(row["human_mean_5"]) for row in rows], dtype=float)
    pred = np.asarray([int(row["pred_label_5"]) for row in rows], dtype=int)
    expected = np.asarray([float(row["pred_score_expected"]) for row in rows], dtype=float)
    probs = np.asarray(
        [[float(row[f"prob_{label}"]) for label in LABELS] for row in rows],
        dtype=float,
    )
    low = gold <= 2
    high = gold >= 4
    diff = pred.astype(float) - human_mean
    counts = Counter(pred.tolist())
    output: dict[str, Any] = {
        "n": len(rows),
        "Exact_rounded": float(np.mean(pred == gold)),
        "MAE_human_mean": float(np.mean(np.abs(diff))),
        "Bias_human_mean": float(np.mean(diff)),
        "Kendall_human_mean": kendall_tau_b(human_mean.tolist(), pred.astype(float).tolist()),
        "BinAgreement_paper_3way": float(np.mean(score_bin(gold) == score_bin(pred))),
        "L2H_count": int(np.sum(low & (pred >= 4))),
        "L2H_rounded": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "L2H_n": int(low.sum()),
        "H2L_count": int(np.sum(high & (pred <= 2))),
        "H2L_rounded": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
        "H2L_n": int(high.sum()),
        "MAE_expected_human_mean": float(np.mean(np.abs(expected - human_mean))),
        "QWK_rounded": qwk(gold.tolist(), pred.tolist()),
    }
    clipped = np.clip(probs, 1e-12, 1.0)
    output["NLL_rounded"] = float(-np.log(clipped[np.arange(len(gold)), gold - 1]).mean())
    for label in LABELS:
        mask = gold == label
        correct = int(np.sum(mask & (pred == label)))
        total = int(mask.sum())
        output[f"Recall_{label}_rounded"] = correct / total if total else float("nan")
        output[f"Recall_{label}_correct"] = correct
        output[f"Recall_{label}_total"] = total
        output[f"Predicted_{label}_count"] = counts[label]
    return output


def add_boundary_diagnostics(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the preregistered hard-head minority-neighbor advantage.

    For a disputed item this is z(minority adjacent label) - z(consensus
    label).  The statistic is undefined on unanimous rows and is reported for
    down, up, and pooled strata without selecting a favorable subgroup.
    """

    values: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        state = str(row["boundary_state"])
        if state == STATE_ZERO:
            continue
        advantage = float(row["minority_neighbor_advantage"])
        values[state].append(advantage)
        values["pooled"].append(advantage)
    for state in ("down", "up", "pooled"):
        state_values = values[state]
        metrics[f"boundary_advantage_{state}_n"] = len(state_values)
        metrics[f"boundary_advantage_{state}_mean"] = (
            float(np.mean(state_values)) if state_values else float("nan")
        )
    return metrics


def finite_primary(metrics: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(metrics[key]))
        for key in ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")
    )
