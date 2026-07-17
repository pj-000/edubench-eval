from __future__ import annotations

import math

from thesis_exp.exp49_cphce.metric_contract import compute_metrics


def test_metric_contract_uses_continuous_human_mean_and_positive_bias_is_overestimate() -> None:
    rows = [
        {"label_5": 5, "human_mean_5": 14 / 3, "pred_label_5": 5, "pred_score_expected": 4.8, **{f"prob_{k}": float(k == 5) for k in range(1, 6)}},
        {"label_5": 2, "human_mean_5": 5 / 3, "pred_label_5": 3, "pred_score_expected": 2.8, **{f"prob_{k}": float(k == 3) for k in range(1, 6)}},
    ]
    metrics = compute_metrics(rows)
    assert metrics["Exact_rounded"] == 0.5
    assert math.isclose(metrics["MAE_human_mean"], (1 / 3 + 4 / 3) / 2)
    assert metrics["Bias_human_mean"] > 0
    assert metrics["L2H_count"] == 0


def test_paper_three_way_bin_mapping() -> None:
    rows = [
        {"label_5": 1, "human_mean_5": 1.0, "pred_label_5": 2},
        {"label_5": 3, "human_mean_5": 3.0, "pred_label_5": 3},
        {"label_5": 4, "human_mean_5": 4.0, "pred_label_5": 5},
        {"label_5": 2, "human_mean_5": 2.0, "pred_label_5": 4},
    ]
    assert compute_metrics(rows)["BinAgreement_paper_3way"] == 0.75
