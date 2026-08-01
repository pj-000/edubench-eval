from __future__ import annotations

import numpy as np
import pytest

from thesis_exp.exp54_rar_sft.analyze_label2_identification import (
    _adjusted_support_coefficient,
    apply_vector_scaler,
    deterministic_group_fold,
    fit_vector_scaler,
    measurement_ambiguous,
    natural_metrics,
    quadratic_weighted_kappa,
)


def test_group_fold_is_deterministic_and_namespaced() -> None:
    first = deterministic_group_fold("q1", namespace="outer", folds=5)
    assert first == deterministic_group_fold("q1", namespace="outer", folds=5)
    assert 0 <= first < 5
    observed = {
        deterministic_group_fold(f"q{i}", namespace="outer", folds=5)
        for i in range(100)
    }
    assert observed == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([2, 2, 2], False),
        ([2, 2, 3], True),
        ([1, 2, 3], True),
        ([1, 2, 2], False),
        ([1, 3, 3], True),
    ],
)
def test_measurement_ambiguity_rule(scores: list[int], expected: bool) -> None:
    row = {f"human_{index}_5": value for index, value in enumerate(scores, 1)}
    assert measurement_ambiguous(row) is expected


def test_vector_scaler_returns_finite_normalized_probabilities() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.10, 0.08, 0.07, 0.05],
            [0.10, 0.65, 0.10, 0.10, 0.05],
            [0.05, 0.10, 0.65, 0.10, 0.10],
            [0.05, 0.10, 0.10, 0.65, 0.10],
            [0.05, 0.05, 0.10, 0.10, 0.70],
        ]
        * 4,
        dtype=np.float64,
    )
    labels = np.asarray(list(range(5)) * 4, dtype=np.int64)
    log_probabilities = np.log(probabilities)
    scales, biases = fit_vector_scaler(
        log_probabilities,
        labels,
        regularization=0.1,
    )
    calibrated = apply_vector_scaler(log_probabilities, scales, biases)
    assert calibrated.shape == probabilities.shape
    assert np.all(np.isfinite(calibrated))
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(calibrated, axis=1), labels)


def test_natural_metrics_are_exact_for_perfect_predictions() -> None:
    labels = np.asarray([1, 2, 3, 4, 5] * 2, dtype=np.int64)
    predictions = labels.copy()
    probabilities = np.eye(5, dtype=np.float64)[labels - 1]
    metrics = natural_metrics(probabilities, labels, predictions)
    assert metrics["MAE"] == 0.0
    assert metrics["Exact"] == 1.0
    assert metrics["QWK"] == 1.0
    assert metrics["Kendall_tau_b"] == pytest.approx(1.0)
    assert metrics["L2H_count"] == 0
    assert metrics["L2H_rate_among_low"] == 0.0
    assert metrics["H2L_count"] == 0
    assert metrics["H2L_rate_among_high"] == 0.0
    assert metrics["Recall"] == {str(score): 1.0 for score in range(1, 6)}
    assert metrics["multiclass_NLL"] == 0.0
    assert metrics["multiclass_Brier"] == 0.0
    assert metrics["RPS_normalized_by_4"] == 0.0


def test_quadratic_weighted_kappa_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal one-dimensional"):
        quadratic_weighted_kappa(np.asarray([1, 2]), np.asarray([1]))


def test_tail_risk_rates_use_conditional_denominators() -> None:
    labels = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
    predictions = np.asarray([4, 2, 3, 1, 5], dtype=np.int64)
    probabilities = np.full((5, 5), 0.05, dtype=np.float64)
    probabilities[np.arange(5), predictions - 1] = 0.8
    metrics = natural_metrics(probabilities, labels, predictions)
    assert metrics["L2H_count"] == 1
    assert metrics["L2H_rate_among_low"] == 0.5
    assert metrics["H2L_count"] == 1
    assert metrics["H2L_rate_among_high"] == 0.5


def test_adjusted_support_coefficient_has_expected_direction() -> None:
    rows = []
    for index in range(80):
        support = 1 + index % 10
        rows.append(
            {
                "metric_id": "A" if index % 2 else "B",
                "language": "en" if index % 3 else "zh",
                "support_count": support,
                "rater_range": float(index % 4),
                "failure": support <= 4,
            }
        )
    coefficient = _adjusted_support_coefficient(rows)
    assert coefficient is not None
    assert coefficient < 0.0
