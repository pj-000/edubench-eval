from __future__ import annotations

import numpy as np
import pytest

from thesis_exp.exp54_rar_sft.analyze_label2_identification import (
    apply_vector_scaler,
    deterministic_group_fold,
    fit_vector_scaler,
    measurement_ambiguous,
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
