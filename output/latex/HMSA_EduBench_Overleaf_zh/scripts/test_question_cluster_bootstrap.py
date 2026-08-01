from __future__ import annotations

import pytest

from build_question_cluster_bootstrap import cluster_bootstrap


def test_cluster_bootstrap_observed_mean_and_shape() -> None:
    deltas = {"a": -1.0, "b": 1.0, "c": -2.0}
    clusters = {"a": "q1", "b": "q1", "c": "q2"}
    result = cluster_bootstrap(
        deltas,
        clusters,
        resamples=100,
        rng_seed=7,
    )
    assert result["mean_delta_mae"] == pytest.approx(-2.0 / 3.0)
    assert result["cluster_count"] == 2
    assert result["row_count"] == 3
    assert result["min_cluster_size"] == 1
    assert result["max_cluster_size"] == 2


def test_cluster_bootstrap_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap(
            {"a": 0.0},
            {"b": "q1"},
            resamples=10,
            rng_seed=7,
        )
