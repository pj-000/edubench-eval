from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from thesis_exp.exp61_soft_sts15_external_confirmation.geometry import select_component
from thesis_exp.exp61_soft_sts15_external_confirmation.mapping import (
    build_maximum_mismatch_mapping,
    mapping_target_lookup,
    theoretical_maximum_changes,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.method import (
    empirical_target,
    hard_soft_ce_identity,
    human_mean,
    quantized_mean_label,
    residual_target,
    target_fifths,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.metrics import (
    component_cluster_bootstrap,
    confirmation_gates,
    evaluate_hard_head,
)


def test_six_class_five_rating_target_contract() -> None:
    scores = [0, 2, 3, 5, 5]
    assert target_fifths(scores) == (1, 0, 1, 1, 0, 2)
    assert empirical_target(scores) == (0.2, 0.0, 0.2, 0.2, 0.0, 0.4)
    assert human_mean(scores) == 3.0
    assert quantized_mean_label(scores) == 3
    assert np.allclose(residual_target(3, empirical_target(scores)), (-0.2, 0, -0.2, 0.8, 0, -0.4))


def test_half_up_quantized_mean_not_bankers_rounding() -> None:
    assert quantized_mean_label([0, 0, 0, 0, 5]) == 1
    assert quantized_mean_label([0, 0, 0, 5, 5]) == 2


def test_soft_ce_identity() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.2, -0.3, 1.1, 0.7, -1.2, 0.4], [1, 2, 3, 4, 5, 6.0]])
    labels = torch.tensor([3, 4])
    targets = torch.tensor([[0.2, 0, 0.2, 0.2, 0, 0.4], [0, 0, 0, 0.2, 0.6, 0.2]])
    assert hard_soft_ce_identity(logits, labels, targets)["absolute_identity_error"] < 1e-6


def _row(index: int, label: int, fifths: tuple[int, ...]) -> dict[str, object]:
    return {
        "record_id": f"r{index}",
        "label": label,
        "target_fifths": list(fifths),
        "component_sha256": f"c{index}",
        "split": "train",
    }


def test_maximum_mismatch_mapping_is_bijective_and_train_only() -> None:
    a, b, c = (5, 0, 0, 0, 0, 0), (4, 1, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0)
    rows = [_row(i, 0, state) for i, state in enumerate([a, a, a, b, c])]
    mapping, audit = build_maximum_mismatch_mapping(rows)
    assert audit["effective_target_changes"] == theoretical_maximum_changes(Counter([a, a, a, b, c])) == 4
    assert len({row["donor_record_id"] for row in mapping}) == len(rows)
    assert all(len(value) == 6 and np.isclose(sum(value), 1) for value in mapping_target_lookup(mapping).values())
    rows[0]["split"] = "dev"
    with pytest.raises(PermissionError):
        build_maximum_mismatch_mapping(rows)


def test_geometry_surface_has_only_three_registered_arms() -> None:
    common = {"backbone.x": np.array([1.0])}
    aligned = {"backbone.x": np.array([2.0])}
    shuffled = {"backbone.x": np.array([3.0])}
    assert float(select_component("quantized_mean_only", common, aligned, shuffled)["backbone.x"][0]) == 0
    assert select_component("aligned_orthogonal_only", common, aligned, shuffled) is aligned
    assert select_component("matched_shuffled_orthogonal_only", common, aligned, shuffled) is shuffled
    with pytest.raises(ValueError):
        select_component("posthoc_new_arm", common, aligned, shuffled)


def test_metrics_use_hard_head_expectation_against_five_rating_mean() -> None:
    probs = np.eye(6)[[0, 2, 5]].astype(float)
    metrics = evaluate_hard_head(probs, np.array([0.0, 2.4, 4.8]), np.array([0, 2, 5]))
    assert np.isclose(metrics["mae"], 0.2)
    assert np.isclose(metrics["nmae"], 0.04)
    assert metrics["exact_match"] == 1.0


def test_component_bootstrap_is_deterministic_and_gate_code_is_fail_closed() -> None:
    human = np.array([0.0, 1.0, 4.0, 5.0])
    aligned = {seed: human + 0.01 for seed in (61, 62, 63)}
    shuffled = {seed: human + 0.2 for seed in (61, 62, 63)}
    first = component_cluster_bootstrap(aligned, shuffled, human, ["a", "a", "b", "c"], n_resamples=100, rng_seed=7)
    second = component_cluster_bootstrap(aligned, shuffled, human, ["a", "a", "b", "c"], n_resamples=100, rng_seed=7)
    assert first == second
    assert first["ci95_upper"] < 0
    aligned_metrics = {seed: {"nmae": 0.10, "spearman": 0.80} for seed in (61, 62, 63)}
    shuffled_metrics = {seed: {"nmae": 0.12, "spearman": 0.80} for seed in (61, 62, 63)}
    main_metrics = {seed: {"nmae": 0.11, "spearman": 0.79} for seed in (61, 62, 63)}
    assert confirmation_gates(aligned_metrics, shuffled_metrics, main_metrics, first)["decision"] == "PASS"
