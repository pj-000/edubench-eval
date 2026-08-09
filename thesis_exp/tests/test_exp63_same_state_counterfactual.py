from __future__ import annotations

import json

import torch
from transformers import get_cosine_schedule_with_warmup

from thesis_exp.exp63_same_state_counterfactual import PROTOCOL_PATH
from thesis_exp.exp63_same_state_counterfactual.counterfactual import (
    candidate_raw_norm,
    candidate_tensor,
    scalar_geometry,
)
from thesis_exp.exp63_same_state_counterfactual.runtime import epoch_order


def test_protocol_is_train_dev_only_and_seed_is_primary_unit() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["allowed_splits"] == ["train", "dev"]
    assert protocol["test_access_count"] == 0
    assert protocol["primary_estimand"]["unit"].startswith("seed;")


def test_epoch_order_is_deterministic_and_epoch_specific() -> None:
    assert epoch_order(20, 67, 3) == epoch_order(20, 67, 3)
    assert epoch_order(20, 67, 3) != epoch_order(20, 67, 4)


def test_geometry_reconstructs_and_is_orthogonal() -> None:
    common = {
        "backbone.a": torch.tensor([1.0, 2.0, -1.0]),
        "hard_head.w": torch.tensor([0.5]),
    }
    residual = {"backbone.a": torch.tensor([2.0, -1.0, 3.0])}
    geometry = scalar_geometry(common, residual)
    assert geometry["reconstruction_relative_error"] <= 1e-12
    assert geometry["normalized_orthogonality_error"] <= 1e-12
    coefficient = geometry["projection_coefficient"]
    full = candidate_tensor(
        "backbone.a", "full_residual", common["backbone.a"], residual["backbone.a"], coefficient
    )
    parallel = candidate_tensor(
        "backbone.a", "parallel_only", common["backbone.a"], residual["backbone.a"], coefficient
    )
    orthogonal = candidate_tensor(
        "backbone.a", "orthogonal_only", common["backbone.a"], residual["backbone.a"], coefficient
    )
    assert torch.allclose(full - common["backbone.a"], (parallel - common["backbone.a"]) + (orthogonal - common["backbone.a"]))


def test_candidate_norm_includes_heads() -> None:
    common = {
        "backbone.a": torch.tensor([3.0, 4.0]),
        "hard_head.w": torch.tensor([12.0]),
    }
    residual = {"backbone.a": torch.zeros(2)}
    assert candidate_raw_norm(common, residual, "blocked", 0.0) == 13.0


def test_scheduler_must_be_constructed_before_optimizer_state_restore() -> None:
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=2e-5)
    source_scheduler = get_cosine_schedule_with_warmup(source_optimizer, 10, 100)
    for _ in range(20):
        source_optimizer.zero_grad(set_to_none=True)
        source(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        source_scheduler.step()
    expected_lr = source_optimizer.param_groups[0]["lr"]
    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=2e-5)
    target_scheduler = get_cosine_schedule_with_warmup(target_optimizer, 10, 100)
    target_optimizer.load_state_dict(source_optimizer.state_dict())
    target_scheduler.load_state_dict(source_scheduler.state_dict())
    assert expected_lr > 0.0
    assert target_optimizer.param_groups[0]["lr"] == expected_lr
    assert target_scheduler.get_last_lr()[0] == expected_lr
