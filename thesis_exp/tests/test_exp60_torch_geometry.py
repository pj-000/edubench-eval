"""Torch CPU test of the memory-streaming Exp60 formal geometry step."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from thesis_exp.exp60_geometry_matched_shuffle.train import compose_geometry_step


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 2, bias=False)
        self.hard_head = torch.nn.Linear(2, 1, bias=False)
        self.soft_head = torch.nn.Linear(2, 1, bias=False)


def tensors() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    common = {
        "backbone.weight": torch.tensor(
            [[0.03, 0.04, 0.01], [0.02, -0.01, 0.05]], dtype=torch.float32
        ),
        "hard_head.weight": torch.tensor([[0.02, -0.01]], dtype=torch.float32),
        "soft_head.weight": torch.tensor([[0.01, 0.02]], dtype=torch.float32),
    }
    aligned = {
        "backbone.weight": torch.tensor(
            [[0.02, -0.01, 0.03], [-0.02, 0.04, 0.01]], dtype=torch.float32
        )
    }
    shuffled = {
        "backbone.weight": torch.tensor(
            [[-0.04, 0.02, 0.01], [0.03, 0.01, -0.02]], dtype=torch.float32
        )
    }
    return common, aligned, shuffled


@pytest.mark.parametrize(
    "variant",
    (
        "consensus_only",
        "aligned_orthogonal_only",
        "matched_shuffled_orthogonal_only",
    ),
)
def test_streaming_geometry_step_passes_all_local_match_gates(variant: str) -> None:
    model = ToyModel()
    common, aligned, shuffled = tensors()
    for name, parameter in model.named_parameters():
        residual = aligned.get(name, torch.zeros_like(parameter))
        parameter.grad = (common[name] + residual).clone()
    audit = compose_geometry_step(
        model,
        {name: value.clone() for name, value in aligned.items()},
        {name: value.clone() for name, value in shuffled.items()},
        variant=variant,
        max_norm=1.0,
    )
    assert audit["aligned_normalized_orthogonality_error"] <= 1e-6
    assert audit["shuffled_normalized_orthogonality_error"] <= 1e-6
    assert audit["component_norm_relative_error"] <= 1e-4
    assert audit["preclip_total_norm_relative_error"] <= 1e-4
    assert audit["clip_coefficient_relative_error"] <= 1e-4
    assert audit["storage_component_norm_relative_error"] <= 1e-6
    assert audit["storage_preclip_total_norm_relative_error"] <= 1e-6
    assert audit["storage_clip_coefficient_relative_error"] <= 1e-6
    assert -1.0 <= audit["aligned_shuffled_component_cosine"] <= 1.0
    assert audit["aligned_shuffled_component_relative_distance"] >= 0.0
    assert audit["storage_component_activity_ratio"] >= 1e-6
    assert audit["postclip_norm"] <= 1.0 + 1e-6


def test_consensus_arm_restores_common_heads_unchanged() -> None:
    model = ToyModel()
    common, aligned, shuffled = tensors()
    for name, parameter in model.named_parameters():
        parameter.grad = (common[name] + aligned.get(name, torch.zeros_like(parameter))).clone()
    compose_geometry_step(
        model,
        {name: value.clone() for name, value in aligned.items()},
        {name: value.clone() for name, value in shuffled.items()},
        variant="consensus_only",
        max_norm=1.0,
    )
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert torch.allclose(gradients["hard_head.weight"], common["hard_head.weight"])
    assert torch.allclose(gradients["soft_head.weight"], common["soft_head.weight"])


def test_bfloat16_storage_space_is_audited_for_both_candidate_arms() -> None:
    model = ToyModel().to(dtype=torch.bfloat16)
    common, aligned, shuffled = tensors()
    for name, parameter in model.named_parameters():
        routed = common[name] + aligned.get(name, torch.zeros_like(parameter, dtype=torch.float32))
        parameter.grad = routed.to(dtype=torch.bfloat16)
    audit = compose_geometry_step(
        model,
        {name: value.clone() for name, value in aligned.items()},
        {name: value.clone() for name, value in shuffled.items()},
        variant="aligned_orthogonal_only",
        max_norm=1.0,
    )
    assert audit["gradient_storage_dtype"] == "torch.bfloat16"
    assert audit["post_cast_relative_tolerance"] == pytest.approx(0.01171875)
    assert audit["storage_component_norm_relative_error"] <= 0.01171875
    assert audit["storage_preclip_total_norm_relative_error"] <= 0.01171875
    assert audit["storage_clip_coefficient_relative_error"] <= 0.01171875
    assert audit["post_cast_orthogonality_tolerance"] == pytest.approx(0.1)


def test_nonfinite_unselected_candidate_fails_closed() -> None:
    model = ToyModel()
    common, aligned, shuffled = tensors()
    shuffled["backbone.weight"][0, 0] = float("nan")
    for name, parameter in model.named_parameters():
        parameter.grad = (
            common[name] + aligned.get(name, torch.zeros_like(parameter))
        ).clone()
    with pytest.raises(RuntimeError, match="EXP60_NONFINITE_TENSOR"):
        compose_geometry_step(
            model,
            {name: value.clone() for name, value in aligned.items()},
            {name: value.clone() for name, value in shuffled.items()},
            variant="consensus_only",
            max_norm=1.0,
        )


def test_zero_treatment_components_fail_instead_of_producing_cosine_zero() -> None:
    model = ToyModel()
    common, _, _ = tensors()
    for name, parameter in model.named_parameters():
        parameter.grad = common[name].clone()
    zero = {"backbone.weight": torch.zeros_like(model.backbone.weight)}
    with pytest.raises(RuntimeError, match="EXP60_DEGENERATE_TREATMENT_COMPONENT"):
        compose_geometry_step(
            model,
            {name: value.clone() for name, value in zero.items()},
            {name: value.clone() for name, value in zero.items()},
            variant="consensus_only",
            max_norm=1.0,
        )
