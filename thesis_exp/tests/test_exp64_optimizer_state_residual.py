from __future__ import annotations

import copy

import pytest

from thesis_exp.exp64_optimizer_state_residual.mechanics import (
    exact_adamw_displacement,
    fixed_denominator_attributable_displacement,
    kendall_tau_b,
    l2_norm,
    shared_scale,
    subtract_displacements,
)


def _initialized_optimizer(torch):
    first = torch.nn.Parameter(torch.tensor([0.4, -0.3], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [first, second],
        lr=2e-3,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.03,
        foreach=False,
        fused=False,
    )
    first.grad = torch.tensor([0.15, -0.07], dtype=torch.float64)
    second.grad = torch.tensor([-0.11], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"first": first, "second": second}, optimizer


def _state_contract(parameters, optimizer):
    parameter_to_name = {id(value): name for name, value in parameters.items()}
    states = {
        parameter_to_name[id(parameter)]: copy.deepcopy(optimizer.state[parameter])
        for parameter in parameters.values()
    }
    groups = {}
    for group in optimizer.param_groups:
        frozen = {key: value for key, value in group.items() if key != "params"}
        for parameter in group["params"]:
            groups[parameter_to_name[id(parameter)]] = frozen
    return states, groups


def test_exact_adamw_formula_matches_direct_optimizer_step():
    torch = pytest.importorskip("torch")
    parameters, optimizer = _initialized_optimizer(torch)
    states, groups = _state_contract(parameters, optimizer)
    gradients = {
        "first": torch.tensor([-0.04, 0.09], dtype=torch.float64),
        "second": torch.tensor([0.13], dtype=torch.float64),
    }
    predicted = exact_adamw_displacement(parameters, gradients, states, groups)
    before = {name: value.detach().clone() for name, value in parameters.items()}
    for name, parameter in parameters.items():
        parameter.grad = gradients[name].clone()
    optimizer.step()
    for name, parameter in parameters.items():
        observed = parameter.detach() - before[name]
        torch.testing.assert_close(predicted[name], observed, rtol=0.0, atol=1e-14)


def test_shared_scale_is_identical_and_keeps_every_arm_below_target():
    torch = pytest.importorskip("torch")
    candidates = {
        "blocked": {"x": torch.tensor([3.0, 4.0], dtype=torch.float64)},
        "full": {"x": torch.tensor([6.0, 8.0], dtype=torch.float64)},
        "signflip": {"x": torch.tensor([0.3, 0.4], dtype=torch.float64)},
    }
    scale, norms = shared_scale(candidates, target_norm=0.95)
    assert scale == pytest.approx(0.095)
    assert norms == pytest.approx({"blocked": 5.0, "full": 10.0, "signflip": 0.5})
    realized = {
        name: l2_norm({key: value * scale for key, value in tensors.items()})
        for name, tensors in candidates.items()
    }
    assert max(realized.values()) == pytest.approx(0.95)
    assert all(value <= 0.95 + 1e-12 for value in realized.values())


def test_attributable_displacement_cancels_weight_decay_and_differs_from_fixed_denominator():
    torch = pytest.importorskip("torch")
    parameters, optimizer = _initialized_optimizer(torch)
    states, groups = _state_contract(parameters, optimizer)
    blocked = {
        "first": torch.tensor([0.02, -0.03], dtype=torch.float64),
        "second": torch.tensor([0.04], dtype=torch.float64),
    }
    candidate = {
        "first": torch.tensor([0.25, 0.19], dtype=torch.float64),
        "second": torch.tensor([-0.21], dtype=torch.float64),
    }
    blocked_step = exact_adamw_displacement(parameters, blocked, states, groups)
    candidate_step = exact_adamw_displacement(parameters, candidate, states, groups)
    attributable = subtract_displacements(candidate_step, blocked_step)
    zero_parameters = {name: value.detach().clone().zero_() for name, value in parameters.items()}
    blocked_zero = exact_adamw_displacement(zero_parameters, blocked, states, groups)
    candidate_zero = exact_adamw_displacement(zero_parameters, candidate, states, groups)
    attributable_zero = subtract_displacements(candidate_zero, blocked_zero)
    for name in attributable:
        torch.testing.assert_close(attributable[name], attributable_zero[name], rtol=0.0, atol=1e-14)
    fixed = fixed_denominator_attributable_displacement(blocked, candidate, states, groups)
    assert any(not torch.allclose(attributable[name], fixed[name]) for name in attributable)


def test_frozen_kendall_tau_b_rewards_direction_not_unsigned_magnitude():
    outcomes = [-0.30, 0.20, -0.10, 0.40]
    exact = [-0.25, 0.15, -0.05, 0.35]
    magnitude_only = [-0.30, -0.20, -0.10, -0.40]
    assert kendall_tau_b(exact, outcomes) == pytest.approx(1.0)
    assert kendall_tau_b(exact, outcomes) > kendall_tau_b(magnitude_only, outcomes)
