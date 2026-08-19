from __future__ import annotations

import math

import pytest

from thesis_exp.exp54_rar_sft.audit_mechanism_control_formal_results import (
    EXPECTED_STEPS,
    _require_finite_vector,
)


def test_formal_audit_step_budget_is_exact() -> None:
    assert EXPECTED_STEPS == {
        "R3_TOKENAVG": 996,
        "P1_FULLSEQ": 27,
        "P1_SYN_LR5E6": 27,
    }
    assert sum(EXPECTED_STEPS.values()) * 3 == 3150


def test_finite_vector_accepts_positive_training_signal() -> None:
    values = _require_finite_vector(
        [0.1, 1.5, 2],
        length=3,
        positive=True,
        label="test",
    )
    assert values == [0.1, 1.5, 2.0]


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_finite_vector_rejects_invalid_training_signal(value: float) -> None:
    with pytest.raises(ValueError):
        _require_finite_vector(
            [value],
            length=1,
            positive=True,
            label="test",
        )
