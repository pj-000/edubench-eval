from __future__ import annotations

import pytest

from thesis_exp.exp49_cphce.build_targets import load_split


def test_train_loader_refuses_test() -> None:
    with pytest.raises(PermissionError):
        load_split("test")


def test_explicit_final_evaluator_can_load_fixed_test() -> None:
    assert len(load_split("test", allow_test=True)) == 2218
