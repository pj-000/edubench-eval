from __future__ import annotations

import pytest

from thesis_exp.exp50_cahs.build_targets import cahs_target, load_split


def test_cahs_unanimous_and_majority_targets() -> None:
    assert cahs_target(4, [0, 0, 0, 1, 0]) == pytest.approx([0, 0, 0, 1, 0])
    assert cahs_target(4, [0, 0, 1 / 3, 2 / 3, 0]) == pytest.approx([0, 0, 1 / 6, 5 / 6, 0])
    assert cahs_target(4, [0, 0, 0, 2 / 3, 1 / 3]) == pytest.approx([0, 0, 0, 5 / 6, 1 / 6])


def test_alpha_is_locked() -> None:
    with pytest.raises(ValueError):
        cahs_target(4, [0, 0, 0, 1, 0], alpha=0.25)


def test_all_train_dev_targets_are_valid() -> None:
    assert len(load_split("train")) == 2654
    assert len(load_split("dev")) == 664


def test_test_is_refused() -> None:
    with pytest.raises(PermissionError):
        load_split("test")
