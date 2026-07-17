from __future__ import annotations

from thesis_exp.exp49_cphce.train import checkpoint_wins


def test_strictly_higher_exact_replaces_checkpoint() -> None:
    assert checkpoint_wins(0.71, 0.70)
    assert not checkpoint_wins(0.70, 0.70)
    assert not checkpoint_wins(0.69, 0.70)
