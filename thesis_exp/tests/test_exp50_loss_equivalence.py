from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from thesis_exp.exp49_cphce.losses import soft_cross_entropy


def test_cahs_loss_equals_half_hard_plus_half_human_soft() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.3, 1.2, 1.8], [0.1, 0.4, 0.9, 0.2, -0.3]])
    hard = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    human = torch.tensor([[0, 0, 0, 2 / 3, 1 / 3], [0, 1 / 3, 2 / 3, 0, 0]], dtype=torch.float32)
    cahs = 0.5 * hard + 0.5 * human
    actual = soft_cross_entropy(logits, cahs)
    expected = 0.5 * soft_cross_entropy(logits, hard) + 0.5 * soft_cross_entropy(logits, human)
    assert torch.allclose(actual, expected, atol=1e-7)
