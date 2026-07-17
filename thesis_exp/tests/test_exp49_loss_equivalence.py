from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from thesis_exp.exp49_cphce.losses import soft_cross_entropy


def test_soft_ce_equals_mean_of_three_rater_cross_entropies() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.3, 1.2, 1.8], [0.1, 0.4, 0.9, 0.2, -0.3]], dtype=torch.float32)
    raters = torch.tensor([[3, 4, 4], [1, 2, 2]], dtype=torch.long)  # zero based
    targets = torch.stack([F.one_hot(row, num_classes=5).float().mean(dim=0) for row in raters])
    soft = soft_cross_entropy(logits, targets)
    repeated = torch.stack([F.cross_entropy(logits, raters[:, index]) for index in range(3)]).mean()
    assert torch.allclose(soft, repeated, atol=1e-6)
