from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")

from thesis_exp.exp49_cphce.losses import soft_cross_entropy
from thesis_exp.exp51_hmsa.model import head_contract, make_dual_head_classifier


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 7)

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None, **kwargs):
        hidden = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()
        self.score = nn.Linear(7, 5, bias=False)
        self.config = SimpleNamespace(pad_token_id=0)
        self.num_labels = 5


def make_inputs():
    return {
        "input_ids": torch.tensor([[2, 3, 0], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }


def test_initialization_is_equal_but_storage_is_independent() -> None:
    torch.manual_seed(42)
    model = make_dual_head_classifier(TinyBase())
    contract = head_contract(model)
    assert contract["hard_head_hash"] == contract["soft_head_hash"]
    assert contract["storage_independent"]
    outputs = model(**make_inputs())
    assert torch.equal(outputs["hard_logits"], outputs["soft_logits"])


def test_hard_path_and_inference_are_isolated_from_soft_parameters() -> None:
    torch.manual_seed(42)
    base = TinyBase()
    inputs = make_inputs()
    hidden = base.model(**inputs).last_hidden_state
    token_logits = base.score(hidden)
    expected = token_logits[torch.arange(2), torch.tensor([1, 2])]
    model = make_dual_head_classifier(base)
    actual = model(**inputs)["hard_logits"]
    assert torch.equal(actual, expected)
    with torch.no_grad():
        model.soft_head.weight.add_(100)
    assert torch.equal(model(**inputs)["hard_logits"], actual)


def test_gradient_routing() -> None:
    torch.manual_seed(42)
    model = make_dual_head_classifier(TinyBase())
    inputs = make_inputs()
    labels = torch.tensor([1, 3])
    targets = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1 / 3, 2 / 3, 0]], dtype=torch.float32)

    model.zero_grad(set_to_none=True)
    hard = F.cross_entropy(model(**inputs)["hard_logits"], labels)
    hard.backward()
    assert model.hard_head.weight.grad is not None
    assert model.soft_head.weight.grad is None
    assert model.backbone.embed.weight.grad is not None
    hard_backbone_grad = model.backbone.embed.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    soft = soft_cross_entropy(model(**inputs)["soft_logits"], targets)
    soft.backward()
    assert model.soft_head.weight.grad is not None
    assert model.hard_head.weight.grad is None
    assert model.backbone.embed.weight.grad is not None
    soft_backbone_grad = model.backbone.embed.weight.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    outputs = model(**inputs)
    total = F.cross_entropy(outputs["hard_logits"], labels) + soft_cross_entropy(outputs["soft_logits"], targets)
    total.backward()
    assert model.hard_head.weight.grad is not None
    assert model.soft_head.weight.grad is not None
    assert model.backbone.embed.weight.grad is not None
    assert torch.allclose(model.backbone.embed.weight.grad, hard_backbone_grad + soft_backbone_grad, atol=1e-7)
