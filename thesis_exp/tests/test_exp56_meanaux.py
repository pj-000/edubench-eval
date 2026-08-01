from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from thesis_exp.exp56_meanaux.build_targets import (
    convert_row,
    load_split,
    validate_human_mean,
)
from thesis_exp.exp56_meanaux.losses import expected_score, mean_aux_smooth_l1
from thesis_exp.exp56_meanaux.model import head_contract, make_hard_mean_classifier
from thesis_exp.exp56_meanaux.train import parse_args


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone(hidden_size=7)
        self.score = nn.Linear(7, 5, bias=False)
        self.config = SimpleNamespace(pad_token_id=0)
        self.num_labels = 5


def test_expected_score_and_smooth_l1() -> None:
    logits = torch.tensor(
        [
            [30.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    predicted = expected_score(logits)
    assert predicted.tolist() == pytest.approx([1.0, 5.0, 3.0], abs=1e-5)
    target = torch.tensor([1.0, 5.0, 3.0])
    assert float(mean_aux_smooth_l1(logits, target)) == pytest.approx(0.0, abs=1e-9)


def test_heads_are_identical_but_storage_independent() -> None:
    torch.manual_seed(42)
    model = make_hard_mean_classifier(TinyClassifier())
    contract = head_contract(model)
    assert contract["hard_head_hash"] == contract["mean_head_hash"]
    assert contract["storage_independent"]
    assert contract["hard_parameter_count"] == contract["mean_parameter_count"]
    assert not contract["hard_bias"]
    assert not contract["mean_bias"]


def test_mean_loss_routes_to_backbone_and_mean_head_only() -> None:
    torch.manual_seed(42)
    model = make_hard_mean_classifier(TinyClassifier())
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    outputs = model(input_ids=input_ids)
    loss = mean_aux_smooth_l1(
        outputs["mean_logits"],
        torch.tensor([3.0, 4.0]),
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.mean_head.parameters())
    assert all(parameter.grad is None for parameter in model.hard_head.parameters())


def test_hard_path_isolated_from_mean_head_mutation() -> None:
    torch.manual_seed(42)
    model = make_hard_mean_classifier(TinyClassifier())
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    before = model(input_ids=input_ids)["hard_logits"].detach().clone()
    with torch.no_grad():
        model.mean_head.weight.add_(10.0)
    after = model(input_ids=input_ids)["hard_logits"].detach()
    assert torch.equal(before, after)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 1.0),
        (4.0 / 3.0, 4.0 / 3.0),
        (11.0 / 3.0, 11.0 / 3.0),
        (5.0, 5.0),
    ],
)
def test_valid_three_rater_mean_grid(value: float, expected: float) -> None:
    assert validate_human_mean(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.99, 5.01, 3.1])
def test_invalid_mean_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        validate_human_mean(value)


def test_row_conversion_preserves_hard_target_and_mean() -> None:
    row = {"label_5": 4, "human_mean_5": 11.0 / 3.0, "record_id": "r1"}
    converted = convert_row(row)
    assert converted["hard_target_5"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert converted["mean_aux_target"] == pytest.approx(11.0 / 3.0)


def test_test_split_is_forbidden_before_any_loader_call(monkeypatch) -> None:
    called = False

    def forbidden_loader(split: str):
        nonlocal called
        called = True
        raise AssertionError("upstream loader must not be reached")

    monkeypatch.setattr(
        "thesis_exp.exp56_meanaux.build_targets.load_exp49_split",
        forbidden_loader,
    )
    with pytest.raises(PermissionError):
        load_split("test")
    assert not called


def test_explicit_checkpoint_root_remains_seed_isolated(monkeypatch) -> None:
    checkpoint_root = Path("thesis_exp/artifacts/exp56_meanaux/test/seed_43")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model_name_or_path",
            "model",
            "--seed",
            "43",
            "--checkpoint_output_dir",
            str(checkpoint_root),
        ],
    )
    config = parse_args()
    assert config.checkpoint_output_dir == checkpoint_root


def test_default_checkpoint_root_is_seed_isolated(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--model_name_or_path", "model", "--seed", "44"],
    )
    config = parse_args()
    assert config.checkpoint_output_dir.name == "seed_44"
