from __future__ import annotations

import math

import pytest

from thesis_exp.exp54_rar_sft.extract_label2_score_probabilities import (
    CanonicalScorePlan,
    SCORES,
    _score_batch,
    build_canonical_score_plan,
    longest_common_token_prefix,
    normalize_logprob_row,
)


class FakeTokenizer:
    def __init__(self, values: dict[str, list[int]]) -> None:
        self.values = values
        self.reverse = {tuple(value): key for key, value in values.items()}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.values[text])

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return self.reverse[tuple(token_ids)]


class PaddingTokenizer:
    def pad(self, value: dict, *, padding: bool, return_tensors: str) -> dict:
        torch = pytest.importorskip("torch")
        assert padding is True
        assert return_tensors == "pt"
        rows = value["input_ids"]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + list(row) for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeOutput:
    def __init__(self, logits) -> None:
        self.logits = logits


class FastPathModel:
    def __call__(self, **kwargs):
        torch = pytest.importorskip("torch")
        assert kwargs["logits_to_keep"] == 1
        batch = int(kwargs["input_ids"].shape[0])
        logits = torch.zeros((batch, 1, 64), dtype=torch.float32)
        for score in SCORES:
            logits[:, 0, 20 + score] = float(score)
        return FakeOutput(logits)


class GenericPathModel:
    def __call__(self, **kwargs):
        torch = pytest.importorskip("torch")
        assert kwargs["logits_to_keep"] == 3
        input_ids = kwargs["input_ids"]
        batch = int(input_ids.shape[0])
        logits = torch.zeros((batch, 3, 64), dtype=torch.float32)
        for index in range(batch):
            first = int(input_ids[index, -2])
            second = int(input_ids[index, -1])
            score = first - 20
            logits[index, 0, first] = float(score)
            logits[index, 1, second] = float(score)
        return FakeOutput(logits)


def test_canonical_plan_supports_multi_token_score_values() -> None:
    tokenizer = FakeTokenizer(
        {
            f'{{"score":{score}': [10, 11, 20 + score, 30 + score]
            for score in SCORES
        }
    )
    plan = build_canonical_score_plan(tokenizer)
    assert plan.common_prefix_ids == [10, 11]
    assert plan.continuation_ids == {
        score: [20 + score, 30 + score] for score in SCORES
    }


def test_common_prefix_rejects_missing_shared_prefix() -> None:
    with pytest.raises(ValueError, match="no shared"):
        longest_common_token_prefix([[1, 2], [3, 4]])


def test_common_prefix_rejects_empty_continuation() -> None:
    with pytest.raises(ValueError, match="continuation is empty"):
        longest_common_token_prefix([[1], [1, 2]])


def test_logprob_normalization_is_stable() -> None:
    probabilities = normalize_logprob_row([-1000, -1001, -1002, -1003, -1004])
    assert len(probabilities) == 5
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[4]


def test_score_batch_fast_path_gathers_all_five_candidates() -> None:
    pytest.importorskip("torch")
    plan = CanonicalScorePlan(
        full_token_ids={score: [10, 20 + score] for score in SCORES},
        common_prefix_ids=[10],
        continuation_ids={score: [20 + score] for score in SCORES},
    )
    result = _score_batch(
        model=FastPathModel(),
        tokenizer=PaddingTokenizer(),
        prompt_token_ids=[[7], [8, 9]],
        plan=plan,
        device="cpu",
    )
    assert len(result) == 2
    assert all(row[0] < row[1] < row[2] < row[3] < row[4] for row in result)


def test_score_batch_generic_path_sums_multi_token_continuations() -> None:
    pytest.importorskip("torch")
    plan = CanonicalScorePlan(
        full_token_ids={
            score: [10, 20 + score, 40 + score] for score in SCORES
        },
        common_prefix_ids=[10],
        continuation_ids={
            score: [20 + score, 40 + score] for score in SCORES
        },
    )
    result = _score_batch(
        model=GenericPathModel(),
        tokenizer=PaddingTokenizer(),
        prompt_token_ids=[[7], [8, 9]],
        plan=plan,
        device="cpu",
    )
    assert len(result) == 2
    assert all(row[0] < row[1] < row[2] < row[3] < row[4] for row in result)


@pytest.mark.parametrize(
    "values",
    [
        [0.0] * 4,
        [0.0] * 6,
        [0.0, 0.0, math.nan, 0.0, 0.0],
        [0.0, 0.0, math.inf, 0.0, 0.0],
    ],
)
def test_logprob_normalization_rejects_invalid_rows(values: list[float]) -> None:
    with pytest.raises(ValueError, match="five finite"):
        normalize_logprob_row(values)
