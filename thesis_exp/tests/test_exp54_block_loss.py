import pytest

torch = pytest.importorskip("torch")

from thesis_exp.exp54_rar_sft.block_loss import (  # noqa: E402
    FixedRARCollator,
    blockwise_causal_loss,
)
from thesis_exp.exp54_rar_sft.mechanism_control_losses import (  # noqa: E402
    token_average_causal_loss,
)


def _logits(input_ids, vocab_size=8):
    return torch.zeros((*input_ids.shape, vocab_size), dtype=torch.float32)


def test_causal_shift_and_inactive_rationale_mask() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = _logits(input_ids)
    score_mask = torch.tensor([[False, False, True, False, False]])
    rationale_mask = torch.zeros_like(score_mask)
    rationale_active = torch.tensor([False])

    baseline = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        rationale_active,
    )["loss"]
    logits[0, 1, 3] = 20.0
    improved = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        rationale_active,
    )["loss"]

    assert improved < baseline
    assert improved.item() < 1e-5


def test_prompt_json_suffix_and_padding_logits_do_not_change_loss() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 0]])
    score_mask = torch.tensor([[False, False, True, False, False, False]])
    rationale_mask = torch.tensor([[False, False, False, True, False, False]])
    active = torch.tensor([True])
    logits = _logits(input_ids)
    baseline = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )["loss"]

    changed = logits.clone()
    changed[0, 0, :] = torch.arange(8) * 100
    changed[0, 3, :] = torch.arange(8) * 100
    changed[0, 4, :] = torch.arange(8) * 100
    result = blockwise_causal_loss(
        changed,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )["loss"]

    assert torch.equal(result, baseline)


def test_each_block_is_normalized_before_the_two_losses_are_added() -> None:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    )
    logits = _logits(input_ids)
    score_mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, False, False, False],
        ]
    )
    rationale_mask = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, True, True, True],
        ]
    )
    result = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        torch.tensor([True, True]),
    )
    expected_block_loss = torch.log(torch.tensor(8.0))

    assert torch.allclose(
        result["per_sample_loss"],
        torch.full((2,), 2 * expected_block_loss),
    )


def test_token_average_matches_two_block_scale_for_uniform_token_loss() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = _logits(input_ids)
    score_mask = torch.tensor([[False, True, False, False, False]])
    rationale_mask = torch.tensor([[False, False, True, True, True]])
    active = torch.tensor([True])

    block = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )
    token_average = token_average_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )

    assert torch.allclose(token_average["loss"], block["loss"])
    assert token_average["active_sample_scale"].item() == pytest.approx(2.0)


def test_token_average_weights_long_rationale_more_than_short_score() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = _logits(input_ids)
    score_mask = torch.tensor([[False, True, False, False, False]])
    rationale_mask = torch.tensor([[False, False, True, True, True]])
    active = torch.tensor([True])
    for position in (2, 3, 4):
        logits[0, position - 1, input_ids[0, position]] = 20.0

    block = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )["loss"]
    token_average = token_average_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        active,
    )["loss"]

    assert token_average < block
    assert token_average.item() == pytest.approx(block.item() / 2, abs=1e-5)


def test_token_average_inactive_row_is_exactly_score_only() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = _logits(input_ids)
    score_mask = torch.tensor([[False, False, True, False]])
    rationale_mask = torch.zeros_like(score_mask)
    inactive = torch.tensor([False])

    block = blockwise_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        inactive,
    )
    token_average = token_average_causal_loss(
        logits,
        input_ids,
        score_mask,
        rationale_mask,
        inactive,
    )

    assert torch.equal(token_average["loss"], block["loss"])
    assert torch.equal(
        token_average["per_sample_loss"],
        block["per_sample_loss"],
    )


def test_fixed_collator_never_supervises_padding() -> None:
    collator = FixedRARCollator(pad_token_id=0, cutoff_len=8)
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "score_token_positions": [2],
                "rationale_token_positions": [3],
                "rationale_active": True,
                "cutoff_len": 8,
            },
            {
                "input_ids": [1, 2, 3],
                "score_token_positions": [2],
                "rationale_token_positions": [],
                "rationale_active": False,
                "cutoff_len": 8,
            },
        ]
    )

    assert batch["attention_mask"].sum(dim=1).tolist() == [5, 3]
    assert batch["score_mask"].sum(dim=1).tolist() == [1, 1]
    assert batch["rationale_mask"].sum(dim=1).tolist() == [1, 0]
    assert not torch.any(batch["score_mask"] & ~batch["attention_mask"].bool())
    assert not torch.any(
        batch["rationale_mask"] & ~batch["attention_mask"].bool()
    )
    assert batch["score_mask"].sum().item() == 2
    assert batch["rationale_mask"].sum().item() == 1


def test_fixed_collator_rejects_duplicate_positions() -> None:
    collator = FixedRARCollator(pad_token_id=0, cutoff_len=8)

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        collator(
            [
                {
                    "input_ids": [1, 2, 3, 4],
                    "score_token_positions": [2, 2],
                    "rationale_token_positions": [],
                    "rationale_active": False,
                    "cutoff_len": 8,
                }
            ]
        )


@pytest.mark.parametrize(
    ("length", "position"),
    [
        (4, -1),
        (4, 4),
        (4, 8),
        (8, -1),
        (8, 8),
    ],
)
def test_fixed_collator_rejects_positions_outside_unpadded_sequence(
    length,
    position,
) -> None:
    collator = FixedRARCollator(pad_token_id=0, cutoff_len=8)

    with pytest.raises(ValueError, match="outside unpadded sequence"):
        collator(
            [
                {
                    "input_ids": [1] * length,
                    "score_token_positions": [position],
                    "rationale_token_positions": [],
                    "rationale_active": False,
                    "cutoff_len": 8,
                }
            ]
        )
