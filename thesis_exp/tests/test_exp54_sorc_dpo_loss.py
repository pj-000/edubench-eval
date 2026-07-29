from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
    exact_objective_weights,
    field_dpo_per_pair,
    field_mean_logps,
    weighted_objective,
)


def _logits_for_targets(
    input_ids: torch.Tensor,
    *,
    vocab_size: int = 17,
    target_logit: float = 3.0,
) -> torch.Tensor:
    logits = torch.zeros((*input_ids.shape, vocab_size), dtype=torch.float64)
    for batch_index in range(input_ids.shape[0]):
        for position in range(1, input_ids.shape[1]):
            logits[
                batch_index,
                position - 1,
                input_ids[batch_index, position],
            ] = target_logit
    return logits


def test_field_mean_logps_uses_causal_predecessor_and_length_mean() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
    logits = _logits_for_targets(input_ids)
    mask = torch.tensor(
        [[False, True, True, False], [False, False, True, False]]
    )
    result = field_mean_logps(logits, input_ids, mask)
    expected = 3.0 - math.log(math.exp(3.0) + 16.0)
    assert result["per_sequence_logp"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert result["active_token_count"].tolist() == [2, 1]


def test_field_mean_logps_ignores_logits_outside_active_field() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, False, True, False]])
    first = _logits_for_targets(input_ids)
    second = first.clone()
    second[0, 0, :] = torch.arange(17, dtype=torch.float64) * 100
    second[0, 2, :] = -torch.arange(17, dtype=torch.float64) * 100
    assert field_mean_logps(first, input_ids, mask)[
        "per_sequence_logp"
    ].item() == pytest.approx(
        field_mean_logps(second, input_ids, mask)[
            "per_sequence_logp"
        ].item()
    )


def test_field_mean_logps_rejects_empty_and_position_zero_masks() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    logits = _logits_for_targets(input_ids)
    with pytest.raises(ValueError, match="nonempty"):
        field_mean_logps(logits, input_ids, torch.zeros_like(input_ids).bool())
    mask = torch.tensor([[True, False, False]])
    with pytest.raises(ValueError, match="position zero"):
        field_mean_logps(logits, input_ids, mask)


def test_dpo_offset_is_subtracted_after_beta_scaling_once() -> None:
    result = field_dpo_per_pair(
        policy_chosen_logps=torch.tensor([-1.0]),
        policy_rejected_logps=torch.tensor([-3.0]),
        reference_chosen_logps=torch.tensor([-2.0]),
        reference_rejected_logps=torch.tensor([-3.0]),
        beta=0.1,
        offsets=torch.tensor([0.5]),
    )
    # Contrast = 1.0, dpo logit = 0.1, margin logit = -0.4.
    assert result["policy_reference_contrast"].item() == pytest.approx(1.0)
    assert result["dpo_logit"].item() == pytest.approx(0.1)
    assert result["margin_logit"].item() == pytest.approx(-0.4)
    assert result["per_pair_loss"].item() == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(0.4)).item()
    )


def test_zero_offset_is_ordinary_field_dpo() -> None:
    result = field_dpo_per_pair(
        policy_chosen_logps=torch.tensor([-1.0, -3.0]),
        policy_rejected_logps=torch.tensor([-2.0, -1.0]),
        reference_chosen_logps=torch.tensor([-1.5, -1.5]),
        reference_rejected_logps=torch.tensor([-1.5, -1.5]),
        beta=0.1,
        offsets=torch.zeros(2),
    )
    contrast = torch.tensor([1.0, -2.0])
    expected = -torch.nn.functional.logsigmoid(0.1 * contrast)
    assert torch.allclose(result["per_pair_loss"], expected)


def test_exact_three_score_block_weighting() -> None:
    tasks = ["score"] * 8
    types = (
        ["adjacent_score"] * 6
        + ["severe_l2h"]
        + ["h2l_guard"]
    )
    weights = torch.tensor(
        exact_objective_weights(
            tasks,
            types,
            score_share=1.0,
            rationale_share=0.0,
        )
    )
    losses = torch.tensor([1.0] * 6 + [3.0, 5.0])
    result = weighted_objective(losses, weights)
    assert result["loss"].item() == pytest.approx((1.0 + 3.0 + 5.0) / 3.0)
    for pair_type in set(types):
        indices = [index for index, value in enumerate(types) if value == pair_type]
        assert weights[indices].sum().item() == pytest.approx(8.0 / 3.0)


def test_joint_objective_is_half_score_half_rationale() -> None:
    tasks = ["score"] * 5 + ["rationale"] * 4
    types = [
        "adjacent_score",
        "adjacent_score",
        "adjacent_score",
        "severe_l2h",
        "h2l_guard",
        "rationale_alignment",
        "rationale_alignment",
        "rationale_alignment",
        "rationale_alignment",
    ]
    weights = torch.tensor(
        exact_objective_weights(
            tasks,
            types,
            score_share=0.5,
            rationale_share=0.5,
        )
    )
    losses = torch.tensor(
        [1.0, 1.0, 1.0, 3.0, 5.0, 7.0, 7.0, 7.0, 7.0]
    )
    result = weighted_objective(losses, weights)
    expected_score = (1.0 + 3.0 + 5.0) / 3.0
    assert result["loss"].item() == pytest.approx(
        0.5 * expected_score + 0.5 * 7.0
    )


def test_microbatch_chunks_sum_to_exact_short_group_objective() -> None:
    losses = torch.tensor([1.0, 2.0, 4.0])
    weights = torch.tensor([0.5, 1.0, 1.5])
    first = weighted_objective(
        losses[:2],
        weights[:2],
        normalization_pair_count=3,
    )
    second = weighted_objective(
        losses[2:],
        weights[2:],
        normalization_pair_count=3,
    )
    whole = weighted_objective(losses, weights)
    assert (first["loss"] + second["loss"]).item() == pytest.approx(
        whole["loss"].item()
    )


def _feature(
    *,
    task: str = "score",
    pair_type: str = "adjacent_score",
) -> dict[str, object]:
    positions = [2] if task == "score" else [3, 4]
    return {
        "pair_id": "pair-1",
        "pair_task": task,
        "pair_type": pair_type,
        "active_field": task,
        "chosen_input_ids": [1, 2, 3, 4, 5],
        "chosen_field_token_positions": positions,
        "rejected_input_ids": [1, 2, 6, 7, 8, 9],
        "rejected_field_token_positions": positions,
        "odpo_offset": 0.25 if task == "score" else 0.0,
        "objective_weight": 1.0,
        "cutoff_len": 8,
    }


def test_collator_emits_disjoint_exact_chosen_and_rejected_masks() -> None:
    collator = SORCDPOPairCollator(pad_token_id=0, cutoff_len=8)
    score = _feature()
    rationale = _feature(task="rationale", pair_type="rationale_alignment")
    rationale["pair_id"] = "pair-2"
    batch = collator([score, rationale])
    assert batch["chosen_input_ids"].shape == (2, 8)
    assert batch["rejected_input_ids"].shape == (2, 8)
    assert batch["chosen_field_mask"][0].nonzero().flatten().tolist() == [2]
    assert batch["chosen_field_mask"][1].nonzero().flatten().tolist() == [3, 4]
    assert not torch.any(
        batch["chosen_field_mask"] & ~batch["chosen_attention_mask"].bool()
    )
    assert batch["odpo_offset"].tolist() == pytest.approx([0.25, 0.0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(chosen_field_token_positions=[]), "empty"),
        (lambda row: row.update(chosen_field_token_positions=[0]), "predecessor"),
        (lambda row: row.update(chosen_field_token_positions=[5]), "unpadded"),
        (lambda row: row.update(chosen_field_token_positions=[2, 2]), "unique"),
        (lambda row: row.update(active_field="rationale"), "active field"),
        (lambda row: row.update(odpo_offset=-0.1), "offset"),
        (lambda row: row.update(objective_weight=0.0), "weight"),
    ],
)
def test_collator_fails_closed(mutation, message: str) -> None:
    row = _feature()
    mutation(row)
    with pytest.raises(ValueError, match=message):
        SORCDPOPairCollator(pad_token_id=0, cutoff_len=8)([row])
