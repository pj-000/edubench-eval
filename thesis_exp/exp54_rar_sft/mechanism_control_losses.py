"""Losses and collation used only by the post-hoc mechanism controls.

The completed RAR-SFT and Field-DPO modules are frozen result dependencies and
must remain byte-identical. New control logic lives here so implementing the
ablations does not rewrite the source files bound to the completed runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thesis_exp.exp54_rar_sft.sorc_dpo_loss import SCORE_BLOCKS
from thesis_exp.exp54_rar_sft.training_contract import (
    require_canonical_positions,
)


def token_average_causal_loss(
    logits: Any,
    input_ids: Any,
    score_mask: Any,
    rationale_mask: Any,
    rationale_active: Any,
    *,
    active_sample_scale: float = 2.0,
) -> dict[str, Any]:
    """Return the matched R3-TOKENAVG mechanism-control loss.

    Active score and rationale tokens are pooled before one token mean is
    calculated. The default factor of two matches the two unit-weight blocks
    in the frozen R3 loss. Inactive rows remain exactly score-only.
    """
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, length, vocab] and IDs [batch, length]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input IDs have different batch/length shapes")
    if score_mask.shape != input_ids.shape or rationale_mask.shape != input_ids.shape:
        raise ValueError("loss masks must have the same shape as input IDs")
    if rationale_active.ndim != 1 or rationale_active.shape[0] != input_ids.shape[0]:
        raise ValueError("rationale_active must contain one value per sample")
    if not isinstance(active_sample_scale, (int, float)) or active_sample_scale <= 0:
        raise ValueError("active_sample_scale must be positive")

    score_mask = score_mask.to(dtype=torch.bool)
    rationale_mask = rationale_mask.to(dtype=torch.bool)
    rationale_active = rationale_active.to(dtype=torch.bool)
    if torch.any(score_mask & rationale_mask):
        raise ValueError("score and rationale masks overlap")
    if torch.any(score_mask[:, 0]) or torch.any(rationale_mask[:, 0]):
        raise ValueError("position zero cannot be supervised by a causal predictor")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_targets = input_ids[:, 1:].contiguous()
    token_loss = functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_targets,
        reduction="none",
    )
    score_predict_mask = score_mask[:, 1:]
    rationale_predict_mask = rationale_mask[:, 1:]
    score_counts = score_predict_mask.sum(dim=1)
    rationale_counts = rationale_predict_mask.sum(dim=1)
    if torch.any(score_counts == 0):
        raise ValueError("every sample must supervise at least one score token")
    if not torch.equal(rationale_counts > 0, rationale_active):
        raise ValueError("rationale mask activity differs from rationale_active")

    score_numerator = (
        token_loss * score_predict_mask.to(token_loss.dtype)
    ).sum(dim=1)
    rationale_numerator = (
        token_loss * rationale_predict_mask.to(token_loss.dtype)
    ).sum(dim=1)
    combined_counts = score_counts + rationale_counts
    pooled_per_sample = (
        score_numerator + rationale_numerator
    ) / combined_counts.to(token_loss.dtype)
    sample_scale = torch.where(
        rationale_active,
        token_loss.new_full(
            (input_ids.shape[0],),
            float(active_sample_scale),
        ),
        token_loss.new_ones((input_ids.shape[0],)),
    )
    total_per_sample = pooled_per_sample * sample_scale
    score_per_sample = score_numerator / score_counts.to(token_loss.dtype)
    rationale_per_sample = (
        rationale_numerator
        / rationale_counts.clamp_min(1).to(token_loss.dtype)
    ) * rationale_active.to(token_loss.dtype)
    return {
        "loss": total_per_sample.mean(),
        "score_loss": score_per_sample.mean(),
        "rationale_loss_active_mean": (
            rationale_per_sample[rationale_active].mean()
            if torch.any(rationale_active)
            else token_loss.new_zeros(())
        ),
        "per_sample_loss": total_per_sample,
        "active_sample_scale": token_loss.new_tensor(float(active_sample_scale)),
        "score_supervised_token_count": score_counts.sum(),
        "rationale_supervised_token_count": rationale_counts.sum(),
    }


def sequence_sum_logps(
    logits: Any,
    input_ids: Any,
    completion_mask: Any,
) -> dict[str, Any]:
    """Return standard DPO summed log-probability over full completions."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError(
            "expected logits [batch, length, vocab] and input IDs [batch, length]"
        )
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input IDs have different batch/length shapes")
    if completion_mask.shape != input_ids.shape:
        raise ValueError("completion mask must have the same shape as input IDs")
    completion_mask = completion_mask.to(dtype=torch.bool)
    if torch.any(completion_mask[:, 0]):
        raise ValueError("position zero cannot be scored by a causal predictor")
    predict_mask = completion_mask[:, 1:]
    token_counts = predict_mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("every DPO sequence must activate a nonempty completion")

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    token_logps = functional.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=shifted_targets.unsqueeze(-1),
    ).squeeze(-1)
    per_sequence = (
        token_logps * predict_mask.to(dtype=token_logps.dtype)
    ).sum(dim=1)
    return {
        "per_sequence_logp": per_sequence,
        "active_token_count": token_counts,
        "active_token_total": token_counts.sum(),
    }


@dataclass(frozen=True)
class _SideBatch:
    input_ids: Any
    attention_mask: Any
    completion_mask: Any


class FullSequenceDPOPairCollator:
    """Collate the same pairs with the complete assistant response active."""

    def __init__(self, *, pad_token_id: int, cutoff_len: int) -> None:
        if cutoff_len < 1:
            raise ValueError("cutoff_len must be positive")
        self.pad_token_id = int(pad_token_id)
        self.cutoff_len = int(cutoff_len)

    def _collate_side(
        self,
        features: list[dict[str, Any]],
        *,
        side: str,
    ) -> _SideBatch:
        import torch

        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, self.cutoff_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, self.cutoff_len),
            dtype=torch.long,
        )
        completion_mask = torch.zeros(
            (batch_size, self.cutoff_len),
            dtype=torch.bool,
        )
        for batch_index, feature in enumerate(features):
            ids = list(feature[f"{side}_input_ids"])
            if int(feature["cutoff_len"]) != self.cutoff_len:
                raise ValueError("feature and collator cutoff lengths differ")
            if not ids or len(ids) > self.cutoff_len:
                raise ValueError("empty sequences and truncation are forbidden")
            prompt_count = feature.get(f"{side}_prompt_token_count")
            if (
                isinstance(prompt_count, bool)
                or not isinstance(prompt_count, int)
                or prompt_count < 1
                or prompt_count >= len(ids)
            ):
                raise ValueError(
                    "prompt token count must leave a nonempty causal completion"
                )
            positions = require_canonical_positions(
                feature[f"{side}_completion_token_positions"],
                field=f"batch {batch_index}: {side} completion positions",
            )
            expected_positions = list(range(prompt_count, len(ids)))
            if positions != expected_positions:
                raise ValueError(
                    "full-sequence mask must equal the complete assistant suffix"
                )
            input_ids[batch_index, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
            )
            attention_mask[batch_index, : len(ids)] = 1
            completion_mask[batch_index, positions] = True
        if torch.any(completion_mask & ~attention_mask.bool()):
            raise ValueError("padding received full-sequence supervision")
        return _SideBatch(input_ids, attention_mask, completion_mask)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty preference batch")
        chosen = self._collate_side(features, side="chosen")
        rejected = self._collate_side(features, side="rejected")
        tasks = [str(feature["pair_task"]) for feature in features]
        pair_types = [str(feature["pair_type"]) for feature in features]
        if set(tasks) != {"score"}:
            raise ValueError("P1-FULLSEQ accepts score preference pairs only")
        if set(pair_types) - set(SCORE_BLOCKS):
            raise ValueError("batch contains an unknown score pair block")
        offsets = torch.tensor(
            [float(feature["odpo_offset"]) for feature in features],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [float(feature["objective_weight"]) for feature in features],
            dtype=torch.float32,
        )
        if torch.any(offsets != 0):
            raise ValueError("P1-FULLSEQ does not accept an ODPO offset")
        if torch.any(weights <= 0) or not torch.isfinite(weights).all():
            raise ValueError("batch contains an invalid objective weight")
        return {
            "chosen_input_ids": chosen.input_ids,
            "chosen_attention_mask": chosen.attention_mask,
            "chosen_completion_mask": chosen.completion_mask,
            "rejected_input_ids": rejected.input_ids,
            "rejected_attention_mask": rejected.attention_mask,
            "rejected_completion_mask": rejected.completion_mask,
            "odpo_offset": offsets,
            "objective_weight": weights,
            "pair_task": tasks,
            "pair_type": pair_types,
            "pair_id": [str(feature["pair_id"]) for feature in features],
        }
