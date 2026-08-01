"""Field-local DPO/ODPO loss and strict preference-pair collation.

The score and rationale channels are deliberately disjoint:

* score pairs reduce log-probability only over JSON score-value tokens;
* rationale pairs reduce log-probability only over rationale-content tokens;
* prompt, the other task field, JSON syntax, assistant suffix, and padding are
  never active.

The module returns an objective numerator and denominator separately so a
trainer can accumulate an exact dataset-level weighted mean across a final
short gradient-accumulation group.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from thesis_exp.exp54_rar_sft.training_contract import (
    require_canonical_positions,
)


SCORE_BLOCKS = ("adjacent_score", "severe_l2h", "h2l_guard")
PAIR_TASKS = ("score", "rationale")


def _require_tensor_shapes(logits: Any, input_ids: Any, field_mask: Any) -> None:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError(
            "expected logits [batch, length, vocab] and input IDs [batch, length]"
        )
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input IDs have different batch/length shapes")
    if field_mask.shape != input_ids.shape:
        raise ValueError("field mask must have the same shape as input IDs")


def field_mean_logps(
    logits: Any,
    input_ids: Any,
    field_mask: Any,
) -> dict[str, Any]:
    """Return mean causal log-probability over each row's active field tokens."""
    import torch
    import torch.nn.functional as functional

    _require_tensor_shapes(logits, input_ids, field_mask)
    field_mask = field_mask.to(dtype=torch.bool)
    if torch.any(field_mask[:, 0]):
        raise ValueError("position zero cannot be scored by a causal predictor")
    predict_mask = field_mask[:, 1:]
    token_counts = predict_mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("every preference sequence must activate a nonempty field")

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    token_logps = functional.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=shifted_targets.unsqueeze(-1),
    ).squeeze(-1)
    per_sequence = (
        token_logps * predict_mask.to(dtype=token_logps.dtype)
    ).sum(dim=1) / token_counts.to(dtype=token_logps.dtype)
    return {
        "per_sequence_logp": per_sequence,
        "active_token_count": token_counts,
        "active_token_total": token_counts.sum(),
    }


def policy_reference_contrast(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
) -> Any:
    """Compute (pi+ - pi-) - (ref+ - ref-) for every pair."""
    shapes = {
        tuple(value.shape)
        for value in (
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
    }
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("all chosen/rejected log-probabilities must be 1-D peers")
    return (
        policy_chosen_logps
        - policy_rejected_logps
        - reference_chosen_logps
        + reference_rejected_logps
    )


def field_dpo_per_pair(
    *,
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    beta: float,
    offsets: Any,
) -> dict[str, Any]:
    """Compute -log sigmoid(beta * contrast - pair-specific offset)."""
    import torch
    import torch.nn.functional as functional

    if beta <= 0:
        raise ValueError("beta must be positive")
    contrast = policy_reference_contrast(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if offsets.ndim != 1 or offsets.shape != contrast.shape:
        raise ValueError("offsets must contain one scalar per preference pair")
    if not torch.isfinite(offsets).all() or torch.any(offsets < 0):
        raise ValueError("offsets must be finite and nonnegative")
    dpo_logits = float(beta) * contrast
    margin_logits = dpo_logits - offsets.to(dtype=dpo_logits.dtype)
    return {
        "per_pair_loss": functional.softplus(-margin_logits),
        "policy_reference_contrast": contrast,
        "dpo_logit": dpo_logits,
        "margin_logit": margin_logits,
    }


def exact_objective_weights(
    pair_tasks: Sequence[str],
    pair_types: Sequence[str],
    *,
    score_share: float,
    rationale_share: float,
) -> list[float]:
    """Return weights whose dataset mean equals the declared block objective.

    With ``N`` rows, each score row in block ``b`` receives
    ``N * score_share / (3 * n_b)`` and each rationale row receives
    ``N * rationale_share / n_r``. Therefore:

    ``mean(weight * loss) =
       score_share * mean(mean(loss_b) for b in SCORE_BLOCKS)
       + rationale_share * mean(loss_rationale)``.
    """
    if len(pair_tasks) != len(pair_types) or not pair_tasks:
        raise ValueError("pair task/type vectors must be nonempty and aligned")
    if score_share < 0 or rationale_share < 0:
        raise ValueError("objective shares must be nonnegative")
    if abs(score_share + rationale_share - 1.0) > 1e-12:
        raise ValueError("score and rationale objective shares must sum to one")

    counts = Counter(zip(pair_tasks, pair_types))
    unknown_tasks = set(pair_tasks) - set(PAIR_TASKS)
    if unknown_tasks:
        raise ValueError(f"unknown pair tasks: {sorted(unknown_tasks)}")
    score_types = {
        pair_type
        for task, pair_type in zip(pair_tasks, pair_types)
        if task == "score"
    }
    if score_share:
        if score_types != set(SCORE_BLOCKS):
            raise ValueError("score objective must contain all three exact blocks")
    elif score_types:
        raise ValueError("score rows exist while score_share is zero")
    rationale_types = {
        pair_type
        for task, pair_type in zip(pair_tasks, pair_types)
        if task == "rationale"
    }
    if rationale_share:
        if rationale_types != {"rationale_alignment"}:
            raise ValueError(
                "rationale objective must contain only rationale_alignment"
            )
    elif rationale_types:
        raise ValueError("rationale rows exist while rationale_share is zero")

    total = len(pair_tasks)
    output = []
    for task, pair_type in zip(pair_tasks, pair_types):
        if task == "score":
            denominator = counts[(task, pair_type)]
            output.append(total * score_share / (len(SCORE_BLOCKS) * denominator))
        else:
            denominator = counts[(task, pair_type)]
            output.append(total * rationale_share / denominator)
    return output


def weighted_objective(
    per_pair_loss: Any,
    objective_weights: Any,
    *,
    normalization_pair_count: int | None = None,
) -> dict[str, Any]:
    """Return an exactly accumulable weighted objective.

    A trainer that splits one optimizer update across micro-batches must pass
    the total number of real pairs in that accumulation group as
    ``normalization_pair_count`` for every chunk, then sum the chunk losses.
    This also handles the final short group without padding or silent
    under-weighting.
    """
    import torch

    if per_pair_loss.ndim != 1 or objective_weights.shape != per_pair_loss.shape:
        raise ValueError("losses and objective weights must be aligned 1-D tensors")
    if not torch.isfinite(per_pair_loss).all():
        raise ValueError("per-pair loss contains a non-finite value")
    if not torch.isfinite(objective_weights).all() or torch.any(
        objective_weights <= 0
    ):
        raise ValueError("objective weights must be finite and positive")
    if normalization_pair_count is None:
        normalization_pair_count = int(per_pair_loss.numel())
    if normalization_pair_count < int(per_pair_loss.numel()):
        raise ValueError("normalization count cannot be smaller than this chunk")
    numerator = (per_pair_loss * objective_weights).sum()
    denominator = per_pair_loss.new_tensor(float(normalization_pair_count))
    return {
        "loss": numerator / denominator,
        "objective_numerator": numerator,
        "objective_denominator": denominator,
        "unweighted_loss": per_pair_loss.mean(),
    }


@dataclass(frozen=True)
class _SideBatch:
    input_ids: Any
    attention_mask: Any
    field_mask: Any


class SORCDPOPairCollator:
    """Right-pad chosen/rejected sequences and emit one exact field mask each."""

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
        field_mask = torch.zeros(
            (batch_size, self.cutoff_len),
            dtype=torch.bool,
        )
        for batch_index, feature in enumerate(features):
            ids = list(feature[f"{side}_input_ids"])
            positions = require_canonical_positions(
                feature[f"{side}_field_token_positions"],
                field=f"batch {batch_index}: {side} field positions",
            )
            if int(feature["cutoff_len"]) != self.cutoff_len:
                raise ValueError("feature and collator cutoff lengths differ")
            if not ids or len(ids) > self.cutoff_len:
                raise ValueError("empty sequences and truncation are forbidden")
            if not positions:
                raise ValueError("preference field supervision cannot be empty")
            if any(position < 1 or position >= len(ids) for position in positions):
                raise ValueError(
                    "field position must have a causal predecessor and stay "
                    "inside the unpadded sequence"
                )
            input_ids[batch_index, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
            )
            attention_mask[batch_index, : len(ids)] = 1
            field_mask[batch_index, positions] = True
        if torch.any(field_mask & ~attention_mask.bool()):
            raise ValueError("padding received preference supervision")
        return _SideBatch(input_ids, attention_mask, field_mask)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty preference batch")
        chosen = self._collate_side(features, side="chosen")
        rejected = self._collate_side(features, side="rejected")
        tasks = [str(feature["pair_task"]) for feature in features]
        pair_types = [str(feature["pair_type"]) for feature in features]
        if set(tasks) - set(PAIR_TASKS):
            raise ValueError("batch contains an unknown pair task")
        for feature, task in zip(features, tasks):
            mask_name = str(feature["active_field"])
            expected = "score" if task == "score" else "rationale"
            if mask_name != expected:
                raise ValueError("pair task differs from its declared active field")
        offsets = torch.tensor(
            [float(feature["odpo_offset"]) for feature in features],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [float(feature["objective_weight"]) for feature in features],
            dtype=torch.float32,
        )
        if torch.any(offsets < 0) or not torch.isfinite(offsets).all():
            raise ValueError("batch contains an invalid ODPO offset")
        if torch.any(weights <= 0) or not torch.isfinite(weights).all():
            raise ValueError("batch contains an invalid objective weight")
        return {
            "chosen_input_ids": chosen.input_ids,
            "chosen_attention_mask": chosen.attention_mask,
            "chosen_field_mask": chosen.field_mask,
            "rejected_input_ids": rejected.input_ids,
            "rejected_attention_mask": rejected.attention_mask,
            "rejected_field_mask": rejected.field_mask,
            "odpo_offset": offsets,
            "objective_weight": weights,
            "pair_task": tasks,
            "pair_type": pair_types,
            "pair_id": [str(feature["pair_id"]) for feature in features],
        }


def token_budget(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Aggregate public-safe token counts for a materialized manifest."""
    row_list = list(rows)
    return {
        "pairs": len(row_list),
        "chosen_unpadded_tokens": sum(
            len(row["chosen_input_ids"]) for row in row_list
        ),
        "rejected_unpadded_tokens": sum(
            len(row["rejected_input_ids"]) for row in row_list
        ),
        "chosen_field_tokens": sum(
            len(row["chosen_field_token_positions"]) for row in row_list
        ),
        "rejected_field_tokens": sum(
            len(row["rejected_field_token_positions"]) for row in row_list
        ),
    }
