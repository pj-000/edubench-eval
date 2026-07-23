"""Exact two-block causal SFT loss and private manifest collation.

Only score-value tokens contribute to the score block and only active rationale
content tokens contribute to the rationale block. Prompt, fixed JSON syntax,
the unsupervised rationale boundary, assistant suffix, and padding contribute
to neither block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    read_jsonl,
    reject_eval_path,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


def blockwise_causal_loss(
    logits: Any,
    input_ids: Any,
    score_mask: Any,
    rationale_mask: Any,
    rationale_active: Any,
    *,
    rationale_weight: float = 1.0,
) -> dict[str, Any]:
    """Return sample-balanced score + rationale causal language-model loss.

    Masks use input-token coordinates. As in standard causal LM training,
    ``logits[:, position - 1]`` predicts ``input_ids[:, position]``.
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
    if rationale_weight < 0:
        raise ValueError("rationale_weight must be nonnegative")

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

    score_per_sample = (
        token_loss * score_predict_mask.to(token_loss.dtype)
    ).sum(dim=1) / score_counts.to(token_loss.dtype)
    rationale_per_sample = (
        token_loss * rationale_predict_mask.to(token_loss.dtype)
    ).sum(dim=1) / rationale_counts.clamp_min(1).to(token_loss.dtype)
    rationale_per_sample = (
        rationale_per_sample * rationale_active.to(token_loss.dtype)
    )
    total_per_sample = (
        score_per_sample + rationale_weight * rationale_per_sample
    )
    return {
        "loss": total_per_sample.mean(),
        "score_loss": score_per_sample.mean(),
        "rationale_loss_active_mean": (
            rationale_per_sample[rationale_active].mean()
            if torch.any(rationale_active)
            else token_loss.new_zeros(())
        ),
        "per_sample_loss": total_per_sample,
        "score_supervised_token_count": score_counts.sum(),
        "rationale_supervised_token_count": rationale_counts.sum(),
    }


def load_materialized_rows(
    prompt_cache_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Join a private manifest to its shared prompt cache with hard validation."""
    for path in (prompt_cache_path, manifest_path):
        reject_eval_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    prompt_rows = read_jsonl(prompt_cache_path, protect_split=True)
    manifest_rows = read_jsonl(manifest_path, protect_split=True)
    prompt_by_id = {
        str(row["prompt_cache_id"]): row for row in prompt_rows
    }
    if len(prompt_by_id) != len(prompt_rows):
        raise ValueError("prompt cache contains duplicate prompt_cache_id")

    output = []
    seen_events: set[str] = set()
    for row in manifest_rows:
        event_id = str(row["base_event_id"])
        if event_id in seen_events:
            raise ValueError(f"manifest contains duplicate base_event_id: {event_id}")
        seen_events.add(event_id)
        prompt = prompt_by_id.get(str(row["prompt_cache_id"]))
        if prompt is None:
            raise ValueError(f"{event_id}: prompt cache row is missing")
        if str(prompt["record_id"]) != str(row["record_id"]):
            raise ValueError(f"{event_id}: prompt belongs to a different record")
        if (
            prompt["prompt_token_ids_sha256"]
            != row["prompt_token_ids_sha256"]
        ):
            raise ValueError(f"{event_id}: prompt hash differs")
        input_ids = (
            list(prompt["prompt_token_ids"])
            + list(row["target_token_ids"])
            + list(row["assistant_suffix_token_ids"])
        )
        if len(input_ids) != int(row["sequence_token_count"]):
            raise ValueError(f"{event_id}: reconstructed sequence length differs")
        if token_ids_sha256(input_ids) != row["full_token_ids_sha256"]:
            raise ValueError(f"{event_id}: reconstructed sequence hash differs")
        score_positions = [int(value) for value in row["score_token_positions"]]
        rationale_positions = [
            int(value) for value in row["rationale_token_positions"]
        ]
        sequence_positions = set(range(len(input_ids)))
        if not set(score_positions).issubset(sequence_positions):
            raise ValueError(f"{event_id}: score mask is outside sequence")
        if not set(rationale_positions).issubset(sequence_positions):
            raise ValueError(f"{event_id}: rationale mask is outside sequence")
        if set(score_positions) & set(rationale_positions):
            raise ValueError(f"{event_id}: score and rationale masks overlap")
        output.append(
            {
                "base_event_id": event_id,
                "input_ids": input_ids,
                "score_token_positions": score_positions,
                "rationale_token_positions": rationale_positions,
                "rationale_active": bool(row["rationale_active"]),
                "cutoff_len": int(row["cutoff_len"]),
                "full_token_ids_sha256": str(row["full_token_ids_sha256"]),
            }
        )
    return output


class FixedRARCollator:
    """Right-pad reconstructed rows and emit the two exact token masks."""

    def __init__(self, *, pad_token_id: int, cutoff_len: int) -> None:
        if cutoff_len < 1:
            raise ValueError("cutoff_len must be positive")
        self.pad_token_id = int(pad_token_id)
        self.cutoff_len = int(cutoff_len)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
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
        score_mask = torch.zeros(
            (batch_size, self.cutoff_len),
            dtype=torch.bool,
        )
        rationale_mask = torch.zeros_like(score_mask)
        rationale_active = torch.zeros(batch_size, dtype=torch.bool)

        for batch_index, feature in enumerate(features):
            ids = list(feature["input_ids"])
            if int(feature["cutoff_len"]) != self.cutoff_len:
                raise ValueError("feature and collator cutoff lengths differ")
            if len(ids) > self.cutoff_len:
                raise ValueError("truncation is forbidden")
            length = len(ids)
            input_ids[batch_index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_index, :length] = 1
            score_positions = list(feature["score_token_positions"])
            rationale_positions = list(feature["rationale_token_positions"])
            if not score_positions:
                raise ValueError("score supervision cannot be empty")
            score_mask[batch_index, score_positions] = True
            rationale_mask[batch_index, rationale_positions] = True
            rationale_active[batch_index] = bool(feature["rationale_active"])

        if torch.any(score_mask & rationale_mask):
            raise ValueError("collated score and rationale masks overlap")
        if not torch.equal(rationale_mask.any(dim=1), rationale_active):
            raise ValueError("collated rationale activity differs from mask")
        if torch.any(score_mask & ~attention_mask.bool()):
            raise ValueError("padding received score supervision")
        if torch.any(rationale_mask & ~attention_mask.bool()):
            raise ValueError("padding received rationale supervision")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "score_mask": score_mask,
            "rationale_mask": rationale_mask,
            "rationale_active": rationale_active,
        }
