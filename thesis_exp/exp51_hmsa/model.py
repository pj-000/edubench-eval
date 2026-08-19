"""Exact Qwen3 sequence-classification hard path with a copied soft head."""

from __future__ import annotations

import copy
import hashlib
from typing import Any


def module_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_dual_head_classifier(base_model: Any) -> Any:
    """Wrap the loaded classifier without drawing new random parameters."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    if not hasattr(base_model, "model") or not hasattr(base_model, "score"):
        raise TypeError("Exp51 requires the Qwen3 sequence-classification model/model.score path")

    class DualHeadClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = base_model.model
            self.hard_head = base_model.score
            self.soft_head = copy.deepcopy(base_model.score)
            self.config = base_model.config
            self.num_labels = int(base_model.num_labels)

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        def _last_token(self, token_logits: Any, input_ids: Any, inputs_embeds: Any) -> Any:
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 without pad_token_id")
            if self.config.pad_token_id is None or input_ids is None:
                last_non_pad_token: Any = -1
            else:
                non_pad_mask = (input_ids != self.config.pad_token_id).to(token_logits.device, torch.int32)
                token_indices = torch.arange(input_ids.shape[-1], device=token_logits.device, dtype=torch.int32)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            return token_logits[torch.arange(batch_size, device=token_logits.device), last_non_pad_token]

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            labels: Any = None,
            inputs_embeds: Any = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            hidden = outputs.last_hidden_state
            hard_logits = self._last_token(self.hard_head(hidden), input_ids, inputs_embeds)
            soft_logits = self._last_token(self.soft_head(hidden), input_ids, inputs_embeds)
            loss = F.cross_entropy(hard_logits.float(), labels) if labels is not None else None
            return {"loss": loss, "logits": hard_logits, "hard_logits": hard_logits, "soft_logits": soft_logits}

    return DualHeadClassifier()


def head_contract(model: Any) -> dict[str, Any]:
    hard_parameters = list(model.hard_head.parameters())
    soft_parameters = list(model.soft_head.parameters())
    if len(hard_parameters) != len(soft_parameters):
        raise AssertionError("hard/soft head parameter structures differ")
    storage_independent = all(h.data_ptr() != s.data_ptr() for h, s in zip(hard_parameters, soft_parameters))
    return {
        "hard_head_hash": module_hash(model.hard_head),
        "soft_head_hash": module_hash(model.soft_head),
        "storage_independent": storage_independent,
        "hard_parameter_count": sum(value.numel() for value in hard_parameters),
        "soft_parameter_count": sum(value.numel() for value in soft_parameters),
        "hard_bias": model.hard_head.bias is not None,
        "soft_bias": model.soft_head.bias is not None,
    }
