"""Reranker classifier used by the Exp46A teacher and students."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    model_name_or_path: str
    use_lora: bool
    gradient_checkpointing: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    raise ValueError("Backbone config has no recognized hidden size")


def build_model(spec: ModelSpec) -> Any:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class HATOClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            config = AutoConfig.from_pretrained(spec.model_name_or_path, local_files_only=True)
            backbone = AutoModel.from_pretrained(spec.model_name_or_path, config=config, local_files_only=True, torch_dtype=torch.bfloat16)
            if spec.use_lora:
                from peft import LoraConfig, TaskType, get_peft_model
                lora = LoraConfig(
                    r=spec.lora_rank,
                    lora_alpha=spec.lora_alpha,
                    lora_dropout=spec.lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                backbone = get_peft_model(backbone, lora)
            if spec.gradient_checkpointing:
                backbone.gradient_checkpointing_enable()
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()
                if hasattr(backbone, "config"):
                    backbone.config.use_cache = False
            self.backbone = backbone
            self.hidden_size = _hidden_size(config)
            self.classifier = nn.Linear(self.hidden_size, 5)

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = output.last_hidden_state
            last = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            return {"logits": self.classifier(pooled), "pooled": pooled}

        def trainable_parameter_counts(self) -> dict[str, int]:
            total = sum(parameter.numel() for parameter in self.parameters())
            trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            return {"total": total, "trainable": trainable}

    return HATOClassifier()
