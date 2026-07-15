"""E4 scorer with a training-only TACO projection head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TACOModelConfig:
    model_name_or_path: str
    projection_dim: int = 128
    use_projection: bool = True
    local_files_only: bool = True


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    raise ValueError("Backbone config has no recognized hidden size")


def build_model(config: TACOModelConfig) -> Any:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class TACOScoreModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hf_config = AutoConfig.from_pretrained(
                config.model_name_or_path, local_files_only=config.local_files_only
            )
            self.backbone = AutoModel.from_pretrained(
                config.model_name_or_path,
                config=hf_config,
                local_files_only=config.local_files_only,
            )
            self.hidden_size = _hidden_size(hf_config)
            self.global_head = nn.Linear(self.hidden_size, 5)
            self.use_projection = config.use_projection
            self.projection_dim = config.projection_dim
            self.projection = (
                nn.Linear(self.hidden_size, config.projection_dim, bias=False)
                if config.use_projection
                else None
            )
            if self.projection is not None:
                nn.init.xavier_uniform_(self.projection.weight)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            return_projection: bool = False,
        ) -> dict[str, Any]:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            last = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            result = {"logits": self.global_head(pooled), "pooled": pooled}
            if return_projection:
                if self.projection is None:
                    raise RuntimeError("Projection requested for C0 model")
                result["projection"] = torch.nn.functional.normalize(
                    self.projection(pooled), p=2, dim=-1
                )
            return result

        def added_parameter_count(self) -> int:
            parameters = list(self.global_head.parameters())
            if self.projection is not None:
                parameters += list(self.projection.parameters())
            return sum(parameter.numel() for parameter in parameters)

    return TACOScoreModel()

