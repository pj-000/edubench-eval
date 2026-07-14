"""AutoModel-based RubiMOR classifier with optional low-rank metric residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RubiMORConfig:
    model_name_or_path: str
    num_metrics: int
    metric_rank: int = 4
    use_metric_residual: bool = False
    local_files_only: bool = True


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        if getattr(config, name, None):
            return int(getattr(config, name))
    raise ValueError("Backbone config has no recognized hidden size")


def build_model(config: RubiMORConfig) -> Any:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class RubiMORModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hf_config = AutoConfig.from_pretrained(config.model_name_or_path, local_files_only=config.local_files_only)
            self.backbone = AutoModel.from_pretrained(config.model_name_or_path, config=hf_config, local_files_only=config.local_files_only)
            self.hidden_size = _hidden_size(hf_config)
            self.global_head = nn.Linear(self.hidden_size, 5)
            self.use_metric_residual = config.use_metric_residual
            self.num_metrics = config.num_metrics
            self.metric_rank = config.metric_rank
            if self.use_metric_residual:
                self.metric_v = nn.Parameter(torch.empty(config.num_metrics, config.metric_rank, self.hidden_size))
                self.metric_u = nn.Parameter(torch.zeros(config.num_metrics, 5, config.metric_rank))
                self.metric_bias = nn.Parameter(torch.zeros(config.num_metrics, 5))
                nn.init.xavier_uniform_(self.metric_v)

        def forward(self, input_ids: Any, attention_mask: Any, metric_ids: Any | None = None, residual_enabled: Any | None = None) -> dict[str, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = output.last_hidden_state
            last = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            logits = self.global_head(pooled)
            if self.use_metric_residual and metric_ids is not None:
                valid = metric_ids.ge(0) & metric_ids.lt(self.num_metrics)
                if residual_enabled is not None:
                    valid = valid & residual_enabled.bool()
                safe_ids = metric_ids.clamp(0, max(self.num_metrics - 1, 0))
                projected = torch.einsum("brd,bd->br", self.metric_v[safe_ids], pooled)
                residual = torch.einsum("bkr,br->bk", self.metric_u[safe_ids], projected) + self.metric_bias[safe_ids]
                logits = logits + residual * valid.unsqueeze(-1).to(residual.dtype)
            return {"logits": logits, "pooled": pooled}

        def added_parameter_count(self) -> int:
            modules = list(self.global_head.parameters())
            if self.use_metric_residual:
                modules += [self.metric_v, self.metric_u, self.metric_bias]
            return sum(parameter.numel() for parameter in modules)

    return RubiMORModel()

