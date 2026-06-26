"""Boundary-linking ordinal model for Exp16A."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryLinkingOrdinalModel(nn.Module):
    """Shared encoder with a quality scalar and context-conditioned thresholds."""

    def __init__(
        self,
        encoder: Any,
        hidden_size: int,
        variant: str = "qmr_meta",
        min_increment: float = 1e-4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.variant = variant
        self.min_increment = float(min_increment)
        self.quality_head = nn.Linear(self.hidden_size, 1)
        self.start_head = nn.Linear(self.hidden_size, 1)
        self.inc_head = nn.Linear(self.hidden_size, 4)
        self.scale_head = nn.Linear(self.hidden_size, 1)
        self.global_start = nn.Parameter(torch.tensor(-1.5))
        self.global_inc_raw = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_model_name(
        cls,
        model_name_or_path: str,
        variant: str = "qmr_meta",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> "BoundaryLinkingOrdinalModel":
        from transformers import AutoConfig, AutoModel, BertConfig

        if model_name_or_path == "__tiny_random__":
            config = BertConfig(
                vocab_size=30522,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=512,
            )
            encoder = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        hidden_size = int(
            getattr(encoder.config, "hidden_size", None)
            or getattr(encoder.config, "n_embd", None)
            or getattr(encoder.config, "d_model", None)
        )
        return cls(encoder=encoder, hidden_size=hidden_size, variant=variant)

    @staticmethod
    def _pool(outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs, attention_mask)

    def ordered_thresholds(self, boundary_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "global":
            boundary_h = torch.zeros_like(boundary_h)
        start = self.global_start + self.start_head(boundary_h).squeeze(-1)
        inc_raw = self.global_inc_raw.unsqueeze(0) + self.inc_head(boundary_h)
        inc = F.softplus(inc_raw) + self.min_increment
        tau = start.unsqueeze(-1) + torch.cumsum(inc, dim=-1)
        alpha = F.softplus(self.scale_head(boundary_h).squeeze(-1)) + self.min_increment
        return tau, alpha

    def forward(
        self,
        quality_input_ids: torch.Tensor,
        quality_attention_mask: torch.Tensor,
        boundary_input_ids: torch.Tensor,
        boundary_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        quality_h = self.encode(quality_input_ids, quality_attention_mask)
        boundary_h = self.encode(boundary_input_ids, boundary_attention_mask)
        quality_score_s = self.quality_head(quality_h).squeeze(-1)
        thresholds_tau, scale_alpha = self.ordered_thresholds(boundary_h)
        logits = scale_alpha.unsqueeze(-1) * (quality_score_s.unsqueeze(-1) - thresholds_tau)
        probs = torch.sigmoid(logits)
        pred_label = 1 + (probs > 0.5).sum(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred_label": pred_label,
            "quality_score_s": quality_score_s,
            "thresholds_tau": thresholds_tau,
            "scale_alpha": scale_alpha,
        }
