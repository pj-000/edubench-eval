"""GPU model, data-loader, and evaluation helpers for Exp36A."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp36_safer_score.common import build_model_text, metrics, sample_id


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_for(torch: Any, bf16: str) -> Any:
    if bf16 == "true" or (bf16 == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return None


def build_model(model_name_or_path: str, bf16: str, local_files_only: bool, gradient_checkpointing: bool) -> Any:
    import torch
    from torch import nn
    from transformers import AutoModel

    class SaferModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=False,
                torch_dtype=dtype_for(torch, bf16),
            )
            hidden = getattr(self.backbone.config, "hidden_size", None) or getattr(self.backbone.config, "n_embd", None)
            if hidden is None:
                raise ValueError("Cannot infer Qwen hidden size")
            self.score_head = nn.Linear(int(hidden), 5)
            self.failure_head = nn.Linear(int(hidden), 6)
            backbone_dtype = next(self.backbone.parameters()).dtype
            self.score_head.to(dtype=backbone_dtype)
            self.failure_head.to(dtype=backbone_dtype)
            if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
                if hasattr(self.backbone.config, "use_cache"):
                    self.backbone.config.use_cache = False

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = output.last_hidden_state
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            return self.score_head(pooled), self.failure_head(pooled)

    return SaferModel()


def load_tokenizer(model_name_or_path: str, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


class Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def make_loader(
    rows: list[dict[str, Any]], tokenizer: Any, batch_size: int, max_length: int,
    shuffle: bool, seed: int, num_workers: int = 0,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row.get("text") or build_model_text(row) for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["metadata"] = batch
        return encoded

    return DataLoader(
        Rows(rows), batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=num_workers,
        drop_last=False, collate_fn=collate,
    )


def evaluate(model: Any, loader: Any, device: Any) -> list[dict[str, Any]]:
    import torch

    model.eval()
    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            score_logits, failure_logits = model(**{key: value.to(device) for key, value in batch.items()})
            probs = torch.softmax(score_logits.float(), dim=-1).cpu().numpy()
            failure_probs = torch.sigmoid(failure_logits.float()).cpu().numpy()
            for row, prob, fail_prob in zip(metadata, probs, failure_probs):
                pred = int(np.argmax(prob)) + 1
                item = {
                    "sample_id": sample_id(row),
                    "question_key": row.get("question_key"),
                    "gold_label_5": int(row["label_5"]),
                    "pred_label_5": pred,
                    "pred_score_expected": float(sum(label * prob[label - 1] for label in range(1, 6))),
                    "language": row.get("language"),
                    "metric_group": row.get("metric_group"),
                    "subject_canonical": row.get("subject_canonical"),
                }
                item.update({f"prob_{label}": float(prob[label - 1]) for label in range(1, 6)})
                item.update({f"failure_prob_{index}": float(fail_prob[index]) for index in range(6)})
                output.append(item)
    return output


def save_model(model: Any, tokenizer: Any, path: Path, metadata: dict[str, Any]) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "state_dict.pt")
    tokenizer.save_pretrained(path)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model_state(model: Any, path: Path) -> None:
    import torch

    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))


def selected_checkpoint(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Locked order: Exact desc, MAE asc, Kendall desc, epoch asc."""
    return min(
        history,
        key=lambda row: (
            -float(row["Exact_Match"]),
            float(row["MAE_argmax"]),
            -float(row["Kendall_tau"]),
            int(row["epoch"]),
        ),
    )


def metric_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics(predictions)
