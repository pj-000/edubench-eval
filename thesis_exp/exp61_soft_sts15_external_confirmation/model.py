"""Six-class Qwen3 dual-head model construction for Exp61."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.model import head_contract, make_cbrd_dual_head_classifier


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str
    local_files_only: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True


def load_model_and_tokenizer(config: ModelConfig) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = str(Path(config.model_name_or_path))
    dtype = torch.bfloat16 if config.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=False, local_files_only=config.local_files_only
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=6,
        problem_type="single_label_classification",
        trust_remote_code=False,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id
    if int(base.num_labels) != 6:
        raise RuntimeError("Exp61 real model did not initialize six labels")
    model = make_cbrd_dual_head_classifier(base)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    contract = head_contract(model)
    required = {
        "six_classes": model.num_labels == 6,
        "copied_head_hash_equal": contract["hard_head_hash"] == contract["soft_head_hash"],
        "copied_head_storage_independent": contract["storage_independent"],
        "hard_head_has_no_bias": not contract["hard_bias"],
        "soft_head_has_no_bias": not contract["soft_bias"],
    }
    if not all(required.values()):
        raise RuntimeError(f"Exp61 head contract failed: {required}; {contract}")
    return model, tokenizer, {**contract, "checks": required}
