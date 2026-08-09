"""Five-class Qwen3 dual-head construction for Exp62."""

from __future__ import annotations

import hashlib
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


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model_and_tokenizer(config: ModelConfig) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = str(Path(config.model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=False, local_files_only=config.local_files_only
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=5,
        problem_type="single_label_classification",
        trust_remote_code=False,
        local_files_only=config.local_files_only,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id
    if int(base.num_labels) != 5:
        raise RuntimeError("Exp62 model did not initialize five labels")
    model = make_cbrd_dual_head_classifier(base)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    contract = head_contract(model)
    checks = {
        "five_classes": model.num_labels == 5,
        "copied_heads_equal": contract["hard_head_hash"] == contract["soft_head_hash"],
        "copied_head_storage_independent": contract["storage_independent"],
        "hard_head_has_no_bias": not contract["hard_bias"],
        "soft_head_has_no_bias": not contract["soft_bias"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp62 head contract failed: {checks}; {contract}")
    return model, tokenizer, {**contract, "checks": checks}

