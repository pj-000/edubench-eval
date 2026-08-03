"""Real-model Exp51 hard-path, initialization, routing, and isolation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp49_cphce.losses import hard_cross_entropy, soft_cross_entropy
from thesis_exp.exp51_hmsa import OUTPUT_ROOT, SEED42_INITIAL_HEAD_HASH
from thesis_exp.exp51_hmsa.build_targets import load_split
from thesis_exp.exp51_hmsa.model import head_contract, make_dual_head_classifier, module_hash
from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig, load_model_and_tokenizer, make_dataloader, set_seed, write_json


def config(model_path: str, *, gradient_checkpointing: bool = False) -> TrainConfig:
    return TrainConfig(
        model_name_or_path=model_path,
        data_dir=Path("EXP51_PREFLIGHT_NO_TEST"),
        output_dir=OUTPUT_ROOT / "audit" / "preflight_runtime",
        checkpoint_output_dir=Path("UNUSED"),
        max_length=2048,
        num_train_epochs=10.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=42,
        bf16="auto",
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        trust_remote_code=False,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=5,
        progress_bar=False,
        evaluate_test=False,
    )


def grad_digest(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    squared_norm = 0.0
    parameter_count = gradient_count = 0
    for _, parameter in named_parameters:
        parameter_count += 1
        if parameter.grad is None:
            digest.update(b"NONE")
            continue
        gradient_count += 1
        value = parameter.grad.detach().float().cpu()
        digest.update(value.numpy().tobytes())
        squared_norm += float(value.square().sum())
    return {"sha256": digest.hexdigest(), "norm": math.sqrt(squared_norm), "parameter_count": parameter_count, "gradient_count": gradient_count}


def selected_backbone_parameter(model: Any) -> tuple[str, Any]:
    candidates = [(name, parameter) for name, parameter in model.backbone.named_parameters() if parameter.requires_grad and parameter.numel() <= 65536]
    if not candidates:
        raise RuntimeError("No compact backbone parameter available for gradient sum audit")
    return candidates[-1]


def ids_from_loader(loader: Any, batches: int = 4) -> list[list[str]]:
    output: list[list[str]] = []
    for index, batch in enumerate(loader):
        output.append([str(row["record_id"]) for row in batch["metadata"]])
        if index + 1 == batches:
            break
    return output


def run(model_path: str) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    set_seed(42)
    cfg = config(model_path)
    base, tokenizer, mode = load_model_and_tokenizer(cfg)
    if mode != "sequence_classification":
        raise RuntimeError(mode)
    base_head_hash = module_hash(base.score)
    if base_head_hash != SEED42_INITIAL_HEAD_HASH:
        raise AssertionError(f"B0 seed42 head hash mismatch: {base_head_hash}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.to(device)
    rows = load_split("dev")[:2]
    encoded = tokenizer([row["text"] for row in rows], truncation=True, max_length=256, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.tensor([int(row["label_5"]) - 1 for row in rows], dtype=torch.long, device=device)
    targets = torch.tensor([row["soft_target_5"] for row in rows], dtype=torch.float32, device=device)

    base.eval()
    with torch.no_grad():
        base_output = base(**encoded)
        base_logits = base_output.logits.detach().clone()
        base_loss = hard_cross_entropy(base_logits, labels).detach().clone()

    model = make_dual_head_classifier(base)
    contract = head_contract(model)
    model.eval()
    with torch.no_grad():
        dual_output = model(**encoded, labels=labels)
        max_logit_diff = float((base_logits - dual_output["hard_logits"]).abs().max().cpu())
        hard_loss_diff = float((base_loss - dual_output["loss"]).abs().cpu())
        initial_head_logit_diff = float((dual_output["hard_logits"] - dual_output["soft_logits"]).abs().max().cpu())
        saved_soft = model.soft_head.weight.detach().clone()
        model.soft_head.weight.add_(1.0)
        changed_soft = model(**encoded)["hard_logits"]
        inference_isolation_diff = float((changed_soft - dual_output["hard_logits"]).abs().max().cpu())
        model.soft_head.weight.copy_(saved_soft)

    if max_logit_diff > 1e-4 or hard_loss_diff > 1e-6 or initial_head_logit_diff != 0 or inference_isolation_diff != 0:
        raise AssertionError("Hard-path or inference-isolation parity failed")
    if contract["hard_bias"] or contract["soft_bias"] or not contract["storage_independent"]:
        raise AssertionError(contract)

    model.train()
    base.train()
    selected_name, selected_parameter = selected_backbone_parameter(model)
    base.zero_grad(set_to_none=True)
    base_loss_train = hard_cross_entropy(base(**encoded).logits, labels)
    base_loss_train.backward()
    base_backbone_grad = grad_digest(base.model.named_parameters())
    base_head_grad = grad_digest(base.score.named_parameters())
    base_selected = selected_parameter.grad.detach().float().cpu().clone()

    model.zero_grad(set_to_none=True)
    hard_output = model(**encoded)
    hard_loss = F.cross_entropy(hard_output["hard_logits"].float(), labels)
    hard_loss.backward()
    dual_backbone_grad = grad_digest(model.backbone.named_parameters())
    dual_hard_grad = grad_digest(model.hard_head.named_parameters())
    hard_soft_grad_none = all(parameter.grad is None for parameter in model.soft_head.parameters())
    hard_selected = selected_parameter.grad.detach().float().cpu().clone()
    backbone_norm_relative_diff = abs(base_backbone_grad["norm"] - dual_backbone_grad["norm"]) / max(base_backbone_grad["norm"], 1e-12)
    selected_hard_gradient_max_diff = float((base_selected - hard_selected).abs().max())
    if backbone_norm_relative_diff > 5e-4 or selected_hard_gradient_max_diff > 1e-4 or base_head_grad["sha256"] != dual_hard_grad["sha256"]:
        raise AssertionError(
            "lambda=0 hard gradient numerical parity failed: "
            + json.dumps(
                {
                    "base_backbone": base_backbone_grad,
                    "dual_backbone": dual_backbone_grad,
                    "base_head": base_head_grad,
                    "dual_head": dual_hard_grad,
                    "selected_parameter": selected_name,
                    "backbone_norm_relative_diff": backbone_norm_relative_diff,
                    "selected_hard_gradient_max_diff": selected_hard_gradient_max_diff,
                },
                sort_keys=True,
            )
        )
    if not hard_soft_grad_none:
        raise AssertionError("soft head received hard-only gradient")

    model.zero_grad(set_to_none=True)
    soft_output = model(**encoded)
    soft_loss = soft_cross_entropy(soft_output["soft_logits"], targets)
    soft_loss.backward()
    soft_backbone_grad = grad_digest(model.backbone.named_parameters())
    soft_hard_grad_none = all(parameter.grad is None for parameter in model.hard_head.parameters())
    soft_head_grad = grad_digest(model.soft_head.named_parameters())
    soft_selected = selected_parameter.grad.detach().float().cpu().clone()
    if not soft_hard_grad_none or soft_head_grad["gradient_count"] == 0 or soft_backbone_grad["gradient_count"] == 0:
        raise AssertionError("soft-only gradient routing failed")

    model.zero_grad(set_to_none=True)
    total_output = model(**encoded)
    total_hard = F.cross_entropy(total_output["hard_logits"].float(), labels)
    total_soft = soft_cross_entropy(total_output["soft_logits"], targets)
    (total_hard + total_soft).backward()
    total_selected = selected_parameter.grad.detach().float().cpu().clone()
    gradient_sum_max_diff = float((total_selected - hard_selected - soft_selected).abs().max())
    gradient_sum_relative_diff = gradient_sum_max_diff / max(float(total_selected.abs().max()), 1e-12)
    hard_flat = hard_selected.flatten()
    soft_flat = soft_selected.flatten()
    gradient_cosine = float(F.cosine_similarity(hard_flat, soft_flat, dim=0))
    both_heads_have_grad = all(parameter.grad is not None for parameter in model.hard_head.parameters()) and all(parameter.grad is not None for parameter in model.soft_head.parameters())
    if gradient_sum_relative_diff > 0.003 or not both_heads_have_grad:
        raise AssertionError(
            "total gradient routing failed: "
            + json.dumps(
                {
                    "gradient_sum_max_diff": gradient_sum_max_diff,
                    "gradient_sum_relative_diff": gradient_sum_relative_diff,
                    "hard_selected_max_abs": float(hard_selected.abs().max()),
                    "soft_selected_max_abs": float(soft_selected.abs().max()),
                    "total_selected_max_abs": float(total_selected.abs().max()),
                    "both_heads_have_grad": both_heads_have_grad,
                },
                sort_keys=True,
            )
        )
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise AssertionError("duplicate parameter registration")

    train_rows = load_split("train")[:32]
    loader_a = make_dataloader(train_rows, tokenizer, cfg, "train", shuffle=True)
    loader_b = make_dataloader(train_rows, tokenizer, cfg, "train", shuffle=True)
    batches_a = ids_from_loader(loader_a)
    batches_b = ids_from_loader(loader_b)
    if batches_a != batches_b:
        raise AssertionError("seed42 batch order mismatch")

    result = {
        "status": "PASS",
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "base_head_hash": base_head_hash,
        "head_contract": contract,
        "max_hard_logit_diff": max_logit_diff,
        "hard_loss_diff": hard_loss_diff,
        "initial_hard_soft_logit_diff": initial_head_logit_diff,
        "inference_isolation_diff": inference_isolation_diff,
        "base_backbone_gradient": base_backbone_grad,
        "dual_backbone_hard_gradient": dual_backbone_grad,
        "base_head_gradient": base_head_grad,
        "dual_hard_head_gradient": dual_hard_grad,
        "backbone_gradient_hash_bitwise_equal": base_backbone_grad["sha256"] == dual_backbone_grad["sha256"],
        "backbone_gradient_norm_relative_diff": backbone_norm_relative_diff,
        "selected_hard_gradient_max_diff": selected_hard_gradient_max_diff,
        "soft_backbone_gradient": soft_backbone_grad,
        "soft_head_gradient": soft_head_grad,
        "selected_backbone_parameter": selected_name,
        "hard_soft_gradient_cosine": gradient_cosine,
        "gradient_sum_max_diff": gradient_sum_max_diff,
        "gradient_sum_relative_diff": gradient_sum_relative_diff,
        "batch_ids": batches_a,
        "scheduler": "cosine_with_warmup",
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_name_or_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
