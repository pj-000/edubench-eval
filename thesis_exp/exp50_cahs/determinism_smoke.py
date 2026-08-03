"""Short actual-path B0 repeatability probe for Exp50 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp49_cphce.build_targets import load_split
from thesis_exp.exp49_cphce.losses import hard_cross_entropy
from thesis_exp.exp50_cahs.train import classifier_hash
from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig, _get_loss_logits, load_model_and_tokenizer, make_dataloader, set_seed


def rng_hash(torch: Any) -> str:
    digest = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def run(model_path: str, output: Path, seed: int, microbatches: int) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    config = TrainConfig(
        model_name_or_path=model_path,
        data_dir=Path("EXP50_DETERMINISM_TRAIN_ONLY"),
        output_dir=output.parent,
        checkpoint_output_dir=output.parent,
        max_length=2048,
        num_train_epochs=1.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=seed,
        bf16="auto",
        fp16=False,
        gradient_checkpointing=True,
        trust_remote_code=False,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=1000,
        progress_bar=False,
        evaluate_test=False,
    )
    set_seed(seed)
    rows = load_split("train")
    model, tokenizer, _ = load_model_and_tokenizer(config)
    head_hash = classifier_hash(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = make_dataloader(rows, tokenizer, config, "train", shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    updates = max(1, (microbatches + 31) // 32)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * 0.05), updates)
    before_rng = rng_hash(torch)
    trace: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader, start=1):
        if index > microbatches:
            break
        metadata = batch.pop("metadata")
        labels = batch.pop("labels").to(device)
        outputs = model(**{key: value.to(device) for key, value in batch.items()})
        _, logits = _get_loss_logits(outputs)
        loss = hard_cross_entropy(logits, labels)
        (loss / 32).backward()
        trace.append({"microbatch": index, "record_ids": [row["record_id"] for row in metadata], "loss": float(loss.detach().cpu())})
        if index % 32 == 0 or index == microbatches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    value = {"seed": seed, "microbatches": microbatches, "initial_classifier_hash": head_hash, "rng_before": before_rng, "rng_after": rng_hash(torch), "scheduler_last_lr": scheduler.get_last_lr()[0], "trace": trace}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def compare(left: Path, right: Path, output: Path) -> dict[str, Any]:
    a, b = json.loads(left.read_text()), json.loads(right.read_text())
    losses_a = [row["loss"] for row in a["trace"]]
    losses_b = [row["loss"] for row in b["trace"]]
    first_update_end = min(32, len(losses_a), len(losses_b))
    result: dict[str, Any] = {
        "initial_classifier_hash_equal": a["initial_classifier_hash"] == b["initial_classifier_hash"],
        "batch_id_sequence_equal": [row["record_ids"] for row in a["trace"]] == [row["record_ids"] for row in b["trace"]],
        "rng_before_equal": a["rng_before"] == b["rng_before"],
        "rng_after_equal": a["rng_after"] == b["rng_after"],
        "max_abs_loss_delta_before_first_optimizer_update": max(abs(x - y) for x, y in zip(losses_a[:first_update_end], losses_b[:first_update_end])),
        "max_abs_loss_delta": max(abs(x - y) for x, y in zip(losses_a, losses_b)),
        "scheduler_last_lr_equal": a["scheduler_last_lr"] == b["scheduler_last_lr"],
    }
    result["contract_components_equal"] = all(value for key, value in result.items() if key.endswith("_equal"))
    result["bitwise_loss_reproducible"] = result["max_abs_loss_delta"] <= 1e-6
    result["status"] = "BITWISE_REPRODUCIBLE" if result["bitwise_loss_reproducible"] else "CONTRACT_MATCH_NON_BITWISE_CUDA"
    result["preflight_passed"] = result["contract_components_equal"] and result["max_abs_loss_delta_before_first_optimizer_update"] <= 1e-6
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microbatches", type=int, default=64)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    value = compare(args.compare[0], args.compare[1], args.output) if args.compare else run(args.model_name_or_path, args.output, args.seed, args.microbatches)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if args.compare and not value["preflight_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
