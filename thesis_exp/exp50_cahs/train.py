"""Train the single locked Exp50 CAHS-0.5 arm without reading test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash
from thesis_exp.exp49_cphce.losses import soft_cross_entropy
from thesis_exp.exp49_cphce.metric_contract import compute_metrics
from thesis_exp.exp50_cahs import ALPHA, VARIANT, checkpoint_dir, run_output_dir
from thesis_exp.exp50_cahs.build_targets import load_split
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    StepProgressBar,
    TrainConfig,
    _get_loss_logits,
    evaluate,
    format_duration,
    load_model_and_tokenizer,
    make_dataloader,
    save_checkpoint,
    save_predictions,
    set_seed,
    write_json,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Locked Exp50 CAHS-0.5 trainer (dev only).")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    args = parser.parse_args()
    output_dir = args.output_dir or run_output_dir(args.seed)
    best_dir = args.checkpoint_output_dir or checkpoint_dir(args.seed)
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP50_IN_MEMORY_TRAIN_DEV_ONLY"),
        output_dir=output_dir,
        checkpoint_output_dir=best_dir.parent,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
        evaluate_test=False,
    )


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def limit(rows: list[dict[str, Any]], value: int | None) -> list[dict[str, Any]]:
    return rows if value is None else rows[:value]


def classifier_hash(model: Any) -> str:
    digest = hashlib.sha256()
    score = getattr(model, "score", None)
    if score is None:
        raise AttributeError("Expected sequence classifier to expose model.score")
    for name, tensor in sorted(score.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def token_audit(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    for start in range(0, len(rows), 64):
        encoded = tokenizer([str(row["text"]) for row in rows[start : start + 64]], add_special_tokens=True, truncation=False, padding=False)
        lengths.extend(len(value) for value in encoded["input_ids"])
    values = np.asarray(lengths, dtype=float)
    return {
        "n": len(lengths),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "truncation_count": int(np.sum(values > max_length)),
        "max_length": max_length,
    }


def contract_metrics(result: Any, epoch: int, global_step: int) -> dict[str, Any]:
    metrics = compute_metrics(result.predictions)
    metrics["split"] = "dev"
    metrics["eval_hard_ce_loss"] = result.metrics.get("loss")
    metrics["epoch"] = epoch
    metrics["global_step"] = global_step
    result.metrics = metrics
    return metrics


def checkpoint_wins(candidate_exact: float, best_exact: float) -> bool:
    """Actual Exp02/Exp49 rule: strictly higher Exact; ties keep earlier epoch."""
    return float(candidate_exact) > float(best_exact)


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if config.evaluate_test:
        raise PermissionError("Exp50 training must be dev-only")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit(load_split("train"), config.max_train_samples)
    dev_rows = limit(load_split("dev"), config.max_eval_samples)
    train_hash = aggregate_text_hash(train_rows)
    dev_hash = aggregate_text_hash(dev_rows)

    model, tokenizer, model_mode = load_model_and_tokenizer(config)
    initial_head_hash = classifier_hash(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    write_json(
        config.output_dir / "input_contract.json",
        {
            "variant": VARIANT,
            "alpha": ALPHA,
            "fields": ["question", "answer", "metric_canonical"],
            "train_text_hash": train_hash,
            "dev_text_hash": dev_hash,
            "token_lengths": {
                "train": token_audit(tokenizer, train_rows, config.max_length),
                "dev": token_audit(tokenizer, dev_rows, config.max_length),
            },
        },
    )

    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * updates_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_dir = config.checkpoint_output_dir / "best"
    best_exact = -1.0
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    global_step = 0
    micro_step = 0
    start = time.time()
    progress = StepProgressBar(total_steps, config.progress_bar, desc="Exp50 CAHS-0.5")
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_zero in range(int(math.ceil(config.num_train_epochs))):
            if epoch_zero >= config.num_train_epochs:
                break
            epoch = epoch_zero + 1
            model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                metadata = batch.pop("metadata")
                batch.pop("labels")
                inputs = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**inputs)
                _, logits = _get_loss_logits(outputs)
                targets = torch.tensor([row["cahs_target_5"] for row in metadata], dtype=torch.float32, device=device)
                loss = soft_cross_entropy(logits, targets)
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().cpu())
                micro_step += 1
                if len(trace) < 64:
                    trace.append({"micro_step": micro_step, "record_ids": [row["record_id"] for row in metadata], "loss": float(loss.detach().cpu())})
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % config.log_steps == 0 or global_step == total_steps:
                        elapsed = time.time() - start
                        progress.render(global_step, epoch, running_loss / max(1, step), scheduler.get_last_lr()[0], elapsed, max(0, total_steps - global_step) * elapsed / max(1, global_step))
            progress.newline()
            result = evaluate(model, dev_loader, device, "dev")
            metrics = contract_metrics(result, epoch, global_step)
            metrics["train_loss"] = running_loss / max(1, len(train_loader))
            history.append(metrics)
            write_csv(config.output_dir / "tables" / "dev_metrics_history.csv", history)
            write_json(config.output_dir / "dev_metrics_history.json", history)
            print(f"[exp50] epoch={epoch} Exact={metrics['Exact_rounded']:.6f} MAE={metrics['MAE_human_mean']:.6f} Kendall={metrics['Kendall_human_mean']:.6f}", flush=True)
            if checkpoint_wins(float(metrics["Exact_rounded"]), best_exact):
                best_exact = float(metrics["Exact_rounded"])
                best_epoch = epoch
                save_checkpoint(model, tokenizer, best_dir, config, model_mode, metrics)
                write_json(best_dir / "exp50_metadata.json", {"variant": VARIANT, "alpha": ALPHA, "seed": config.seed, "epoch": epoch, "initial_classifier_hash": initial_head_hash})
                save_predictions(config.output_dir, result, suffix="dev_best")
    finally:
        progress.close()

    if best_epoch is None:
        raise RuntimeError("No Exp50 checkpoint selected")
    state_path = best_dir / "state_dict.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    final = evaluate(model, dev_loader, device, "dev")
    selected = contract_metrics(final, best_epoch, global_step)
    save_predictions(config.output_dir, final)
    write_json(config.output_dir / "selected_dev_metrics.json", selected)
    write_csv(config.output_dir / "tables" / "selected_dev_metrics.csv", [{"variant": VARIANT, "seed": config.seed, **selected}])
    write_json(config.output_dir / "training_trace_first64.json", {"initial_classifier_hash": initial_head_hash, "trace": trace})
    elapsed = time.time() - start
    summary = {
        "status": "COMPLETED",
        "variant": VARIANT,
        "alpha": ALPHA,
        "seed": config.seed,
        "runtime_head": git_head(),
        "model_mode": model_mode,
        "model_name_or_path": config.model_name_or_path,
        "initial_classifier_hash": initial_head_hash,
        "selected_epoch": best_epoch,
        "selected_metrics": selected,
        "checkpoint_path": str(best_dir),
        "scheduler": "cosine_with_warmup",
        "checkpoint_rule": "highest Exact_rounded; ties keep earlier epoch",
        "train_text_hash": train_hash,
        "dev_text_hash": dev_hash,
        "test_access_count": 0,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "nan_or_inf": not all(math.isfinite(float(selected[key])) for key in ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean", "Bias_human_mean")),
    }
    write_json(config.output_dir / "run_summary.json", summary)
    write_text(config.output_dir / "run_summary.md", f"# Exp50 CAHS-0.5 seed {config.seed}\n\n- Selected epoch: {best_epoch}\n- Exact: {selected['Exact_rounded']:.6f}\n- MAE: {selected['MAE_human_mean']:.6f}\n- Kendall: {selected['Kendall_human_mean']:.6f}\n- Runtime: {format_duration(elapsed)}\n- Test accessed: no\n")
    return summary


def main() -> None:
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
