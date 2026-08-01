"""Train the locked MeanAux train/dev-only control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash
from thesis_exp.exp49_cphce.losses import hard_cross_entropy
from thesis_exp.exp49_cphce.metric_contract import compute_metrics
from thesis_exp.exp50_cahs.train import (
    checkpoint_global_step,
    checkpoint_wins,
    token_audit,
)
from thesis_exp.exp56_meanaux import (
    AUX_WEIGHT,
    SEED42_INITIAL_HEAD_HASH,
    SMOOTH_L1_BETA,
    VARIANT,
    checkpoint_dir,
    run_output_dir,
)
from thesis_exp.exp56_meanaux.build_targets import load_split
from thesis_exp.exp56_meanaux.losses import expected_score, mean_aux_smooth_l1
from thesis_exp.exp56_meanaux.model import head_contract, make_hard_mean_classifier
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    EvalResult,
    StepProgressBar,
    TrainConfig,
    format_duration,
    load_model_and_tokenizer,
    make_dataloader,
    save_predictions,
    set_seed,
    write_json,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


SOURCE_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "exp56_meanaux" / "source_lock.json"
)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Locked Exp56 MeanAux trainer")
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
    checkpoint_root = (
        args.checkpoint_output_dir
        if args.checkpoint_output_dir is not None
        else checkpoint_dir(args.seed).parent
    )
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP56_IN_MEMORY_TRAIN_DEV_ONLY"),
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint_root,
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def limit(
    rows: list[dict[str, Any]],
    value: int | None,
) -> list[dict[str, Any]]:
    return rows if value is None else rows[:value]


def verify_source_lock(required: bool) -> dict[str, Any] | None:
    if not SOURCE_LOCK_PATH.exists():
        if required:
            raise FileNotFoundError(f"Formal MeanAux requires {SOURCE_LOCK_PATH}")
        return None
    lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    for relative, expected in lock["files"].items():
        actual = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Exp56 source-lock mismatch: {relative}: {actual} != {expected}"
            )
    lock["manifest_sha256"] = hashlib.sha256(
        SOURCE_LOCK_PATH.read_bytes()
    ).hexdigest()
    return lock


def load_meanaux_model(
    config: TrainConfig,
) -> tuple[Any, Any, str, dict[str, Any]]:
    base_model, tokenizer, model_mode = load_model_and_tokenizer(config)
    if model_mode != "sequence_classification":
        raise RuntimeError(
            f"MeanAux hard-path parity requires sequence_classification; got {model_mode}"
        )
    model = make_hard_mean_classifier(base_model)
    contract = head_contract(model)
    if (
        contract["hard_head_hash"] != contract["mean_head_hash"]
        or not contract["storage_independent"]
    ):
        raise AssertionError(f"Invalid copied-head initialization: {contract}")
    if config.seed == 42 and contract["hard_head_hash"] != SEED42_INITIAL_HEAD_HASH:
        raise AssertionError(
            f"Seed42 hard-head hash differs from B0: {contract['hard_head_hash']}"
        )
    return (
        model,
        tokenizer,
        "qwen3_hard_head_mean_aux_sequence_classification",
        contract,
    )


def evaluate_meanaux(
    model: Any,
    dataloader: Any,
    device: Any,
    split: str,
) -> EvalResult:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    hard_chunks: list[np.ndarray] = []
    hard_prob_chunks: list[np.ndarray] = []
    labels_out: list[int] = []
    record_ids: list[str] = []
    mean_predictions: list[float] = []
    mean_targets: list[float] = []
    hard_loss_sum = mean_loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            hard_logits = outputs["hard_logits"]
            mean_logits = outputs["mean_logits"]
            targets = torch.tensor(
                [row["mean_aux_target"] for row in metadata],
                dtype=torch.float32,
                device=device,
            )
            hard_loss = hard_cross_entropy(hard_logits, labels)
            mean_loss = mean_aux_smooth_l1(
                mean_logits,
                targets,
                beta=SMOOTH_L1_BETA,
            )
            mean_scores = expected_score(mean_logits)
            batch_n = len(metadata)
            hard_loss_sum += float(hard_loss.detach().cpu()) * batch_n
            mean_loss_sum += float(mean_loss.detach().cpu()) * batch_n
            n += batch_n
            hard_cpu = hard_logits.float().cpu()
            hard_probs = torch.softmax(hard_cpu, dim=-1)
            mean_cpu = mean_scores.float().cpu()
            hard_chunks.append(hard_cpu.numpy())
            hard_prob_chunks.append(hard_probs.numpy())
            mean_predictions.extend(mean_cpu.tolist())
            mean_targets.extend(targets.float().cpu().tolist())
            for row, hard_logit, hard_prob, mean_score in zip(
                metadata,
                hard_cpu.tolist(),
                hard_probs.tolist(),
                mean_cpu.tolist(),
            ):
                pred_label = int(np.argmax(hard_logit)) + 1
                expected = float(
                    sum((idx + 1) * value for idx, value in enumerate(hard_prob))
                )
                human_mean = float(row["human_mean_5"])
                prediction = {
                    "split": split,
                    "id": row.get("id"),
                    "record_id": row.get("record_id"),
                    "triple_key": row.get("triple_key"),
                    "label_5": int(row["label_5"]),
                    "human_mean_5": human_mean,
                    "pred_label_5": pred_label,
                    "pred_score_5": float(pred_label),
                    "pred_score_expected": expected,
                    "mean_aux_pred": float(mean_score),
                    "confidence": float(max(hard_prob)),
                    "abs_error_label": abs(pred_label - human_mean),
                    "signed_error_label": pred_label - human_mean,
                    "abs_error_expected": abs(expected - human_mean),
                    "signed_error_expected": expected - human_mean,
                    "metric_canonical": row.get("metric_canonical"),
                    "scenario_canonical": row.get("scenario_canonical"),
                    "subject_canonical": row.get("subject_canonical"),
                    "language": row.get("language"),
                    "generator_model": row.get("generator_model"),
                }
                for idx in range(5):
                    prediction[f"logit_{idx + 1}"] = float(hard_logit[idx])
                    prediction[f"prob_{idx + 1}"] = float(hard_prob[idx])
                predictions.append(prediction)
                labels_out.append(int(row["label_5"]))
                record_ids.append(str(row.get("record_id") or row.get("id")))
    hard_logits_arr = np.concatenate(hard_chunks, axis=0)
    hard_probs_arr = np.concatenate(hard_prob_chunks, axis=0)
    predicted = np.asarray(mean_predictions, dtype=float)
    target = np.asarray(mean_targets, dtype=float)
    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "split": split,
            "eval_hard_ce_loss": hard_loss_sum / n,
            "mean_aux_smooth_l1": mean_loss_sum / n,
            "mean_aux_mae": float(np.mean(np.abs(predicted - target))),
            "mean_aux_rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
            "mean_aux_bias": float(np.mean(predicted - target)),
        }
    )
    return EvalResult(
        metrics,
        predictions,
        hard_logits_arr,
        hard_probs_arr,
        np.asarray(labels_out),
        record_ids,
    )


def contract_metrics(
    result: EvalResult,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    metrics = compute_metrics(result.predictions)
    for key in (
        "eval_hard_ce_loss",
        "mean_aux_smooth_l1",
        "mean_aux_mae",
        "mean_aux_rmse",
        "mean_aux_bias",
    ):
        metrics[key] = result.metrics[key]
    metrics.update({"split": "dev", "epoch": epoch, "global_step": global_step})
    result.metrics = metrics
    return metrics


def save_meanaux_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    config: TrainConfig,
    metrics: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "state_dict.pt")
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(
        output_dir / "training_config.json",
        {
            **asdict(config),
            "model_mode": "qwen3_hard_head_mean_aux_sequence_classification",
            "aux_weight": AUX_WEIGHT,
            "smooth_l1_beta": SMOOTH_L1_BETA,
        },
    )
    write_json(output_dir / "dev_metrics.json", metrics)
    write_json(output_dir / "head_contract.json", contract)


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if config.evaluate_test:
        raise PermissionError("Exp56 MeanAux must remain train/dev-only")
    source_lock = verify_source_lock(
        required=os.environ.get("EXP56_REQUIRE_SOURCE_LOCK") == "1"
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit(load_split("train"), config.max_train_samples)
    dev_rows = limit(load_split("dev"), config.max_eval_samples)
    train_hash = aggregate_text_hash(train_rows)
    dev_hash = aggregate_text_hash(dev_rows)
    model, tokenizer, model_mode, initial_contract = load_meanaux_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    write_json(
        config.output_dir / "input_contract.json",
        {
            "variant": VARIANT,
            "aux_weight": AUX_WEIGHT,
            "smooth_l1_beta": SMOOTH_L1_BETA,
            "mean_projection": "expected score over copied five-logit auxiliary head",
            "fields": ["question", "answer", "metric_canonical"],
            "train_text_hash": train_hash,
            "dev_text_hash": dev_hash,
            "token_lengths": {
                "train": token_audit(tokenizer, train_rows, config.max_length),
                "dev": token_audit(tokenizer, dev_rows, config.max_length),
            },
        },
    )
    train_loader = make_dataloader(
        train_rows,
        tokenizer,
        config,
        "train",
        shuffle=True,
    )
    dev_loader = make_dataloader(
        dev_rows,
        tokenizer,
        config,
        "dev",
        shuffle=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_steps = max(
        1,
        int(math.ceil(config.num_train_epochs * updates_per_epoch)),
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )
    best_dir = config.checkpoint_output_dir / "best"
    best_exact = -1.0
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    global_step = micro_step = clip_events = 0
    max_pre_clip_grad_norm = 0.0
    start = time.time()
    progress = StepProgressBar(
        total_steps,
        config.progress_bar,
        desc="Exp56 MeanAux",
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_zero in range(int(math.ceil(config.num_train_epochs))):
            if epoch_zero >= config.num_train_epochs:
                break
            epoch = epoch_zero + 1
            model.train()
            hard_running = mean_running = total_running = 0.0
            for step, batch in enumerate(train_loader, start=1):
                metadata = batch.pop("metadata")
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**inputs)
                targets = torch.tensor(
                    [row["mean_aux_target"] for row in metadata],
                    dtype=torch.float32,
                    device=device,
                )
                hard_loss = hard_cross_entropy(outputs["hard_logits"], labels)
                mean_loss = mean_aux_smooth_l1(
                    outputs["mean_logits"],
                    targets,
                    beta=SMOOTH_L1_BETA,
                )
                loss = hard_loss + AUX_WEIGHT * mean_loss
                (loss / config.gradient_accumulation_steps).backward()
                hard_running += float(hard_loss.detach().cpu())
                mean_running += float(mean_loss.detach().cpu())
                total_running += float(loss.detach().cpu())
                micro_step += 1
                if len(trace) < 64:
                    trace.append(
                        {
                            "micro_step": micro_step,
                            "record_ids": [
                                str(row["record_id"]) for row in metadata
                            ],
                            "hard_loss": float(hard_loss.detach().cpu()),
                            "mean_loss": float(mean_loss.detach().cpu()),
                            "total_loss": float(loss.detach().cpu()),
                        }
                    )
                if (
                    step % config.gradient_accumulation_steps == 0
                    or step == len(train_loader)
                ):
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.max_grad_norm,
                        )
                        .detach()
                        .cpu()
                    )
                    max_pre_clip_grad_norm = max(
                        max_pre_clip_grad_norm,
                        grad_norm,
                    )
                    clip_events += int(grad_norm > config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if (
                        global_step % config.log_steps == 0
                        or global_step == total_steps
                    ):
                        elapsed = time.time() - start
                        progress.render(
                            global_step,
                            epoch,
                            total_running / max(1, step),
                            scheduler.get_last_lr()[0],
                            elapsed,
                            max(0, total_steps - global_step)
                            * elapsed
                            / max(1, global_step),
                        )
            progress.newline()
            result = evaluate_meanaux(model, dev_loader, device, "dev")
            metrics = contract_metrics(result, epoch, global_step)
            metrics.update(
                {
                    "train_hard_loss": hard_running / len(train_loader),
                    "train_mean_loss": mean_running / len(train_loader),
                    "train_total_loss": total_running / len(train_loader),
                }
            )
            history.append(metrics)
            write_csv(
                config.output_dir / "tables" / "dev_metrics_history.csv",
                history,
            )
            write_json(config.output_dir / "dev_metrics_history.json", history)
            print(
                f"[exp56] epoch={epoch} "
                f"Exact={metrics['Exact_rounded']:.6f} "
                f"MAE={metrics['MAE_human_mean']:.6f} "
                f"Kendall={metrics['Kendall_human_mean']:.6f}",
                flush=True,
            )
            if checkpoint_wins(float(metrics["Exact_rounded"]), best_exact):
                best_exact = float(metrics["Exact_rounded"])
                best_epoch = epoch
                save_meanaux_checkpoint(
                    model,
                    tokenizer,
                    best_dir,
                    config,
                    metrics,
                    initial_contract,
                )
                save_predictions(config.output_dir, result, suffix="dev_best")
    finally:
        progress.close()
    if best_epoch is None:
        raise RuntimeError("No MeanAux checkpoint selected")
    model.load_state_dict(
        torch.load(best_dir / "state_dict.pt", map_location=device, weights_only=True)
    )
    final = evaluate_meanaux(model, dev_loader, device, "dev")
    selected = contract_metrics(
        final,
        best_epoch,
        checkpoint_global_step(history, best_epoch),
    )
    save_predictions(config.output_dir, final)
    write_json(config.output_dir / "selected_dev_metrics.json", selected)
    write_csv(
        config.output_dir / "tables" / "selected_dev_metrics.csv",
        [{"variant": VARIANT, "seed": config.seed, **selected}],
    )
    write_json(
        config.output_dir / "training_trace_first64.json",
        {"head_contract": initial_contract, "trace": trace},
    )
    elapsed = time.time() - start
    summary = {
        "status": "COMPLETED",
        "variant": VARIANT,
        "aux_weight": AUX_WEIGHT,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "mean_projection": "expected score over copied five-logit auxiliary head",
        "seed": config.seed,
        "runtime_head": git_head(),
        "locked_source_commit": (
            source_lock.get("source_commit") if source_lock else None
        ),
        "source_lock_manifest_sha256": (
            source_lock.get("manifest_sha256") if source_lock else None
        ),
        "model_mode": model_mode,
        "model_name_or_path": config.model_name_or_path,
        "initial_head_contract": initial_contract,
        "selected_epoch": best_epoch,
        "selected_metrics": selected,
        "checkpoint_path": str(best_dir),
        "scheduler": "cosine_with_warmup",
        "checkpoint_rule": "highest hard Exact_rounded; ties keep earlier epoch",
        "inference": "hard_head_raw_logit_argmax",
        "train_text_hash": train_hash,
        "dev_text_hash": dev_hash,
        "test_access_count": 0,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "clip_events": clip_events,
        "max_pre_clip_grad_norm": max_pre_clip_grad_norm,
        "nan_or_inf": not all(
            math.isfinite(float(selected[key]))
            for key in (
                "MAE_human_mean",
                "Exact_rounded",
                "Kendall_human_mean",
                "Bias_human_mean",
                "mean_aux_smooth_l1",
            )
        ),
    }
    write_json(config.output_dir / "run_summary.json", summary)
    write_text(
        config.output_dir / "run_summary.md",
        f"# Exp56 MeanAux seed {config.seed}\n\n"
        f"- Selected epoch: {best_epoch}\n"
        f"- Exact: {selected['Exact_rounded']:.6f}\n"
        f"- MAE: {selected['MAE_human_mean']:.6f}\n"
        f"- Kendall: {selected['Kendall_human_mean']:.6f}\n"
        f"- Mean-aux MAE: {selected['mean_aux_mae']:.6f}\n"
        f"- Runtime: {format_duration(elapsed)}\n"
        "- Test accessed: no\n",
    )
    return summary


def main() -> None:
    print(
        json.dumps(
            train(parse_args()),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
