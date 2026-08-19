"""Frozen train/dev-only trainer for the Exp57 CBRD intervention matrix."""

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

from thesis_exp.exp57_cbrd import CONFIG_ROOT, OUTPUT_ROOT, REPO_ROOT, SHUFFLE_SEED
from thesis_exp.exp57_cbrd.data_audit import (
    load_rows,
    model_rows,
    shuffled_residual_audit,
)
from thesis_exp.exp57_cbrd.losses import VARIANTS, cbrd_objective
from thesis_exp.exp57_cbrd.method import STATE_ZERO, target_state, thirds
from thesis_exp.exp57_cbrd.metrics import add_boundary_diagnostics, compute_metrics, finite_primary
from thesis_exp.exp57_cbrd.model import head_contract, make_cbrd_dual_head_classifier
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


TRAIN_VARIANTS = tuple(value for value in VARIANTS if value != "ordinary_hmsa")
EXPECTED_SEED42_HEAD_HASH = "d7c922a1956118af437c52189ffc993465277a72698cef38290e47404247f9e2"
PROTOCOL_PATH = CONFIG_ROOT / "stage1_protocol.json"
SOURCE_LOCK_PATH = CONFIG_ROOT / "stage1_source_lock.json"
STAGE0_DECISION_PATH = OUTPUT_ROOT / "decision" / "stage0_decision.json"


def parse_args() -> tuple[TrainConfig, str]:
    parser = argparse.ArgumentParser(description="Frozen Exp57 CBRD trainer (train/dev only).")
    parser.add_argument("--variant", required=True, choices=TRAIN_VARIANTS)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--checkpoint_output_dir", type=Path)
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
    parser.add_argument("--bf16", choices=("auto", "true", "false"), default="true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    args = parser.parse_args()
    output_dir = args.output_dir or OUTPUT_ROOT / "runs" / args.variant / f"seed_{args.seed}"
    checkpoint_root = (
        args.checkpoint_output_dir
        or REPO_ROOT / "thesis_exp" / "artifacts" / "exp57_cbrd" / args.variant / f"seed_{args.seed}"
    )
    config = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP57_IN_MEMORY_TRAIN_DEV_ONLY"),
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
    return config, args.variant


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_text_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['record_id']}\t{row['text']}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_frozen_contract(*, required: bool) -> dict[str, Any]:
    if not STAGE0_DECISION_PATH.is_file():
        raise FileNotFoundError(f"Missing Stage 0 decision: {STAGE0_DECISION_PATH}")
    stage0 = json.loads(STAGE0_DECISION_PATH.read_text(encoding="utf-8"))
    if stage0.get("status") != "STAGE0_PASS_READY_TO_FREEZE_STAGE1":
        raise RuntimeError(f"Stage 0 did not authorize Stage 1: {stage0.get('status')}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("test_access_count") != 0 or protocol.get("allowed_splits") != ["train", "dev"]:
        raise RuntimeError("Invalid Stage 1 data-access contract")
    lock: dict[str, Any] | None = None
    if SOURCE_LOCK_PATH.is_file():
        lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
        for relative, expected in lock["files"].items():
            actual = sha256_file(REPO_ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"Stage 1 source-lock mismatch: {relative}: {actual} != {expected}")
    elif required:
        raise FileNotFoundError(f"Formal Stage 1 requires {SOURCE_LOCK_PATH}")
    return {
        "stage0_decision_sha256": sha256_file(STAGE0_DECISION_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_lock_sha256": sha256_file(SOURCE_LOCK_PATH) if lock else None,
    }


def limit(rows: list[dict[str, Any]], value: int | None) -> list[dict[str, Any]]:
    return rows if value is None else rows[:value]


def shuffled_target_lookup() -> tuple[dict[str, list[float]], dict[str, Any]]:
    audit = shuffled_residual_audit(load_rows("train"), shuffle_seed=SHUFFLE_SEED)
    if not audit["checks"]["mapping_matches_exp55"]:
        raise RuntimeError("Frozen residual permutation no longer matches Exp55")
    lookup = {
        str(item["recipient_record_id"]): [value / 3.0 for value in item["shuffled_target_thirds"]]
        for item in audit["mapping"]
    }
    summary = {key: value for key, value in audit.items() if key != "mapping"}
    return lookup, summary


def load_model(config: TrainConfig) -> tuple[Any, Any, str, dict[str, Any]]:
    base, tokenizer, model_mode = load_model_and_tokenizer(config)
    if model_mode != "sequence_classification":
        raise RuntimeError(f"CBRD requires sequence_classification; got {model_mode}")
    model = make_cbrd_dual_head_classifier(base)
    contract = head_contract(model)
    if contract["hard_head_hash"] != contract["soft_head_hash"] or not contract["storage_independent"]:
        raise AssertionError(f"Invalid copied-head initialization: {contract}")
    if config.seed == 42 and config.bf16 == "true" and contract["hard_head_hash"] != EXPECTED_SEED42_HEAD_HASH:
        raise AssertionError(f"Seed42 initial head hash mismatch: {contract['hard_head_hash']}")
    return model, tokenizer, "qwen3_cbrd_dual_head_sequence_classification", contract


def evaluate(model: Any, dataloader: Any, device: Any, split: str) -> EvalResult:
    import torch

    if split != "dev":
        raise PermissionError("Exp57 evaluation is dev-only")
    model.eval()
    predictions: list[dict[str, Any]] = []
    logit_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    labels_out: list[int] = []
    record_ids: list[str] = []
    hard_loss_sum = aux_loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs, labels=labels, aux_route="ordinary_hmsa")
            hard_logits = outputs["hard_logits"].float()
            aux_logits = outputs["aux_logits"].float()
            targets = torch.tensor(
                [row["soft_target_5"] for row in metadata],
                dtype=torch.float32,
                device=device,
            )
            hard_loss = torch.nn.functional.cross_entropy(hard_logits, labels)
            aux_loss = -(targets * torch.log_softmax(aux_logits, dim=-1)).sum(dim=-1).mean()
            batch_n = len(metadata)
            hard_loss_sum += float(hard_loss.cpu()) * batch_n
            aux_loss_sum += float(aux_loss.cpu()) * batch_n
            n += batch_n
            hard_cpu = hard_logits.cpu()
            probs_cpu = torch.softmax(hard_cpu, dim=-1)
            logit_chunks.append(hard_cpu.numpy())
            prob_chunks.append(probs_cpu.numpy())
            for row, logit, probs in zip(metadata, hard_cpu.tolist(), probs_cpu.tolist()):
                label = int(row["label_5"])
                target = [float(value) for value in row["soft_target_5"]]
                state = target_state(label, target)
                minority_label: int | None = None
                advantage: float | None = None
                if state != STATE_ZERO:
                    minority_label = next(
                        index + 1
                        for index, value in enumerate(thirds(target))
                        if value == 1
                    )
                    advantage = float(logit[minority_label - 1] - logit[label - 1])
                pred = int(np.argmax(logit)) + 1
                expected = float(sum((index + 1) * value for index, value in enumerate(probs)))
                prediction: dict[str, Any] = {
                    "split": split,
                    "id": row["id"],
                    "record_id": row["record_id"],
                    "triple_key": row.get("triple_key"),
                    "question_key": row.get("question_key"),
                    "label_5": label,
                    "human_mean_5": float(row["human_mean_5"]),
                    "pred_label_5": pred,
                    "pred_score_5": float(pred),
                    "pred_score_expected": expected,
                    "boundary_state": state,
                    "minority_neighbor_label": minority_label,
                    "minority_neighbor_advantage": advantage,
                    "metric_canonical": row.get("metric_canonical"),
                    "scenario_canonical": row.get("scenario_canonical"),
                    "subject_canonical": row.get("subject_canonical"),
                    "language": row.get("language"),
                    "generator_model": row.get("generator_model"),
                }
                for index in range(5):
                    prediction[f"logit_{index + 1}"] = float(logit[index])
                    prediction[f"prob_{index + 1}"] = float(probs[index])
                predictions.append(prediction)
                labels_out.append(label)
                record_ids.append(str(row["record_id"]))
    metrics = add_boundary_diagnostics(compute_metrics(predictions), predictions)
    metrics.update(
        {
            "split": split,
            "eval_hard_ce_loss": hard_loss_sum / n,
            "eval_aux_soft_ce": aux_loss_sum / n,
        }
    )
    return EvalResult(
        metrics,
        predictions,
        np.concatenate(logit_chunks, axis=0),
        np.concatenate(prob_chunks, axis=0),
        np.asarray(labels_out),
        record_ids,
    )


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    config: TrainConfig,
    metrics: dict[str, Any],
    contract: dict[str, Any],
    variant: str,
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "state_dict.pt")
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "training_config.json", {**asdict(config), "variant": variant})
    write_json(output_dir / "dev_metrics.json", metrics)
    write_json(output_dir / "head_contract.json", contract)


def train(config: TrainConfig, variant: str) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if config.evaluate_test:
        raise PermissionError("Exp57 Stage 1 must never evaluate test")
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Unsupported training variant: {variant}")
    contract_files = verify_frozen_contract(required=os.environ.get("EXP57_REQUIRE_SOURCE_LOCK") == "1")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit(model_rows("train"), config.max_train_samples)
    dev_rows = limit(model_rows("dev"), config.max_eval_samples)
    shuffled_lookup, shuffle_summary = shuffled_target_lookup()
    model, tokenizer, model_mode, initial_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
    global_step = micro_step = clip_events = 0
    max_pre_clip_grad_norm = 0.0
    start = time.time()
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp57 {variant}")
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_zero in range(int(math.ceil(config.num_train_epochs))):
            if epoch_zero >= config.num_train_epochs:
                break
            epoch = epoch_zero + 1
            model.train()
            hard_running = aux_running = total_running = 0.0
            for step, batch in enumerate(train_loader, start=1):
                metadata = batch.pop("metadata")
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                targets = torch.tensor(
                    [row["soft_target_5"] for row in metadata],
                    dtype=torch.float32,
                    device=device,
                )
                residual_targets = (
                    torch.tensor(
                        [shuffled_lookup[str(row["record_id"])] for row in metadata],
                        dtype=torch.float32,
                        device=device,
                    )
                    if variant == "shuffled_residual"
                    else None
                )
                loss_scale = 1.0 / float(config.gradient_accumulation_steps)
                outputs = model(
                    **inputs,
                    labels=labels,
                    soft_targets=targets,
                    residual_targets=residual_targets,
                    aux_route=variant,
                    route_loss_scale=loss_scale,
                )
                objective = cbrd_objective(
                    outputs,
                    labels,
                    targets,
                    variant=variant,
                    residual_targets=residual_targets,
                )
                loss = objective["optimization_loss"]
                (loss * loss_scale).backward()
                hard_value = float(objective["reported_hard_loss"].detach().cpu())
                aux_value = float(objective["reported_aux_soft_ce"].detach().cpu())
                total_value = float(loss.detach().cpu())
                hard_running += hard_value
                aux_running += aux_value
                total_running += total_value
                micro_step += 1
                if len(trace) < 64:
                    trace.append(
                        {
                            "micro_step": micro_step,
                            "record_ids": [row["record_id"] for row in metadata],
                            "hard_loss": hard_value,
                            "reported_original_soft_ce": aux_value,
                            "optimization_loss": total_value,
                            "route_loss_scale": loss_scale,
                        }
                    )
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        .detach()
                        .cpu()
                    )
                    max_pre_clip_grad_norm = max(max_pre_clip_grad_norm, grad_norm)
                    clip_events += int(grad_norm > config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % config.log_steps == 0 or global_step == total_steps:
                        elapsed = time.time() - start
                        progress.render(
                            global_step,
                            epoch,
                            total_running / max(1, step),
                            scheduler.get_last_lr()[0],
                            elapsed,
                            max(0, total_steps - global_step) * elapsed / max(1, global_step),
                        )
            progress.newline()
            result = evaluate(model, dev_loader, device, "dev")
            metrics = {
                **result.metrics,
                "epoch": epoch,
                "global_step": global_step,
                "train_hard_loss": hard_running / len(train_loader),
                "train_reported_aux_soft_ce": aux_running / len(train_loader),
                "train_optimization_loss": total_running / len(train_loader),
            }
            history.append(metrics)
            write_json(config.output_dir / "dev_metrics_history.json", history)
            write_csv(config.output_dir / "tables" / "dev_metrics_history.csv", history)
            print(
                f"[exp57:{variant}] epoch={epoch} Exact={metrics['Exact_rounded']:.6f} "
                f"MAE={metrics['MAE_human_mean']:.6f} Kendall={metrics['Kendall_human_mean']:.6f}",
                flush=True,
            )
            candidate_exact = float(metrics["Exact_rounded"])
            if candidate_exact > best_exact:
                best_exact = candidate_exact
                best_epoch = epoch
                save_checkpoint(model, tokenizer, best_dir, config, metrics, initial_contract, variant)
                save_predictions(config.output_dir, result, suffix="dev_best")
    finally:
        progress.close()
    if best_epoch is None:
        raise RuntimeError("No CBRD checkpoint was selected")
    model.load_state_dict(torch.load(best_dir / "state_dict.pt", map_location=device, weights_only=True))
    final = evaluate(model, dev_loader, device, "dev")
    selected = {
        **final.metrics,
        "epoch": best_epoch,
        "global_step": next(int(row["global_step"]) for row in history if int(row["epoch"]) == best_epoch),
    }
    save_predictions(config.output_dir, final)
    write_json(config.output_dir / "selected_dev_metrics.json", selected)
    write_csv(
        config.output_dir / "tables" / "selected_dev_metrics.csv",
        [{"variant": variant, "seed": config.seed, **selected}],
    )
    write_json(config.output_dir / "training_trace_first64.json", {"trace": trace})
    elapsed = time.time() - start
    summary = {
        "status": "COMPLETED" if finite_primary(selected) else "FAILED_NONFINITE",
        "variant": variant,
        "seed": config.seed,
        "runtime_head": git_head(),
        "model_mode": model_mode,
        "model_name_or_path": config.model_name_or_path,
        "initial_head_contract": initial_contract,
        "selected_epoch": best_epoch,
        "selected_metrics": selected,
        "checkpoint_path": str(best_dir),
        "checkpoint_rule": "highest hard-head dev Exact_rounded; ties keep earlier epoch",
        "inference": "hard_head_raw_logit_argmax",
        "route_implementation": "single_soft_ce_hidden_gradient_hook",
        "route_loss_scale": f"1/{config.gradient_accumulation_steps}",
        "train_text_hash": aggregate_text_hash(train_rows),
        "dev_text_hash": aggregate_text_hash(dev_rows),
        "shuffle_contract": shuffle_summary if variant == "shuffled_residual" else None,
        "frozen_contract_files": contract_files,
        "test_access_count": 0,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "clip_events": clip_events,
        "max_pre_clip_grad_norm": max_pre_clip_grad_norm,
    }
    write_json(config.output_dir / "run_summary.json", summary)
    write_text(
        config.output_dir / "run_summary.md",
        f"# Exp57 {variant} seed {config.seed}\n\n"
        f"- Selected epoch: {best_epoch}\n"
        f"- Exact: {selected['Exact_rounded']:.6f}\n"
        f"- MAE: {selected['MAE_human_mean']:.6f}\n"
        f"- Kendall: {selected['Kendall_human_mean']:.6f}\n"
        f"- Runtime: {format_duration(elapsed)}\n"
        "- Test accessed: no\n",
    )
    return summary


def main() -> None:
    config, variant = parse_args()
    result = train(config, variant)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
