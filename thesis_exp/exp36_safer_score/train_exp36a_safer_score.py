"""Train one locked Exp36A soft-target or SAFER-Score variant."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import (
    DEV_PATH, PRIVATE, ROOT, VARIANTS, curriculum_rho, dynamic_target, read_jsonl,
    sample_id, write_json, write_jsonl,
)
from thesis_exp.exp36_safer_score.model_utils import (
    build_model, evaluate, load_model_state, load_tokenizer, make_loader, metric_summary,
    save_model, selected_checkpoint, set_seed,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import StepProgressBar


DYNAMIC = {"v5_safer_score", "v7_shuffled_teacher_control", "v6a_no_failure_aux", "v6b_no_student_uncertainty", "v6c_no_curriculum"}
FAILURE_ACTIVE = {"v5_safer_score", "v7_shuffled_teacher_control", "v6b_no_student_uncertainty", "v6c_no_curriculum"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model-name-or-path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--dev-jsonl", type=Path, default=DEV_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--failure-loss-weight", type=float, default=0.1)
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--log-steps", type=int, default=5)
    values = parser.parse_args()
    values.train_jsonl = values.train_jsonl or values.out_dir / "private/data" / f"exp36a_{values.variant}_train.jsonl"
    values.run_dir = values.run_dir or Path("thesis_exp/runs/exp36_safer_score") / values.variant / f"seed_{values.seed}"
    values.artifact_dir = values.artifact_dir or Path("thesis_exp/artifacts/exp36_safer_score") / values.variant / f"seed_{values.seed}"
    if "test" in str(values.train_jsonl).lower() or "test" in str(values.dev_jsonl).lower():
        raise ValueError("Exp36A strictly forbids test paths")
    return values


def soft_ce(logits: Any, targets: Any) -> Any:
    import torch

    return -(targets.float() * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def failure_loss(logits: Any, targets: Any, mask: Any) -> Any:
    import torch

    active = mask.float().sum()
    if float(active.detach().cpu()) == 0:
        return logits.sum() * 0.0
    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return (per_cell.mean(dim=-1) * mask.float()).sum() / active


def failure_train_metrics(model: Any, rows: list[dict[str, Any]], tokenizer: Any, config: argparse.Namespace, device: Any) -> dict[str, Any]:
    import torch

    active_rows = [row for row in rows if int(row.get("failure_mask", 0)) == 1]
    if not active_rows:
        return {"masked_rows": 0, "micro_f1": float("nan"), "exact_set_match": float("nan")}
    loader = make_loader(active_rows, tokenizer, config.eval_batch_size, config.max_length, False, config.seed)
    truth: list[int] = []
    pred: list[int] = []
    exact = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            _, logits = model(**{key: value.to(device) for key, value in batch.items()})
            binary = (torch.sigmoid(logits.float()) >= 0.5).int().cpu().tolist()
            for row, values in zip(metadata, binary):
                gold = [int(value) for value in row["failure_target_6"]]
                truth.extend(gold); pred.extend(values); exact.append(gold == values)
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, pred))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"masked_rows": len(active_rows), "micro_f1": f1, "exact_set_match": float(np.mean(exact))}


def main() -> None:
    config = parse_args()
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    set_seed(config.seed)
    train_rows = read_jsonl(config.train_jsonl)
    dev_rows = read_jsonl(config.dev_jsonl)
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_eval_samples:
        dev_rows = dev_rows[: config.max_eval_samples]
    if not train_rows or not dev_rows:
        raise ValueError("Empty train or dev rows")
    if any(float(row.get("sample_weight", 1)) != 1.0 for row in train_rows):
        raise ValueError("Exp36A requires all sample weights to equal one")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config.model_name_or_path, True)
    model = build_model(config.model_name_or_path, config.bf16, True, config.gradient_checkpointing)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp36A training requires CUDA")
    device = torch.device("cuda")
    model.to(device)
    dev_loader = make_loader(dev_rows, tokenizer, config.eval_batch_size, config.max_length, False, config.seed)
    batches = math.ceil(len(train_rows) / config.batch_size)
    updates_per_epoch = math.ceil(batches / config.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * config.epochs)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * config.warmup_ratio), total_updates)
    priors = torch.tensor(train_rows[0].get("class_priors_5", [0.2] * 5), device=device).clamp_min(1e-8)
    history = []
    selected_path: Path | None = None
    selected_epoch = None
    global_step = 0
    start = time.time()
    progress = StepProgressBar(total_updates, True, desc=f"Exp36A {config.variant}")
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(1, config.epochs + 1):
            loader = make_loader(train_rows, tokenizer, config.batch_size, config.max_length, True, config.seed + epoch)
            model.train()
            window = 0
            score_sum = fail_sum = total_sum = 0.0
            mask_count = 0
            lambda_sum = 0.0
            for index, batch in enumerate(loader, 1):
                metadata = batch.pop("metadata")
                if config.variant in DYNAMIC:
                    target_values = [dynamic_target(row, epoch, config.variant) for row in metadata]
                else:
                    target_values = [row["soft_target_5"] for row in metadata]
                targets = torch.tensor(target_values, dtype=torch.float32, device=device)
                failure_targets = torch.tensor([row["failure_target_6"] for row in metadata], dtype=torch.float32, device=device)
                masks = torch.tensor([int(row["failure_mask"]) for row in metadata], dtype=torch.float32, device=device)
                score_logits, failure_logits = model(**{key: value.to(device) for key, value in batch.items()})
                adjusted = score_logits
                if config.variant == "v4_human_soft_logit_adjustment":
                    adjusted = score_logits + config.logit_adjustment_tau * torch.log(priors)
                score = soft_ce(adjusted, targets)
                fail = failure_loss(failure_logits, failure_targets, masks) if config.variant in FAILURE_ACTIVE else failure_logits.sum() * 0.0
                total = score + config.failure_loss_weight * fail
                (total / config.gradient_accumulation_steps).backward()
                score_sum += float(score.detach().cpu()); fail_sum += float(fail.detach().cpu()); total_sum += float(total.detach().cpu())
                mask_count += int(masks.sum().detach().cpu())
                lambda_sum += sum(float(row.get("teacher_lambda", 0)) for row in metadata)
                window += 1
                if window == config.gradient_accumulation_steps or index == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1; window = 0
                    if global_step % config.log_steps == 0 or global_step == total_updates:
                        elapsed = time.time() - start
                        progress.render(global_step, epoch, total_sum / index, scheduler.get_last_lr()[0], elapsed,
                                        (total_updates-global_step) * elapsed / max(global_step, 1))
            progress.newline()
            predictions = evaluate(model, dev_loader, device)
            row_metrics = {"variant": config.variant, "seed": config.seed, "epoch": epoch, "global_step": global_step,
                           "train_score_loss": score_sum / len(loader), "train_failure_loss": fail_sum / len(loader),
                           "train_total_loss": total_sum / len(loader), "failure_mask_exposures": mask_count,
                           "curriculum_rho": curriculum_rho(epoch) if config.variant in DYNAMIC and config.variant != "v6c_no_curriculum" else (1.0 if config.variant in DYNAMIC else 0.0),
                           "mean_teacher_lambda_seen": lambda_sum / len(train_rows), **metric_summary(predictions)}
            history.append(row_metrics)
            chosen = selected_checkpoint(history)
            if int(chosen["epoch"]) == epoch:
                new_path = config.artifact_dir / f"epoch_{epoch:02d}"
                save_model(model, tokenizer, new_path, {"variant": config.variant, "epoch": epoch, "metrics": row_metrics})
                if selected_path and selected_path != new_path:
                    shutil.rmtree(selected_path, ignore_errors=True)
                selected_path, selected_epoch = new_path, epoch
            write_json(config.run_dir / "epoch_metrics.json", history)
            print(f"[exp36a] {config.variant} epoch={epoch} exact={row_metrics['Exact_Match']:.4f} MAE={row_metrics['MAE_argmax']:.4f}", flush=True)
    finally:
        progress.close()
    chosen = selected_checkpoint(history)
    if selected_path is None or selected_epoch != int(chosen["epoch"]):
        raise RuntimeError("Selected checkpoint tracking failed")
    load_model_state(model, selected_path)
    selected_predictions = evaluate(model, dev_loader, device)
    pred_path = config.out_dir / "private/dev_predictions" / config.variant / f"seed_{config.seed}.jsonl"
    write_jsonl(pred_path, selected_predictions)
    fail_metrics = failure_train_metrics(model, train_rows, tokenizer, config, device) if config.variant in FAILURE_ACTIVE else {"masked_rows": 0, "micro_f1": float("nan"), "exact_set_match": float("nan")}
    summary = {
        "status": "COMPLETED", "variant": config.variant, "seed": config.seed,
        "selected_epoch": selected_epoch, "selected_metrics": chosen,
        "failure_head_train_metrics": fail_metrics, "checkpoint_path": str(selected_path),
        "prediction_path": str(pred_path), "nan_count": 0, "oom_count": 0, "test_access_count": 0,
    }
    write_json(config.run_dir / "run_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
