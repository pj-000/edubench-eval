"""Train Exp5 L3b weighted threshold-aligned ordinal loss."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04.train_objective import (
    StepProgressBar,
    format_duration,
    get_loss_logits,
    save_dev_test_npz,
    save_eval_arrays,
    save_metrics_tables,
    save_predictions,
    set_seed,
)
from thesis_exp.src.edujudge.exp05 import (
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    EXP04_A4_DATASET_DIR,
    EXP05_TABLES_DIR,
    L3B_RUN_CONFIG,
    L3B_RUN_ID,
    ensure_exp05_dirs,
    l3b_checkpoint_dir,
    l3b_run_dir,
)
from thesis_exp.src.edujudge.exp05.build_exp05_dataset import ensure_exp05_dataset
from thesis_exp.src.edujudge.exp05.class_weights import load_class_weight_vector, write_class_weights
from thesis_exp.src.edujudge.exp05.losses import weighted_threshold_ordinal_loss
from thesis_exp.src.edujudge.exp05.train_l1_weighted_ordinal import evaluate
from thesis_exp.src.edujudge.exp05.train_l2_asymmetric_ordinal import (
    limit_rows,
    make_dataloader,
    read_json,
    score_is_better,
    select_dtype,
    selection_metric_name,
    selection_metric_value,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_text


@dataclass
class L3bTrainConfig:
    model_name_or_path: str
    data_dir: Path
    output_dir: Path
    checkpoint_output_dir: Path
    max_length: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    seed: int
    bf16: str
    fp16: bool
    gradient_checkpointing: bool
    trust_remote_code: bool
    local_files_only: bool
    max_train_samples: int | None
    max_eval_samples: int | None
    selection_metric: str
    selection_mode: str
    num_workers: int
    log_steps: int
    progress_bar: bool
    mu_thr: float
    w_min: float
    w_max: float
    class_weights_path: Path


def make_weighted_threshold_head_class(torch: Any, nn: Any):
    class WeightedThresholdOrdinalHead(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int, class_weights: list[float], mu_thr: float) -> None:
            super().__init__()
            self.backbone = backbone
            self.score = nn.Linear(hidden_size, 4)
            self.hidden_size = int(hidden_size)
            self.num_outputs = 4
            self.mu_thr = float(mu_thr)
            self.config = backbone.config
            self.config.num_labels = 4
            self.config.problem_type = "multi_label_classification"
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None, **kwargs: Any) -> dict[str, Any]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = outputs.last_hidden_state
            if attention_mask is None:
                pooled = hidden[:, -1]
            else:
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            if pooled.dtype != self.score.weight.dtype:
                pooled = pooled.to(dtype=self.score.weight.dtype)
            logits = self.score(pooled)
            loss = None
            loss_debug = {}
            if labels is not None:
                loss, loss_debug = weighted_threshold_ordinal_loss(
                    logits.float(),
                    labels.long(),
                    self.class_weights,
                    mu_thr=self.mu_thr,
                )
            return {"loss": loss, "logits": logits, "loss_debug": loss_debug}

    return WeightedThresholdOrdinalHead


def build_model(torch: Any, nn: Any, AutoModel: Any, config: L3bTrainConfig, dtype: Any, class_weights: list[float]) -> Any:
    source = str(config.model_name_or_path)
    metadata_path = config.checkpoint_output_dir / "best" / "exp05_l3b_head_metadata.json"
    if metadata_path.exists():
        source = read_json(metadata_path)["base_model_name_or_path"]
    backbone = AutoModel.from_pretrained(
        source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for Exp5 L3b weighted threshold ordinal head.")
    return make_weighted_threshold_head_class(torch, nn)(backbone, int(hidden_size), class_weights, config.mu_thr)


def load_model_and_tokenizer(config: L3bTrainConfig):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    if not config.class_weights_path.exists():
        write_class_weights(train_path=config.data_dir / "train.jsonl", w_min=config.w_min, w_max=config.w_max)
    class_weights = load_class_weight_vector(config.class_weights_path)
    dtype = select_dtype(config, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model = build_model(torch, nn, AutoModel, config, dtype, class_weights)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: L3bTrainConfig, metrics: dict[str, Any]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    tokenizer.save_pretrained(output_dir)
    if hasattr(model.backbone, "config"):
        model.backbone.config.save_pretrained(output_dir)
    write_json(
        output_dir / "exp05_l3b_head_metadata.json",
        {
            "base_model_name_or_path": config.model_name_or_path,
            "experiment": "Exp5 L3b weighted threshold-aligned ordinal",
            "loss_type": "weighted_threshold_ordinal",
            "run_id": L3B_RUN_ID,
            "mu_thr": config.mu_thr,
            "class_weights_path": relpath(config.class_weights_path),
            "use_class_weights": True,
            "use_expected_score_penalty": False,
            "use_high_score_preservation": False,
            "use_threshold_suppression": True,
            "num_outputs": getattr(model, "num_outputs", 4),
            "hidden_size": getattr(model, "hidden_size", None),
            "state_dict": state_dict_path.name,
            "best_epoch": metrics.get("epoch"),
            "best_global_step": metrics.get("global_step"),
            "best_selection_metric_name": selection_metric_name(config),
            "best_selection_metric_value": selection_metric_value(metrics, config),
            "selection_direction": config.selection_mode,
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "dev_metrics.json", metrics)


def save_loss_debug_history(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "epoch",
        "step",
        "loss",
        "weighted_base_loss",
        "threshold_penalty",
        "mu_thr",
        "mean_total_loss",
        "mean_sample_weight",
        "min_sample_weight",
        "max_sample_weight",
        "active_low_count",
        "mean_p_gt_3_low",
        "mean_p_gt_4_low",
        "max_p_gt_3_low",
        "max_p_gt_4_low",
    ]
    write_csv(output_dir / "tables" / "loss_debug_history.csv", rows, fieldnames=fieldnames)
    write_csv(output_dir / "logs" / "loss_debug_history.csv", rows, fieldnames=fieldnames)


def save_dev_metrics_history_to_logs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_csv(output_dir / "logs" / "dev_metrics_history.csv", rows)


def normalize_run_files(output_dir: Path, config: L3bTrainConfig, status: str) -> None:
    best_metadata_path = config.checkpoint_output_dir / "best" / "exp05_l3b_head_metadata.json"
    best_metadata = read_json(best_metadata_path) if best_metadata_path.exists() else {}
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": "Exp5 low-score loss ablation",
            "loss_id": L3B_RUN_ID,
            "loss_type": "weighted_threshold_ordinal",
            "status": status,
            "fixed_input_template": "A4_question_answer_metric_rubric_metadata",
            "model_name_or_path": config.model_name_or_path,
            "mu_thr": config.mu_thr,
            "class_weights_path": relpath(config.class_weights_path),
            "use_class_weights": True,
            "use_expected_score_penalty": False,
            "use_high_score_preservation": False,
            "use_threshold_suppression": True,
            "checkpoint_selection": f"dev {config.selection_metric} ({config.selection_mode})",
            "best_epoch": best_metadata.get("best_epoch"),
            "best_global_step": best_metadata.get("best_global_step"),
            "best_selection_metric_name": best_metadata.get("best_selection_metric_name", selection_metric_name(config)),
            "best_selection_metric_value": best_metadata.get("best_selection_metric_value"),
            "selection_direction": best_metadata.get("selection_direction", config.selection_mode),
            "test_evaluated_after_best_checkpoint": status == "completed",
        },
    )
    lines = [
        "# Exp5 L3b Weighted Threshold-Aligned Run Summary",
        "",
        f"Run: `{L3B_RUN_ID}`",
        f"Status: `{status}`",
        f"Model: `{config.model_name_or_path}`",
        "Fixed input: `A4_question_answer_metric_rubric_metadata`",
        "Objective: ordinal threshold prediction",
        "Loss: weighted ordinal BCE + low-score p_gt_3/p_gt_4 threshold penalty",
        f"mu_thr: `{config.mu_thr}`",
        "use_class_weights: `true`",
        "use_expected_score_penalty: `false`",
        "use_high_score_preservation: `false`",
        "use_threshold_suppression: `true`",
        f"class_weights_path: `{relpath(config.class_weights_path)}`",
        f"Checkpoint selection: dev `{config.selection_metric}` ({config.selection_mode})",
        "Test evaluation: after best checkpoint selection",
    ]
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def train(config: L3bTrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp05_dirs()
    ensure_exp05_dataset()
    write_class_weights(train_path=config.data_dir / "train.jsonl", w_min=config.w_min, w_max=config.w_max)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = limit_rows(read_jsonl(config.data_dir / "train.jsonl"), config.max_train_samples)
    dev_rows = limit_rows(read_jsonl(config.data_dir / "dev.jsonl"), config.max_eval_samples)
    test_rows = limit_rows(read_jsonl(config.data_dir / "test.jsonl"), config.max_eval_samples)

    model, tokenizer = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_dataloader(test_rows, tokenizer, config, "test", shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_dir = config.checkpoint_output_dir / "best"
    best_score: float | None = None
    global_step = 0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    metrics_history: list[dict[str, Any]] = []
    loss_debug_history: list[dict[str, Any]] = []
    num_epochs = int(math.ceil(config.num_train_epochs))
    print(
        "training schedule: "
        f"run={L3B_RUN_ID} mu_thr={config.mu_thr} train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} total_update_steps={total_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp5 {L3B_RUN_ID}")
    try:
        for epoch in range(num_epochs):
            if epoch >= config.num_train_epochs:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            latest_debug: dict[str, float] = {}
            for step, batch in enumerate(train_loader, start=1):
                batch.pop("metadata")
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss, _ = get_loss_logits(outputs)
                if loss is None:
                    raise RuntimeError("Model returned no loss.")
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().cpu())
                latest_debug = {key: float(value) for key, value in outputs.get("loss_debug", {}).items()}
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    now = time.time()
                    elapsed = now - start
                    avg_loss = running_loss / max(1, step)
                    avg_steps_per_second = global_step / max(1e-9, elapsed)
                    eta_seconds = max(0, total_steps - global_step) / avg_steps_per_second
                    current_lr = scheduler.get_last_lr()[0]
                    progress.render(global_step, epoch + 1, avg_loss, current_lr, elapsed, eta_seconds)
                    loss_debug_history.append({"epoch": epoch + 1, "step": global_step, "loss": avg_loss, **latest_debug})
                    if global_step % config.log_steps == 0:
                        interval_steps = max(1, global_step - last_log_step)
                        interval_seconds = max(1e-9, now - last_log_time)
                        steps_per_second = interval_steps / interval_seconds
                        remaining_steps = max(0, total_steps - global_step)
                        last_log_time = now
                        last_log_step = global_step
                        progress.newline()
                        print(
                            f"epoch={epoch + 1} step={global_step}/{total_steps} "
                            f"loss={avg_loss:.4f} "
                            f"weighted_base_loss={latest_debug.get('weighted_base_loss', float('nan')):.4f} "
                            f"threshold_penalty={latest_debug.get('threshold_penalty', float('nan')):.4f} "
                            f"active_low_count={latest_debug.get('active_low_count', float('nan')):.0f} "
                            f"mean_p_gt_3_low={latest_debug.get('mean_p_gt_3_low', float('nan')):.4f} "
                            f"mean_p_gt_4_low={latest_debug.get('mean_p_gt_4_low', float('nan')):.4f} "
                            f"lr={current_lr:.2e} elapsed={format_duration(elapsed)} "
                            f"eta={format_duration(remaining_steps / steps_per_second)} "
                            f"steps_per_min={steps_per_second * 60:.2f}",
                            flush=True,
                        )
                    if global_step >= total_steps:
                        break
            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev", loss_variant=L3B_RUN_ID)
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            metrics_history.append(dev_result.metrics)
            selected = bool(score_is_better(dev_result.metrics, best_score, config))
            elapsed = time.time() - start
            print(
                f"dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"MAE_expected={dev_result.metrics['MAE_expected']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"threshold_penalty={dev_result.metrics.get('threshold_penalty', float('nan')):.4f}, "
                f"selected={selected}, elapsed={format_duration(elapsed)}",
                flush=True,
            )
            if selected:
                best_score = selection_metric_value(dev_result.metrics, config)
                save_checkpoint(model, tokenizer, best_dir, config, dev_result.metrics)
                save_predictions(config.output_dir, dev_result, suffix="dev_best")
            if global_step >= total_steps:
                break
    finally:
        progress.close()

    state_dict_path = best_dir / "state_dict.pt"
    if not state_dict_path.exists():
        raise RuntimeError(f"Best checkpoint was not saved: {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    dev_final = evaluate(model, dev_loader, device, "dev", loss_variant=L3B_RUN_ID)
    test_result = evaluate(model, test_loader, device, "test", loss_variant=L3B_RUN_ID)
    for result in [dev_final, test_result]:
        result.metrics["epoch"] = "best"
        result.metrics["global_step"] = global_step
        save_predictions(config.output_dir, result)
        save_eval_arrays(config.output_dir, result.metrics["split"], result)
    save_dev_test_npz(config.output_dir, dev_final, test_result)
    test_result.metrics["elapsed_seconds"] = time.time() - start
    write_csv(config.output_dir / "tables" / "dev_metrics_history.csv", metrics_history)
    write_csv(config.output_dir / "logs" / "training_log.csv", metrics_history)
    save_dev_metrics_history_to_logs(config.output_dir, metrics_history)
    save_loss_debug_history(config.output_dir, loss_debug_history)
    save_metrics_tables(config.output_dir, [dev_final, test_result], "ordinal")
    normalize_run_files(config.output_dir, config, "completed")
    return {"metrics": [dev_final.metrics, test_result.metrics], "best_dir": str(best_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp5 L3b weighted threshold ordinal.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir", type=Path, default=EXP04_A4_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--selection_metric", default="MAE_label")
    parser.add_argument("--selection_mode", choices=["min", "max"], default="min")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument("--mu_thr", type=float, default=L3B_RUN_CONFIG["mu_thr"])
    parser.add_argument("--w_min", type=float, default=DEFAULT_W_MIN)
    parser.add_argument("--w_max", type=float, default=DEFAULT_W_MAX)
    parser.add_argument("--class_weights_path", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> L3bTrainConfig:
    return L3bTrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir or l3b_run_dir(smoke=args.smoke),
        checkpoint_output_dir=args.checkpoint_output_dir or l3b_checkpoint_dir(smoke=args.smoke),
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
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
        mu_thr=float(args.mu_thr),
        w_min=float(args.w_min),
        w_max=float(args.w_max),
        class_weights_path=args.class_weights_path or (EXP05_TABLES_DIR / "class_weights.json"),
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
