"""Train one Exp27P Qwen3-Reranker variant with globally weighted soft CE."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    StepProgressBar,
    load_model_and_tokenizer,
    save_checkpoint,
)
from thesis_exp.src.edujudge.exp27p import ARTIFACT_ROOT, EXP27O_DIR, OUTPUT_DIR, RUN_ROOT, VARIANTS
from thesis_exp.src.edujudge.exp27p.common import (
    build_model_text,
    prediction_metrics,
    read_jsonl,
    sample_id,
    select_checkpoint,
    stable_hash,
    stratified_metrics,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")


@dataclass
class TrainConfig:
    variant: str
    model_name_or_path: str
    train_jsonl: Path
    dev_jsonl: Path
    run_dir: Path
    output_dir: Path
    checkpoint_output_dir: Path
    high_impact_manifest: Path
    max_length: int = 2048
    num_train_epochs: float = 10.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: str = "auto"
    fp16: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    checkpoint_dir: Path | None = None
    num_workers: int = 0
    log_steps: int = 5
    progress_bar: bool = True


class RowDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def global_scaled_soft_ce(
    logits: Any,
    soft_targets: Any,
    sample_weights: Any,
    dataset_rows: int,
    global_weight_sum: float,
) -> Any:
    """Unbiased uniform-minibatch estimator of the locked full-data objective."""
    import torch

    if logits.ndim != 2 or logits.shape[-1] != 5:
        raise ValueError(f"Expected [batch,5] logits, found {tuple(logits.shape)}")
    if soft_targets.shape != logits.shape:
        raise ValueError("soft_targets must have the same [batch,5] shape as logits")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != logits.shape[0]:
        raise ValueError("sample_weights must have shape [batch]")
    if dataset_rows <= 0 or global_weight_sum <= 0:
        raise ValueError("dataset_rows and global_weight_sum must be positive")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    per_example = -(soft_targets.float() * log_probs).sum(dim=-1)
    batch_size = int(logits.shape[0])
    return (float(dataset_rows) / (float(global_weight_sum) * batch_size)) * (
        sample_weights.float() * per_example
    ).sum()


def prepare_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        target = [float(value) for value in row["soft_target_5"]]
        if len(target) != 5 or min(target) < 0 or abs(sum(target) - 1.0) > 1e-8:
            raise ValueError(f"Invalid soft target: {sample_id(row)}")
        output.append(
            {
                **row,
                "text": build_model_text(row),
                "soft_target_5": target,
                "sample_weight": float(row["sample_weight"]),
                "gold_label_5": int(row["original_label_5"]),
                "training_label_5": int(row["label_5"]),
            }
        )
    return output


def prepare_dev_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        label = int(row["label_5"])
        output.append(
            {
                **row,
                "sample_id": sample_id(row),
                "text": build_model_text(row),
                "soft_target_5": [1.0 if index == label else 0.0 for index in range(1, 6)],
                "sample_weight": 1.0,
                "gold_label_5": label,
                "training_label_5": label,
                "exp27o_training_tier": "dev_not_applicable",
            }
        )
    return output


def collate_fn(tokenizer: Any, max_length: int):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["soft_targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["sample_weights"] = torch.tensor([row["sample_weight"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return collate


def make_loader(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    config: TrainConfig,
    train: bool,
    epoch: int = 0,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(config.seed + epoch)
    return DataLoader(
        RowDataset(rows),
        batch_size=config.per_device_train_batch_size if train else config.per_device_eval_batch_size,
        shuffle=train,
        generator=generator if train else None,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_fn(tokenizer, config.max_length),
    )


def output_logits(outputs: Any) -> Any:
    return outputs["logits"] if isinstance(outputs, dict) else outputs.logits


def evaluate(model: Any, loader: Any, device: Any) -> list[dict[str, Any]]:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch.pop("soft_targets")
            batch.pop("sample_weights")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = output_logits(model(**inputs)).float().cpu()
            probs = torch.softmax(logits, dim=-1).tolist()
            for row, prob in zip(metadata, probs):
                pred = int(np.argmax(prob)) + 1
                prediction = {
                    "sample_id": row["sample_id"],
                    "sample_id_hash": stable_hash(row["sample_id"]),
                    "question_key": row.get("question_key"),
                    "gold_label_5": int(row["gold_label_5"]),
                    "training_label_5": int(row["training_label_5"]),
                    "pred_label_5": pred,
                    "pred_score_expected": float(sum((index + 1) * value for index, value in enumerate(prob))),
                    "language": row.get("language"),
                    "metric_group": row.get("metric_group"),
                    "subject_canonical": row.get("subject_canonical"),
                    "scenario_canonical": row.get("scenario_canonical"),
                    "exp27o_training_tier": row.get("exp27o_training_tier"),
                    "original_label_5": row.get("original_label_5", row.get("gold_label_5")),
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                    "soft_target_5": row.get("soft_target_5"),
                }
                for index, value in enumerate(prob, 1):
                    prediction[f"prob_{index}"] = float(value)
                predictions.append(prediction)
    return predictions


def diagnostics_by_tier(predictions: list[dict[str, Any]], variant: str, seed: int) -> list[dict[str, Any]]:
    output = []
    tiers = sorted({str(row.get("exp27o_training_tier")) for row in predictions})
    for tier in tiers:
        subset = [row for row in predictions if str(row.get("exp27o_training_tier")) == tier]
        output.append({"variant": variant, "seed": seed, "training_tier": tier, **prediction_metrics(subset)})
    return output


def high_impact_diagnostics(
    predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    by_id = {row["sample_id"]: row for row in predictions}
    output = []
    for item in manifest:
        pred = by_id.get(item["sample_id"])
        if pred is None:
            continue
        output.append(
            {
                "variant": variant,
                "seed": seed,
                "sample_id": item["sample_id"],
                "sample_id_hash": item["sample_id_hash"],
                "original_score": int(item["original_label_5"]),
                "silver_hard_score": int(item["silver_hard_score"]),
                "silver_score_range": item["silver_score_range"],
                "soft_target_5": item["soft_target_5"],
                "sample_weight": float(pred["sample_weight"]),
                "pred_label_5": int(pred["pred_label_5"]),
                "pred_score_expected": float(pred["pred_score_expected"]),
                "agrees_original": int(pred["pred_label_5"]) == int(item["original_label_5"]),
                "agrees_silver": int(pred["pred_label_5"]) == int(item["silver_hard_score"]),
                **{f"prob_{label}": pred[f"prob_{label}"] for label in range(1, 6)},
            }
        )
    return output


def matched_dev_diagnostic(
    dev_predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    strata = {
        (str(row.get("language")), str(row.get("metric_group")), str(row.get("subject")))
        for row in manifest
    }
    subset = [
        row
        for row in dev_predictions
        if int(row["gold_label_5"]) <= 2
        and (str(row.get("language")), str(row.get("metric_group")), str(row.get("subject_canonical"))) in strata
    ]
    return {"variant": variant, "seed": seed, "slice": "dev_low_matched_high_impact16_strata", **prediction_metrics(subset)}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--dev_jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
    parser.add_argument("--high_impact_manifest", type=Path, default=EXP27O_DIR / "tables" / "exp27o_high_impact16_manifest_light.csv")
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
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_local_files_only", action="store_false", dest="local_files_only")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(local_files_only=True, progress_bar=True)
    values = vars(parser.parse_args())
    if values["run_dir"] is None:
        values["run_dir"] = RUN_ROOT / values["variant"] / f"seed_{values['seed']}"
    if values["checkpoint_output_dir"] is None:
        values["checkpoint_output_dir"] = ARTIFACT_ROOT / values["variant"] / f"seed_{values['seed']}"
    values["checkpoint_dir"] = None
    return TrainConfig(**values)


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    if "test" in str(config.dev_jsonl).lower():
        raise ValueError("Exp27P dev_jsonl must not reference test")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    raw_train = read_jsonl(config.train_jsonl)
    raw_dev = read_jsonl(config.dev_jsonl)
    if config.max_train_samples:
        raw_train = raw_train[: config.max_train_samples]
    if config.max_eval_samples:
        raw_dev = raw_dev[: config.max_eval_samples]
    train_rows = prepare_train_rows(raw_train)
    dev_rows = prepare_dev_rows(raw_dev)
    dataset_rows = len(train_rows)
    global_weight_sum = sum(float(row["sample_weight"]) for row in train_rows)
    if dataset_rows == 0 or global_weight_sum <= 0:
        raise ValueError("Empty or zero-weight training dataset")

    model, tokenizer, model_mode = load_model_and_tokenizer(config)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Exp27P GPU training requires CUDA")
    model.to(device)
    dev_loader = make_loader(dev_rows, tokenizer, config, train=False)

    batches_per_epoch = math.ceil(dataset_rows / config.per_device_train_batch_size)
    total_micro_batches = max(1, math.ceil(config.num_train_epochs * batches_per_epoch))
    full_epochs, partial_batches = divmod(total_micro_batches, batches_per_epoch)
    total_update_steps = full_epochs * math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    if partial_batches:
        total_update_steps += math.ceil(partial_batches / config.gradient_accumulation_steps)
    total_update_steps = max(1, total_update_steps)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_update_steps * config.warmup_ratio),
        num_training_steps=total_update_steps,
    )

    history: list[dict[str, Any]] = []
    checkpoint_paths: dict[int, Path] = {}
    global_step = 0
    processed_batches = 0
    zero_weight_window_count = 0
    start = time.time()
    progress = StepProgressBar(total_update_steps, config.progress_bar, desc=f"Exp27P {config.variant}")
    optimizer.zero_grad(set_to_none=True)
    try:
        epoch = 0
        while processed_batches < total_micro_batches:
            epoch += 1
            loader = make_loader(train_rows, tokenizer, config, train=True, epoch=epoch - 1)
            epoch_limit = min(len(loader), total_micro_batches - processed_batches)
            model.train()
            running_loss = 0.0
            iterator = iter(loader)
            step = 0
            while step < epoch_limit:
                window_size = min(config.gradient_accumulation_steps, epoch_limit - step)
                window_weight = 0.0
                for _ in range(window_size):
                    batch = next(iterator)
                    batch.pop("metadata")
                    targets = batch.pop("soft_targets").to(device)
                    weights = batch.pop("sample_weights").to(device)
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    logits = output_logits(model(**inputs))
                    loss = global_scaled_soft_ce(logits, targets, weights, dataset_rows, global_weight_sum)
                    (loss / window_size).backward()
                    running_loss += float(loss.detach().cpu())
                    window_weight += float(weights.sum().detach().cpu())
                    step += 1
                    processed_batches += 1
                if window_weight <= 0:
                    optimizer.zero_grad(set_to_none=True)
                    zero_weight_window_count += 1
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.log_steps == 0 or global_step == total_update_steps:
                    elapsed = time.time() - start
                    rate = global_step / max(elapsed, 1e-9)
                    eta = (total_update_steps - global_step) / max(rate, 1e-9)
                    progress.render(
                        global_step,
                        epoch,
                        running_loss / max(1, step),
                        scheduler.get_last_lr()[0],
                        elapsed,
                        eta,
                    )
            progress.newline()
            dev_predictions = evaluate(model, dev_loader, device)
            metrics = {
                "variant": config.variant,
                "seed": config.seed,
                "epoch": epoch,
                "global_step": global_step,
                "processed_micro_batches": processed_batches,
                "train_loss_batch_estimator_mean": running_loss / max(1, epoch_limit),
                "zero_weight_window_count": zero_weight_window_count,
                **prediction_metrics(dev_predictions),
            }
            history.append(metrics)
            _, _, eligible_epochs = select_checkpoint(history)
            if epoch in eligible_epochs:
                path = config.checkpoint_output_dir / f"epoch_{epoch:02d}"
                save_checkpoint(model, tokenizer, path, config, model_mode, metrics)
                if model_mode != "fallback_last_token_classifier":
                    (path / "state_dict.pt").unlink(missing_ok=True)
                checkpoint_paths[epoch] = path
            for saved_epoch in list(checkpoint_paths):
                if saved_epoch not in eligible_epochs:
                    shutil.rmtree(checkpoint_paths.pop(saved_epoch), ignore_errors=True)
            write_json(config.run_dir / "epoch_metrics.json", history)
            print(
                f"[exp27p] {config.variant} epoch={epoch} MAE={metrics['MAE_argmax']:.4f} "
                f"QWK={metrics['QWK']:.4f} l2h={metrics['low_to_high_count']}/{metrics['low_n']}",
                flush=True,
            )
    finally:
        progress.close()

    selected, pure_min, eligible_epochs = select_checkpoint(history)
    selected_epoch = int(selected["epoch"])
    selected_dir = checkpoint_paths.get(selected_epoch)
    if selected_dir is None or not selected_dir.exists():
        raise RuntimeError(f"Selected checkpoint is missing: epoch {selected_epoch}")
    reload_config = replace(config, checkpoint_dir=selected_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selected_model, selected_tokenizer, selected_mode = load_model_and_tokenizer(reload_config)
    selected_tokenizer.padding_side = "right"
    selected_tokenizer.truncation_side = "right"
    selected_model.to(device)
    selected_dev_loader = make_loader(dev_rows, selected_tokenizer, config, train=False)
    selected_dev_predictions = evaluate(selected_model, selected_dev_loader, device)
    reloaded_metrics = prediction_metrics(selected_dev_predictions)
    for key in ("MAE_argmax", "MAE_expected", "QWK", "low_to_high_rate"):
        if abs(float(reloaded_metrics[key]) - float(selected[key])) > 1e-8:
            raise RuntimeError(f"Checkpoint reload changed {key}: {reloaded_metrics[key]} vs {selected[key]}")

    selected_train_loader = make_loader(train_rows, selected_tokenizer, config, train=False)
    selected_train_predictions = evaluate(selected_model, selected_train_loader, device)
    with config.high_impact_manifest.open("r", encoding="utf-8", newline="") as handle:
        import csv

        manifest = list(csv.DictReader(handle))
    tier_rows = diagnostics_by_tier(selected_train_predictions, config.variant, config.seed)
    high_rows = high_impact_diagnostics(selected_train_predictions, manifest, config.variant, config.seed)
    matched_row = matched_dev_diagnostic(selected_dev_predictions, manifest, config.variant, config.seed)
    strata_rows = stratified_metrics(selected_dev_predictions, config.variant, config.seed)

    write_jsonl(config.run_dir / "predictions_private" / "selected_dev_predictions.jsonl", selected_dev_predictions)
    write_jsonl(config.run_dir / "predictions_private" / "selected_train_predictions.jsonl", selected_train_predictions)
    write_csv(config.run_dir / "train_tier_fit_diagnostics.csv", tier_rows)
    write_csv(config.run_dir / "high_impact16_fit_diagnostics.csv", high_rows)
    write_csv(config.run_dir / "dev_high_impact_matched_strata.csv", [matched_row])
    write_csv(config.run_dir / "dev_stratified_metrics.csv", strata_rows)
    summary = {
        "status": "COMPLETED",
        "variant": config.variant,
        "seed": config.seed,
        "model_mode": selected_mode,
        "dataset_rows": dataset_rows,
        "global_weight_sum": global_weight_sum,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(selected_dir),
        "selected_metrics": selected,
        "pure_min_mae_epoch": int(pure_min["epoch"]),
        "pure_min_mae_metrics": pure_min,
        "eligible_epochs": eligible_epochs,
        "zero_weight_window_count": zero_weight_window_count,
        "checkpoint_reload_pass": True,
        "test_access_count": 0,
        "test_predictions_generated": False,
        "config": asdict(config),
    }
    write_json(config.run_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
