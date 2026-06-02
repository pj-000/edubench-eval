"""Train the Exp2 0.6B CE baseline.

The training target is the locked human ``label_5`` class from Exp0.1. This is
not a generative SFT script: it optimizes a 5-class cross-entropy objective and
reports evaluator-audit metrics directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thesis_exp.src.edujudge.exp02 import EXP02_DATA_DIR, EXP02_MODEL_DIR, EXP02_OUTPUT_DIR, EXP02_TABLES_DIR, ensure_exp02_dirs
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import build_exp02_dataset
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_jsonl, write_text


@dataclass
class TrainConfig:
    model_name_or_path: str
    data_dir: Path
    output_dir: Path
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
    eval_only: bool
    checkpoint_dir: Path | None
    num_workers: int
    log_steps: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train/evaluate Exp2 EduBenchEvaluator CE baseline.")
    parser.add_argument("--model_name_or_path", required=True, help="Base 0.6B model path or HF id.")
    parser.add_argument("--data_dir", type=Path, default=EXP02_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP02_MODEL_DIR)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=20)
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_data_exists(data_dir: Path) -> None:
    if not all((data_dir / f"{split}.jsonl").exists() for split in ["train", "dev", "test"]):
        build_exp02_dataset()


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit else rows


class JsonlTextDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def select_dtype(config: TrainConfig, torch: Any) -> Any:
    if config.fp16:
        return torch.float16
    if config.bf16 == "true":
        return torch.bfloat16
    if config.bf16 == "false":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def build_fallback_classifier(torch: Any, nn: Any, AutoModel: Any, model_name_or_path: str, config: TrainConfig, dtype: Any):
    class LastTokenClassifier(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int, num_labels: int = 5) -> None:
            super().__init__()
            self.backbone = backbone
            self.score = nn.Linear(hidden_size, num_labels)
            self.num_labels = num_labels
            self.config = backbone.config
            self.config.num_labels = num_labels
            self.config.problem_type = "single_label_classification"

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        def save_pretrained(self, output_dir: str | Path) -> None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.backbone.config.save_pretrained(output_dir)
            torch.save(self.state_dict(), output_dir / "pytorch_model.bin")
            write_json(output_dir / "classification_head.json", {"num_labels": self.num_labels, "fallback_classifier": True})

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None, **kwargs: Any) -> dict[str, Any]:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = outputs.last_hidden_state
            if attention_mask is None:
                pooled = hidden[:, -1]
            else:
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            logits = self.score(pooled)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.float(), labels)
            return {"loss": loss, "logits": logits}

    backbone = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for fallback classifier.")
    return LastTokenClassifier(backbone, int(hidden_size), num_labels=5)


def load_model_and_tokenizer(config: TrainConfig):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    dtype = select_dtype(config, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model_source = str(config.checkpoint_dir or config.model_name_or_path)
    model_mode = "sequence_classification"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=5,
            problem_type="single_label_classification",
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            torch_dtype=dtype,
        )
    except Exception as exc:
        if config.checkpoint_dir:
            raise RuntimeError(
                "AutoModelForSequenceClassification could not load checkpoint_dir. "
                "For fallback checkpoints, evaluate by passing the original base model path and the saved state is "
                "loaded only when using a fallback-aware checkpoint in this script."
            ) from exc
        print(f"[WARN] Falling back to AutoModel + last-token classifier: {type(exc).__name__}: {exc}")
        model = build_fallback_classifier(torch, nn, AutoModel, config.model_name_or_path, config, dtype)
        model_mode = "fallback_last_token_classifier"

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer, model_mode


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
        encoded["labels"] = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        encoded["metadata"] = batch
        return encoded

    return collate


def make_dataloader(rows: list[dict[str, Any]], tokenizer: Any, config: TrainConfig, split: str, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    batch_size = config.per_device_train_batch_size if split == "train" else config.per_device_eval_batch_size
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        JsonlTextDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        collate_fn=collate_fn(tokenizer, config.max_length),
    )


def _get_loss_logits(outputs: Any) -> tuple[Any, Any]:
    if isinstance(outputs, dict):
        return outputs.get("loss"), outputs["logits"]
    return outputs.loss, outputs.logits


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    f1s = []
    for label in [1, 2, 3, 4, 5]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return float(np.mean(f1s))


def kendall_tau_b(x_values: list[float], y_values: list[float]) -> float:
    n = len(x_values)
    if n < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = (x_values[j] > x_values[i]) - (x_values[j] < x_values[i])
            dy = (y_values[j] > y_values[i]) - (y_values[j] < y_values[i])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float("nan") if denom == 0 else (concordant - discordant) / denom


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    y_true = [int(row["label_5"]) for row in predictions]
    y_pred = [int(row["pred_label_5"]) for row in predictions]
    human_mean = [float(row["human_mean_5"]) for row in predictions]
    diffs = np.array(y_pred, dtype=float) - np.array(human_mean, dtype=float)
    labels = np.array(y_true, dtype=int)
    preds = np.array(y_pred, dtype=int)
    low_mask = labels <= 2
    high_mask = labels == 5
    return {
        "n": len(predictions),
        "MAE": float(np.mean(np.abs(diffs))),
        "RMSE": float(np.sqrt(np.mean(np.square(diffs)))),
        "Signed Bias": float(np.mean(diffs)),
        "Exact Match": float(np.mean(labels == preds)),
        "Within-1 Accuracy": float(np.mean(np.abs(labels - preds) <= 1)),
        "Macro-F1": macro_f1(y_true, y_pred),
        "Kendall tau": kendall_tau_b(human_mean, [float(v) for v in y_pred]),
        "low_n": int(low_mask.sum()),
        "low_exact_match": float(np.mean(labels[low_mask] == preds[low_mask])) if low_mask.any() else float("nan"),
        "low_to_high_rate": float(np.mean(preds[low_mask] >= 4)) if low_mask.any() else float("nan"),
        "high_n": int(high_mask.sum()),
        "acc_at_5": float(np.mean(preds[high_mask] == 5)) if high_mask.any() else float("nan"),
    }


def evaluate(model: Any, dataloader: Any, device: Any, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, logits = _get_loss_logits(outputs)
            if loss is not None:
                total_loss += float(loss.detach().cpu())
            pred_zero = logits.argmax(dim=-1).detach().cpu().tolist()
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
            for row, pred, prob in zip(metadata, pred_zero, probs):
                pred_label = int(pred) + 1
                predictions.append(
                    {
                        "split": split,
                        "id": row["id"],
                        "record_id": row.get("record_id"),
                        "label_5": int(row["label_5"]),
                        "human_mean_5": float(row["human_mean_5"]),
                        "pred_label_5": pred_label,
                        "pred_score_5": float(pred_label),
                        "confidence": float(max(prob)),
                        "metric_canonical": row.get("metric_canonical"),
                        "scenario_canonical": row.get("scenario_canonical"),
                        "subject_canonical": row.get("subject_canonical"),
                        "language": row.get("language"),
                        "generator_model": row.get("generator_model"),
                    }
                )
            steps += 1
    metrics = compute_metrics(predictions)
    metrics["split"] = split
    metrics["loss"] = total_loss / steps if steps else float("nan")
    return metrics, predictions


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: TrainConfig, model_mode: str, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "training_config.json", {**asdict(config), "model_mode": model_mode})
    write_json(output_dir / "dev_metrics.json", metrics)


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp02_dirs()
    ensure_data_exists(config.data_dir)
    set_seed(config.seed)
    train_rows = limit_rows(read_jsonl(config.data_dir / "train.jsonl"), config.max_train_samples)
    dev_rows = limit_rows(read_jsonl(config.data_dir / "dev.jsonl"), config.max_eval_samples)
    test_rows = limit_rows(read_jsonl(config.data_dir / "test.jsonl"), config.max_eval_samples)

    model, tokenizer, model_mode = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_dataloader(test_rows, tokenizer, config, "test", shuffle=False)

    best_dir = config.output_dir / "best"
    metrics_history: list[dict[str, Any]] = []
    if config.eval_only:
        for split, loader in [("dev", dev_loader), ("test", test_loader)]:
            metrics, predictions = evaluate(model, loader, device, split)
            write_jsonl(config.output_dir / f"predictions_{split}.jsonl", predictions)
            metrics_history.append(metrics)
        write_csv(config.output_dir / "metrics.csv", metrics_history)
        write_json(config.output_dir / "metrics.json", metrics_history)
        return {"model_mode": model_mode, "metrics": metrics_history}

    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_dev_mae = float("inf")
    global_step = 0
    start = time.time()
    num_epochs = int(math.ceil(config.num_train_epochs))
    for epoch in range(num_epochs):
        if epoch >= config.num_train_epochs:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("metadata")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, _ = _get_loss_logits(outputs)
            if loss is None:
                raise RuntimeError("Model returned no loss.")
            (loss / config.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach().cpu())
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.log_steps == 0:
                    avg_loss = running_loss / max(1, step)
                    print(
                        f"epoch={epoch + 1} step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        dev_metrics, dev_predictions = evaluate(model, dev_loader, device, "dev")
        dev_metrics["epoch"] = epoch + 1
        dev_metrics["global_step"] = global_step
        metrics_history.append(dev_metrics)
        print(
            f"dev epoch={epoch + 1}: MAE={dev_metrics['MAE']:.4f}, "
            f"Exact={dev_metrics['Exact Match']:.4f}, low_to_high={dev_metrics['low_to_high_rate']:.4f}"
        )
        if dev_metrics["MAE"] < best_dev_mae:
            best_dev_mae = dev_metrics["MAE"]
            save_checkpoint(model, tokenizer, best_dir, config, model_mode, dev_metrics)
            write_jsonl(config.output_dir / "predictions_dev_best.jsonl", dev_predictions)

    test_metrics, test_predictions = evaluate(model, test_loader, device, "test")
    test_metrics["epoch"] = "final"
    test_metrics["global_step"] = global_step
    test_metrics["elapsed_seconds"] = time.time() - start
    metrics_history.append(test_metrics)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "predictions_test.jsonl", test_predictions)
    write_csv(config.output_dir / "metrics.csv", metrics_history)
    write_json(config.output_dir / "metrics.json", metrics_history)
    write_text(
        config.output_dir / "run_summary.md",
        "\n".join(
            [
                "# Exp2 CE Baseline Run Summary",
                "",
                f"Model: `{config.model_name_or_path}`",
                f"Model mode: `{model_mode}`",
                f"Best checkpoint: `{relpath(best_dir)}`",
                f"Final test MAE: {test_metrics['MAE']:.6f}",
                f"Final test Exact Match: {test_metrics['Exact Match']:.6f}",
                f"Final low-to-high rate: {test_metrics['low_to_high_rate']:.6f}",
            ]
        ),
    )
    return {"model_mode": model_mode, "metrics": metrics_history, "best_dir": str(best_dir)}


def main() -> None:
    config = parse_args()
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
