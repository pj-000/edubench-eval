"""Train or reuse one Exp4 scoring-objective variant."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    compute_metrics,
    high_score_metrics,
    low_score_metrics,
    per_bin_metrics,
)
from thesis_exp.src.edujudge.exp04 import (
    A4_FIXED_DATASET_DIR,
    EXP03_A4_RUN_DIR,
    OBJECTIVE_LABELS,
    checkpoint_dir,
    ensure_exp04_dirs,
    run_dir,
)
from thesis_exp.src.edujudge.exp04.build_exp04_dataset import build_exp04_dataset
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_jsonl, write_text


LABELS = [1, 2, 3, 4, 5]
ORDINAL_THRESHOLDS = [1, 2, 3, 4]


@dataclass
class ObjectiveTrainConfig:
    objective_id: str
    objective_type: str
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
    eval_only: bool
    checkpoint_dir: Path | None
    regression_loss: str
    selection_metric: str
    selection_mode: str
    num_workers: int
    log_steps: int
    progress_bar: bool


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    logits: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    target_values: np.ndarray
    record_ids: list[str]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_record_ids(path: Path, record_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record_ids) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class StepProgressBar:
    def __init__(self, total: int, enabled: bool, desc: str) -> None:
        self.total = max(1, total)
        self.enabled = enabled
        self.desc = desc
        self.stream: TextIO | None = None
        self.owns_stream = False
        self.last_current = 0
        if not enabled:
            return
        try:
            self.stream = open("/dev/tty", "w", encoding="utf-8", buffering=1)
            self.owns_stream = True
        except OSError:
            self.stream = sys.stderr

    def render(self, current: int, epoch: int, loss: float | None, lr: float | None, elapsed: float, eta: float) -> None:
        if not self.enabled or self.stream is None:
            return
        self.last_current = current
        ratio = min(1.0, max(0.0, current / self.total))
        width = 28
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        loss_text = "nan" if loss is None else f"{loss:.4f}"
        lr_text = "nan" if lr is None else f"{lr:.2e}"
        line = (
            f"{self.desc} |{bar}| {current}/{self.total} {ratio * 100:5.1f}% "
            f"epoch={epoch} loss={loss_text} lr={lr_text} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
        )
        self.stream.write("\r" + line + "\x1b[K")
        self.stream.flush()

    def newline(self) -> None:
        if self.enabled and self.stream is not None and self.last_current:
            self.stream.write("\n")
            self.stream.flush()

    def close(self) -> None:
        if self.enabled and self.stream is not None:
            if self.last_current:
                self.stream.write("\n")
                self.stream.flush()
            if self.owns_stream:
                self.stream.close()


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit else rows


def ensure_data_exists(data_dir: Path) -> None:
    if not all((data_dir / f"{split}.jsonl").exists() for split in ["train", "dev", "test"]):
        build_exp04_dataset()


class JsonlTextDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def select_dtype(config: ObjectiveTrainConfig, torch: Any) -> Any:
    if config.fp16:
        return torch.float16
    if config.bf16 == "true":
        return torch.bfloat16
    if config.bf16 == "false":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def make_last_token_head_class(torch: Any, nn: Any):
    class LastTokenScoringHead(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int, objective_type: str, regression_loss: str) -> None:
            super().__init__()
            self.backbone = backbone
            self.objective_type = objective_type
            self.regression_loss = regression_loss
            self.num_outputs = 1 if objective_type == "regression" else 4
            self.score = nn.Linear(hidden_size, self.num_outputs)
            self.hidden_size = int(hidden_size)
            self.config = backbone.config
            self.config.num_labels = self.num_outputs
            self.config.problem_type = "regression" if objective_type == "regression" else "multi_label_classification"

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        def save_pretrained(self, output_dir: str | Path) -> None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.backbone.config.save_pretrained(output_dir)
            torch.save(self.state_dict(), output_dir / "state_dict.pt")

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
                if self.objective_type == "regression":
                    pred = logits.squeeze(-1).float()
                    target = labels.float()
                    if self.regression_loss == "mse":
                        loss = nn.MSELoss()(pred, target)
                    else:
                        loss = nn.SmoothL1Loss()(pred, target)
                else:
                    loss = nn.BCEWithLogitsLoss()(logits.float(), labels.float())
            return {"loss": loss, "logits": logits}

    return LastTokenScoringHead


def build_model(torch: Any, nn: Any, AutoModel: Any, config: ObjectiveTrainConfig, dtype: Any):
    source = str(config.model_name_or_path)
    checkpoint_metadata_path = config.checkpoint_dir / "exp04_head_metadata.json" if config.checkpoint_dir else None
    if checkpoint_metadata_path and checkpoint_metadata_path.exists():
        metadata = read_json(checkpoint_metadata_path)
        source = metadata["base_model_name_or_path"]
    backbone = AutoModel.from_pretrained(
        source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for Exp4 scoring head.")
    head_cls = make_last_token_head_class(torch, nn)
    model = head_cls(backbone, int(hidden_size), config.objective_type, config.regression_loss)
    if config.checkpoint_dir:
        state_dict_path = config.checkpoint_dir / "state_dict.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Missing checkpoint state dict: {relpath(state_dict_path)}")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model


def load_model_and_tokenizer(config: ObjectiveTrainConfig):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    dtype = select_dtype(config, torch)
    tokenizer_source = config.checkpoint_dir or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model = build_model(torch, nn, AutoModel, config, dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def ordinal_targets(label_5: int) -> list[float]:
    return [1.0 if label_5 > threshold else 0.0 for threshold in ORDINAL_THRESHOLDS]


def collate_fn(tokenizer: Any, config: ObjectiveTrainConfig):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=config.max_length,
            padding=True,
            return_tensors="pt",
        )
        if config.objective_type == "regression":
            encoded["labels"] = torch.tensor([float(row["human_mean_5"]) for row in batch], dtype=torch.float32)
        else:
            encoded["labels"] = torch.tensor([ordinal_targets(int(row["label_5"])) for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return collate


def make_dataloader(rows: list[dict[str, Any]], tokenizer: Any, config: ObjectiveTrainConfig, split: str, shuffle: bool):
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
        collate_fn=collate_fn(tokenizer, config),
    )


def get_loss_logits(outputs: Any) -> tuple[Any, Any]:
    if isinstance(outputs, dict):
        return outputs.get("loss"), outputs["logits"]
    return outputs.loss, outputs.logits


def round_half_up(score: float) -> int:
    return int(min(5, max(1, math.floor(float(score) + 0.5))))


def monotonic_violation(probs: list[float]) -> bool:
    return any(probs[idx] < probs[idx + 1] for idx in range(len(probs) - 1))


def add_extra_metrics(metrics: dict[str, Any], predictions: list[dict[str, Any]], objective_type: str) -> dict[str, Any]:
    if not predictions:
        return metrics
    y_true = np.array([int(row["label_5"]) for row in predictions], dtype=int)
    y_pred = np.array([int(row["pred_label_5"]) for row in predictions], dtype=int)
    score = np.array([float(row["pred_score_expected"]) for row in predictions], dtype=float)
    human_mean = np.array([float(row["human_mean_5"]) for row in predictions], dtype=float)
    metrics["MAE_score"] = float(np.mean(np.abs(score - human_mean)))
    metrics["severe_error_rate"] = float(np.mean(np.abs(y_pred - y_true) >= 2))
    for label in LABELS:
        metrics[f"pred_label_{label}_rate"] = float(np.mean(y_pred == label))
    if objective_type == "ordinal":
        violations = [bool(row.get("monotonic_violation")) for row in predictions]
        metrics["monotonic_violation_rate"] = float(np.mean(violations)) if violations else float("nan")
    else:
        metrics["monotonic_violation_rate"] = float("nan")
    return metrics


def evaluate(model: Any, dataloader: Any, device: Any, split: str, config: ObjectiveTrainConfig) -> EvalResult:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    logits_chunks = []
    probs_chunks = []
    labels = []
    target_values = []
    record_ids = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, logits = get_loss_logits(outputs)
            if loss is not None:
                total_loss += float(loss.detach().cpu())
            logits_cpu = logits.float().detach().cpu()
            logits_chunks.append(logits_cpu.numpy())
            if config.objective_type == "regression":
                raw_scores = logits_cpu.squeeze(-1).tolist()
                clipped_scores = [float(min(5.0, max(1.0, score))) for score in raw_scores]
                probs_cpu = torch.tensor([[score] for score in clipped_scores], dtype=torch.float32)
                probs_chunks.append(probs_cpu.numpy())
                for row, raw_score, score in zip(metadata, raw_scores, clipped_scores):
                    pred_label = round_half_up(score)
                    human_mean = float(row["human_mean_5"])
                    label_5 = int(row["label_5"])
                    prediction = base_prediction_row(split, row, label_5, human_mean, pred_label, score, score)
                    prediction["pred_score_raw"] = float(raw_score)
                    prediction["regression_loss"] = config.regression_loss
                    predictions.append(prediction)
                    labels.append(label_5)
                    target_values.append(human_mean)
                    record_ids.append(str(row.get("record_id") or row.get("id")))
            else:
                probs_cpu = torch.sigmoid(logits_cpu)
                probs_chunks.append(probs_cpu.numpy())
                for row, probs, logit in zip(metadata, probs_cpu.tolist(), logits_cpu.tolist()):
                    pred_label = 1 + sum(1 for prob in probs if prob > 0.5)
                    expected_score = 1.0 + float(sum(probs))
                    human_mean = float(row["human_mean_5"])
                    label_5 = int(row["label_5"])
                    prediction = base_prediction_row(split, row, label_5, human_mean, pred_label, float(pred_label), expected_score)
                    prediction["confidence"] = float(np.mean([abs(prob - 0.5) * 2 for prob in probs]))
                    prediction["monotonic_violation"] = monotonic_violation(probs)
                    for idx, threshold in enumerate(ORDINAL_THRESHOLDS):
                        prediction[f"prob_gt_{threshold}"] = float(probs[idx])
                        prediction[f"logit_gt_{threshold}"] = float(logit[idx])
                    predictions.append(prediction)
                    labels.append(label_5)
                    target_values.append(label_5)
                    record_ids.append(str(row.get("record_id") or row.get("id")))
            steps += 1
    logits_arr = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 1), dtype=np.float32)
    probs_arr = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.empty((0, 1), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    targets_arr = np.array(target_values, dtype=np.float32)
    metrics = compute_metrics(predictions)
    metrics = add_extra_metrics(metrics, predictions, config.objective_type)
    metrics["split"] = split
    metrics["loss"] = total_loss / steps if steps else float("nan")
    return EvalResult(metrics, predictions, logits_arr, probs_arr, labels_arr, targets_arr, record_ids)


def base_prediction_row(
    split: str,
    row: dict[str, Any],
    label_5: int,
    human_mean: float,
    pred_label: int,
    pred_score: float,
    expected_score: float,
) -> dict[str, Any]:
    return {
        "split": split,
        "id": row["id"],
        "record_id": row.get("record_id"),
        "triple_key": row.get("triple_key"),
        "label_5": label_5,
        "human_mean_5": human_mean,
        "pred_label_5": int(pred_label),
        "pred_score_5": float(pred_score),
        "pred_score_expected": float(expected_score),
        "confidence": float("nan"),
        "abs_error_label": abs(float(pred_label) - human_mean),
        "signed_error_label": float(pred_label) - human_mean,
        "abs_error_expected": abs(float(expected_score) - human_mean),
        "signed_error_expected": float(expected_score) - human_mean,
        "metric_canonical": row.get("metric_canonical"),
        "scenario_canonical": row.get("scenario_canonical"),
        "subject_canonical": row.get("subject_canonical"),
        "education_level_canonical": row.get("education_level_canonical"),
        "language": row.get("language"),
        "generator_model": row.get("generator_model"),
    }


def grouped_metrics(predictions: list[dict[str, Any]], split: str, group_field: str, objective_type: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get(group_field) or "unknown"), []).append(row)
    rows = []
    for group_value, group_rows in sorted(grouped.items()):
        metrics = compute_metrics(group_rows)
        metrics = add_extra_metrics(metrics, group_rows, objective_type)
        rows.append({"split": split, group_field: group_value, **metrics})
    return rows


def score_distribution_rows(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    y_pred = [int(row["pred_label_5"]) for row in predictions]
    y_true = [int(row["label_5"]) for row in predictions]
    rows = []
    for label in LABELS:
        rows.append(
            {
                "split": split,
                "label_5": label,
                "true_count": sum(1 for value in y_true if value == label),
                "pred_count": sum(1 for value in y_pred if value == label),
                "pred_rate": sum(1 for value in y_pred if value == label) / len(y_pred) if y_pred else float("nan"),
            }
        )
    return rows


def save_eval_arrays(output_dir: Path, split: str, result: EvalResult) -> None:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / f"logits_{split}.npy", result.logits)
    np.save(arrays_dir / f"probs_{split}.npy", result.probs)
    np.save(arrays_dir / f"labels_{split}.npy", result.labels)
    np.save(arrays_dir / f"targets_{split}.npy", result.target_values)
    write_record_ids(arrays_dir / f"record_ids_{split}.txt", result.record_ids)


def save_dev_test_npz(output_dir: Path, dev: EvalResult, test: EvalResult) -> None:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_dir / "dev_test_arrays.npz",
        logits_dev=dev.logits,
        logits_test=test.logits,
        probs_dev=dev.probs,
        probs_test=test.probs,
        labels_dev=dev.labels,
        labels_test=test.labels,
        targets_dev=dev.target_values,
        targets_test=test.target_values,
        record_ids_dev=np.array(dev.record_ids, dtype=object),
        record_ids_test=np.array(test.record_ids, dtype=object),
    )


def save_predictions(output_dir: Path, result: EvalResult, suffix: str | None = None) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    name = f"predictions_{suffix or result.metrics['split']}.jsonl"
    write_jsonl(predictions_dir / name, result.predictions)
    write_jsonl(output_dir / name, result.predictions)


def save_metrics_tables(output_dir: Path, results: list[EvalResult], objective_type: str) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [result.metrics for result in results]
    per_bin_rows = []
    low_rows = []
    high_rows = []
    metric_rows = []
    scenario_rows = []
    distribution_rows = []
    for result in results:
        split = result.metrics["split"]
        per_bin_rows.extend(per_bin_metrics(result.predictions, split))
        low_rows.append(low_score_metrics(result.metrics, split))
        high_rows.append(high_score_metrics(result.metrics, split))
        metric_rows.extend(grouped_metrics(result.predictions, split, "metric_canonical", objective_type))
        scenario_rows.extend(grouped_metrics(result.predictions, split, "scenario_canonical", objective_type))
        distribution_rows.extend(score_distribution_rows(result.predictions, split))
    write_csv(tables_dir / "metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "per_bin_metrics.csv", per_bin_rows)
    write_csv(tables_dir / "low_score_metrics.csv", low_rows)
    write_csv(tables_dir / "high_score_metrics.csv", high_rows)
    write_csv(tables_dir / "metric_level_metrics.csv", metric_rows)
    write_csv(tables_dir / "scenario_level_metrics.csv", scenario_rows)
    write_csv(tables_dir / "score_distribution.csv", distribution_rows)
    write_csv(output_dir / "metrics.csv", summary_rows)
    write_json(output_dir / "metrics.json", summary_rows)


def refresh_tables_from_predictions(output_dir: Path, objective_type: str) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_bin_rows = []
    low_rows = []
    high_rows = []
    metric_rows = []
    scenario_rows = []
    distribution_rows = []
    for split in ["dev", "test"]:
        predictions_path = output_dir / "predictions" / f"predictions_{split}.jsonl"
        if not predictions_path.exists():
            continue
        predictions = read_jsonl(predictions_path)
        metrics = compute_metrics(predictions)
        metrics = add_extra_metrics(metrics, predictions, objective_type)
        metrics["split"] = split
        summary_rows.append(metrics)
        per_bin_rows.extend(per_bin_metrics(predictions, split))
        low_rows.append(low_score_metrics(metrics, split))
        high_rows.append(high_score_metrics(metrics, split))
        metric_rows.extend(grouped_metrics(predictions, split, "metric_canonical", objective_type))
        scenario_rows.extend(grouped_metrics(predictions, split, "scenario_canonical", objective_type))
        distribution_rows.extend(score_distribution_rows(predictions, split))
    if not summary_rows:
        return
    write_csv(tables_dir / "metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "per_bin_metrics.csv", per_bin_rows)
    write_csv(tables_dir / "low_score_metrics.csv", low_rows)
    write_csv(tables_dir / "high_score_metrics.csv", high_rows)
    write_csv(tables_dir / "metric_level_metrics.csv", metric_rows)
    write_csv(tables_dir / "scenario_level_metrics.csv", scenario_rows)
    write_csv(tables_dir / "score_distribution.csv", distribution_rows)
    write_csv(output_dir / "metrics.csv", summary_rows)
    write_json(output_dir / "metrics.json", summary_rows)


def save_metrics_history(output_dir: Path, metrics_history: list[dict[str, Any]]) -> None:
    if not metrics_history:
        return
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "dev_metrics_history.csv", metrics_history)
    write_csv(output_dir / "logs" / "training_log.csv", metrics_history)
    write_json(output_dir / "metrics_history.json", metrics_history)


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: ObjectiveTrainConfig, metrics: dict[str, Any]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    tokenizer.save_pretrained(output_dir)
    if hasattr(model.backbone, "config"):
        model.backbone.config.save_pretrained(output_dir)
    write_json(
        output_dir / "exp04_head_metadata.json",
        {
            "base_model_name_or_path": config.model_name_or_path,
            "objective_id": config.objective_id,
            "objective_type": config.objective_type,
            "regression_loss": config.regression_loss,
            "num_outputs": getattr(model, "num_outputs", None),
            "hidden_size": getattr(model, "hidden_size", None),
            "state_dict": state_dict_path.name,
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "dev_metrics.json", metrics)


def score_is_better(metrics: dict[str, Any], best_score: float | None, config: ObjectiveTrainConfig) -> bool:
    value = metrics.get(config.selection_metric)
    if value is None:
        raise KeyError(f"Selection metric {config.selection_metric!r} not found in dev metrics.")
    score = float(value)
    if best_score is None:
        return True
    if config.selection_mode == "max":
        return score > best_score
        return score < best_score


def loss_display_name(config: ObjectiveTrainConfig) -> str:
    if config.objective_type == "classification":
        return "CrossEntropyLoss"
    if config.objective_type == "regression":
        return "MSELoss" if config.regression_loss == "mse" else "SmoothL1Loss"
    return "BCEWithLogitsLoss"


def normalize_run_files(output_dir: Path, config: ObjectiveTrainConfig, status: str) -> None:
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": "Exp4 target objective comparison",
            "objective_id": config.objective_id,
            "objective_type": config.objective_type,
            "objective_label": OBJECTIVE_LABELS.get(config.objective_id, config.objective_id),
            "status": status,
            "fixed_input_template": "A4_question_answer_metric_rubric_metadata",
            "model_name_or_path": config.model_name_or_path,
            "loss": loss_display_name(config),
            "checkpoint_selection": f"dev {config.selection_metric} ({config.selection_mode})",
        },
    )

    rows = []
    metrics_path = output_dir / "tables" / "metrics_summary.csv"
    if metrics_path.exists():
        import csv

        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    test = next((row for row in rows if row.get("split") == "test"), {})
    lines = [
        "# Exp4 Target Objective Run Summary",
        "",
        f"Objective: `{config.objective_id}`",
        f"Status: `{status}`",
        f"Model: `{config.model_name_or_path}`",
        "Fixed input: `A4_question_answer_metric_rubric_metadata`",
        f"Checkpoint selection: dev `{config.selection_metric}` ({config.selection_mode})",
    ]
    if test:
        lines.extend(
            [
                "",
                "## Test Metrics",
                "",
                f"- Accuracy: {test.get('Accuracy', 'NA')}",
                f"- MAE_label: {test.get('MAE_label', 'NA')}",
                f"- MAE_expected: {test.get('MAE_expected', 'NA')}",
                f"- QWK: {test.get('Quadratic Weighted Kappa', 'NA')}",
                f"- Kendall tau: {test.get('Kendall tau', 'NA')}",
                f"- severe_error_rate: {test.get('severe_error_rate', 'NA')}",
                f"- low_to_high_rate: {test.get('low_to_high_rate', 'NA')}",
                f"- monotonic_violation_rate: {test.get('monotonic_violation_rate', 'NA')}",
            ]
        )
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def reuse_classification_baseline(output_dir: Path, model_name_or_path: str) -> dict[str, Any]:
    if not EXP03_A4_RUN_DIR.exists():
        raise FileNotFoundError(f"Exp3 A4 run not found: {relpath(EXP03_A4_RUN_DIR)}")
    required = [
        "predictions/predictions_dev.jsonl",
        "predictions/predictions_test.jsonl",
        "tables/metrics_summary.csv",
        "tables/per_bin_metrics.csv",
        "tables/low_score_metrics.csv",
        "tables/high_score_metrics.csv",
        "tables/metric_level_metrics.csv",
        "tables/scenario_level_metrics.csv",
        "arrays/dev_test_arrays.npz",
    ]
    missing = [rel for rel in required if not (EXP03_A4_RUN_DIR / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp3 A4 baseline outputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in required + ["run_summary.md"]:
        src = EXP03_A4_RUN_DIR / rel
        if copy_if_exists(src, output_dir / rel):
            copied.append(rel)
    refresh_tables_from_predictions(output_dir, "classification")
    config = ObjectiveTrainConfig(
        objective_id="O1_classification",
        objective_type="classification",
        model_name_or_path=model_name_or_path,
        data_dir=A4_FIXED_DATASET_DIR,
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint_dir("O1_classification"),
        max_length=2048,
        num_train_epochs=0.0,
        learning_rate=0.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        per_device_train_batch_size=0,
        per_device_eval_batch_size=0,
        gradient_accumulation_steps=0,
        max_grad_norm=0.0,
        seed=42,
        bf16="auto",
        fp16=False,
        gradient_checkpointing=False,
        trust_remote_code=True,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=True,
        checkpoint_dir=None,
        regression_loss="none",
        selection_metric="Exact Match",
        selection_mode="max",
        num_workers=0,
        log_steps=0,
        progress_bar=False,
    )
    normalize_run_files(output_dir, config, "reused_exp03_a4")
    return {"status": "reused_exp03_a4", "copied": copied, "output_dir": str(output_dir)}


def train(config: ObjectiveTrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp04_dirs()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_data_exists(config.data_dir)
    set_seed(config.seed)
    train_rows = limit_rows(read_jsonl(config.data_dir / "train.jsonl"), config.max_train_samples)
    dev_rows = limit_rows(read_jsonl(config.data_dir / "dev.jsonl"), config.max_eval_samples)
    test_rows = limit_rows(read_jsonl(config.data_dir / "test.jsonl"), config.max_eval_samples)

    model, tokenizer = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_dataloader(test_rows, tokenizer, config, "test", shuffle=False)
    best_dir = config.checkpoint_output_dir / "best"
    metrics_history: list[dict[str, Any]] = []

    if config.eval_only:
        dev_result = evaluate(model, dev_loader, device, "dev", config)
        test_result = evaluate(model, test_loader, device, "test", config)
        for result in [dev_result, test_result]:
            save_predictions(config.output_dir, result)
            save_eval_arrays(config.output_dir, result.metrics["split"], result)
        save_dev_test_npz(config.output_dir, dev_result, test_result)
        save_metrics_tables(config.output_dir, [dev_result, test_result], config.objective_type)
        normalize_run_files(config.output_dir, config, "eval_only")
        return {"metrics": [dev_result.metrics, test_result.metrics]}

    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_score: float | None = None
    global_step = 0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    num_epochs = int(math.ceil(config.num_train_epochs))
    print(
        "training schedule: "
        f"objective={config.objective_id} "
        f"train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} "
        f"total_update_steps={total_steps} "
        f"log_steps={config.log_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp4 {config.objective_id}")
    try:
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
                loss, _ = get_loss_logits(outputs)
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
                    now = time.time()
                    elapsed = now - start
                    avg_loss = running_loss / max(1, step)
                    avg_steps_per_second = global_step / max(1e-9, elapsed)
                    eta_seconds = max(0, total_steps - global_step) / avg_steps_per_second
                    current_lr = scheduler.get_last_lr()[0]
                    progress.render(global_step, epoch + 1, avg_loss, current_lr, elapsed, eta_seconds)
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
                            f"loss={avg_loss:.4f} lr={current_lr:.2e} "
                            f"elapsed={format_duration(elapsed)} "
                            f"eta={format_duration(remaining_steps / steps_per_second)} "
                            f"steps_per_min={steps_per_second * 60:.2f}",
                            flush=True,
                        )

            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev", config)
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            metrics_history.append(dev_result.metrics)
            elapsed = time.time() - start
            avg_seconds_per_step = elapsed / max(1, global_step)
            eta_seconds = max(0, total_steps - global_step) * avg_seconds_per_step
            selected = score_is_better(dev_result.metrics, best_score, config)
            print(
                f"dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"MAE_expected={dev_result.metrics['MAE_expected']:.4f}, "
                f"Exact={dev_result.metrics['Exact Match']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"selected={selected}, elapsed={format_duration(elapsed)}, eta={format_duration(eta_seconds)}",
                flush=True,
            )
            if selected:
                best_score = float(dev_result.metrics[config.selection_metric])
                save_checkpoint(model, tokenizer, best_dir, config, dev_result.metrics)
                save_predictions(config.output_dir, dev_result, suffix="dev_best")
    finally:
        progress.close()

    state_dict_path = best_dir / "state_dict.pt"
    if state_dict_path.exists():
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    dev_final = evaluate(model, dev_loader, device, "dev", config)
    test_result = evaluate(model, test_loader, device, "test", config)
    for result in [dev_final, test_result]:
        result.metrics["epoch"] = "best"
        result.metrics["global_step"] = global_step
        save_predictions(config.output_dir, result)
        save_eval_arrays(config.output_dir, result.metrics["split"], result)
    save_dev_test_npz(config.output_dir, dev_final, test_result)
    test_result.metrics["elapsed_seconds"] = time.time() - start
    save_metrics_history(config.output_dir, metrics_history)
    save_metrics_tables(config.output_dir, [dev_final, test_result], config.objective_type)
    normalize_run_files(config.output_dir, config, "completed")
    return {"metrics": [dev_final.metrics, test_result.metrics], "best_dir": str(best_dir)}


def default_objective_id(objective_type: str, regression_loss: str) -> str:
    if objective_type == "regression":
        return f"O2_regression_{regression_loss}"
    if objective_type == "ordinal":
        return "O3_ordinal"
    return "O1_classification"


def default_selection(objective_type: str) -> tuple[str, str]:
    if objective_type == "regression":
        return "MAE_expected", "min"
    if objective_type == "ordinal":
        return "MAE_label", "min"
    return "Exact Match", "max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or reuse one Exp4 objective.")
    parser.add_argument("--objective_type", required=True, choices=["classification", "regression", "ordinal"])
    parser.add_argument("--objective_id", default=None)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir", type=Path, default=A4_FIXED_DATASET_DIR)
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
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--regression_loss", choices=["smoothl1", "mse"], default="smoothl1")
    parser.add_argument("--selection_metric", default=None)
    parser.add_argument("--selection_mode", choices=["min", "max"], default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument("--reuse_exp03_a4", action="store_true")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ObjectiveTrainConfig:
    objective_id = args.objective_id or default_objective_id(args.objective_type, args.regression_loss)
    selection_metric, selection_mode = default_selection(args.objective_type)
    selection_metric = args.selection_metric or selection_metric
    selection_mode = args.selection_mode or selection_mode
    return ObjectiveTrainConfig(
        objective_id=objective_id,
        objective_type=args.objective_type,
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir or run_dir(objective_id),
        checkpoint_output_dir=args.checkpoint_output_dir or checkpoint_dir(objective_id),
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
        eval_only=args.eval_only,
        checkpoint_dir=args.checkpoint_dir,
        regression_loss=args.regression_loss,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )


def main() -> None:
    args = parse_args()
    ensure_exp04_dirs()
    if args.objective_type == "classification" or args.reuse_exp03_a4:
        output_dir = args.output_dir or run_dir(args.objective_id or "O1_classification")
        result = reuse_classification_baseline(output_dir, args.model_name_or_path)
    else:
        config = make_config(args)
        result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
