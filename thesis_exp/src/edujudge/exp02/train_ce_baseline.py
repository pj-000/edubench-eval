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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from thesis_exp.src.edujudge.exp02 import (
    EXP02_ARRAYS_DIR,
    EXP02_CHECKPOINT_DIR,
    EXP02_DATA_DIR,
    EXP02_LOG_DIR,
    EXP02_OUTPUT_DIR,
    EXP02_PREDICTIONS_DIR,
    EXP02_SUMMARIES_DIR,
    EXP02_TABLES_DIR,
    ensure_exp02_dirs,
)
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import build_exp02_dataset
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_jsonl, write_text


LABELS = [1, 2, 3, 4, 5]


@dataclass
class TrainConfig:
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
    num_workers: int
    log_steps: int
    progress_bar: bool
    evaluate_test: bool


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    predictions: list[dict[str, Any]]
    logits: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    record_ids: list[str]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train/evaluate Exp2 EduBenchEvaluator CE baseline.")
    parser.add_argument("--model_name_or_path", required=True, help="Base 0.6B model path or HF id.")
    parser.add_argument("--data_dir", type=Path, default=EXP02_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP02_OUTPUT_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=EXP02_CHECKPOINT_DIR)
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
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument(
        "--dev_only",
        action="store_false",
        dest="evaluate_test",
        help="Train/select on dev without reading or evaluating the held-out test split.",
    )
    parser.set_defaults(progress_bar=True, evaluate_test=True)
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


def write_record_ids(path: Path, record_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record_ids) + "\n", encoding="utf-8")


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
    def __init__(self, total: int, enabled: bool, desc: str = "Exp2 train") -> None:
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


def ensure_data_exists(data_dir: Path, evaluate_test: bool = True) -> None:
    required = ["train", "dev", "test"] if evaluate_test else ["train", "dev"]
    if not all((data_dir / f"{split}.jsonl").exists() for split in required):
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


def make_last_token_classifier_class(torch: Any, nn: Any):
    class LastTokenClassifier(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int, num_labels: int = 5) -> None:
            super().__init__()
            self.backbone = backbone
            self.score = nn.Linear(hidden_size, num_labels)
            self.hidden_size = int(hidden_size)
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
                loss = nn.CrossEntropyLoss()(logits.float(), labels)
            return {"loss": loss, "logits": logits}

    return LastTokenClassifier


def build_fallback_classifier(
    torch: Any,
    nn: Any,
    AutoModel: Any,
    base_model_name_or_path: str,
    config: TrainConfig,
    dtype: Any,
):
    backbone = AutoModel.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for fallback classifier.")
    classifier_cls = make_last_token_classifier_class(torch, nn)
    return classifier_cls(backbone, int(hidden_size), num_labels=5)


def load_model_and_tokenizer(config: TrainConfig):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    dtype = select_dtype(config, torch)
    checkpoint_dir = config.checkpoint_dir
    fallback_metadata_path = checkpoint_dir / "fallback_metadata.json" if checkpoint_dir else None

    if fallback_metadata_path and fallback_metadata_path.exists():
        metadata = json.loads(fallback_metadata_path.read_text(encoding="utf-8"))
        base_model_name_or_path = metadata["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model = build_fallback_classifier(torch, nn, AutoModel, base_model_name_or_path, config, dtype)
        state_dict_path = checkpoint_dir / "state_dict.pt"
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
        model_mode = "fallback_last_token_classifier"
    else:
        tokenizer_source = checkpoint_dir or config.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model_source = str(checkpoint_dir or config.model_name_or_path)
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
            if checkpoint_dir:
                raise RuntimeError(
                    "Could not load checkpoint_dir as a Hugging Face sequence-classification checkpoint "
                    "and no fallback_metadata.json was found."
                ) from exc
            print(f"[WARN] Falling back to AutoModel + last-token classifier: {type(exc).__name__}: {exc}")
            model = build_fallback_classifier(torch, nn, AutoModel, config.model_name_or_path, config, dtype)
            model_mode = "fallback_last_token_classifier"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
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


def macro_weighted_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float]:
    f1s = []
    weights = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
        weights.append(sum(1 for t in y_true if t == label))
    macro = float(np.mean(f1s))
    weighted = float(np.average(f1s, weights=weights)) if sum(weights) else float("nan")
    return macro, weighted


def qwk(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("nan")
    observed = np.zeros((5, 5), dtype=float)
    for true, pred in zip(y_true, y_pred):
        observed[int(true) - 1, int(pred) - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            weights[i, j] = ((i - j) ** 2) / 16
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return float("nan")
    return 1 - float((weights * observed).sum()) / denominator


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


def rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg
        i = j + 1
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(np.array(x_values, dtype=float), np.array(y_values, dtype=float))[0, 1])


def spearman(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return float("nan")
    return pearson(rank_average(x_values), rank_average(y_values))


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {}
    y_true = np.array([int(row["label_5"]) for row in predictions], dtype=int)
    y_pred = np.array([int(row["pred_label_5"]) for row in predictions], dtype=int)
    y_expected = np.array([float(row["pred_score_expected"]) for row in predictions], dtype=float)
    human_mean = np.array([float(row["human_mean_5"]) for row in predictions], dtype=float)
    label_diffs = y_pred.astype(float) - human_mean
    expected_diffs = y_expected - human_mean
    macro_f1, weighted_f1 = macro_weighted_f1(y_true.tolist(), y_pred.tolist())
    low_mask = y_true <= 2
    high_mask = y_true == 5
    return {
        "n": len(predictions),
        "Accuracy": float(np.mean(y_true == y_pred)),
        "Exact Match": float(np.mean(y_true == y_pred)),
        "MAE_label": float(np.mean(np.abs(label_diffs))),
        "MAE_expected": float(np.mean(np.abs(expected_diffs))),
        "RMSE_label": float(np.sqrt(np.mean(np.square(label_diffs)))),
        "Signed Bias label": float(np.mean(label_diffs)),
        "Signed Bias expected": float(np.mean(expected_diffs)),
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Quadratic Weighted Kappa": qwk(y_true.tolist(), y_pred.tolist()),
        "Kendall tau": kendall_tau_b(human_mean.tolist(), y_pred.astype(float).tolist()),
        "Spearman rho": spearman(human_mean.tolist(), y_pred.astype(float).tolist()),
        "Within-1 Accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "Acc@1": float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan"),
        "Acc@2": float(np.mean(y_pred[y_true == 2] == 2)) if np.any(y_true == 2) else float("nan"),
        "Acc@5": float(np.mean(y_pred[y_true == 5] == 5)) if np.any(y_true == 5) else float("nan"),
        "low_n": int(low_mask.sum()),
        "low_exact_match": float(np.mean(y_true[low_mask] == y_pred[low_mask])) if low_mask.any() else float("nan"),
        "low_MAE": float(np.mean(np.abs(label_diffs[low_mask]))) if low_mask.any() else float("nan"),
        "low_signed_bias": float(np.mean(label_diffs[low_mask])) if low_mask.any() else float("nan"),
        "low_to_high_rate": float(np.mean(y_pred[low_mask] >= 4)) if low_mask.any() else float("nan"),
        "low_severe_overestimation_rate": float(np.mean((y_pred[low_mask] - y_true[low_mask]) >= 2)) if low_mask.any() else float("nan"),
        "mean_pred_low": float(np.mean(y_pred[low_mask])) if low_mask.any() else float("nan"),
        "high_n": int(high_mask.sum()),
        "high_to_mid_or_low_rate": float(np.mean(y_pred[high_mask] <= 3)) if high_mask.any() else float("nan"),
        "high_signed_bias": float(np.mean(label_diffs[high_mask])) if high_mask.any() else float("nan"),
    }


def per_bin_metrics(predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = []
    for label in LABELS:
        subset = [row for row in predictions if int(row["label_5"]) == label]
        if not subset:
            rows.append({"split": split, "label_5": label, "n": 0})
            continue
        y_pred = np.array([int(row["pred_label_5"]) for row in subset], dtype=int)
        y_expected = np.array([float(row["pred_score_expected"]) for row in subset], dtype=float)
        human_mean = np.array([float(row["human_mean_5"]) for row in subset], dtype=float)
        rows.append(
            {
                "split": split,
                "label_5": label,
                "n": len(subset),
                "accuracy": float(np.mean(y_pred == label)),
                "MAE": float(np.mean(np.abs(y_pred.astype(float) - human_mean))),
                "signed_bias": float(np.mean(y_pred.astype(float) - human_mean)),
                "mean_pred_label": float(np.mean(y_pred)),
                "mean_pred_expected": float(np.mean(y_expected)),
                "overestimation_rate": float(np.mean(y_pred > label)),
                "underestimation_rate": float(np.mean(y_pred < label)),
            }
        )
    return rows


def low_score_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    keys = [
        "low_n",
        "Acc@1",
        "Acc@2",
        "low_exact_match",
        "low_MAE",
        "low_signed_bias",
        "low_to_high_rate",
        "low_severe_overestimation_rate",
        "mean_pred_low",
    ]
    return {"split": split, **{key: metrics.get(key) for key in keys}}


def high_score_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    keys = ["high_n", "Acc@5", "high_to_mid_or_low_rate", "high_signed_bias"]
    return {"split": split, **{key: metrics.get(key) for key in keys}}


def evaluate(model: Any, dataloader: Any, device: Any, split: str) -> EvalResult:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    logits_chunks = []
    probs_chunks = []
    labels = []
    record_ids = []
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
            logits_cpu = logits.float().detach().cpu()
            probs_cpu = torch.softmax(logits_cpu, dim=-1)
            pred_zero = logits_cpu.argmax(dim=-1).tolist()
            logits_chunks.append(logits_cpu.numpy())
            probs_chunks.append(probs_cpu.numpy())
            for row, pred, prob, logit in zip(metadata, pred_zero, probs_cpu.tolist(), logits_cpu.tolist()):
                pred_label = int(pred) + 1
                expected_score = float(sum((idx + 1) * value for idx, value in enumerate(prob)))
                human_mean = float(row["human_mean_5"])
                label_5 = int(row["label_5"])
                prediction = {
                    "split": split,
                    "id": row["id"],
                    "record_id": row.get("record_id"),
                    "triple_key": row.get("triple_key"),
                    "label_5": label_5,
                    "human_mean_5": human_mean,
                    "pred_label_5": pred_label,
                    "pred_score_5": float(pred_label),
                    "pred_score_expected": expected_score,
                    "confidence": float(max(prob)),
                    "abs_error_label": abs(float(pred_label) - human_mean),
                    "signed_error_label": float(pred_label) - human_mean,
                    "abs_error_expected": abs(expected_score - human_mean),
                    "signed_error_expected": expected_score - human_mean,
                    "metric_canonical": row.get("metric_canonical"),
                    "scenario_canonical": row.get("scenario_canonical"),
                    "subject_canonical": row.get("subject_canonical"),
                    "language": row.get("language"),
                    "generator_model": row.get("generator_model"),
                }
                for idx in range(5):
                    prediction[f"prob_{idx + 1}"] = float(prob[idx])
                    prediction[f"logit_{idx + 1}"] = float(logit[idx])
                predictions.append(prediction)
                labels.append(label_5)
                record_ids.append(str(row.get("record_id") or row.get("id")))
            steps += 1
    logits_arr = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 5), dtype=np.float32)
    probs_arr = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.empty((0, 5), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    metrics = compute_metrics(predictions)
    metrics["split"] = split
    metrics["loss"] = total_loss / steps if steps else float("nan")
    return EvalResult(metrics, predictions, logits_arr, probs_arr, labels_arr, record_ids)


def save_eval_arrays(output_dir: Path, split: str, result: EvalResult) -> None:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / f"logits_{split}.npy", result.logits)
    np.save(arrays_dir / f"probs_{split}.npy", result.probs)
    np.save(arrays_dir / f"labels_{split}.npy", result.labels)
    write_record_ids(arrays_dir / f"record_ids_{split}.txt", result.record_ids)


def save_dev_test_npz(output_dir: Path, dev: EvalResult, test: EvalResult) -> None:
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_dir / "exp02_dev_test_arrays.npz",
        logits_dev=dev.logits,
        logits_test=test.logits,
        probs_dev=dev.probs,
        probs_test=test.probs,
        labels_dev=dev.labels,
        labels_test=test.labels,
        record_ids_dev=np.array(dev.record_ids, dtype=object),
        record_ids_test=np.array(test.record_ids, dtype=object),
    )


def save_metrics_tables(output_dir: Path, results: list[EvalResult]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [result.metrics for result in results]
    per_bin_rows = []
    low_rows = []
    high_rows = []
    for result in results:
        split = result.metrics["split"]
        per_bin_rows.extend(per_bin_metrics(result.predictions, split))
        low_rows.append(low_score_metrics(result.metrics, split))
        high_rows.append(high_score_metrics(result.metrics, split))
    write_csv(tables_dir / "metrics_summary.csv", summary_rows)
    write_csv(tables_dir / "per_bin_metrics.csv", per_bin_rows)
    write_csv(tables_dir / "low_score_metrics.csv", low_rows)
    write_csv(tables_dir / "high_score_metrics.csv", high_rows)
    write_csv(output_dir / "metrics.csv", summary_rows)
    write_json(output_dir / "metrics.json", summary_rows)


def save_metrics_history(output_dir: Path, metrics_history: list[dict[str, Any]]) -> None:
    if not metrics_history:
        return
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "dev_metrics_history.csv", metrics_history)
    write_json(output_dir / "metrics_history.json", metrics_history)


def save_predictions(output_dir: Path, result: EvalResult, suffix: str | None = None) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    name = f"predictions_{suffix or result.metrics['split']}.jsonl"
    write_jsonl(predictions_dir / name, result.predictions)
    write_jsonl(output_dir / name, result.predictions)


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: TrainConfig, model_mode: str, metrics: dict[str, Any]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / "state_dict.pt"
    if model_mode == "fallback_last_token_classifier":
        torch.save(model.state_dict(), state_dict_path)
        metadata = {
            "base_model_name_or_path": config.model_name_or_path,
            "model_mode": model_mode,
            "hidden_size": getattr(model, "hidden_size", None),
            "num_labels": getattr(model, "num_labels", 5),
            "state_dict": state_dict_path.name,
        }
        write_json(output_dir / "fallback_metadata.json", metadata)
        model.backbone.config.save_pretrained(output_dir)
    else:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        torch.save(model.state_dict(), state_dict_path)
    tokenizer.save_pretrained(output_dir)
    write_json(output_dir / "training_config.json", {**asdict(config), "model_mode": model_mode})
    write_json(output_dir / "dev_metrics.json", metrics)


def train(config: TrainConfig) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp02_dirs()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_data_exists(config.data_dir, config.evaluate_test)
    set_seed(config.seed)
    train_rows = limit_rows(read_jsonl(config.data_dir / "train.jsonl"), config.max_train_samples)
    dev_rows = limit_rows(read_jsonl(config.data_dir / "dev.jsonl"), config.max_eval_samples)
    test_rows = (
        limit_rows(read_jsonl(config.data_dir / "test.jsonl"), config.max_eval_samples)
        if config.evaluate_test
        else []
    )

    model, tokenizer, model_mode = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = (
        make_dataloader(test_rows, tokenizer, config, "test", shuffle=False)
        if config.evaluate_test
        else None
    )

    best_dir = config.checkpoint_output_dir / "best"
    metrics_history: list[dict[str, Any]] = []
    if config.eval_only:
        dev_result = evaluate(model, dev_loader, device, "dev")
        results = [dev_result]
        if config.evaluate_test:
            if test_loader is None:
                raise RuntimeError("evaluate_test=True but test loader is missing")
            results.append(evaluate(model, test_loader, device, "test"))
        for result in results:
            save_predictions(config.output_dir, result)
            save_eval_arrays(config.output_dir, result.metrics["split"], result)
        if config.evaluate_test:
            save_dev_test_npz(config.output_dir, results[0], results[1])
        save_metrics_tables(config.output_dir, results)
        return {"model_mode": model_mode, "metrics": [result.metrics for result in results]}

    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_dev_accuracy = -1.0
    global_step = 0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    num_epochs = int(math.ceil(config.num_train_epochs))
    print(
        "training schedule: "
        f"train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} "
        f"total_update_steps={total_steps} "
        f"log_steps={config.log_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar)
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
                    now = time.time()
                    elapsed = now - start
                    avg_loss = running_loss / max(1, step)
                    avg_steps_per_second = global_step / max(1e-9, elapsed)
                    avg_eta_seconds = max(0, total_steps - global_step) / avg_steps_per_second
                    current_lr = scheduler.get_last_lr()[0]
                    progress.render(global_step, epoch + 1, avg_loss, current_lr, elapsed, avg_eta_seconds)
                    if global_step % config.log_steps == 0:
                        interval_steps = max(1, global_step - last_log_step)
                        interval_seconds = max(1e-9, now - last_log_time)
                        steps_per_second = interval_steps / interval_seconds
                        remaining_steps = max(0, total_steps - global_step)
                        eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else float("nan")
                        last_log_time = now
                        last_log_step = global_step
                        progress.newline()
                        print(
                            f"epoch={epoch + 1} step={global_step}/{total_steps} "
                            f"loss={avg_loss:.4f} lr={current_lr:.2e} "
                            f"elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)} "
                            f"steps_per_min={steps_per_second * 60:.2f}",
                            flush=True,
                        )

            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev")
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            metrics_history.append(dev_result.metrics)
            elapsed = time.time() - start
            avg_seconds_per_step = elapsed / max(1, global_step)
            eta_seconds = max(0, total_steps - global_step) * avg_seconds_per_step
            print(
                f"dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"Exact={dev_result.metrics['Exact Match']:.4f}, low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"elapsed={format_duration(elapsed)}, eta={format_duration(eta_seconds)}",
                flush=True,
            )
            if dev_result.metrics["Exact Match"] > best_dev_accuracy:
                best_dev_accuracy = dev_result.metrics["Exact Match"]
                save_checkpoint(model, tokenizer, best_dir, config, model_mode, dev_result.metrics)
                save_predictions(config.output_dir, dev_result, suffix="dev_best")
    finally:
        progress.close()

    state_dict_path = best_dir / "state_dict.pt"
    if state_dict_path.exists():
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    dev_final = evaluate(model, dev_loader, device, "dev")
    results = [dev_final]
    if config.evaluate_test:
        if test_loader is None:
            raise RuntimeError("evaluate_test=True but test loader is missing")
        results.append(evaluate(model, test_loader, device, "test"))
    for result in results:
        result.metrics["epoch"] = "best"
        result.metrics["global_step"] = global_step
        save_predictions(config.output_dir, result)
        save_eval_arrays(config.output_dir, result.metrics["split"], result)
    if config.evaluate_test:
        save_dev_test_npz(config.output_dir, results[0], results[1])
    results[-1].metrics["elapsed_seconds"] = time.time() - start
    save_metrics_history(config.output_dir, metrics_history)
    save_metrics_tables(config.output_dir, results)
    run_summary_lines = [
        "# Exp2 CE Baseline Run Summary",
        "",
        f"Model: `{config.model_name_or_path}`",
        f"Model mode: `{model_mode}`",
        f"Best checkpoint: `{relpath(best_dir)}`",
        f"Evaluation scope: `{'dev+test' if config.evaluate_test else 'dev-only'}`",
        f"Final dev MAE_label: {dev_final.metrics['MAE_label']:.6f}",
        f"Final dev Exact Match: {dev_final.metrics['Exact Match']:.6f}",
        f"Final dev low-to-high rate: {dev_final.metrics['low_to_high_rate']:.6f}",
    ]
    if config.evaluate_test:
        test_result = results[1]
        run_summary_lines.extend(
            [
                f"Final test MAE_label: {test_result.metrics['MAE_label']:.6f}",
                f"Final test MAE_expected: {test_result.metrics['MAE_expected']:.6f}",
                f"Final test Exact Match: {test_result.metrics['Exact Match']:.6f}",
                f"Final test low-to-high rate: {test_result.metrics['low_to_high_rate']:.6f}",
            ]
        )
    run_summary = "\n".join(run_summary_lines)
    write_text(config.output_dir / "run_summary.md", run_summary)
    write_text(config.output_dir / "summaries" / "run_summary.md", run_summary)
    return {"model_mode": model_mode, "metrics": [result.metrics for result in results], "best_dir": str(best_dir)}


def main() -> None:
    config = parse_args()
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
