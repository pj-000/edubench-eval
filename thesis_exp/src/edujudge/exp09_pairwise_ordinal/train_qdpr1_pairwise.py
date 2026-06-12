"""Train Exp9 QD-PR1 risk-aware pairwise ordinal model."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp04.train_objective import (
    EvalResult,
    StepProgressBar,
    base_prediction_row,
    format_duration,
    get_loss_logits,
    monotonic_violation,
    save_dev_test_npz,
    save_eval_arrays,
    save_metrics_history,
    save_predictions,
    set_seed,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_DEV_PAIR_COUNT,
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LAMBDA_PAIR,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_MAX_PAIRS_PER_LOW_RECORD,
    DEFAULT_MAX_PAIRS_PER_RECORD,
    DEFAULT_PAIR_SAMPLING_SEED,
    DEFAULT_TRAIN_PAIR_COUNT,
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    EXP09_DATASET_DIR,
    EXP09_PAIRS_DIR,
    EXP09_RUN_ID,
    ensure_exp09_dirs,
    exp09_checkpoint_dir,
    exp09_run_dir,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import (
    class_weight_vector,
    dataset_sanity_rows,
    limit_rows,
    read_split,
    write_pointwise_class_weights,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import total_pairwise_training_loss
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.metrics import (
    compute_ordinal_metrics,
    save_metric_tables,
    save_pairwise_diagnostics,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.pair_builder import PairBuildConfig, read_pair_shards, write_pair_outputs
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


@dataclass
class PairwiseTrainConfig:
    run_id: str
    dataset_id: str
    split: str
    input_template: str
    model_name_or_path: str
    data_dir: Path
    output_dir: Path
    checkpoint_output_dir: Path
    head_type: str
    objective: str
    loss: str
    use_synthetic: bool
    use_coral: bool
    use_edurisk: bool
    pair_dataset_size_train: int
    pair_dataset_size_dev: int
    pair_sampling_seed: int
    max_pairs_per_record: int
    max_pairs_per_low_record: int
    margin_scale: float
    low_high_margin: float
    low_high_weight: float
    gap_weight: float
    lambda_pair: float
    w_min: float
    w_max: float
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
    max_train_pairs: int | None
    max_dev_pairs: int | None
    eval_only: bool
    checkpoint_dir: Path | None
    selection_metric: str
    selection_mode: str
    num_workers: int
    log_steps: int
    progress_bar: bool
    smoke: bool


class JsonlTextDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class PairJsonlDataset(JsonlTextDataset):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - server env should have PyYAML.
        raise ModuleNotFoundError("PyYAML is required to read Exp9 config files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Exp9 config must be a mapping: {relpath(path)}")
    return data


def select_dtype(config: PairwiseTrainConfig, torch: Any) -> Any:
    if config.fp16:
        return torch.float16
    if config.bf16 == "true":
        return torch.bfloat16
    if config.bf16 == "false":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def make_pairwise_head_class(torch: Any, nn: Any, class_weights: list[float]):
    class PairwiseOrdinalHead(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.score = nn.Linear(hidden_size, 4)
            self.hidden_size = int(hidden_size)
            self.num_outputs = 4
            self.config = backbone.config
            self.config.num_labels = 4
            self.config.problem_type = "multi_label_classification"
            self.register_buffer("pointwise_class_weights", torch.tensor(class_weights, dtype=torch.float32))
            self.last_loss_debug: dict[str, float] = {}

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone.config, "use_cache"):
                self.backbone.config.use_cache = False
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
            debug: dict[str, float] = {}
            if labels is not None:
                from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import weighted_ordinal_bce

                loss, debug = weighted_ordinal_bce(logits.float(), labels.long(), self.pointwise_class_weights)
                self.last_loss_debug = debug
            return {"loss": loss, "logits": logits, "loss_debug": debug}

    return PairwiseOrdinalHead


def build_model(torch: Any, nn: Any, AutoModel: Any, config: PairwiseTrainConfig, dtype: Any, class_weights: list[float]) -> Any:
    source = str(config.model_name_or_path)
    metadata_path = config.checkpoint_dir / "exp09_head_metadata.json" if config.checkpoint_dir else None
    if metadata_path and metadata_path.exists():
        source = read_json(metadata_path)["base_model_name_or_path"]
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    backbone = AutoModel.from_pretrained(source, **model_kwargs)
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for Exp9 pairwise ordinal head.")
    model = make_pairwise_head_class(torch, nn, class_weights)(backbone, int(hidden_size))
    if config.checkpoint_dir:
        state_dict_path = config.checkpoint_dir / "state_dict.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Missing checkpoint state dict: {relpath(state_dict_path)}")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model


def load_model_and_tokenizer(config: PairwiseTrainConfig, class_weights: list[float]):
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


def pointwise_collate_fn(tokenizer: Any, config: PairwiseTrainConfig):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=config.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["label_5"]) for row in batch], dtype=torch.long)
        encoded["metadata"] = batch
        return encoded

    return collate


def pair_collate_fn(tokenizer: Any, config: PairwiseTrainConfig):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        win_texts = [row["win_text"] for row in batch]
        lose_texts = [row["lose_text"] for row in batch]
        encoded = tokenizer(
            win_texts + lose_texts,
            truncation=True,
            max_length=config.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["win_labels"] = torch.tensor([int(row["win_label_5"]) for row in batch], dtype=torch.long)
        encoded["lose_labels"] = torch.tensor([int(row["lose_label_5"]) for row in batch], dtype=torch.long)
        encoded["label_gap"] = torch.tensor([int(row["label_gap"]) for row in batch], dtype=torch.float32)
        encoded["low_high"] = torch.tensor([1.0 if row["risk_pair_low_high"] in (True, "True", "true", "1", 1) else 0.0 for row in batch])
        encoded["pair_metadata"] = batch
        return encoded

    return collate


def make_pointwise_dataloader(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    config: PairwiseTrainConfig,
    split: str,
    shuffle: bool,
):
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
        collate_fn=pointwise_collate_fn(tokenizer, config),
    )


def make_pair_dataloader(pair_rows: list[dict[str, Any]], tokenizer: Any, config: PairwiseTrainConfig, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        PairJsonlDataset(pair_rows),
        batch_size=config.per_device_train_batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.num_workers,
        collate_fn=pair_collate_fn(tokenizer, config),
    )


def evaluate(model: Any, dataloader: Any, device: Any, split: str) -> EvalResult:
    import torch

    model.eval()
    predictions: list[dict[str, Any]] = []
    logits_chunks = []
    probs_chunks = []
    labels = []
    target_values = []
    record_ids = []
    total_loss = 0.0
    debug_totals: dict[str, float] = {}
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, logits = get_loss_logits(outputs)
            if loss is not None:
                total_loss += float(loss.detach().cpu())
            for key, value in outputs.get("loss_debug", {}).items():
                debug_totals[key] = debug_totals.get(key, 0.0) + float(value)
            logits_cpu = logits.float().detach().cpu()
            probs_cpu = torch.sigmoid(logits_cpu)
            logits_chunks.append(logits_cpu.numpy())
            probs_chunks.append(probs_cpu.numpy())
            for row, probs, logit in zip(metadata, probs_cpu.tolist(), logits_cpu.tolist()):
                pred_label = 1 + sum(1 for prob in probs if prob > 0.5)
                expected_score = 1.0 + float(sum(probs))
                human_mean = float(row["human_mean_5"])
                label_5 = int(row["label_5"])
                prediction = base_prediction_row(
                    split,
                    row,
                    label_5,
                    human_mean,
                    int(pred_label),
                    float(pred_label),
                    expected_score,
                )
                prediction["pred_label"] = int(pred_label)
                prediction["head_type"] = "independent_ordinal"
                prediction["objective"] = "risk_aware_pairwise_ordinal"
                prediction["loss"] = "weighted_ordinal_bce_plus_pairwise_softplus"
                prediction["confidence"] = float(np.mean([abs(prob - 0.5) * 2 for prob in probs]))
                prediction["monotonic_violation"] = monotonic_violation(probs)
                for idx, threshold in enumerate([1, 2, 3, 4]):
                    prediction[f"prob_gt_{threshold}"] = float(probs[idx])
                    prediction[f"logit_gt_{threshold}"] = float(logit[idx])
                predictions.append(prediction)
                labels.append(label_5)
                target_values.append(label_5)
                record_ids.append(str(row.get("record_id") or row.get("id")))
            steps += 1
    logits_arr = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 4), dtype=np.float32)
    probs_arr = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.empty((0, 4), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    targets_arr = np.array(target_values, dtype=np.float32)
    metrics = compute_ordinal_metrics(predictions)
    metrics["split"] = split
    metrics["loss"] = total_loss / steps if steps else float("nan")
    for key, value in debug_totals.items():
        metrics[key] = value / steps if steps else float("nan")
    return EvalResult(metrics, predictions, logits_arr, probs_arr, labels_arr, targets_arr, record_ids)


def selection_metric_name(config: PairwiseTrainConfig) -> str:
    return f"dev_{config.selection_metric}"


def selection_metric_value(metrics: dict[str, Any], config: PairwiseTrainConfig) -> float:
    if config.selection_metric not in metrics:
        raise KeyError(f"Selection metric {config.selection_metric!r} missing from dev metrics.")
    return float(metrics[config.selection_metric])


def score_is_better(metrics: dict[str, Any], best_score: float | None, config: PairwiseTrainConfig) -> bool:
    score = selection_metric_value(metrics, config)
    if best_score is None:
        return True
    if config.selection_mode == "min":
        return score < best_score
    if config.selection_mode == "max":
        return score > best_score
    raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    config: PairwiseTrainConfig,
    metrics: dict[str, Any],
    class_weights: list[float],
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    tokenizer.save_pretrained(output_dir)
    if hasattr(model.backbone, "config"):
        model.backbone.config.save_pretrained(output_dir)
    write_json(
        output_dir / "exp09_head_metadata.json",
        {
            "base_model_name_or_path": config.model_name_or_path,
            "experiment": "Exp9 risk-aware pairwise ordinal training",
            "run_id": EXP09_RUN_ID,
            "head_type": "independent_ordinal",
            "objective": config.objective,
            "loss": config.loss,
            "num_outputs": getattr(model, "num_outputs", 4),
            "hidden_size": getattr(model, "hidden_size", None),
            "state_dict": state_dict_path.name,
            "best_epoch": metrics.get("epoch"),
            "best_global_step": metrics.get("global_step"),
            "best_selection_metric_name": selection_metric_name(config),
            "best_selection_metric_value": selection_metric_value(metrics, config),
            "selection_direction": config.selection_mode,
            "synthetic_used": False,
            "class_weight_formula": "clip(N / (5 * N_c), w_min, w_max)",
            "class_weights": class_weights,
            "pairwise_loss": {
                "margin_scale": config.margin_scale,
                "low_high_margin": config.low_high_margin,
                "low_high_weight": config.low_high_weight,
                "gap_weight": config.gap_weight,
                "lambda_pair": config.lambda_pair,
            },
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "dev_metrics.json", metrics)


def copy_dev_history_to_logs(output_dir: Path) -> None:
    src = output_dir / "tables" / "dev_metrics_history.csv"
    dst = output_dir / "logs" / "dev_metrics_history.csv"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_metrics_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_run_files(output_dir: Path, config: PairwiseTrainConfig, status: str) -> None:
    best_metadata_path = config.checkpoint_output_dir / "best" / "exp09_head_metadata.json"
    best_metadata = read_json(best_metadata_path) if best_metadata_path.exists() else {}
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": "Exp9 risk-aware pairwise ordinal training",
            "run_id": EXP09_RUN_ID,
            "dataset_id": config.dataset_id,
            "split": config.split,
            "status": status,
            "input_template": config.input_template,
            "model_name_or_path": config.model_name_or_path,
            "head_type": config.head_type,
            "objective": config.objective,
            "loss": config.loss,
            "synthetic_used": False,
            "coral_used": False,
            "edurisk_used": False,
            "checkpoint_selection": f"dev {config.selection_metric} ({config.selection_mode})",
            "best_epoch": best_metadata.get("best_epoch"),
            "best_global_step": best_metadata.get("best_global_step"),
            "best_selection_metric_name": best_metadata.get("best_selection_metric_name", selection_metric_name(config)),
            "best_selection_metric_value": best_metadata.get("best_selection_metric_value"),
            "selection_direction": best_metadata.get("selection_direction", config.selection_mode),
        },
    )
    lines = [
        "# Exp9 QD-PR1 Pairwise Ordinal Run Summary",
        "",
        f"Status: `{status}`",
        f"Run ID: `{EXP09_RUN_ID}`",
        f"Model: `{config.model_name_or_path}`",
        "Dataset: `QD-S0_human_only`, question_seed42 human-only train/dev/test.",
        "Fixed input: `A4_question_answer_metric_rubric_metadata` text field.",
        "Head: independent ordinal threshold head.",
        "Loss: QD-B1-style weighted ordinal BCE plus risk-aware pairwise softplus.",
        "Synthetic rows: none.",
        f"lambda_pair: `{config.lambda_pair}`.",
        f"Checkpoint selection: dev `{config.selection_metric}` ({config.selection_mode})",
    ]
    test = next((row for row in _read_metrics_rows(output_dir / "tables" / "metrics_summary.csv") if row.get("split") == "test"), {})
    if test:
        lines.extend(
            [
                "",
                "## Test Metrics",
                "",
                f"- MAE_label: {test.get('MAE_label', 'NA')}",
                f"- QWK: {test.get('Quadratic Weighted Kappa', 'NA')}",
                f"- Acc@5: {test.get('Acc@5', 'NA')}",
                f"- low_to_high_rate: {test.get('low_to_high_rate', 'NA')}",
                f"- monotonic_violation_rate: {test.get('monotonic_violation_rate', 'NA')}",
            ]
        )
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def require_cuda_if_requested() -> None:
    if os.environ.get("REQUIRE_CUDA", "0") != "1":
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - server environment check.
        raise RuntimeError(f"REQUIRE_CUDA=1 but torch import failed: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is false.")


def _failures_to_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {row['check_name']}: {row['status']} ({row.get('details', '')})" for row in rows if row["status"] != "PASS")


def pair_build_config(config: PairwiseTrainConfig) -> PairBuildConfig:
    return PairBuildConfig(
        train_pairs=config.pair_dataset_size_train,
        dev_pairs=config.pair_dataset_size_dev,
        seed=config.pair_sampling_seed,
        max_pairs_per_record=config.max_pairs_per_record,
        max_pairs_per_low_record=config.max_pairs_per_low_record,
        margin_scale=config.margin_scale,
        low_high_margin=config.low_high_margin,
        low_high_weight=config.low_high_weight,
        gap_weight=config.gap_weight,
    )


def ensure_pair_artifacts(config: PairwiseTrainConfig) -> dict[str, Any]:
    train_rows = read_split(config.data_dir, "train")
    dev_rows = read_split(config.data_dir, "dev")
    test_rows = read_split(config.data_dir, "test")
    return write_pair_outputs(train_rows, dev_rows, test_rows, pair_build_config(config))


def postprocess_outputs(output_dir: Path) -> None:
    copy_dev_history_to_logs(output_dir)
    try:
        from thesis_exp.src.edujudge.exp09_pairwise_ordinal.collect_exp09_results import collect

        collect()
    except FileNotFoundError:
        pass


def loss_config_row(config: PairwiseTrainConfig, class_weights: list[float]) -> dict[str, Any]:
    return {
        "run_id": EXP09_RUN_ID,
        "pointwise_loss": "weighted_ordinal_bce",
        "pairwise_loss": "softplus(m_ij - (r(win)-r(lose)))",
        "r_x": "1 + sum_t sigmoid(z_t)",
        "margin_scale": config.margin_scale,
        "low_high_margin": config.low_high_margin,
        "low_high_weight": config.low_high_weight,
        "gap_weight": config.gap_weight,
        "lambda_pair": config.lambda_pair,
        "w_min": config.w_min,
        "w_max": config.w_max,
        "class_weights": class_weights,
    }


def train(config: PairwiseTrainConfig, preflight_only: bool = False, postprocess_only: bool = False) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp09_dirs()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    if config.run_id != EXP09_RUN_ID:
        raise RuntimeError(f"Unexpected run_id for Exp9 first run: {config.run_id}")
    if config.use_synthetic or config.use_coral or config.use_edurisk:
        raise RuntimeError("Exp9 QD-PR1 first run must be human-only, non-CORAL, non-EduRisk.")
    if config.smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if formal and any(
        value is not None
        for value in [config.max_train_samples, config.max_eval_samples, config.max_train_pairs, config.max_dev_pairs]
    ):
        raise RuntimeError("FORMAL_RUN=1 cannot use max sample or max pair limits.")
    if not config.smoke and not formal and not config.eval_only and not preflight_only and not postprocess_only:
        raise RuntimeError("Formal QD-PR1 training requires FORMAL_RUN=1.")

    sanity_rows = dataset_sanity_rows(config.data_dir)
    sanity_name = "smoke_preflight_qdpr1.csv" if config.smoke else "preflight_qdpr1.csv"
    write_csv(config.output_dir / "tables" / sanity_name, sanity_rows)
    failures = [row for row in sanity_rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QD-PR1 preflight failed:\n{_failures_to_text(failures)}")

    pair_summary = ensure_pair_artifacts(config)
    train_rows_full = read_split(config.data_dir, "train")
    weight_rows = write_pointwise_class_weights(train_rows_full, config.output_dir / "tables", config.w_min, config.w_max)
    write_pointwise_class_weights(train_rows_full, config.output_dir.parent.parent / "tables", config.w_min, config.w_max)
    class_weights = class_weight_vector(weight_rows)
    write_csv(config.output_dir / "tables" / "loss_config.csv", [loss_config_row(config, class_weights)])
    if preflight_only:
        write_text(
            config.output_dir / "qdpr1_preflight_status.md",
            "\n".join(
                [
                    "# QD-PR1 Preflight Status",
                    "",
                    "Status: PASS",
                    f"Formal mode: {formal}",
                    f"Smoke mode: {config.smoke}",
                    "Training executed: no",
                    "API called: no",
                    "Synthetic generated: no",
                    f"Train pairs: {pair_summary['train_pairs']}",
                    f"Dev pairs: {pair_summary['dev_pairs']}",
                    f"Class weight rows: {len(weight_rows)}",
                ]
            ),
        )
        return {"status": "preflight_pass", "pair_summary": pair_summary, "class_weights": class_weights}
    if postprocess_only:
        postprocess_outputs(config.output_dir)
        return {"status": "postprocessed"}

    require_cuda_if_requested()
    set_seed(config.seed)
    train_rows = limit_rows(train_rows_full, config.max_train_samples)
    dev_rows = limit_rows(read_split(config.data_dir, "dev"), config.max_eval_samples)
    test_rows = limit_rows(read_split(config.data_dir, "test"), config.max_eval_samples)
    train_pairs = limit_rows(read_pair_shards("train", EXP09_PAIRS_DIR), config.max_train_pairs)
    dev_pairs = limit_rows(read_pair_shards("dev", EXP09_PAIRS_DIR), config.max_dev_pairs)

    model, tokenizer = load_model_and_tokenizer(config, class_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dev_loader = make_pointwise_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_pointwise_dataloader(test_rows, tokenizer, config, "test", shuffle=False)
    best_dir = config.checkpoint_output_dir / "best"
    metrics_history: list[dict[str, Any]] = []
    loss_debug_history: list[dict[str, Any]] = []

    if config.eval_only:
        dev_result = evaluate(model, dev_loader, device, "dev")
        test_result = evaluate(model, test_loader, device, "test")
        for result in [dev_result, test_result]:
            save_predictions(config.output_dir, result)
            save_eval_arrays(config.output_dir, result.metrics["split"], result)
        save_dev_test_npz(config.output_dir, dev_result, test_result)
        save_metric_tables(config.output_dir, [dev_result, test_result])
        save_pairwise_diagnostics(config.output_dir, dev_pairs, dev_result, "dev")
        normalize_run_files(config.output_dir, config, "eval_only")
        postprocess_outputs(config.output_dir)
        return {"metrics": [dev_result.metrics, test_result.metrics]}

    train_loader = make_pair_dataloader(train_pairs, tokenizer, config, shuffle=True)
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
        f"run_id={EXP09_RUN_ID} head=independent_ordinal loss=pairwise "
        f"train_pairs={len(train_pairs)} train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} total_update_steps={total_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp9 {EXP09_RUN_ID}")
    try:
        for epoch in range(num_epochs):
            if epoch >= config.num_train_epochs:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                pair_metadata = batch.pop("pair_metadata")
                batch_size = len(pair_metadata)
                win_labels = batch.pop("win_labels").to(device)
                lose_labels = batch.pop("lose_labels").to(device)
                label_gap = batch.pop("label_gap").to(device)
                low_high = batch.pop("low_high").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                _, logits = get_loss_logits(outputs)
                win_logits = logits[:batch_size]
                lose_logits = logits[batch_size:]
                loss, debug = total_pairwise_training_loss(
                    win_logits.float(),
                    lose_logits.float(),
                    win_labels,
                    lose_labels,
                    label_gap,
                    low_high,
                    model.pointwise_class_weights,
                    lambda_pair=config.lambda_pair,
                    margin_scale=config.margin_scale,
                    low_high_margin=config.low_high_margin,
                    low_high_weight=config.low_high_weight,
                    gap_weight=config.gap_weight,
                )
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().cpu())
                if debug:
                    loss_debug_history.append({"epoch": epoch + 1, "step": step, "global_step": global_step, **debug})
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
                    if global_step >= total_steps:
                        break
            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev")
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            metrics_history.append(dev_result.metrics)
            selected = bool(score_is_better(dev_result.metrics, best_score, config))
            elapsed = time.time() - start
            print(
                f"dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"selected={selected}, elapsed={format_duration(elapsed)}",
                flush=True,
            )
            if selected:
                best_score = selection_metric_value(dev_result.metrics, config)
                save_checkpoint(model, tokenizer, best_dir, config, dev_result.metrics, class_weights)
                save_predictions(config.output_dir, dev_result, suffix="dev_best")
            if global_step >= total_steps:
                break
    finally:
        progress.close()

    state_dict_path = best_dir / "state_dict.pt"
    if not state_dict_path.exists():
        raise RuntimeError(f"Best checkpoint was not saved: {state_dict_path}")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    dev_final = evaluate(model, dev_loader, device, "dev")
    test_result = evaluate(model, test_loader, device, "test")
    for result in [dev_final, test_result]:
        result.metrics["epoch"] = "best"
        result.metrics["global_step"] = global_step
        save_predictions(config.output_dir, result)
        save_eval_arrays(config.output_dir, result.metrics["split"], result)
    save_dev_test_npz(config.output_dir, dev_final, test_result)
    test_result.metrics["elapsed_seconds"] = time.time() - start
    save_metrics_history(config.output_dir, metrics_history)
    if loss_debug_history:
        write_csv(config.output_dir / "tables" / "loss_debug_history.csv", loss_debug_history)
        write_csv(config.output_dir / "logs" / "loss_debug_history.csv", loss_debug_history)
        write_csv(config.output_dir / "tables" / "loss_component_scale.csv", loss_debug_history)
    save_metric_tables(config.output_dir, [dev_final, test_result])
    save_pairwise_diagnostics(config.output_dir, dev_pairs, dev_final, "dev")
    normalize_run_files(config.output_dir, config, "completed")
    postprocess_outputs(config.output_dir)
    return {"metrics": [dev_final.metrics, test_result.metrics], "best_dir": str(best_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp9 QD-PR1 pairwise ordinal scorer.")
    parser.add_argument("--config_path", type=Path, default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=None)
    parser.add_argument("--max_dev_pairs", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--selection_metric", default=None)
    parser.add_argument("--selection_mode", choices=["min", "max"], default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> PairwiseTrainConfig:
    raw = load_yaml_config(args.config_path)
    smoke_raw = raw.get("smoke", {}) if isinstance(raw.get("smoke", {}), dict) else {}
    smoke = bool(args.smoke or raw.get("smoke_mode", False))
    section = smoke_raw if smoke and smoke_raw else {}
    batch_size = args.per_device_train_batch_size or int(section.get("batch_size", raw.get("per_device_train_batch_size", 4)))
    grad_accum = args.gradient_accumulation_steps
    if grad_accum is None:
        grad_accum = int(section.get("gradient_accumulation_steps", raw.get("gradient_accumulation_steps", 0)) or 0)
    if grad_accum <= 0:
        effective_batch = int(raw.get("effective_batch_size", 128))
        grad_accum = max(1, math.ceil(effective_batch / max(1, batch_size)))
    epochs = args.num_train_epochs
    if epochs is None:
        epochs = float(section.get("epochs", raw.get("epochs", 10.0)))
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else section.get("max_train_samples")
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else section.get("max_eval_samples")
    max_train_pairs = args.max_train_pairs if args.max_train_pairs is not None else section.get("max_train_pairs")
    max_dev_pairs = args.max_dev_pairs if args.max_dev_pairs is not None else section.get("max_dev_pairs")
    return PairwiseTrainConfig(
        run_id=str(raw.get("run_id", EXP09_RUN_ID)),
        dataset_id=str(raw.get("dataset_id", "QD-S0_human_only")),
        split=str(raw.get("split", "question_seed42")),
        input_template=str(raw.get("input_template", "A4_question_answer_metric_rubric_metadata")),
        model_name_or_path=args.model_name_or_path or str(raw.get("model_name_or_path", "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")),
        data_dir=args.data_dir or Path(raw.get("data_dir", EXP09_DATASET_DIR)),
        output_dir=args.output_dir or exp09_run_dir(smoke=smoke),
        checkpoint_output_dir=args.checkpoint_output_dir or exp09_checkpoint_dir(smoke=smoke),
        head_type=str(raw.get("head_type", "independent_ordinal")),
        objective=str(raw.get("objective", "risk_aware_pairwise_ordinal")),
        loss=str(raw.get("loss", "weighted_ordinal_bce_plus_pairwise_softplus")),
        use_synthetic=bool(raw.get("use_synthetic", False)),
        use_coral=bool(raw.get("use_coral", False)),
        use_edurisk=bool(raw.get("use_edurisk", False)),
        pair_dataset_size_train=int(raw.get("pair_dataset_size_train", DEFAULT_TRAIN_PAIR_COUNT)),
        pair_dataset_size_dev=int(raw.get("pair_dataset_size_dev", DEFAULT_DEV_PAIR_COUNT)),
        pair_sampling_seed=int(raw.get("pair_sampling_seed", DEFAULT_PAIR_SAMPLING_SEED)),
        max_pairs_per_record=int(raw.get("max_pairs_per_record", DEFAULT_MAX_PAIRS_PER_RECORD)),
        max_pairs_per_low_record=int(raw.get("max_pairs_per_low_record", DEFAULT_MAX_PAIRS_PER_LOW_RECORD)),
        margin_scale=float(raw.get("margin_scale", DEFAULT_MARGIN_SCALE)),
        low_high_margin=float(raw.get("low_high_margin", DEFAULT_LOW_HIGH_MARGIN)),
        low_high_weight=float(raw.get("low_high_weight", DEFAULT_LOW_HIGH_WEIGHT)),
        gap_weight=float(raw.get("gap_weight", DEFAULT_GAP_WEIGHT)),
        lambda_pair=float(raw.get("lambda_pair", DEFAULT_LAMBDA_PAIR)),
        w_min=float(raw.get("w_min", DEFAULT_W_MIN)),
        w_max=float(raw.get("w_max", DEFAULT_W_MAX)),
        max_length=args.max_length or int(raw.get("max_length", 2048)),
        num_train_epochs=float(epochs),
        learning_rate=float(args.learning_rate or raw.get("learning_rate", 2e-5)),
        weight_decay=float(args.weight_decay),
        warmup_ratio=float(args.warmup_ratio),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size or raw.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(grad_accum),
        max_grad_norm=float(args.max_grad_norm),
        seed=int(args.seed),
        bf16=str(args.bf16),
        fp16=bool(args.fp16),
        gradient_checkpointing=bool(args.gradient_checkpointing or raw.get("gradient_checkpointing", False)),
        trust_remote_code=bool(args.trust_remote_code or raw.get("trust_remote_code", True)),
        local_files_only=bool(args.local_files_only or raw.get("local_files_only", False)),
        max_train_samples=int(max_train_samples) if max_train_samples is not None else None,
        max_eval_samples=int(max_eval_samples) if max_eval_samples is not None else None,
        max_train_pairs=int(max_train_pairs) if max_train_pairs is not None else None,
        max_dev_pairs=int(max_dev_pairs) if max_dev_pairs is not None else None,
        eval_only=bool(args.eval_only),
        checkpoint_dir=args.checkpoint_dir,
        selection_metric=str(args.selection_metric or raw.get("selection_metric", "dev_MAE_label")).replace("dev_", ""),
        selection_mode=str(args.selection_mode or raw.get("selection_direction", raw.get("checkpoint_selection_direction", "min"))),
        num_workers=int(args.num_workers),
        log_steps=int(args.log_steps),
        progress_bar=bool(args.progress_bar),
        smoke=smoke,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    result = train(config, preflight_only=args.preflight_only, postprocess_only=args.postprocess_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
