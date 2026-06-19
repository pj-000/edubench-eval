"""Train QD-PR2 anchored pairwise ordinal fine-tuning from QD-B1."""

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
    DEFAULT_GAP_WEIGHT,
    DEFAULT_LOW_HIGH_MARGIN,
    DEFAULT_LOW_HIGH_WEIGHT,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_PAIR_SAMPLING_SEED,
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    QD_B1_CHECKPOINT_DIR,
    QDPR2_DATASET_DIR,
    QDPR2_DEFAULT_DEV_PAIR_COUNT,
    QDPR2_DEFAULT_EPOCHS,
    QDPR2_DEFAULT_LAMBDA_ANCHOR,
    QDPR2_DEFAULT_LAMBDA_MONO,
    QDPR2_DEFAULT_LAMBDA_PAIR,
    QDPR2_DEFAULT_LAMBDA_POINT,
    QDPR2_DEFAULT_LEARNING_RATE,
    QDPR2_DEFAULT_MAX_PAIRS_PER_LOW_RECORD,
    QDPR2_DEFAULT_MAX_PAIRS_PER_RECORD,
    QDPR2_DEFAULT_TRAIN_PAIR_COUNT,
    QDPR2_PAIRS_DIR,
    QDPR2_RUN_ID,
    ensure_exp09_dirs,
    qdpr2_checkpoint_dir,
    qdpr2_run_dir,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import (
    class_weight_vector,
    limit_rows,
    read_split,
    write_pointwise_class_weights,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import (
    total_anchored_pairwise_training_loss,
    total_anchored_pointwise_training_loss,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.metrics import (
    compute_ordinal_metrics,
    save_metric_tables,
    save_pairwise_diagnostics,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.qdpr2_setup import (
    QDPR2PairBuildConfig,
    prepare_qdpr2_setup,
    read_qdpr2_pair_shards,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr1_pairwise import (
    JsonlTextDataset,
    PairJsonlDataset,
    load_yaml_config,
    pair_collate_fn,
    pointwise_collate_fn,
    read_json,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


@dataclass
class QDPR2TrainConfig:
    exp_name: str
    ablation_name: str
    display_name: str
    run_id: str
    dataset_id: str
    split: str
    input_template: str
    model_name_or_path: str
    data_dir: Path
    output_dir: Path
    checkpoint_output_dir: Path
    qd_b1_checkpoint_dir: Path
    head_type: str
    objective: str
    loss: str
    use_synthetic: bool
    pair_dataset_size_train: int
    pair_dataset_size_dev: int
    pair_sampling_seed: int
    max_pairs_per_record: int
    max_pairs_per_low_record: int
    margin_scale: float
    low_high_margin: float
    low_high_weight: float
    gap_weight: float
    lambda_point: float
    lambda_pair: float
    lambda_anchor: float
    lambda_mono: float
    force_pair_training: bool
    dataloader_mode: str
    strict_module_ablation: bool
    removed_module: str
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
    selection_metric: str
    selection_mode: str
    num_workers: int
    log_steps: int
    progress_bar: bool
    smoke: bool


def require_qd_b1_checkpoint(path: Path) -> None:
    required = ["state_dict.pt", "exp05_head_metadata.json", "tokenizer.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"BLOCKED_MISSING_QDB1_CHECKPOINT: {relpath(path)} missing {missing}")


def select_dtype(config: QDPR2TrainConfig, torch: Any) -> Any:
    if config.fp16:
        return torch.float16
    if config.bf16 == "true":
        return torch.bfloat16
    if config.bf16 == "false":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def make_qdpr2_head_class(torch: Any, nn: Any, class_weights: list[float]):
    class AnchoredPairwiseOrdinalHead(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.score = nn.Linear(hidden_size, 4)
            self.hidden_size = int(hidden_size)
            self.num_outputs = 4
            self.config = backbone.config
            self.config.num_labels = 4
            self.config.problem_type = "multi_label_classification"
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
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

                loss, debug = weighted_ordinal_bce(logits.float(), labels.long(), self.class_weights)
                self.last_loss_debug = debug
            return {"loss": loss, "logits": logits, "loss_debug": debug}

    return AnchoredPairwiseOrdinalHead


def build_model(torch: Any, nn: Any, AutoModel: Any, config: QDPR2TrainConfig, dtype: Any, class_weights: list[float]) -> Any:
    require_qd_b1_checkpoint(config.qd_b1_checkpoint_dir)
    metadata = read_json(config.qd_b1_checkpoint_dir / "exp05_head_metadata.json")
    source = metadata.get("base_model_name_or_path") or config.model_name_or_path
    model_kwargs = {"trust_remote_code": config.trust_remote_code, "local_files_only": config.local_files_only}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    backbone = AutoModel.from_pretrained(source, **model_kwargs)
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for QD-PR2 anchored ordinal head.")
    model = make_qdpr2_head_class(torch, nn, class_weights)(backbone, int(hidden_size))
    state_dict = torch.load(config.qd_b1_checkpoint_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def load_model_and_tokenizer(config: QDPR2TrainConfig, class_weights: list[float]):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    dtype = select_dtype(config, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.qd_b1_checkpoint_dir,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model = build_model(torch, nn, AutoModel, config, dtype, class_weights)
    reference_model = None
    if config.lambda_anchor != 0.0:
        reference_model = build_model(torch, nn, AutoModel, config, dtype, class_weights)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, reference_model, tokenizer


def make_pointwise_dataloader(rows: list[dict[str, Any]], tokenizer: Any, config: QDPR2TrainConfig, split: str, shuffle: bool):
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


def make_pair_dataloader(pair_rows: list[dict[str, Any]], tokenizer: Any, config: QDPR2TrainConfig, shuffle: bool):
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


def evaluate(model: Any, dataloader: Any, device: Any, split: str, config: QDPR2TrainConfig | None = None) -> EvalResult:
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
                prediction = base_prediction_row(split, row, label_5, human_mean, int(pred_label), float(pred_label), expected_score)
                prediction["pred_label"] = int(pred_label)
                prediction["head_type"] = "independent_ordinal"
                prediction["objective"] = config.objective if config is not None else "anchored_risk_aware_pairwise_ordinal"
                prediction["loss"] = config.loss if config is not None else "weighted_ordinal_bce_plus_pairwise_anchor_monotonic"
                if config is not None:
                    prediction["run_id"] = config.run_id
                    prediction["ablation_name"] = config.ablation_name
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


def selection_metric_value(metrics: dict[str, Any], config: QDPR2TrainConfig) -> float:
    if config.selection_metric not in metrics:
        raise KeyError(f"Selection metric {config.selection_metric!r} missing from dev metrics.")
    return float(metrics[config.selection_metric])


def score_is_better(metrics: dict[str, Any], best_score: float | None, config: QDPR2TrainConfig) -> bool:
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
    config: QDPR2TrainConfig,
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
        output_dir / "exp09_qdpr2_head_metadata.json",
        {
            "base_model_name_or_path": config.model_name_or_path,
            "anchor_checkpoint_dir": relpath(config.qd_b1_checkpoint_dir),
            "experiment": config.exp_name,
            "ablation_name": config.ablation_name,
            "display_name": config.display_name,
            "run_id": config.run_id,
            "head_type": config.head_type,
            "objective": config.objective,
            "loss": config.loss,
            "removed_module": config.removed_module,
            "strict_module_ablation": config.strict_module_ablation,
            "force_pair_training": config.force_pair_training,
            "use_pair_training": use_pair_training(config),
            "dataloader_mode": resolved_dataloader_mode(config),
            "num_outputs": getattr(model, "num_outputs", 4),
            "hidden_size": getattr(model, "hidden_size", None),
            "state_dict": state_dict_path.name,
            "best_epoch": metrics.get("epoch"),
            "best_global_step": metrics.get("global_step"),
            "best_selection_metric_name": f"dev_{config.selection_metric}",
            "best_selection_metric_value": selection_metric_value(metrics, config),
            "selection_direction": config.selection_mode,
            "synthetic_used": False,
            "class_weights": class_weights,
            "loss_weights": {
                "lambda_point": config.lambda_point,
                "lambda_pair": config.lambda_pair,
                "lambda_anchor": config.lambda_anchor,
                "lambda_mono": config.lambda_mono,
            },
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "dev_metrics.json", metrics)


def normalize_run_files(output_dir: Path, config: QDPR2TrainConfig, status: str) -> None:
    best_metadata_path = config.checkpoint_output_dir / "best" / "exp09_qdpr2_head_metadata.json"
    best_metadata = read_json(best_metadata_path) if best_metadata_path.exists() else {}
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": config.exp_name,
            "ablation_name": config.ablation_name,
            "display_name": config.display_name,
            "run_id": config.run_id,
            "status": status,
            "dataset_id": config.dataset_id,
            "split": config.split,
            "input_template": config.input_template,
            "model_name_or_path": config.model_name_or_path,
            "anchor_checkpoint_dir": relpath(config.qd_b1_checkpoint_dir),
            "head_type": config.head_type,
            "objective": config.objective,
            "loss": config.loss,
            "removed_module": config.removed_module,
            "strict_module_ablation": config.strict_module_ablation,
            "force_pair_training": config.force_pair_training,
            "use_pair_training": use_pair_training(config),
            "dataloader_mode": resolved_dataloader_mode(config),
            "expected_dataloader_mode": config.dataloader_mode,
            "synthetic_used": False,
            "best_epoch": best_metadata.get("best_epoch"),
            "best_global_step": best_metadata.get("best_global_step"),
            "selection_metric": config.selection_metric,
            "selection_direction": config.selection_mode,
        },
    )
    active_losses = []
    if config.lambda_point != 0.0:
        active_losses.append("L_point")
    if config.lambda_pair != 0.0:
        active_losses.append("L_pair")
    if config.lambda_anchor != 0.0:
        active_losses.append("L_anchor")
    if config.lambda_mono != 0.0:
        active_losses.append("L_mono")
    lines = [
        f"# {config.exp_name} Run Summary",
        "",
        f"Status: `{status}`",
        f"Run ID: `{config.run_id}`",
        f"Ablation: `{config.ablation_name}`",
        f"Display name: `{config.display_name}`",
        "Initialization: QD-B1 checkpoint only.",
        "Dataset: question_seed42 human-only A4 rows.",
        "Synthetic rows: none.",
        f"Removed module: `{config.removed_module}`",
        f"Strict module ablation: `{config.strict_module_ablation}`",
        f"Force pair training: `{config.force_pair_training}`",
        f"Use pair training: `{use_pair_training(config)}`",
        f"Dataloader mode: `{resolved_dataloader_mode(config)}`",
        f"Active losses: `{' + '.join(active_losses) if active_losses else 'none'}`",
        f"lambda_point: `{config.lambda_point}`",
        f"lambda_pair: `{config.lambda_pair}`",
        f"lambda_anchor: `{config.lambda_anchor}`",
        f"lambda_mono: `{config.lambda_mono}`",
    ]
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def copy_dev_history_to_logs(output_dir: Path) -> None:
    src = output_dir / "tables" / "dev_metrics_history.csv"
    dst = output_dir / "logs" / "dev_metrics_history.csv"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def require_cuda_if_requested() -> None:
    if os.environ.get("REQUIRE_CUDA", "0") != "1":
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"REQUIRE_CUDA=1 but torch import failed: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is false.")


def pair_build_config(config: QDPR2TrainConfig) -> QDPR2PairBuildConfig:
    return QDPR2PairBuildConfig(
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


def use_pair_training(config: QDPR2TrainConfig) -> bool:
    return bool(config.lambda_pair != 0.0 or config.force_pair_training)


def resolved_dataloader_mode(config: QDPR2TrainConfig) -> str:
    return "pair" if use_pair_training(config) else "pointwise"


def loss_config_row(config: QDPR2TrainConfig, class_weights: list[float]) -> dict[str, Any]:
    active_losses = []
    if config.lambda_point != 0.0:
        active_losses.append("L_point")
    if config.lambda_pair != 0.0:
        active_losses.append("L_pair")
    if config.lambda_anchor != 0.0:
        active_losses.append("L_anchor")
    if config.lambda_mono != 0.0:
        active_losses.append("L_mono")
    return {
        "exp_name": config.exp_name,
        "ablation_name": config.ablation_name,
        "display_name": config.display_name,
        "run_id": config.run_id,
        "initialization": "QD-B1 checkpoint",
        "active_losses": " + ".join(active_losses) if active_losses else "none",
        "removed_module": config.removed_module,
        "strict_module_ablation": config.strict_module_ablation,
        "force_pair_training": config.force_pair_training,
        "use_pair_training": use_pair_training(config),
        "dataloader_mode": resolved_dataloader_mode(config),
        "expected_dataloader_mode": config.dataloader_mode,
        "pointwise_loss": "weighted_ordinal_bce",
        "pairwise_loss": "softplus(m_ij - (r(win)-r(lose)))",
        "anchor_loss": "BCEWithLogits(current_logits, sigmoid(qd_b1_ref_logits))",
        "monotonic_regularization": "mean_t max(0, p_{t+1}-p_t)",
        "lambda_point": config.lambda_point,
        "lambda_pair": config.lambda_pair,
        "lambda_anchor": config.lambda_anchor,
        "lambda_mono": config.lambda_mono,
        "class_weights": class_weights,
    }


def train(config: QDPR2TrainConfig, preflight_only: bool = False) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp09_dirs()
    require_qd_b1_checkpoint(config.qd_b1_checkpoint_dir)
    if config.use_synthetic:
        raise RuntimeError("QD-PR2 must be human-only; synthetic data is disabled.")
    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    if config.smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if not config.smoke and not formal and not config.eval_only and not preflight_only:
        raise RuntimeError("QD-PR2 formal training requires FORMAL_RUN=1.")
    if formal and any(
        value is not None
        for value in [config.max_train_samples, config.max_eval_samples, config.max_train_pairs, config.max_dev_pairs]
    ):
        raise RuntimeError("FORMAL_RUN=1 cannot use max sample or max pair limits.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    pair_training_enabled = use_pair_training(config)
    actual_dataloader_mode = "pair" if pair_training_enabled else "pointwise"
    if config.dataloader_mode and config.dataloader_mode != actual_dataloader_mode:
        raise ValueError(
            f"dataloader_mode mismatch for {config.ablation_name}: "
            f"config={config.dataloader_mode}, resolved={actual_dataloader_mode}"
        )
    setup: dict[str, Any] = {"train_pairs": [], "dev_pairs": []}
    if pair_training_enabled:
        setup = prepare_qdpr2_setup(pair_build_config(config), write_shards=True)
    train_rows_full = read_split(config.data_dir, "train")
    weight_rows = write_pointwise_class_weights(train_rows_full, config.output_dir / "tables", config.w_min, config.w_max)
    class_weights = class_weight_vector(weight_rows)
    write_csv(config.output_dir / "tables" / "loss_config.csv", [loss_config_row(config, class_weights)])
    if preflight_only:
        write_text(
            config.output_dir / "qdpr2_preflight_status.md",
            "\n".join(
                [
                    "# QD-PR2 Preflight Status",
                    "",
                    "Status: PASS",
                    "Training executed: no",
                    "API called: no",
                    "Synthetic generated: no",
                    f"QD-B1 checkpoint: {relpath(config.qd_b1_checkpoint_dir)}",
                    f"Display name: {config.display_name}",
                    f"Removed module: {config.removed_module}",
                    f"Strict module ablation: {config.strict_module_ablation}",
                    f"Force pair training: {config.force_pair_training}",
                    f"Pair training enabled: {pair_training_enabled}",
                    f"Dataloader mode: {actual_dataloader_mode}",
                    f"Train pairs: {len(setup.get('train_pairs', []))}",
                    f"Dev pairs: {len(setup.get('dev_pairs', []))}",
                    f"lambda_point: {config.lambda_point}",
                    f"lambda_pair: {config.lambda_pair}",
                    f"lambda_anchor: {config.lambda_anchor}",
                    f"lambda_mono: {config.lambda_mono}",
                ]
            ),
        )
        return {
            "status": "preflight_pass",
            "pair_training_enabled": pair_training_enabled,
            "force_pair_training": config.force_pair_training,
            "dataloader_mode": actual_dataloader_mode,
            "train_pairs": len(setup.get("train_pairs", [])),
            "dev_pairs": len(setup.get("dev_pairs", [])),
        }

    require_cuda_if_requested()
    set_seed(config.seed)
    train_rows = limit_rows(train_rows_full, config.max_train_samples)
    dev_rows = limit_rows(read_split(config.data_dir, "dev"), config.max_eval_samples)
    test_rows = limit_rows(read_split(config.data_dir, "test"), config.max_eval_samples)
    train_pairs = limit_rows(read_qdpr2_pair_shards("train"), config.max_train_pairs) if pair_training_enabled else []
    dev_pairs = limit_rows(read_qdpr2_pair_shards("dev"), config.max_dev_pairs) if pair_training_enabled else []
    model, reference_model, tokenizer = load_model_and_tokenizer(config, class_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if reference_model is not None:
        reference_model.to(device)
    dev_loader = make_pointwise_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_pointwise_dataloader(test_rows, tokenizer, config, "test", shuffle=False)
    train_loader = (
        make_pair_dataloader(train_pairs, tokenizer, config, shuffle=True)
        if pair_training_enabled
        else make_pointwise_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    )
    best_dir = config.checkpoint_output_dir / "best"
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    metrics_history: list[dict[str, Any]] = []
    loss_debug_history: list[dict[str, Any]] = []
    best_score: float | None = None
    global_step = 0
    start = time.time()
    num_epochs = int(math.ceil(config.num_train_epochs))
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"{config.exp_name} {config.ablation_name}")
    try:
        for epoch in range(num_epochs):
            if epoch >= config.num_train_epochs:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                if pair_training_enabled:
                    pair_metadata = batch.pop("pair_metadata")
                    batch_size = len(pair_metadata)
                    win_labels = batch.pop("win_labels").to(device)
                    lose_labels = batch.pop("lose_labels").to(device)
                    label_gap = batch.pop("label_gap").to(device)
                    low_high = batch.pop("low_high").to(device)
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**batch)
                    _, logits = get_loss_logits(outputs)
                    ref_logits = None
                    if reference_model is not None:
                        with torch.no_grad():
                            ref_outputs = reference_model(**batch)
                            _, ref_logits = get_loss_logits(ref_outputs)
                    loss, debug = total_anchored_pairwise_training_loss(
                        logits[:batch_size].float(),
                        logits[batch_size:].float(),
                        ref_logits[:batch_size].float() if ref_logits is not None else None,
                        ref_logits[batch_size:].float() if ref_logits is not None else None,
                        win_labels,
                        lose_labels,
                        label_gap,
                        low_high,
                        model.class_weights,
                        lambda_point=config.lambda_point,
                        lambda_pair=config.lambda_pair,
                        lambda_anchor=config.lambda_anchor,
                        lambda_mono=config.lambda_mono,
                        margin_scale=config.margin_scale,
                        low_high_margin=config.low_high_margin,
                        low_high_weight=config.low_high_weight,
                        gap_weight=config.gap_weight,
                    )
                else:
                    batch.pop("metadata")
                    labels = batch["labels"].to(device)
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**batch)
                    _, logits = get_loss_logits(outputs)
                    ref_logits = None
                    if reference_model is not None:
                        with torch.no_grad():
                            ref_outputs = reference_model(**batch)
                            _, ref_logits = get_loss_logits(ref_outputs)
                    loss, debug = total_anchored_pointwise_training_loss(
                        logits.float(),
                        ref_logits.float() if ref_logits is not None else None,
                        labels,
                        model.class_weights,
                        lambda_point=config.lambda_point,
                        lambda_anchor=config.lambda_anchor,
                        lambda_mono=config.lambda_mono,
                    )
                (loss / config.gradient_accumulation_steps).backward()
                running_loss += float(loss.detach().cpu())
                debug.update(
                    {
                        "lambda_point": config.lambda_point,
                        "lambda_pair": config.lambda_pair,
                        "lambda_anchor": config.lambda_anchor,
                        "lambda_mono": config.lambda_mono,
                        "force_pair_training": config.force_pair_training,
                        "use_pair_training": pair_training_enabled,
                        "dataloader_mode": actual_dataloader_mode,
                        "strict_module_ablation": config.strict_module_ablation,
                    }
                )
                loss_debug_history.append({"epoch": epoch + 1, "step": step, "global_step": global_step, **debug})
                if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    elapsed = time.time() - start
                    avg_loss = running_loss / max(1, step)
                    steps_per_second = global_step / max(1e-9, elapsed)
                    eta = max(0, total_steps - global_step) / max(1e-9, steps_per_second)
                    progress.render(global_step, epoch + 1, avg_loss, scheduler.get_last_lr()[0], elapsed, eta)
                    if global_step >= total_steps:
                        break
            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev", config)
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            metrics_history.append(dev_result.metrics)
            selected = score_is_better(dev_result.metrics, best_score, config)
            print(
                f"dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"monotonic={dev_result.metrics['monotonic_violation_rate']:.4f}, selected={selected}",
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
    dev_final = evaluate(model, dev_loader, device, "dev", config)
    test_result = evaluate(model, test_loader, device, "test", config)
    for result in [dev_final, test_result]:
        result.metrics["epoch"] = "best"
        result.metrics["global_step"] = global_step
        save_predictions(config.output_dir, result)
        save_eval_arrays(config.output_dir, result.metrics["split"], result)
    save_dev_test_npz(config.output_dir, dev_final, test_result)
    save_metrics_history(config.output_dir, metrics_history)
    if loss_debug_history:
        write_csv(config.output_dir / "tables" / "loss_debug_history.csv", loss_debug_history)
        write_csv(config.output_dir / "logs" / "loss_debug_history.csv", loss_debug_history)
    save_metric_tables(config.output_dir, [dev_final, test_result])
    save_pairwise_diagnostics(config.output_dir, dev_pairs, dev_final, "dev")
    copy_dev_history_to_logs(config.output_dir)
    normalize_run_files(config.output_dir, config, "completed")
    return {"metrics": [dev_final.metrics, test_result.metrics], "best_dir": str(best_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QD-PR2 anchored pairwise ordinal scorer.")
    parser.add_argument("--config_path", type=Path, default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
    parser.add_argument("--qd_b1_checkpoint_dir", type=Path, default=None)
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
    parser.add_argument("--selection_metric", default=None)
    parser.add_argument("--selection_mode", choices=["min", "max"], default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> QDPR2TrainConfig:
    raw = load_yaml_config(args.config_path)
    smoke_raw = raw.get("smoke", {}) if isinstance(raw.get("smoke", {}), dict) else {}
    smoke = bool(args.smoke or raw.get("smoke_mode", False))
    section = smoke_raw if smoke and smoke_raw else {}
    batch_size = args.per_device_train_batch_size or int(section.get("batch_size", raw.get("per_device_train_batch_size", 32)))
    grad_accum = args.gradient_accumulation_steps
    if grad_accum is None:
        grad_accum = int(section.get("gradient_accumulation_steps", raw.get("gradient_accumulation_steps", 0)) or 0)
    if grad_accum <= 0:
        grad_accum = max(1, math.ceil(int(raw.get("effective_batch_size", 128)) / max(1, batch_size)))
    epochs = args.num_train_epochs
    if epochs is None:
        epochs = float(section.get("epochs", raw.get("epochs", QDPR2_DEFAULT_EPOCHS)))
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else section.get("max_train_samples")
    max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else section.get("max_eval_samples")
    max_train_pairs = args.max_train_pairs if args.max_train_pairs is not None else section.get("max_train_pairs")
    max_dev_pairs = args.max_dev_pairs if args.max_dev_pairs is not None else section.get("max_dev_pairs")
    default_output_dir = qdpr2_run_dir(smoke=smoke)
    default_checkpoint_dir = qdpr2_checkpoint_dir(smoke=smoke)
    raw_output_dir = Path(raw["output_dir"]) if raw.get("output_dir") else default_output_dir
    raw_checkpoint_dir = Path(raw["checkpoint_output_dir"]) if raw.get("checkpoint_output_dir") else default_checkpoint_dir
    ablation_name = str(raw.get("ablation_name", "full_qdpr2"))
    lambda_point = float(raw.get("lambda_point", QDPR2_DEFAULT_LAMBDA_POINT))
    lambda_pair = float(raw.get("lambda_pair", QDPR2_DEFAULT_LAMBDA_PAIR))
    lambda_anchor = float(raw.get("lambda_anchor", QDPR2_DEFAULT_LAMBDA_ANCHOR))
    lambda_mono = float(raw.get("lambda_mono", QDPR2_DEFAULT_LAMBDA_MONO))
    force_pair_training = bool(raw.get("force_pair_training", False))
    resolved_mode = "pair" if lambda_pair != 0.0 or force_pair_training else "pointwise"
    return QDPR2TrainConfig(
        exp_name=str(raw.get("exp_name", "Exp9 QD-PR2 anchored pairwise ordinal fine-tuning")),
        ablation_name=ablation_name,
        display_name=str(raw.get("display_name", ablation_name)),
        run_id=str(raw.get("run_id", QDPR2_RUN_ID)),
        dataset_id=str(raw.get("dataset_id", "A4_question_seed42_human_only")),
        split=str(raw.get("split", "question_seed42")),
        input_template=str(raw.get("input_template", "A4_question_answer_metric_rubric_metadata")),
        model_name_or_path=args.model_name_or_path or str(raw.get("model_name_or_path", "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")),
        data_dir=args.data_dir or Path(raw.get("data_dir", QDPR2_DATASET_DIR)),
        output_dir=args.output_dir or raw_output_dir,
        checkpoint_output_dir=args.checkpoint_output_dir or raw_checkpoint_dir,
        qd_b1_checkpoint_dir=args.qd_b1_checkpoint_dir or Path(raw.get("qd_b1_checkpoint_dir", QD_B1_CHECKPOINT_DIR)),
        head_type=str(raw.get("head_type", "independent_ordinal")),
        objective=str(raw.get("objective", "anchored_risk_aware_pairwise_ordinal")),
        loss=str(raw.get("loss", "weighted_ordinal_bce_plus_pairwise_anchor_monotonic")),
        use_synthetic=bool(raw.get("use_synthetic", False)),
        pair_dataset_size_train=int(raw.get("pair_dataset_size_train", QDPR2_DEFAULT_TRAIN_PAIR_COUNT)),
        pair_dataset_size_dev=int(raw.get("pair_dataset_size_dev", QDPR2_DEFAULT_DEV_PAIR_COUNT)),
        pair_sampling_seed=int(raw.get("pair_sampling_seed", DEFAULT_PAIR_SAMPLING_SEED)),
        max_pairs_per_record=int(raw.get("max_pairs_per_record", QDPR2_DEFAULT_MAX_PAIRS_PER_RECORD)),
        max_pairs_per_low_record=int(raw.get("max_pairs_per_low_record", QDPR2_DEFAULT_MAX_PAIRS_PER_LOW_RECORD)),
        margin_scale=float(raw.get("margin_scale", DEFAULT_MARGIN_SCALE)),
        low_high_margin=float(raw.get("low_high_margin", DEFAULT_LOW_HIGH_MARGIN)),
        low_high_weight=float(raw.get("low_high_weight", DEFAULT_LOW_HIGH_WEIGHT)),
        gap_weight=float(raw.get("gap_weight", DEFAULT_GAP_WEIGHT)),
        lambda_point=lambda_point,
        lambda_pair=lambda_pair,
        lambda_anchor=lambda_anchor,
        lambda_mono=lambda_mono,
        force_pair_training=force_pair_training,
        dataloader_mode=str(raw.get("dataloader_mode", resolved_mode)),
        strict_module_ablation=bool(raw.get("strict_module_ablation", False)),
        removed_module=str(raw.get("removed_module", "none")),
        w_min=float(raw.get("w_min", DEFAULT_W_MIN)),
        w_max=float(raw.get("w_max", DEFAULT_W_MAX)),
        max_length=args.max_length or int(raw.get("max_length", 2048)),
        num_train_epochs=float(epochs),
        learning_rate=float(args.learning_rate or raw.get("learning_rate", QDPR2_DEFAULT_LEARNING_RATE)),
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
    result = train(config, preflight_only=args.preflight_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
