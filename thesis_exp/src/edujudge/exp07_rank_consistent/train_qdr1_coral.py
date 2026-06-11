"""Train Exp7 QD-R1 CORAL-style rank-consistent ordinal model."""

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

from thesis_exp.src.edujudge.exp02.train_ce_baseline import compute_metrics
from thesis_exp.src.edujudge.exp04.train_objective import (
    EvalResult,
    StepProgressBar,
    add_extra_metrics,
    base_prediction_row,
    format_duration,
    get_loss_logits,
    grouped_metrics,
    save_dev_test_npz,
    save_eval_arrays,
    save_metrics_history,
    save_metrics_tables,
    save_predictions,
    set_seed,
)
from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_DATASET_DIR,
    EXP07_RUN_ID,
    ensure_exp07_dirs,
    exp07_checkpoint_dir,
    exp07_run_dir,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.coral_head import CoralOrdinalHead
from thesis_exp.src.edujudge.exp07_rank_consistent.data import dataset_sanity_rows, limit_rows, read_split
from thesis_exp.src.edujudge.exp07_rank_consistent.losses import coral_ordinal_loss
from thesis_exp.src.edujudge.exp07_rank_consistent.metrics import (
    monotonic_violation_for_probs,
    monotonicity_metrics,
    prediction_from_probs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_text


@dataclass
class CoralTrainConfig:
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_dtype(config: CoralTrainConfig, torch: Any) -> Any:
    if config.fp16:
        return torch.float16
    if config.bf16 == "true":
        return torch.bfloat16
    if config.bf16 == "false":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def make_coral_scorer_class(torch: Any, nn: Any):
    class CoralScoringModel(nn.Module):
        def __init__(self, backbone: Any, hidden_size: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.coral_head = CoralOrdinalHead(hidden_size)
            self.hidden_size = int(hidden_size)
            self.num_outputs = 4
            self.config = backbone.config
            self.config.num_labels = 4
            self.config.problem_type = "multi_label_classification"

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
            logits = self.coral_head(pooled)
            loss = coral_ordinal_loss(logits.float(), labels.long()) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return CoralScoringModel


def build_model(torch: Any, nn: Any, AutoModel: Any, config: CoralTrainConfig, dtype: Any) -> Any:
    source = str(config.model_name_or_path)
    metadata_path = config.checkpoint_dir / "exp07_head_metadata.json" if config.checkpoint_dir else None
    if metadata_path and metadata_path.exists():
        source = read_json(metadata_path)["base_model_name_or_path"]
    backbone = AutoModel.from_pretrained(
        source,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size for Exp7 CORAL head.")
    model = make_coral_scorer_class(torch, nn)(backbone, int(hidden_size))
    if config.checkpoint_dir:
        state_dict_path = config.checkpoint_dir / "state_dict.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Missing checkpoint state dict: {relpath(state_dict_path)}")
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return model


def load_model_and_tokenizer(config: CoralTrainConfig):
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
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def collate_fn(tokenizer: Any, config: CoralTrainConfig):
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


def make_dataloader(rows: list[dict[str, Any]], tokenizer: Any, config: CoralTrainConfig, split: str, shuffle: bool):
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
            probs_cpu = torch.sigmoid(logits_cpu)
            logits_chunks.append(logits_cpu.numpy())
            probs_chunks.append(probs_cpu.numpy())
            for row, probs, logit in zip(metadata, probs_cpu.tolist(), logits_cpu.tolist()):
                pred_label, expected_score = prediction_from_probs(probs)
                human_mean = float(row["human_mean_5"])
                label_5 = int(row["label_5"])
                prediction = base_prediction_row(
                    split,
                    row,
                    label_5,
                    human_mean,
                    pred_label,
                    float(pred_label),
                    expected_score,
                )
                prediction["pred_label"] = int(pred_label)
                prediction["head_type"] = "coral"
                prediction["confidence"] = float(np.mean([abs(prob - 0.5) * 2 for prob in probs]))
                prediction["monotonic_violation"] = monotonic_violation_for_probs(probs)
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
    metrics = compute_metrics(predictions)
    metrics = add_extra_metrics(metrics, predictions, "ordinal")
    metrics.update(monotonicity_metrics(probs_arr))
    metrics["split"] = split
    metrics["loss"] = total_loss / steps if steps else float("nan")
    return EvalResult(metrics, predictions, logits_arr, probs_arr, labels_arr, targets_arr, record_ids)


def selection_metric_name(config: CoralTrainConfig) -> str:
    return f"dev_{config.selection_metric}"


def selection_metric_value(metrics: dict[str, Any], config: CoralTrainConfig) -> float:
    if config.selection_metric not in metrics:
        raise KeyError(f"Selection metric {config.selection_metric!r} missing from dev metrics.")
    return float(metrics[config.selection_metric])


def score_is_better(metrics: dict[str, Any], best_score: float | None, config: CoralTrainConfig) -> bool:
    score = selection_metric_value(metrics, config)
    if best_score is None:
        return True
    if config.selection_mode == "min":
        return score < best_score
    if config.selection_mode == "max":
        return score > best_score
    raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, config: CoralTrainConfig, metrics: dict[str, Any]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / "state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    tokenizer.save_pretrained(output_dir)
    if hasattr(model.backbone, "config"):
        model.backbone.config.save_pretrained(output_dir)
    write_json(
        output_dir / "exp07_head_metadata.json",
        {
            "base_model_name_or_path": config.model_name_or_path,
            "experiment": "Exp7 rank-consistent ordinal scorer",
            "run_id": EXP07_RUN_ID,
            "head_type": "coral",
            "objective": "CORAL_rank_consistent_ordinal",
            "loss": "BCEWithLogits ordinal",
            "threshold_parameterization": "tau_1=tau_base; tau_t=tau_{t-1}+softplus(delta_t)",
            "num_outputs": getattr(model, "num_outputs", 4),
            "hidden_size": getattr(model, "hidden_size", None),
            "state_dict": state_dict_path.name,
            "best_epoch": metrics.get("epoch"),
            "best_global_step": metrics.get("global_step"),
            "best_selection_metric_name": selection_metric_name(config),
            "best_selection_metric_value": selection_metric_value(metrics, config),
            "selection_direction": config.selection_mode,
            "synthetic_used": False,
            "class_weights_used": False,
            "calibration_used": False,
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "dev_metrics.json", metrics)


def _read_metrics_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_monotonicity_table(output_dir: Path, results: list[EvalResult]) -> None:
    rows = [monotonicity_metrics(result.probs, result.metrics["split"]) for result in results]
    write_csv(output_dir / "tables" / "monotonicity_metrics.csv", rows)


def write_language_metrics(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for split in ["dev", "test"]:
        path = output_dir / "predictions" / f"predictions_{split}.jsonl"
        if path.exists():
            rows.extend(grouped_metrics(read_jsonl(path), split, "language", "ordinal"))
    if rows:
        write_csv(output_dir / "tables" / "language_level_metrics.csv", rows)


def copy_dev_history_to_logs(output_dir: Path) -> None:
    src = output_dir / "tables" / "dev_metrics_history.csv"
    dst = output_dir / "logs" / "dev_metrics_history.csv"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def normalize_run_files(output_dir: Path, config: CoralTrainConfig, status: str) -> None:
    best_metadata_path = config.checkpoint_output_dir / "best" / "exp07_head_metadata.json"
    best_metadata = read_json(best_metadata_path) if best_metadata_path.exists() else {}
    write_json(
        output_dir / "run_metadata.json",
        {
            "experiment": "Exp7 rank-consistent ordinal scorer",
            "run_id": EXP07_RUN_ID,
            "dataset_id": "QD-S0_human_only",
            "split": "question_seed42",
            "status": status,
            "input_template": "A4_question_answer_metric_rubric_metadata",
            "model_name_or_path": config.model_name_or_path,
            "head_type": "coral",
            "objective": "CORAL_rank_consistent_ordinal",
            "loss": "BCEWithLogits ordinal",
            "synthetic_used": False,
            "class_weights_used": False,
            "calibration_used": False,
            "checkpoint_selection": f"dev {config.selection_metric} ({config.selection_mode})",
            "best_epoch": best_metadata.get("best_epoch"),
            "best_global_step": best_metadata.get("best_global_step"),
            "best_selection_metric_name": best_metadata.get("best_selection_metric_name", selection_metric_name(config)),
            "best_selection_metric_value": best_metadata.get("best_selection_metric_value"),
            "selection_direction": best_metadata.get("selection_direction", config.selection_mode),
        },
    )
    lines = [
        "# Exp7 QD-R1 CORAL Run Summary",
        "",
        f"Status: `{status}`",
        f"Run ID: `{EXP07_RUN_ID}`",
        f"Model: `{config.model_name_or_path}`",
        "Dataset: `QD-S0_human_only`, question_seed42 human-only train/dev/test.",
        "Fixed input: `A4_question_answer_metric_rubric_metadata` text field.",
        "Head: CORAL-style shared quality score with ordered thresholds.",
        "Loss: unweighted ordinal BCEWithLogits loss.",
        "Synthetic rows: none.",
        "Class weights, focal loss, and calibration: disabled.",
        f"Checkpoint selection: dev `{config.selection_metric}` ({config.selection_mode})",
    ]
    test = next((row for row in _read_metrics_rows(output_dir / "tables" / "metrics_summary.csv") if row.get("split") == "test"), {})
    mono = next((row for row in _read_metrics_rows(output_dir / "tables" / "monotonicity_metrics.csv") if row.get("split") == "test"), {})
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
                f"- low_to_high_rate: {test.get('low_to_high_rate', 'NA')}",
                f"- monotonic_violation_rate: {mono.get('monotonic_violation_rate', 'NA')}",
            ]
        )
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def require_cuda_if_requested() -> None:
    if os.environ.get("REQUIRE_CUDA", "0") != "1":
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - server environment check
        raise RuntimeError(f"REQUIRE_CUDA=1 but torch import failed: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is false.")


def _failures_to_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {row['check_name']}: {row['status']} ({row.get('details', '')})" for row in rows if row["status"] != "PASS")


def postprocess_outputs(output_dir: Path) -> None:
    write_language_metrics(output_dir)
    copy_dev_history_to_logs(output_dir)
    try:
        from thesis_exp.src.edujudge.exp07_rank_consistent.collect_exp07_results import collect

        collect()
    except FileNotFoundError:
        pass


def train(config: CoralTrainConfig, preflight_only: bool = False, postprocess_only: bool = False) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    ensure_exp07_dirs()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    if config.smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if formal and (config.max_train_samples is not None or config.max_eval_samples is not None):
        raise RuntimeError("FORMAL_RUN=1 cannot use max_train_samples or max_eval_samples.")
    if not config.smoke and not formal and not config.eval_only and not preflight_only and not postprocess_only:
        raise RuntimeError("Formal QD-R1 training requires FORMAL_RUN=1.")

    sanity_rows = dataset_sanity_rows(config.data_dir)
    sanity_name = "smoke_preflight_qdr1.csv" if config.smoke else "preflight_qdr1.csv"
    write_csv(config.output_dir / "tables" / sanity_name, sanity_rows)
    failures = [row for row in sanity_rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QD-R1 preflight failed:\n{_failures_to_text(failures)}")
    if preflight_only:
        write_text(
            config.output_dir / "qdr1_preflight_status.md",
            "\n".join(
                [
                    "# QD-R1 Preflight Status",
                    "",
                    "Status: PASS",
                    f"Formal mode: {formal}",
                    f"Smoke mode: {config.smoke}",
                    "Training executed: no",
                    "API called: no",
                    "Synthetic generated: no",
                ]
            ),
        )
        return {"status": "preflight_pass"}
    if postprocess_only:
        postprocess_outputs(config.output_dir)
        return {"status": "postprocessed"}

    require_cuda_if_requested()
    set_seed(config.seed)
    train_rows = limit_rows(read_split(config.data_dir, "train"), config.max_train_samples)
    dev_rows = limit_rows(read_split(config.data_dir, "dev"), config.max_eval_samples)
    test_rows = limit_rows(read_split(config.data_dir, "test"), config.max_eval_samples)
    model, tokenizer = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    test_loader = make_dataloader(test_rows, tokenizer, config, "test", shuffle=False)
    best_dir = config.checkpoint_output_dir / "best"
    metrics_history: list[dict[str, Any]] = []

    if config.eval_only:
        dev_result = evaluate(model, dev_loader, device, "dev")
        test_result = evaluate(model, test_loader, device, "test")
        for result in [dev_result, test_result]:
            save_predictions(config.output_dir, result)
            save_eval_arrays(config.output_dir, result.metrics["split"], result)
        save_dev_test_npz(config.output_dir, dev_result, test_result)
        save_metrics_tables(config.output_dir, [dev_result, test_result], "ordinal")
        write_monotonicity_table(config.output_dir, [dev_result, test_result])
        normalize_run_files(config.output_dir, config, "eval_only")
        postprocess_outputs(config.output_dir)
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
        f"run_id={EXP07_RUN_ID} head=coral train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} total_update_steps={total_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar, desc=f"Exp7 {EXP07_RUN_ID}")
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
                f"MAE_expected={dev_result.metrics['MAE_expected']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"monotonic_violation={dev_result.metrics['monotonic_violation_rate']:.6f}, "
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
    save_metrics_tables(config.output_dir, [dev_final, test_result], "ordinal")
    write_monotonicity_table(config.output_dir, [dev_final, test_result])
    normalize_run_files(config.output_dir, config, "completed")
    postprocess_outputs(config.output_dir)
    return {"metrics": [dev_final.metrics, test_result.metrics], "best_dir": str(best_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp7 QD-R1 CORAL ordinal scorer.")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--data_dir", type=Path, default=EXP07_DATASET_DIR)
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
    parser.add_argument("--selection_metric", default="MAE_label")
    parser.add_argument("--selection_mode", choices=["min", "max"], default="min")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> CoralTrainConfig:
    return CoralTrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir or exp07_run_dir(smoke=args.smoke),
        checkpoint_output_dir=args.checkpoint_output_dir or exp07_checkpoint_dir(smoke=args.smoke),
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
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
        smoke=args.smoke,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    result = train(config, preflight_only=args.preflight_only, postprocess_only=args.postprocess_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
