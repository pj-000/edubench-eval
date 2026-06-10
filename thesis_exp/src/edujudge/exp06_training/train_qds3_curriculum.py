"""Train Exp6 QD-S3 synthetic pretrain then human fine-tune model."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04.train_objective import (
    ObjectiveTrainConfig,
    StepProgressBar,
    evaluate,
    format_duration,
    get_loss_logits,
    load_model_and_tokenizer,
    make_dataloader,
    save_checkpoint,
    save_metrics_history,
    save_metrics_tables,
    save_predictions,
    score_is_better,
    selection_metric_value,
    set_seed,
    train,
)
from thesis_exp.src.edujudge.exp06_training import (
    QD_S3_CHECKPOINT_DIR,
    QD_S3_DATASET_DIR,
    QD_S3_RUN_DIR,
    QD_S3_RUN_ID,
    QUESTION_SPLIT_DIR,
    ensure_exp06_training_dirs,
)
from thesis_exp.src.edujudge.exp06_training.collect_qds3_results import collect
from thesis_exp.src.edujudge.exp06_training.train_qds1_ordinal import (
    _api_key_hits,
    _failures_to_text,
    copy_dev_history_to_logs,
    require_cuda_if_requested,
    write_language_metrics,
)
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_json, write_text


FORBIDDEN_TEXT_MARKERS = [
    "rationale_for_label",
    "expected_failure_against_rubric",
    "error_type",
    "label_source",
]
CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def _read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / f"{split}.jsonl")


def _split_keys(split: str, field: str) -> set[str]:
    return {str(row.get(field)) for row in read_jsonl(QUESTION_SPLIT_DIR / f"{split}.jsonl") if row.get(field)}


def _tracked_weight_files() -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS)


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


def qds3_sanity_rows(args: argparse.Namespace, formal: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "") -> None:
        rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})

    stage1_rows = _read_split(args.data_dir, "stage1_train")
    stage2_rows = _read_split(args.data_dir, "stage2_train")
    train_rows = _read_split(args.data_dir, "train")
    dev_rows = _read_split(args.data_dir, "dev")
    test_rows = _read_split(args.data_dir, "test")

    add("stage1 rows = 384", len(stage1_rows) == 384, len(stage1_rows))
    add("stage2 rows = 3326", len(stage2_rows) == 3326, len(stage2_rows))
    add("dev rows = 1107", len(dev_rows) == 1107, len(dev_rows))
    add("test rows = 1103", len(test_rows) == 1103, len(test_rows))
    add("stage1 synthetic only", all(row.get("source_type") == "synthetic" for row in stage1_rows))
    add("stage2 human only", all(row.get("source_type") == "human" for row in stage2_rows))
    add("train mirror is human only", len(train_rows) == 3326 and all(row.get("source_type") == "human" for row in train_rows))
    add("dev synthetic rows = 0", sum(1 for row in dev_rows if row.get("source_type") == "synthetic") == 0)
    add("test synthetic rows = 0", sum(1 for row in test_rows if row.get("source_type") == "synthetic") == 0)
    add("stage1 label provenance pseudo_label", all(row.get("label_provenance") == "pseudo_label" for row in stage1_rows))
    add("stage2/dev/test label provenance human_score", all(row.get("label_provenance") == "human_score" for row in stage2_rows + dev_rows + test_rows))

    dev_test_q = _split_keys("dev", "question_key") | _split_keys("test", "question_key")
    dev_test_t = _split_keys("dev", "triple_key") | _split_keys("test", "triple_key")
    q_overlap = sorted({str(row.get("source_question_key")) for row in stage1_rows if str(row.get("source_question_key")) in dev_test_q})
    t_overlap = sorted({str(row.get("source_triple_key")) for row in stage1_rows if str(row.get("source_triple_key")) in dev_test_t})
    add("synthetic source question overlap with dev/test = 0", not q_overlap, len(q_overlap))
    add("synthetic source triple overlap with dev/test = 0", not t_overlap, len(t_overlap))

    marker_hits = {marker: 0 for marker in FORBIDDEN_TEXT_MARKERS}
    for row in stage1_rows + stage2_rows:
        text = str(row.get("text") or "")
        for marker in FORBIDDEN_TEXT_MARKERS:
            if marker in text:
                marker_hits[marker] += 1
    add("training text excludes rationale/error_type/label_source markers", all(value == 0 for value in marker_hits.values()), marker_hits)

    add(
        "formal run has no max sample limits",
        not formal or (args.stage1_max_train_samples is None and args.stage2_max_train_samples is None and args.max_eval_samples is None),
        (
            f"stage1_max_train_samples={args.stage1_max_train_samples}, "
            f"stage2_max_train_samples={args.stage2_max_train_samples}, max_eval_samples={args.max_eval_samples}"
        ),
    )
    add("no tracked checkpoint/weight files", not _tracked_weight_files(), ",".join(_tracked_weight_files()))
    return rows


def stage1_config(args: argparse.Namespace) -> ObjectiveTrainConfig:
    return ObjectiveTrainConfig(
        objective_id=f"{QD_S3_RUN_ID}_stage1_synthetic_pretrain",
        objective_type="ordinal",
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir / "stage1_synthetic_pretrain",
        checkpoint_output_dir=args.checkpoint_output_dir / "stage1_synthetic_pretrain",
        max_length=args.max_length,
        num_train_epochs=args.stage1_num_train_epochs,
        learning_rate=args.stage1_learning_rate,
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
        max_train_samples=args.stage1_max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_only=False,
        checkpoint_dir=args.checkpoint_dir,
        regression_loss="none",
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )


def stage2_config(args: argparse.Namespace, stage1_best_dir: Path) -> ObjectiveTrainConfig:
    return ObjectiveTrainConfig(
        objective_id=QD_S3_RUN_ID,
        objective_type="ordinal",
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_output_dir=args.checkpoint_output_dir / "stage2_human_finetune",
        max_length=args.max_length,
        num_train_epochs=args.stage2_num_train_epochs,
        learning_rate=args.stage2_learning_rate,
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
        max_train_samples=args.stage2_max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_only=args.eval_only,
        checkpoint_dir=stage1_best_dir,
        regression_loss="none",
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )


def run_stage1_pretrain(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    config = stage1_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    train_rows = _read_split(args.data_dir, "stage1_train")
    dev_rows = _read_split(args.data_dir, "dev")
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_eval_samples:
        dev_rows = dev_rows[: config.max_eval_samples]

    model, tokenizer = load_model_and_tokenizer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = make_dataloader(train_rows, tokenizer, config, "train", shuffle=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = max(1, int(math.ceil(config.num_train_epochs * update_steps_per_epoch)))
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_dir = config.checkpoint_output_dir / "best"
    best_score: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = None
    global_step = 0
    metrics_history: list[dict[str, Any]] = []
    start = time.time()
    last_log_time = start
    last_log_step = 0
    num_epochs = int(math.ceil(config.num_train_epochs))
    print(
        "stage1 schedule: "
        f"objective={config.objective_id} train_batches_per_epoch={len(train_loader)} "
        f"update_steps_per_epoch={update_steps_per_epoch} total_update_steps={total_steps} "
        f"log_steps={config.log_steps}",
        flush=True,
    )
    progress = StepProgressBar(total_steps, config.progress_bar, desc="Exp6 QD-S3 stage1")
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
                    eta_seconds = max(0, total_steps - global_step) / max(1e-9, avg_steps_per_second)
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
                            f"stage1 epoch={epoch + 1} step={global_step}/{total_steps} "
                            f"loss={avg_loss:.4f} lr={current_lr:.2e} "
                            f"elapsed={format_duration(elapsed)} "
                            f"eta={format_duration(remaining_steps / max(1e-9, steps_per_second))} "
                            f"steps_per_min={steps_per_second * 60:.2f}",
                            flush=True,
                        )

            progress.newline()
            dev_result = evaluate(model, dev_loader, device, "dev", config)
            dev_result.metrics["epoch"] = epoch + 1
            dev_result.metrics["global_step"] = global_step
            dev_result.metrics["stage"] = "stage1_synthetic_pretrain"
            metrics_history.append(dev_result.metrics)
            selected = bool(score_is_better(dev_result.metrics, best_score, config))
            elapsed = time.time() - start
            print(
                f"stage1 dev epoch={epoch + 1}: MAE_label={dev_result.metrics['MAE_label']:.4f}, "
                f"MAE_expected={dev_result.metrics['MAE_expected']:.4f}, "
                f"Exact={dev_result.metrics['Exact Match']:.4f}, "
                f"low_to_high={dev_result.metrics['low_to_high_rate']:.4f}, "
                f"selected={selected}, elapsed={format_duration(elapsed)}",
                flush=True,
            )
            if selected:
                best_score = selection_metric_value(dev_result.metrics, config)
                best_metrics = dict(dev_result.metrics)
                best_epoch = epoch + 1
                save_checkpoint(model, tokenizer, best_dir, config, dev_result.metrics)
                save_predictions(config.output_dir, dev_result, suffix="stage1_dev_best")
                save_metrics_tables(config.output_dir, [dev_result], config.objective_type)
    finally:
        progress.close()

    if not (best_dir / "state_dict.pt").exists():
        raise RuntimeError(f"Stage1 best checkpoint was not saved: {best_dir}")
    save_metrics_history(config.output_dir, metrics_history)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_csv(logs_dir / "stage1_dev_metrics_history.csv", metrics_history)
    shutil.copy2(config.output_dir / "logs" / "training_log.csv", logs_dir / "stage1_training_log.csv")
    write_text(
        config.output_dir / "stage1_run_summary.md",
        "\n".join(
            [
                "# Exp6 QD-S3 Stage 1 Synthetic Pretrain",
                "",
                "Status: completed",
                "Train: 384 synthetic low-score pseudo-label rows",
                "Dev: question_seed42 human-only",
                "Test evaluation: no",
                f"Epochs configured: {config.num_train_epochs}",
                f"Selected epoch: {best_epoch}",
                f"Best dev {config.selection_metric}: {best_score}",
                f"Best checkpoint: {relpath(best_dir)}",
            ]
        ),
    )
    return {
        "status": "completed",
        "best_dir": best_dir,
        "best_epoch": best_epoch,
        "best_global_step": best_metrics.get("global_step") if best_metrics else None,
        "best_metric": best_score,
        "metrics_history": metrics_history,
    }


def overwrite_qds3_metadata(run_dir: Path, args: argparse.Namespace, stage1_result: dict[str, Any] | None, status: str) -> None:
    stage2_best_metadata_path = args.checkpoint_output_dir / "stage2_human_finetune" / "best" / "exp04_head_metadata.json"
    stage2_metadata = json.loads(stage2_best_metadata_path.read_text(encoding="utf-8")) if stage2_best_metadata_path.exists() else {}
    stage1_metadata_path = args.checkpoint_output_dir / "stage1_synthetic_pretrain" / "best" / "exp04_head_metadata.json"
    stage1_metadata = json.loads(stage1_metadata_path.read_text(encoding="utf-8")) if stage1_metadata_path.exists() else {}
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": "Exp6 synthetic low-score training",
            "run_id": QD_S3_RUN_ID,
            "status": status,
            "fixed_input_template": "A4_question_answer_metric_rubric_metadata",
            "model_name_or_path": args.model_name_or_path,
            "objective_type": "ordinal",
            "loss": "ordinary ordinal BCEWithLogitsLoss",
            "class_weights": "disabled",
            "asymmetric_loss": "disabled",
            "synthetic_label_provenance": "pseudo_label",
            "stage1": {
                "train": "384 synthetic low-score pseudo-label rows",
                "test_evaluation": "no",
                "epochs_configured": args.stage1_num_train_epochs,
                "learning_rate": args.stage1_learning_rate,
                "best_epoch": stage1_metadata.get("best_epoch") or (stage1_result or {}).get("best_epoch"),
                "best_global_step": stage1_metadata.get("best_global_step") or (stage1_result or {}).get("best_global_step"),
                "best_selection_metric_value": stage1_metadata.get("best_selection_metric_value") or (stage1_result or {}).get("best_metric"),
            },
            "stage2": {
                "train": "3326 question_seed42 human-only rows",
                "epochs_configured": args.stage2_num_train_epochs,
                "learning_rate": args.stage2_learning_rate,
                "checkpoint_selection": f"dev {args.selection_metric} ({args.selection_mode})",
                "best_epoch": stage2_metadata.get("best_epoch"),
                "best_global_step": stage2_metadata.get("best_global_step"),
                "best_selection_metric_name": stage2_metadata.get("best_selection_metric_name", f"dev_{args.selection_metric}"),
                "best_selection_metric_value": stage2_metadata.get("best_selection_metric_value"),
                "selection_direction": stage2_metadata.get("selection_direction", args.selection_mode),
            },
            "best_epoch": stage2_metadata.get("best_epoch"),
            "best_global_step": stage2_metadata.get("best_global_step"),
            "stage2_best_epoch": stage2_metadata.get("best_epoch"),
            "stage2_best_global_step": stage2_metadata.get("best_global_step"),
            "stage1_checkpoint": relpath(args.checkpoint_output_dir / "stage1_synthetic_pretrain" / "best"),
            "stage2_checkpoint": relpath(args.checkpoint_output_dir / "stage2_human_finetune" / "best"),
        },
    )
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    rows = read_csv(metrics_path) if metrics_path.exists() else []
    test = next((row for row in rows if row.get("split") == "test"), {})
    lines = [
        "# Exp6 QD-S3 Synthetic Pretrain then Human Fine-tune",
        "",
        f"Status: `{status}`",
        f"Model: `{args.model_name_or_path}`",
        "Stage 1 train: 384 synthetic low-score pseudo-label rows",
        "Stage 2 train: 3326 question_seed42 human-only rows",
        "Dev/test: question_seed42 human-only",
        "Loss: ordinary ordinal BCEWithLogitsLoss",
        "Class weights: disabled",
        "Asymmetric loss: disabled",
        f"Stage 1 epochs: {args.stage1_num_train_epochs}",
        f"Stage 2 epochs: {args.stage2_num_train_epochs}",
        f"Checkpoint selection: dev `{args.selection_metric}` ({args.selection_mode})",
        f"Selected stage2 epoch: {stage2_metadata.get('best_epoch', 'NA')}",
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
    write_text(run_dir / "run_summary.md", "\n".join(lines))


def postprocess_outputs(run_dir: Path, args: argparse.Namespace, stage1_result: dict[str, Any] | None = None, strict: bool = True) -> None:
    write_language_metrics(run_dir)
    copy_dev_history_to_logs(run_dir)
    logs_dir = run_dir / "logs"
    training_log = logs_dir / "training_log.csv"
    if training_log.exists():
        shutil.copy2(training_log, logs_dir / "stage2_training_log.csv")
    overwrite_qds3_metadata(run_dir, args, stage1_result, "completed")
    api_hits = _api_key_hits(run_dir)
    if api_hits and strict:
        raise RuntimeError(f"API key-like strings found in outputs: {api_hits}")
    if not args.smoke:
        collect(run_dir)


def train_qds3(args: argparse.Namespace) -> None:
    ensure_exp06_training_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)

    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    smoke = bool(args.smoke)
    if smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if not smoke and not formal and not args.postprocess_only and not args.preflight_only:
        raise RuntimeError("Formal QD-S3 training requires FORMAL_RUN=1.")

    sanity_rows = qds3_sanity_rows(args, formal)
    sanity_path = args.output_dir / "tables" / ("smoke_preflight_qds3.csv" if smoke else "preflight_qds3.csv")
    write_csv(sanity_path, sanity_rows)
    failures = [row for row in sanity_rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QD-S3 preflight failed:\n{_failures_to_text(failures)}")
    if args.preflight_only:
        write_text(
            args.output_dir / "qds3_preflight_status.md",
            "\n".join(
                [
                    "# QD-S3 Preflight Status",
                    "",
                    "Status: PASS",
                    f"Formal mode: {formal}",
                    f"Smoke mode: {smoke}",
                    "Training executed: no",
                    "API called: no",
                    "Synthetic generated: no",
                ]
            ),
        )
        return
    if args.postprocess_only:
        postprocess_outputs(args.output_dir, args)
        return

    require_cuda_if_requested()
    stage1_result = run_stage1_pretrain(args)
    stage2_cfg = stage2_config(args, Path(stage1_result["best_dir"]))
    stage2_result = train(stage2_cfg)
    postprocess_outputs(args.output_dir, args, stage1_result)
    write_text(
        args.output_dir / "qds3_training_status.md",
        "\n".join(
            [
                "# QD-S3 Training Status",
                "",
                "Status: completed",
                f"Smoke: {smoke}",
                "API called: no",
                "Synthetic generated: no",
                "Stage 1: synthetic low-score pseudo-label pretrain",
                "Stage 2: question_seed42 human-only fine-tune",
                "Loss: ordinary ordinal BCEWithLogitsLoss",
                "Synthetic labels: pseudo_label metadata only",
            ]
        ),
    )
    print(json.dumps({"stage1": _jsonify(stage1_result), "stage2": _jsonify(stage2_result)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp6 QD-S3 synthetic pretrain then human fine-tune model.")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--data_dir", type=Path, default=QD_S3_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=QD_S3_RUN_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=QD_S3_CHECKPOINT_DIR)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stage1_num_train_epochs", type=float, default=3.0)
    parser.add_argument("--stage2_num_train_epochs", type=float, default=10.0)
    parser.add_argument("--stage1_learning_rate", type=float, default=2e-5)
    parser.add_argument("--stage2_learning_rate", type=float, default=2e-5)
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
    parser.add_argument("--stage1_max_train_samples", type=int, default=None)
    parser.add_argument("--stage2_max_train_samples", type=int, default=None)
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


def main() -> None:
    args = parse_args()
    train_qds3(args)
    print(f"QD-S3 output: {relpath(args.output_dir)}")


if __name__ == "__main__":
    main()
