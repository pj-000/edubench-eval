"""Train or reuse one Exp3 input-ablation setting."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp02.train_ce_baseline import TrainConfig, compute_metrics, train
from thesis_exp.src.edujudge.exp03 import (
    EXP02_OUTPUT_DIR,
    TEMPLATE_NAMES,
    ensure_exp03_dirs,
    template_checkpoint_dir,
    template_dataset_dir,
    template_run_dir,
)
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import build_exp03_datasets
from thesis_exp.src.edujudge.exp03.templates import get_template_spec
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_json, write_text


LABELS = [1, 2, 3, 4, 5]


def write_record_ids(path: Path, record_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record_ids) + "\n", encoding="utf-8")


def read_predictions(output_dir: Path, split: str) -> list[dict[str, Any]]:
    candidates = [
        output_dir / "predictions" / f"predictions_{split}.jsonl",
        output_dir / f"predictions_{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return read_jsonl(path)
    return []


def grouped_metrics(predictions: list[dict[str, Any]], split: str, group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(group_field) or "unknown")].append(row)
    out = []
    for group_value, rows in sorted(grouped.items()):
        metrics = compute_metrics(rows)
        out.append({"split": split, group_field: group_value, **metrics})
    return out


def write_group_metrics(output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for split in ["dev", "test"]:
        predictions = read_predictions(output_dir, split)
        metric_rows.extend(grouped_metrics(predictions, split, "metric_canonical"))
        scenario_rows.extend(grouped_metrics(predictions, split, "scenario_canonical"))
    write_csv(tables_dir / "metric_level_metrics.csv", metric_rows)
    write_csv(tables_dir / "scenario_level_metrics.csv", scenario_rows)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def normalize_run_files(output_dir: Path, template_name: str, model_name_or_path: str, status: str) -> None:
    arrays_dir = output_dir / "arrays"
    copy_if_exists(arrays_dir / "exp02_dev_test_arrays.npz", arrays_dir / "dev_test_arrays.npz")
    copy_if_exists(output_dir / "tables" / "dev_metrics_history.csv", output_dir / "logs" / "dev_metrics_history.csv")

    history_path = output_dir / "tables" / "dev_metrics_history.csv"
    if history_path.exists():
        history = read_csv(history_path)
        write_csv(output_dir / "logs" / "training_log.csv", history)
    elif (output_dir / "tables" / "metrics_summary.csv").exists():
        write_csv(output_dir / "logs" / "training_log.csv", read_csv(output_dir / "tables" / "metrics_summary.csv"))
    else:
        write_csv(output_dir / "logs" / "training_log.csv", [])

    write_group_metrics(output_dir)
    metadata = {
        "experiment": "Exp3 input ablation",
        "template_name": template_name,
        "ablation_id": get_template_spec(template_name).ablation_id,
        "status": status,
        "model_name_or_path": model_name_or_path,
        "objective": "5-class sequence classification",
        "loss": "cross_entropy",
        "checkpoint_selection": "dev_exact_match",
        "reuse_exp02": template_name == "A2_question_answer_metric" and status == "reused_exp02",
    }
    write_json(output_dir / "run_metadata.json", metadata)

    test_rows = [row for row in read_csv(output_dir / "tables" / "metrics_summary.csv")] if (output_dir / "tables" / "metrics_summary.csv").exists() else []
    test = next((row for row in test_rows if row.get("split") == "test"), {})
    lines = [
        "# Exp3 Input Ablation Run Summary",
        "",
        f"Template: `{template_name}`",
        f"Status: `{status}`",
        f"Model: `{model_name_or_path}`",
        "Objective: 5-class Cross Entropy sequence classification",
        "Checkpoint selection: dev Exact Match",
    ]
    if test:
        lines.extend(
            [
                "",
                "## Test Metrics",
                "",
                f"- Exact Match: {test.get('Exact Match', 'NA')}",
                f"- MAE_label: {test.get('MAE_label', 'NA')}",
                f"- Kendall tau: {test.get('Kendall tau', 'NA')}",
                f"- low_to_high_rate: {test.get('low_to_high_rate', 'NA')}",
            ]
        )
    write_text(output_dir / "run_summary.md", "\n".join(lines))


def reuse_exp02_outputs(output_dir: Path, model_name_or_path: str) -> dict[str, Any]:
    if not EXP02_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Exp2 output dir not found: {EXP02_OUTPUT_DIR}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["predictions", "arrays", "tables"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    files = [
        "predictions/predictions_dev.jsonl",
        "predictions/predictions_test.jsonl",
        "arrays/exp02_dev_test_arrays.npz",
        "arrays/logits_dev.npy",
        "arrays/logits_test.npy",
        "arrays/probs_dev.npy",
        "arrays/probs_test.npy",
        "arrays/labels_dev.npy",
        "arrays/labels_test.npy",
        "arrays/record_ids_dev.txt",
        "arrays/record_ids_test.txt",
        "tables/metrics_summary.csv",
        "tables/per_bin_metrics.csv",
        "tables/low_score_metrics.csv",
        "tables/high_score_metrics.csv",
    ]
    copied = []
    missing = []
    for rel in files:
        src = EXP02_OUTPUT_DIR / rel
        dst = output_dir / rel
        if copy_if_exists(src, dst):
            copied.append(rel)
        else:
            missing.append(rel)
    if missing:
        raise FileNotFoundError(f"Missing required Exp2 reuse outputs: {missing}")
    normalize_run_files(output_dir, "A2_question_answer_metric", model_name_or_path, "reused_exp02")
    return {"status": "reused_exp02", "copied": copied, "output_dir": str(output_dir)}


def ensure_template_data(template_name: str, model_name_or_path: str | None, max_length: int) -> Path:
    data_dir = template_dataset_dir(template_name)
    if not all((data_dir / f"{split}.jsonl").exists() for split in ["train", "dev", "test"]):
        build_exp03_datasets(template_names=TEMPLATE_NAMES, model_name_or_path=model_name_or_path, max_length=max_length)
    return data_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Exp3 input-ablation template.")
    parser.add_argument("--template_name", required=True, choices=TEMPLATE_NAMES)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=None)
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
    parser.add_argument("--reuse_exp02", action="store_true", help="Reuse Exp2 outputs when template_name is A2.")
    parser.add_argument("--force_train_a2", action="store_true", help="Allow retraining A2 instead of reusing Exp2.")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def make_train_config(args: argparse.Namespace, data_dir: Path, output_dir: Path, checkpoint_output_dir: Path) -> TrainConfig:
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_output_dir=checkpoint_output_dir,
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
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )


def main() -> None:
    args = parse_args()
    ensure_exp03_dirs()
    template_name = args.template_name
    get_template_spec(template_name)
    output_dir = args.output_dir or template_run_dir(template_name)
    checkpoint_output_dir = args.checkpoint_output_dir or template_checkpoint_dir(template_name)

    if template_name == "A2_question_answer_metric" and (args.reuse_exp02 or not args.force_train_a2):
        result = reuse_exp02_outputs(output_dir, args.model_name_or_path)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    data_dir = args.data_dir or ensure_template_data(template_name, args.model_name_or_path if args.local_files_only else None, args.max_length)
    config = make_train_config(args, data_dir, output_dir, checkpoint_output_dir)
    result = train(config)
    normalize_run_files(output_dir, template_name, args.model_name_or_path, "completed")
    print(json.dumps({"template_name": template_name, "train_result": result}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
