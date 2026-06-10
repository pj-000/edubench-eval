"""Train Exp6 QD-S2 human + synthetic low-score L1 weighted ordinal model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import DEFAULT_W_MAX, DEFAULT_W_MIN
from thesis_exp.src.edujudge.exp05.class_weights import compute_class_weights
from thesis_exp.src.edujudge.exp05.train_l1_weighted_ordinal import L1TrainConfig
from thesis_exp.src.edujudge.exp05 import train_l1_weighted_ordinal as l1_module
from thesis_exp.src.edujudge.exp06_training import (
    QD_S2_CHECKPOINT_DIR,
    QD_S2_DATASET_DIR,
    QD_S2_RUN_DIR,
    QD_S2_RUN_ID,
    ensure_exp06_training_dirs,
)
from thesis_exp.src.edujudge.exp06_training.collect_qds2_results import collect
from thesis_exp.src.edujudge.exp06_training.train_qds1_ordinal import (
    _api_key_hits,
    _failures_to_text,
    copy_dev_history_to_logs,
    dataset_sanity_rows,
    require_cuda_if_requested,
    write_language_metrics,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_text


def write_qds2_class_weights(train_path: Path, output_json: Path, output_csv: Path, w_min: float, w_max: float) -> list[dict[str, Any]]:
    rows = compute_class_weights(train_path=train_path, w_min=w_min, w_max=w_max)
    fieldnames = [
        "label_5",
        "train_count",
        "raw_weight",
        "clipped_weight",
        "w_min",
        "w_max",
        "notes",
    ]
    write_csv(output_csv, rows, fieldnames)
    write_json(
        output_json,
        {
            "source_split": relpath(train_path),
            "w_min": w_min,
            "w_max": w_max,
            "weights": {str(row["label_5"]): row["clipped_weight"] for row in rows},
            "rows": rows,
        },
    )
    return rows


def patch_exp05_side_effects(class_weights_path: Path, class_weights_csv: Path):
    def ensure_qds2_dataset_reference() -> list[dict[str, str]]:
        return [{"status": "SKIP", "notes": "Exp6 QD-S2 wrapper uses prebuilt question_seed42 synthetic-augmented dataset"}]

    def write_qds2_weights_only(train_path: Path, w_min: float, w_max: float):
        return write_qds2_class_weights(train_path, class_weights_path, class_weights_csv, w_min, w_max)

    l1_module.train.__globals__["ensure_exp05_dataset"] = ensure_qds2_dataset_reference
    l1_module.train.__globals__["write_class_weights"] = write_qds2_weights_only
    l1_module.load_model_and_tokenizer.__globals__["write_class_weights"] = write_qds2_weights_only


def qds2_sanity_rows(args: argparse.Namespace, formal: bool) -> list[dict[str, Any]]:
    rows = dataset_sanity_rows(args.data_dir, args.max_train_samples, args.max_eval_samples, formal)
    for row in rows:
        row["check_name"] = str(row["check_name"]).replace("QD-S1", "QD-S2")
    return rows


def overwrite_qds2_metadata(run_dir: Path, args: argparse.Namespace, status: str) -> None:
    best_metadata_path = args.checkpoint_output_dir / "best" / "exp05_head_metadata.json"
    best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8")) if best_metadata_path.exists() else {}
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": "Exp6 synthetic low-score training",
            "run_id": QD_S2_RUN_ID,
            "status": status,
            "fixed_input_template": "A4_question_answer_metric_rubric_metadata",
            "model_name_or_path": args.model_name_or_path,
            "objective_type": "ordinal",
            "loss": "L1 weighted ordinal BCEWithLogitsLoss",
            "class_weights_path": relpath(args.class_weights_path),
            "class_weight_source": "QD-S2 train split only; human and synthetic rows included",
            "synthetic_label_provenance": "pseudo_label",
            "w_min": args.w_min,
            "w_max": args.w_max,
            "checkpoint_selection": f"dev {args.selection_metric} ({args.selection_mode})",
            "best_epoch": best_metadata.get("best_epoch"),
            "best_global_step": best_metadata.get("best_global_step"),
            "best_selection_metric_name": best_metadata.get("best_selection_metric_name", f"dev_{args.selection_metric}"),
            "best_selection_metric_value": best_metadata.get("best_selection_metric_value"),
            "selection_direction": best_metadata.get("selection_direction", args.selection_mode),
        },
    )
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    lines = [
        "# Exp6 QD-S2 Human + Synthetic L1 Weighted Ordinal",
        "",
        f"Status: `{status}`",
        f"Model: `{args.model_name_or_path}`",
        "Train: 3326 human + 384 synthetic low-score pseudo-label rows",
        "Dev/test: question_seed42 human-only",
        "Loss: L1 weighted ordinal BCE",
        "Class weights: computed from QD-S2 train only",
        f"Checkpoint selection: dev `{args.selection_metric}` ({args.selection_mode})",
    ]
    if metrics_path.exists():
        import csv

        rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline="")))
        test = next((row for row in rows if row.get("split") == "test"), {})
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
                    f"- mean_sample_weight: {test.get('mean_sample_weight', 'NA')}",
                ]
            )
    write_text(run_dir / "run_summary.md", "\n".join(lines))


def postprocess_outputs(run_dir: Path, args: argparse.Namespace, strict: bool = True) -> None:
    write_language_metrics(run_dir)
    copy_dev_history_to_logs(run_dir)
    overwrite_qds2_metadata(run_dir, args, "completed")
    api_hits = _api_key_hits(run_dir)
    if api_hits and strict:
        raise RuntimeError(f"API key-like strings found in outputs: {api_hits}")
    if not args.smoke:
        collect(run_dir)


def train_qds2(args: argparse.Namespace) -> None:
    ensure_exp06_training_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.class_weights_path is None:
        args.class_weights_path = tables_dir / "class_weights.json"
    class_weights_csv = tables_dir / "class_weights.csv"

    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    smoke = bool(args.smoke)
    if smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if not smoke and not formal and not args.postprocess_only and not args.preflight_only:
        raise RuntimeError("Formal QD-S2 training requires FORMAL_RUN=1.")

    sanity_rows = qds2_sanity_rows(args, formal)
    sanity_path = tables_dir / ("smoke_preflight_qds2.csv" if smoke else "preflight_qds2.csv")
    write_csv(sanity_path, sanity_rows)
    failures = [row for row in sanity_rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QD-S2 preflight failed:\n{_failures_to_text(failures)}")

    class_weight_rows = write_qds2_class_weights(args.data_dir / "train.jsonl", args.class_weights_path, class_weights_csv, args.w_min, args.w_max)
    write_csv(tables_dir / "preflight_qds2_class_weights.csv", class_weight_rows)
    if args.preflight_only:
        write_text(
            args.output_dir / "qds2_preflight_status.md",
            "\n".join(
                [
                    "# QD-S2 Preflight Status",
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

    patch_exp05_side_effects(args.class_weights_path, class_weights_csv)
    require_cuda_if_requested()
    config = L1TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_output_dir=args.checkpoint_output_dir,
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
        w_min=args.w_min,
        w_max=args.w_max,
        class_weights_path=args.class_weights_path,
    )
    result = l1_module.train(config)
    postprocess_outputs(args.output_dir, args)
    write_text(
        args.output_dir / "qds2_training_status.md",
        "\n".join(
            [
                "# QD-S2 Training Status",
                "",
                "Status: completed",
                f"Smoke: {smoke}",
                "API called: no",
                "Synthetic generated: no",
                "Loss: L1 weighted ordinal BCEWithLogitsLoss",
                "Class weights: QD-S2 train only",
                "Synthetic labels: pseudo_label metadata only",
            ]
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp6 QD-S2 L1 weighted ordinal model.")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--data_dir", type=Path, default=QD_S2_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=QD_S2_RUN_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=QD_S2_CHECKPOINT_DIR)
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
    parser.add_argument("--w_min", type=float, default=DEFAULT_W_MIN)
    parser.add_argument("--w_max", type=float, default=DEFAULT_W_MAX)
    parser.add_argument("--class_weights_path", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_qds2(args)
    print(f"QD-S2 output: {relpath(args.output_dir)}")


if __name__ == "__main__":
    main()
