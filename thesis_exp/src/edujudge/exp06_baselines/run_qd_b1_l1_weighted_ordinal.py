"""Run Exp6 QD-B1 human-only L1 weighted ordinal baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.src.edujudge.exp05 import DEFAULT_W_MAX, DEFAULT_W_MIN
from thesis_exp.src.edujudge.exp05.class_weights import compute_class_weights
from thesis_exp.src.edujudge.exp05.train_l1_weighted_ordinal import L1TrainConfig, train
from thesis_exp.src.edujudge.exp06_baselines import (
    A4_QD_DATASET_DIR,
    EXP06_QD_CHECKPOINTS_DIR,
    EXP06_QD_RUNS_DIR,
    EXP06_QD_TABLES_DIR,
    QD_B1_RUN_ID,
    ensure_exp06_qd_dirs,
)
from thesis_exp.src.edujudge.exp06_baselines.build_qd_a4_dataset import build_dataset
from thesis_exp.src.edujudge.utils.io import write_csv, write_json


def write_qd_class_weights(train_path: Path, output_json: Path, w_min: float, w_max: float) -> None:
    rows = compute_class_weights(train_path=train_path, w_min=w_min, w_max=w_max)
    write_csv(EXP06_QD_TABLES_DIR / "qd_b1_class_weights.csv", rows)
    write_json(
        output_json,
        {
            "source_split": str(train_path),
            "w_min": w_min,
            "w_max": w_max,
            "weights": {str(row["label_5"]): row["clipped_weight"] for row in rows},
            "rows": rows,
        },
    )


def patch_exp05_side_effects(class_weights_path: Path):
    def ensure_qd_dataset_reference() -> list[dict[str, str]]:
        return [{"status": "SKIP", "notes": "Exp6 QD wrapper uses question_seed42 A4 dataset"}]

    def write_qd_weights_only(train_path: Path, w_min: float, w_max: float):
        write_qd_class_weights(train_path, class_weights_path, w_min, w_max)
        return compute_class_weights(train_path=train_path, w_min=w_min, w_max=w_max)

    train.__globals__["ensure_exp05_dataset"] = ensure_qd_dataset_reference
    train.__globals__["write_class_weights"] = write_qd_weights_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_dir", type=Path, default=A4_QD_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=EXP06_QD_RUNS_DIR / QD_B1_RUN_ID)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=EXP06_QD_CHECKPOINTS_DIR / QD_B1_RUN_ID)
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
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.add_argument("--w_min", type=float, default=DEFAULT_W_MIN)
    parser.add_argument("--w_max", type=float, default=DEFAULT_W_MAX)
    parser.add_argument("--class_weights_path", type=Path, default=EXP06_QD_TABLES_DIR / "qd_b1_class_weights.json")
    parser.set_defaults(progress_bar=True)
    args = parser.parse_args()

    ensure_exp06_qd_dirs()
    build_dataset(args.model_name_or_path, args.max_length)
    write_qd_class_weights(args.data_dir / "train.jsonl", args.class_weights_path, args.w_min, args.w_max)
    patch_exp05_side_effects(args.class_weights_path)
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
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
