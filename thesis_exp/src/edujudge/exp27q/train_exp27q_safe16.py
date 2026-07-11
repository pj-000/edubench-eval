"""Train the single locked Exp27Q Safe16 variant through the Exp27P trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker import TrainConfig, train
from thesis_exp.src.edujudge.exp27q import ARTIFACT_ROOT, EXP27O_DIR, OUTPUT_DIR, RUN_ROOT, SAFE16_DATASET, VARIANT


DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_jsonl", type=Path, default=SAFE16_DATASET)
    parser.add_argument("--dev_jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_local_files_only", action="store_false", dest="local_files_only")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(local_files_only=True, progress_bar=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in str(args.dev_jsonl).lower():
        raise ValueError("Exp27Q forbids test paths")
    run_dir = args.run_dir or RUN_ROOT / VARIANT / f"seed_{args.seed}"
    artifact_dir = args.checkpoint_output_dir or ARTIFACT_ROOT / VARIANT / f"seed_{args.seed}"
    config = TrainConfig(
        variant=VARIANT,
        model_name_or_path=args.model_name_or_path,
        train_jsonl=args.train_jsonl,
        dev_jsonl=args.dev_jsonl,
        run_dir=run_dir,
        output_dir=args.output_dir,
        checkpoint_output_dir=artifact_dir,
        high_impact_manifest=EXP27O_DIR / "tables/exp27o_high_impact16_manifest_light.csv",
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
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )
    print(json.dumps(train(config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

