"""Train one fixed-final-epoch Exp39A GroupCV fold with standard soft CE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from thesis_exp.exp39_educfa.common import ROOT, VARIANTS, read_jsonl, sample_id, write_json, write_jsonl  # noqa: E402
from thesis_exp.src.edujudge.exp02.train_ce_baseline import load_model_and_tokenizer  # noqa: E402
from thesis_exp.src.edujudge.exp03.templates import make_prompt  # noqa: E402


@dataclass
class ModelConfig:
    model_name_or_path: str
    checkpoint_dir: Path | None = None
    trust_remote_code: bool = False
    local_files_only: bool = True
    bf16: str = "auto"
    fp16: bool = False
    gradient_checkpointing: bool = False


class Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def collator(tokenizer: Any, max_length: int):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return collate


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, train: bool, epoch: int = 0) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(args.seed + args.fold * 100 + epoch)
    return DataLoader(
        Rows(rows), batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train, generator=generator if train else None, drop_last=False, num_workers=0,
        collate_fn=collator(tokenizer, args.max_length),
    )


def output_logits(output: Any) -> Any:
    return output["logits"] if isinstance(output, dict) else output.logits


def main() -> None:
    args = parse_args()
    if args.epochs != 10 and not (args.max_train_rows or args.max_eval_rows):
        raise ValueError("Formal Exp39A uses fixed final epoch 10")
    assignment_path = args.out_dir / "private/data/exp39a_groupcv_fold_assignment.csv"
    with assignment_path.open("r", encoding="utf-8", newline="") as handle:
        assignment_rows = list(csv.DictReader(handle))
    qkey_folds: dict[str, int] = {}
    for row in assignment_rows:
        qkey, fold = row["question_key"], int(row["fold"])
        if qkey in qkey_folds and qkey_folds[qkey] != fold:
            raise RuntimeError("Question-key fold assignment is inconsistent")
        qkey_folds[qkey] = fold
    rows = read_jsonl(args.out_dir / f"private/data/exp39a_{args.variant}.jsonl")
    prepared = []
    for row in rows:
        qkey = str(row["question_key"])
        if qkey not in qkey_folds:
            raise ValueError(f"Missing fold assignment for question_key: {qkey}")
        prepared.append({
            **row,
            "text": make_prompt(row, "A2_question_answer_metric"),
            "gold_label_5": int(row["original_label_5"]),
        })
    train_rows = [row for row in prepared if qkey_folds[str(row["question_key"])] != args.fold]
    heldout_rows = [
        row for row in prepared
        if qkey_folds[str(row["question_key"])] == args.fold and row.get("original_evaluation_row") and not row.get("synthetic")
    ]
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows:
        heldout_rows = heldout_rows[: args.max_eval_rows]
    train_qkeys = {row["question_key"] for row in train_rows}
    heldout_qkeys = {row["question_key"] for row in heldout_rows}
    if train_qkeys & heldout_qkeys or any(row.get("synthetic") for row in heldout_rows):
        raise RuntimeError("Exp39A GroupCV leakage or synthetic evaluation detected")

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp39A GroupCV training requires CUDA")
    model, tokenizer, model_mode = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda")
    model.to(device)
    batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    total_updates = max(1, args.epochs * updates_per_epoch)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_loader(train_rows, tokenizer, args, train=True, epoch=epoch)
        running = 0.0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("metadata")
            targets = batch.pop("targets").to(device)
            logits = output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float()
            loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            running += float(loss.detach().cpu())
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        history.append({"epoch": epoch, "train_loss": running / max(len(loader), 1), "global_step": global_step})
        elapsed = time.time() - started
        eta = elapsed / epoch * (args.epochs - epoch)
        print(f"[exp39a-groupcv] {args.variant} fold={args.fold} epoch={epoch}/{args.epochs} loss={history[-1]['train_loss']:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in make_loader(heldout_rows, tokenizer, args, train=False):
            metadata = batch.pop("metadata")
            batch.pop("targets")
            probabilities = torch.softmax(output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float(), dim=-1).cpu().tolist()
            for row, probs in zip(metadata, probabilities):
                predictions.append({
                    "variant": args.variant, "fold": args.fold, "sample_id": sample_id(row),
                    "question_key": row["question_key"], "gold_label_5": row["gold_label_5"],
                    "human_distribution_5": row["soft_target_5"],
                    "pred_label_5": int(np.argmax(probs)) + 1,
                    "pred_score_expected": sum((index + 1) * value for index, value in enumerate(probs)),
                    "language": row.get("language"), "metric_group": row.get("metric_group"),
                    "subject_canonical": row.get("subject_canonical"),
                    **{f"prob_{index}": value for index, value in enumerate(probs, 1)},
                })
    run_dir = args.out_dir / f"private/groupcv_predictions/{args.variant}/fold_{args.fold}"
    write_jsonl(run_dir / "heldout_predictions.jsonl", predictions)
    write_json(run_dir / "run_summary.json", {
        "status": "COMPLETED", "variant": args.variant, "fold": args.fold,
        "train_rows": len(train_rows), "heldout_original_rows": len(heldout_rows),
        "train_question_keys": len(train_qkeys), "heldout_question_keys": len(heldout_qkeys),
        "question_key_overlap": 0, "synthetic_heldout_rows": 0,
        "fixed_final_epoch": args.epochs, "global_step": global_step, "model_mode": model_mode,
        "history": history, "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "fold": args.fold, "heldout_rows": len(predictions)}, sort_keys=True))


if __name__ == "__main__":
    main()
