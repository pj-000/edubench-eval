"""Train one fixed-final-epoch Exp41A train-only GroupCV fold."""

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp41_rubric_bridge.common import ROOT, VARIANTS, read_jsonl, write_json, write_jsonl  # noqa: E402
from thesis_exp.src.edujudge.exp02.train_ce_baseline import load_model_and_tokenizer  # noqa: E402


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
    parser.add_argument("--model-name-or-path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logit-adjustment-tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    return parser.parse_args()


def output_logits(output: Any) -> Any:
    return output["logits"] if isinstance(output, dict) else output.logits


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, train: bool, epoch: int = 0) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(args.seed + args.fold * 100 + epoch)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return DataLoader(Rows(rows), batch_size=args.batch_size if train else args.eval_batch_size,
                      shuffle=train, generator=generator if train else None, drop_last=False,
                      num_workers=0, collate_fn=collate)


def main() -> None:
    args = parse_args()
    formal = not (args.max_train_rows or args.max_eval_rows)
    if formal and args.epochs != 10:
        raise ValueError("Formal Exp41A uses fixed final epoch 10")
    if args.variant == "v5_human_soft_logit_adjustment" and args.logit_adjustment_tau != 1.0:
        raise ValueError("Exp41A V5 freezes logit adjustment tau=1.0")
    decision = json.loads((args.out_dir / "decision/exp41a_compiler_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("Training blocked by compiler qualification NO-GO")
    data_path = args.out_dir / f"private/data/exp41a_{args.variant}.jsonl"
    rows = read_jsonl(data_path)
    if len(rows) != 2654:
        raise ValueError(f"Expected 2654 rows for {args.variant}, found {len(rows)}")
    assignment_path = args.out_dir / "private/data/exp41a_groupcv_fold_assignment.csv"
    with assignment_path.open("r", encoding="utf-8", newline="") as handle:
        assignments = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    train_rows = [row for row in rows if assignments[row["sample_id"]] != args.fold]
    heldout_rows = [row for row in rows if assignments[row["sample_id"]] == args.fold]
    train_qkeys = {row["question_key"] for row in train_rows}
    heldout_qkeys = {row["question_key"] for row in heldout_rows}
    if train_qkeys & heldout_qkeys or len(train_rows) + len(heldout_rows) != 2654:
        raise RuntimeError("Question-key GroupCV leakage or row loss")
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows:
        heldout_rows = heldout_rows[: args.max_eval_rows]
    if not train_rows or not heldout_rows:
        raise ValueError("Empty train or heldout rows")

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp41A GroupCV training requires CUDA")
    model, tokenizer, model_mode = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda")
    model.to(device)
    train_label_counts = np.asarray([sum(int(row["gold_label_5"]) == label for row in train_rows) for label in range(1, 6)], dtype=float)
    priors = torch.tensor(train_label_counts / train_label_counts.sum(), dtype=torch.float32, device=device).clamp_min(1e-8)
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
            training_logits = logits
            if args.variant == "v5_human_soft_logit_adjustment":
                training_logits = logits + args.logit_adjustment_tau * torch.log(priors)
            loss = -(targets * torch.log_softmax(training_logits, dim=-1)).sum(dim=-1).mean()
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            running += float(loss.detach().cpu())
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
        elapsed = time.time() - started
        history.append({"epoch": epoch, "train_loss": running / len(loader), "global_step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0]})
        print(f"[exp41a-groupcv] {args.variant} fold={args.fold} epoch={epoch}/{args.epochs} "
              f"loss={history[-1]['train_loss']:.4f} elapsed={elapsed/60:.1f}m eta={elapsed/epoch*(args.epochs-epoch)/60:.1f}m", flush=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in make_loader(heldout_rows, tokenizer, args, train=False):
            metadata = batch.pop("metadata")
            batch.pop("targets")
            logits = output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for row, probs in zip(metadata, probabilities):
                predictions.append({
                    "variant": args.variant, "fold": args.fold, "sample_id": row["sample_id"],
                    "question_key": row["question_key"], "gold_label_5": row["gold_label_5"],
                    "human_distribution_5": row["human_distribution_5"],
                    "expected_human_score": row["expected_human_score"], "human_entropy": row["human_entropy"],
                    "human_score_range": row["human_score_range"], "pred_label_5": int(np.argmax(probs)) + 1,
                    "pred_score_expected": sum((index + 1) * value for index, value in enumerate(probs)),
                    "language": row.get("language"), "metric_group": row.get("metric_group"),
                    "subject_canonical": row.get("subject_canonical"),
                    **{f"prob_{index}": value for index, value in enumerate(probs, 1)},
                })
    run_dir = args.out_dir / f"private/groupcv_predictions/{args.variant}/fold_{args.fold}"
    write_jsonl(run_dir / "heldout_predictions.jsonl", predictions)
    write_json(run_dir / "run_summary.json", {
        "status": "COMPLETED", "variant": args.variant, "fold": args.fold,
        "train_rows": len(train_rows), "heldout_rows": len(heldout_rows),
        "train_question_keys": len(train_qkeys), "heldout_question_keys": len(heldout_qkeys),
        "question_key_overlap": 0, "fixed_final_epoch": args.epochs, "global_step": global_step,
        "model_mode": model_mode, "loss": "hard_ce" if args.variant == "v0_original_hard" else "human_soft_ce",
        "logit_adjustment_tau": args.logit_adjustment_tau if args.variant == "v5_human_soft_logit_adjustment" else 0,
        "checkpoint_selection": "none_fixed_final_epoch", "history": history,
        "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "fold": args.fold,
                      "heldout_rows": len(predictions), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
