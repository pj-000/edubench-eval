"""Train one fixed-final-epoch Exp40A GroupCV fold with soft CE and RankNet."""

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

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    ROOT,
    TRAIN_PATH,
    VARIANTS,
    human_distribution,
    read_jsonl,
    sample_id,
    write_json,
    write_jsonl,
)
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
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lambda-pair", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def output_logits(output: Any) -> Any:
    return output["logits"] if isinstance(output, dict) else output.logits


def ce_collator(tokenizer: Any, max_length: int):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return collate


def make_ce_loader(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, train: bool, epoch: int = 0) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(args.seed + args.fold * 100 + epoch)
    return DataLoader(
        Rows(rows), batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train, generator=generator if train else None, drop_last=False, num_workers=0,
        collate_fn=ce_collator(tokenizer, args.max_length),
    )


def prepare_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "better_text": make_prompt(pair["better"], "A2_question_answer_metric"),
        "worse_text": make_prompt(pair["worse"], "A2_question_answer_metric"),
    }


def sample_pair_batch(pairs: list[dict[str, Any]], batch_size: int, rng: random.Random) -> list[dict[str, Any]]:
    return [pairs[rng.randrange(len(pairs))] for _ in range(batch_size)]


def pair_loss(model: Any, tokenizer: Any, pairs: list[dict[str, Any]], device: Any, max_length: int) -> Any:
    import torch

    texts = [row["better_text"] for row in pairs] + [row["worse_text"] for row in pairs]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    logits = output_logits(model(**{key: value.to(device) for key, value in encoded.items()})).float()
    probabilities = torch.softmax(logits, dim=-1)
    score_values = torch.arange(1, 6, dtype=probabilities.dtype, device=device)
    expected = probabilities @ score_values
    better, worse = expected[: len(pairs)], expected[len(pairs) :]
    return -torch.nn.functional.logsigmoid(better - worse).mean()


def pair_ranking_accuracy(
    model: Any, tokenizer: Any, pairs: list[dict[str, Any]], device: Any, max_length: int, batch_size: int,
) -> tuple[int, int]:
    import torch

    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [row["better_text"] for row in batch] + [row["worse_text"] for row in batch]
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            logits = output_logits(model(**{key: value.to(device) for key, value in encoded.items()})).float()
            probabilities = torch.softmax(logits, dim=-1)
            score_values = torch.arange(1, 6, dtype=probabilities.dtype, device=device)
            expected = probabilities @ score_values
            correct += int(torch.sum(expected[: len(batch)] > expected[len(batch) :]).item())
            total += len(batch)
    return correct, total


def main() -> None:
    args = parse_args()
    if args.lambda_pair != 0.25:
        raise ValueError("Exp40A freezes lambda_pair=0.25")
    if args.epochs != 10 and not (args.max_train_rows or args.max_eval_rows or args.max_pairs):
        raise ValueError("Formal Exp40A uses fixed final epoch 10")
    if args.variant == "v0h_human_soft" and args.lambda_pair != 0.25:
        raise ValueError("V0H retains the locked CLI value even though no pair term is applied")
    assignment_path = args.out_dir / "private/data/exp40a_groupcv_fold_assignment.csv"
    with assignment_path.open("r", encoding="utf-8", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    qkey_folds: dict[str, int] = {}
    for row in assignments:
        qkey, fold = str(row["question_key"]), int(row["fold"])
        if qkey in qkey_folds and qkey_folds[qkey] != fold:
            raise RuntimeError("Question-key fold assignment is inconsistent")
        qkey_folds[qkey] = fold

    originals = read_jsonl(args.train_jsonl)
    prepared = [
        {
            **row,
            "sample_id": sample_id(row),
            "text": make_prompt(row, "A2_question_answer_metric"),
            "soft_target_5": human_distribution(row),
            "gold_label_5": int(row["label_5"]),
        }
        for row in originals
    ]
    train_rows = [row for row in prepared if qkey_folds[str(row["question_key"])] != args.fold]
    heldout_rows = [row for row in prepared if qkey_folds[str(row["question_key"])] == args.fold]
    train_qkeys = {str(row["question_key"]) for row in train_rows}
    heldout_qkeys = {str(row["question_key"]) for row in heldout_rows}
    if train_qkeys & heldout_qkeys or len(train_rows) + len(heldout_rows) != 2654:
        raise RuntimeError("Exp40A original-row GroupCV leakage or row loss detected")

    train_pairs: list[dict[str, Any]] = []
    if args.variant != "v0h_human_soft":
        all_pairs = read_jsonl(args.out_dir / f"private/data/exp40a_{args.variant}.jsonl")
        train_pairs = [prepare_pair(row) for row in all_pairs if str(row["question_key"]) in train_qkeys]
        if not train_pairs:
            raise RuntimeError(f"No train-fold pairs for {args.variant} fold {args.fold}")
    diagnostic_pairs = [
        prepare_pair(row)
        for row in read_jsonl(args.out_dir / "private/data/exp40a_v3_edupair_cf.jsonl")
        if str(row["question_key"]) in heldout_qkeys
    ]
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows:
        heldout_rows = heldout_rows[: args.max_eval_rows]
        limited_qkeys = {str(row["question_key"]) for row in heldout_rows}
        diagnostic_pairs = [row for row in diagnostic_pairs if str(row["question_key"]) in limited_qkeys]
    if args.max_pairs:
        train_pairs = train_pairs[: args.max_pairs]
        diagnostic_pairs = diagnostic_pairs[: args.max_pairs]

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp40A GroupCV training requires CUDA")
    model, tokenizer, model_mode = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda")
    model.to(device)
    ce_batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(ce_batches / args.gradient_accumulation)
    total_updates = max(1, args.epochs * updates_per_epoch)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_ce_loader(train_rows, tokenizer, args, train=True, epoch=epoch)
        pair_rng = random.Random(args.seed + args.fold * 1000 + epoch)
        running_total = running_ce = running_pair = 0.0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("metadata")
            targets = batch.pop("targets").to(device)
            logits = output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float()
            ce = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            if train_pairs:
                pair_batch = sample_pair_batch(train_pairs, args.batch_size, pair_rng)
                rank_loss = pair_loss(model, tokenizer, pair_batch, device, args.max_length)
                loss = ce + args.lambda_pair * rank_loss
            else:
                rank_loss = torch.zeros((), dtype=ce.dtype, device=device)
                loss = ce
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            running_total += float(loss.detach().cpu())
            running_ce += float(ce.detach().cpu())
            running_pair += float(rank_loss.detach().cpu())
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": running_total / max(len(loader), 1),
                "train_human_soft_ce": running_ce / max(len(loader), 1),
                "train_pair_loss": running_pair / max(len(loader), 1),
                "pair_updates": len(loader) if train_pairs else 0,
                "global_step": global_step,
            }
        )
        elapsed = time.time() - started
        eta = elapsed / epoch * (args.epochs - epoch)
        print(
            f"[exp40a-groupcv] {args.variant} fold={args.fold} epoch={epoch}/{args.epochs} "
            f"loss={history[-1]['train_total_loss']:.4f} ce={history[-1]['train_human_soft_ce']:.4f} "
            f"pair={history[-1]['train_pair_loss']:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in make_ce_loader(heldout_rows, tokenizer, args, train=False):
            metadata = batch.pop("metadata")
            batch.pop("targets")
            probabilities = torch.softmax(
                output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float(), dim=-1
            ).cpu().tolist()
            for row, probs in zip(metadata, probabilities):
                predictions.append(
                    {
                        "variant": args.variant, "fold": args.fold, "sample_id": sample_id(row),
                        "question_key": row["question_key"], "gold_label_5": row["gold_label_5"],
                        "human_distribution_5": row["soft_target_5"],
                        "pred_label_5": int(np.argmax(probs)) + 1,
                        "pred_score_expected": sum((index + 1) * value for index, value in enumerate(probs)),
                        "language": row.get("language"), "metric_group": row.get("metric_group"),
                        "subject_canonical": row.get("subject_canonical"),
                        **{f"prob_{index}": value for index, value in enumerate(probs, 1)},
                    }
                )
    diagnostic_correct, diagnostic_total = pair_ranking_accuracy(
        model, tokenizer, diagnostic_pairs, device, args.max_length, args.eval_batch_size
    ) if diagnostic_pairs else (0, 0)
    run_dir = args.out_dir / f"private/groupcv_predictions/{args.variant}/fold_{args.fold}"
    write_jsonl(run_dir / "heldout_predictions.jsonl", predictions)
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "COMPLETED", "variant": args.variant, "fold": args.fold,
            "train_original_rows": len(train_rows), "heldout_original_rows": len(heldout_rows),
            "train_pair_pool": len(train_pairs), "train_question_keys": len(train_qkeys),
            "heldout_question_keys": len(heldout_qkeys), "question_key_overlap": 0,
            "synthetic_heldout_rows": 0, "fixed_final_epoch": args.epochs,
            "global_step": global_step, "model_mode": model_mode, "lambda_pair": args.lambda_pair,
            "heldout_pair_diagnostic_correct": diagnostic_correct,
            "heldout_pair_diagnostic_total": diagnostic_total,
            "heldout_pair_ranking_accuracy": diagnostic_correct / diagnostic_total if diagnostic_total else None,
            "history": history, "dev_access_count": 0, "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED", "variant": args.variant, "fold": args.fold,
                "heldout_rows": len(predictions), "heldout_pair_diagnostic_total": diagnostic_total,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
