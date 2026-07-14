"""Train one fixed-final-epoch Exp42A train-only GroupCV run."""

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

from thesis_exp.exp42_rubidist.common import (  # noqa: E402
    ARTIFACT_ROOT,
    ROOT,
    RUN_ROOT,
    VARIANTS,
    read_jsonl,
    write_json,
    write_jsonl,
)
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
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--model-name-or-path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--smoke-save-reload", action="store_true")
    return parser.parse_args()


def output_logits(output: Any) -> Any:
    return output["logits"] if isinstance(output, dict) else output.logits


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, train: bool, epoch: int = 0) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(args.seed + args.fold * 100 + epoch)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    return DataLoader(
        Rows(rows),
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        generator=generator if train else None,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )


def output_head(model: Any) -> tuple[str, Any]:
    for name in ("score", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            return name, module
    raise RuntimeError("Could not locate a small classifier head for smoke save/reload")


def main() -> None:
    args = parse_args()
    smoke = bool(args.max_train_rows or args.max_eval_rows or args.max_updates)
    if not smoke and args.epochs != 10:
        raise ValueError("Formal Exp42A uses fixed final epoch 10")
    if not smoke and (args.batch_size, args.eval_batch_size, args.gradient_accumulation) != (4, 4, 32):
        raise ValueError("Formal Exp42A locks batch/eval/gradient accumulation to 4/4/32")

    rows = read_jsonl(args.out_dir / f"private/data/exp42a_{args.variant}.jsonl")
    if len(rows) != 2654:
        raise ValueError(f"Expected 2654 rows for {args.variant}, found {len(rows)}")
    assignment_path = args.out_dir / "private/data/exp42a_groupcv_fold_assignment.csv"
    with assignment_path.open("r", encoding="utf-8", newline="") as handle:
        assignments = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if set(assignments) != {row["sample_id"] for row in rows}:
        raise RuntimeError("Fold assignment and variant sample IDs differ")
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp42A GroupCV training requires CUDA")
    model, tokenizer, model_mode = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path, gradient_checkpointing=args.gradient_checkpointing)
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    device = torch.device("cuda")
    model.to(device)

    batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    planned_updates = max(1, args.epochs * updates_per_epoch)
    total_updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_updates * args.warmup_ratio),
        total_updates,
    )
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_loader(train_rows, tokenizer, args, train=True, epoch=epoch)
        running = 0.0
        batches_seen = 0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("metadata")
            targets = batch.pop("targets").to(device)
            logits = output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float()
            loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {args.variant} seed={args.seed} fold={args.fold}")
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            running += float(loss.detach().cpu())
            batches_seen += 1
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_updates and global_step >= args.max_updates:
                    stop = True
                    break
        elapsed = time.time() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(batches_seen, 1),
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"[exp42a] {args.variant} seed={args.seed} fold={args.fold} epoch={epoch}/{args.epochs} "
            f"loss={history[-1]['train_loss']:.4f} step={global_step}/{total_updates} "
            f"elapsed={elapsed/60:.1f}m eta={elapsed/max(global_step, 1)*(total_updates-global_step)/60:.1f}m",
            flush=True,
        )
        if stop:
            break

    save_reload_status = "not_requested"
    if args.smoke_save_reload:
        name, head = output_head(model)
        state_path = args.artifact_root / "smoke" / args.variant / f"seed_{args.seed}_fold_{args.fold}_head.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, state_path)
        before = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        head.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        after = {key: value.detach().cpu() for key, value in head.state_dict().items()}
        if before.keys() != after.keys() or any(not torch.equal(before[key], after[key]) for key in before):
            raise RuntimeError("Smoke classifier-head save/reload mismatch")
        save_reload_status = f"PASS:{name}"

    model.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in make_loader(heldout_rows, tokenizer, args, train=False):
            metadata = batch.pop("metadata")
            batch.pop("targets")
            logits = output_logits(model(**{key: value.to(device) for key, value in batch.items()})).float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            for row, probs in zip(metadata, probabilities):
                predictions.append(
                    {
                        "variant": args.variant,
                        "seed": args.seed,
                        "fold": args.fold,
                        "sample_id": row["sample_id"],
                        "question_key": row["question_key"],
                        "gold_label_5": row["gold_label_5"],
                        "human_distribution_5": row["human_distribution_5"],
                        "expected_human_score": row["expected_human_score"],
                        "human_entropy": row["human_entropy"],
                        "human_score_range": row["human_score_range"],
                        "pred_label_5": int(np.argmax(probs)) + 1,
                        "pred_score_expected": sum((index + 1) * value for index, value in enumerate(probs)),
                        "language": row.get("language"),
                        "metric_group": row.get("metric_group"),
                        "subject": row.get("subject_canonical") or "unknown",
                        **{f"prob_{index}": value for index, value in enumerate(probs, 1)},
                    }
                )
    if any(not all(math.isfinite(float(value)) for value in [row["pred_score_expected"], *[row[f"prob_{k}"] for k in range(1, 6)]]) for row in predictions):
        raise FloatingPointError("Non-finite prediction encountered")
    run_dir = args.run_root / args.variant / f"seed_{args.seed}" / f"fold_{args.fold}"
    write_jsonl(run_dir / "heldout_predictions.jsonl", predictions)
    fixed_epoch = args.epochs if not smoke else history[-1]["epoch"]
    write_json(
        run_dir / "run_summary.json",
        {
            "status": "COMPLETED",
            "variant": args.variant,
            "seed": args.seed,
            "fold": args.fold,
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "train_question_keys": len(train_qkeys),
            "heldout_question_keys": len(heldout_qkeys),
            "question_key_overlap": 0,
            "fixed_final_epoch": fixed_epoch,
            "global_step": global_step,
            "model_mode": model_mode,
            "loss": "hard_one_hot_ce" if "hard" in args.variant else "human_soft_ce",
            "checkpoint_selection": "none_fixed_final_epoch",
            "formal": not smoke,
            "smoke_save_reload": save_reload_status,
            "history": history,
            "nan_count": 0,
            "oom_count": 0,
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "variant": args.variant,
                "seed": args.seed,
                "fold": args.fold,
                "heldout_rows": len(predictions),
                "global_step": global_step,
                "dev_access_count": 0,
                "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
