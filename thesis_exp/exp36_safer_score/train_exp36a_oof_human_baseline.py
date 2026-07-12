"""Train one final-epoch human-hard OOF baseline and export held-out probabilities."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import ROOT, TRAIN_PATH, read_jsonl, sample_id, write_json, write_jsonl
from thesis_exp.exp36_safer_score.model_utils import build_model, evaluate, load_tokenizer, make_loader, set_seed
from thesis_exp.src.edujudge.exp02.train_ce_baseline import StepProgressBar


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--model-name-or-path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--fold-assignment", type=Path, default=ROOT / "data/exp36a_oof_fold_assignment.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def main() -> None:
    config = args()
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    set_seed(config.seed + config.fold)
    rows = read_jsonl(config.train_jsonl)
    with config.fold_assignment.open("r", encoding="utf-8", newline="") as handle:
        assignments = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if set(assignments) != {sample_id(row) for row in rows}:
        raise ValueError("Fold assignment does not exactly cover train")
    train_rows = [row for row in rows if assignments[sample_id(row)] != config.fold]
    heldout_rows = [row for row in rows if assignments[sample_id(row)] == config.fold]
    if {row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows}:
        raise ValueError("OOF question-key leakage")
    if config.max_train_samples:
        train_rows = train_rows[: config.max_train_samples]
    if config.max_eval_samples:
        heldout_rows = heldout_rows[: config.max_eval_samples]

    tokenizer = load_tokenizer(config.model_name_or_path, True)
    model = build_model(config.model_name_or_path, config.bf16, True, config.gradient_checkpointing)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp36A OOF requires CUDA")
    device = torch.device("cuda")
    model.to(device)
    batches = math.ceil(len(train_rows) / config.batch_size)
    updates_per_epoch = math.ceil(batches / config.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * config.epochs)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * config.warmup_ratio), total_updates,
    )
    progress = StepProgressBar(total_updates, True, desc=f"Exp36A OOF fold{config.fold}")
    global_step = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(1, config.epochs + 1):
            loader = make_loader(train_rows, tokenizer, config.batch_size, config.max_length, True, config.seed + epoch)
            model.train()
            window = 0
            running = 0.0
            for index, batch in enumerate(loader, 1):
                metadata = batch.pop("metadata")
                labels = torch.tensor([int(row["label_5"]) - 1 for row in metadata], device=device)
                logits, _ = model(**{key: value.to(device) for key, value in batch.items()})
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                (loss / config.gradient_accumulation_steps).backward()
                running += float(loss.detach().cpu())
                window += 1
                if window == config.gradient_accumulation_steps or index == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1; window = 0
                    elapsed = time.time() - start
                    progress.render(global_step, epoch, running / index, scheduler.get_last_lr()[0], elapsed,
                                    (total_updates-global_step) * elapsed / max(global_step, 1))
            progress.newline()
    finally:
        progress.close()
    heldout_loader = make_loader(heldout_rows, tokenizer, config.eval_batch_size, config.max_length, False, config.seed)
    predictions = evaluate(model, heldout_loader, device)
    path = config.out_dir / "private" / "oof_folds" / f"fold_{config.fold}_predictions.jsonl"
    write_jsonl(path, predictions)
    write_json(config.out_dir / "private" / "oof_folds" / f"fold_{config.fold}_summary.json", {
        "status": "COMPLETED", "fold": config.fold, "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows), "epochs": config.epochs, "final_epoch_only": True,
        "question_key_leakage_count": 0, "test_access_count": 0, "prediction_path": str(path),
    })
    print({"status": "COMPLETED", "fold": config.fold, "predictions": len(predictions)})


if __name__ == "__main__":
    main()
