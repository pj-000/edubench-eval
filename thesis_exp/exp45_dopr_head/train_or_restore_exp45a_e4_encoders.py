"""Retrain or strictly restore one locked Exp45A E4 fold encoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp44_taco_score.losses_exp44a_taco import base_loss
from thesis_exp.exp44_taco_score.modeling_exp44a_taco import TACOModelConfig, build_model
from thesis_exp.exp45_dopr_head.common import (
    ARTIFACT_ROOT, EXP44_RUN_ROOT, FOLDS, ROOT, RUN_ROOT, atomic_json,
    encoder_dir, fold_map, read_jsonl, sha256_file, stable_hash,
)


class Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--mode", choices=("groupcv", "smoke"), default="groupcv")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fingerprint-only", action="store_true")
    return parser.parse_args()


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, epoch: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        return encoded

    generator = torch.Generator().manual_seed(42 + epoch * 100 + args.fold)
    return DataLoader(Rows(rows), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0, collate_fn=collate)


def identity(args: argparse.Namespace) -> dict[str, Any]:
    data = args.out_dir / "private/data/exp45a_train_e4.jsonl"
    folds = args.out_dir / "private/data/exp45a_groupcv_fold_assignment.csv"
    config = args.out_dir / "configs/exp45a_encoder_protocol_lock.json"
    code = [
        Path("thesis_exp/exp44_taco_score/modeling_exp44a_taco.py"),
        Path("thesis_exp/exp44_taco_score/losses_exp44a_taco.py"),
        Path("thesis_exp/exp45_dopr_head/train_or_restore_exp45a_e4_encoders.py"),
    ]
    value = {
        "fold": args.fold, "mode": args.mode, "seed": 42, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation, "max_length": args.max_length,
        "max_train_rows": args.max_train_rows, "max_updates": args.max_updates,
        "data_hashes": {str(path): sha256_file(path) for path in (data, folds, config)},
        "code_hashes": {str(path): sha256_file(path) for path in code},
        "model_hashes": json.loads((args.out_dir / "hashes/exp45a_model_hashes.json").read_text(encoding="utf-8")),
    }
    value["run_fingerprint"] = stable_hash(value)
    return value


def main() -> None:
    args = parse_args()
    formal = args.mode == "groupcv"
    if formal and (args.epochs, args.learning_rate, args.weight_decay, args.warmup_ratio, args.batch_size, args.gradient_accumulation, args.max_length) != (10, 2e-5, 0.01, 0.05, 4, 32, 2048):
        raise ValueError("Exp45A E4 encoder protocol is locked")
    run_identity = identity(args)
    if args.fingerprint_only:
        print(json.dumps(run_identity, sort_keys=True))
        return
    target_dir = encoder_dir(args.out_dir, args.fold, args.mode)
    checkpoint = target_dir / "final_encoder_head.pt"
    summary_path = target_dir / "encoder_summary.json"
    if args.skip_completed and checkpoint.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "COMPLETED" and summary.get("run_fingerprint") == run_identity["run_fingerprint"]:
            print(json.dumps({"status": "REUSED_EXP45", "fold": args.fold, "checkpoint": str(checkpoint)}, sort_keys=True))
            return

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Exp45A encoder training requires CUDA")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = read_jsonl(args.out_dir / "private/data/exp45a_train_e4.jsonl")
    folds = fold_map(args.out_dir / "private/data/exp45a_groupcv_fold_assignment.csv")
    train_rows = [row for row in rows if folds[row["sample_id"]] != args.fold]
    heldout_rows = [row for row in rows if folds[row["sample_id"]] == args.fold]
    if {row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows}:
        raise RuntimeError("Exp45A encoder question-key leakage")
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(TACOModelConfig(args.model_name_or_path, use_projection=False)).cuda()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(math.ceil(len(train_rows) / args.batch_size) / args.gradient_accumulation)
    total_updates = args.max_updates or args.epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    optimizer.zero_grad(set_to_none=True)
    history = []
    global_step = 0
    started = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_loader(train_rows, tokenizer, args, epoch)
        sums: Counter[str] = Counter()
        for batch_index, batch in enumerate(loader, 1):
            targets = batch.pop("targets").cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{key: value.cuda() for key, value in batch.items()})["logits"]
                loss, components = base_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Exp45A encoder loss")
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            sums.update({"loss": float(loss.detach()), "distribution": float(components["distribution"].detach()), "ordinal": float(components["ordinal"].detach())})
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.time() - started
                print(f"[exp45a-encoder] fold={args.fold} epoch={epoch}/{args.epochs} batch={batch_index}/{len(loader)} step={global_step}/{total_updates} loss={sums['loss']/batch_index:.4f} elapsed={elapsed/60:.1f}m eta={elapsed/max(global_step,1)*(total_updates-global_step)/60:.1f}m", flush=True)
                if args.max_updates and global_step >= args.max_updates:
                    stop = True
                    break
        history.append({"epoch": epoch, "global_step": global_step, "learning_rate": scheduler.get_last_lr()[0], **{f"train_{key}": sums[key] / max(len(loader), 1) for key in ("loss", "distribution", "ordinal")}})
        if stop:
            break

    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "run_fingerprint": run_identity["run_fingerprint"], "fold": args.fold, "epoch": history[-1]["epoch"], "global_step": global_step}, temporary)
    os.replace(temporary, checkpoint)
    clone = build_model(TACOModelConfig(args.model_name_or_path, use_projection=False)).cuda()
    clone.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True)["model"])
    summary = {
        "status": "COMPLETED", "source": "retrained_exp45", "fold": args.fold,
        "mode": args.mode, "seed": 42, "train_rows": len(train_rows), "heldout_rows": len(heldout_rows),
        "fixed_final_epoch": history[-1]["epoch"], "global_step": global_step,
        "run_fingerprint": run_identity["run_fingerprint"], "checkpoint_sha256": sha256_file(checkpoint),
        "save_reload": "PASS", "question_key_overlap": 0, "history": history,
        "oom_count": 0, "nan_count": 0, "dev_access_count": 0, "test_access_count": 0,
    }
    atomic_json(summary_path, summary)
    print(json.dumps({"status": "COMPLETED", "fold": args.fold, "checkpoint": str(checkpoint), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
