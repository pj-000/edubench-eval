"""Train one locked Exp44A GroupCV or smoke run."""

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

from thesis_exp.exp44_taco_score.common import (
    ARTIFACT_ROOT,
    CONTRASTIVE_VARIANTS,
    ROOT,
    RUN_ROOT,
    VARIANTS,
    atomic_json,
    fold_map,
    prediction_metrics,
    read_jsonl,
    run_dir,
    sha256_file,
    stable_hash,
    triplet_path,
    write_jsonl,
)
from thesis_exp.exp44_taco_score.losses_exp44a_taco import base_loss, taco_loss
from thesis_exp.exp44_taco_score.modeling_exp44a_taco import TACOModelConfig, build_model


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
    parser.add_argument("--mode", choices=("groupcv", "smoke"), default="groupcv")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--model-name-or-path", required=True)
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
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fingerprint-only", action="store_true")
    return parser.parse_args()


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    files = [
        args.out_dir / "private/data/exp44a_train_e4.jsonl",
        args.out_dir / "private/data/exp44a_groupcv_fold_assignment.csv",
        args.out_dir / "configs/exp44a_model_loss_lock.json",
    ]
    if args.variant in CONTRASTIVE_VARIANTS:
        files.extend(triplet_path(args.out_dir, args.fold, epoch) for epoch in range(1, args.epochs + 1))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp44A fingerprint inputs: {missing}")
    code_files = [
        Path("thesis_exp/exp44_taco_score/common.py"),
        Path("thesis_exp/exp44_taco_score/modeling_exp44a_taco.py"),
        Path("thesis_exp/exp44_taco_score/losses_exp44a_taco.py"),
        Path("thesis_exp/exp44_taco_score/train_exp44a_groupcv.py"),
    ]
    config = {
        key: getattr(args, key)
        for key in (
            "variant", "mode", "fold", "seed", "epochs", "learning_rate",
            "weight_decay", "warmup_ratio", "batch_size", "eval_batch_size",
            "gradient_accumulation", "max_length", "max_train_rows",
            "max_eval_rows", "max_updates",
        )
    }
    identity = {
        "config": config,
        "data_hashes": {str(path): sha256_file(path) for path in files},
        "model_hashes": json.loads((args.out_dir / "hashes/exp44a_model_hashes.json").read_text(encoding="utf-8")),
        "code_hashes": {str(path): sha256_file(path) for path in code_files},
    }
    identity["run_fingerprint"] = stable_hash(identity)
    return identity


def make_loader(
    rows: list[dict[str, Any]], tokenizer: Any, batch_size: int, shuffle: bool, seed: int, max_length: int
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Rows(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        collate_fn=collate,
    )


def encode_triplet(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_length: int, device: Any) -> Any:
    encoded = tokenizer(
        [row["text"] for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    output = model(
        **{key: value.to(device) for key, value in encoded.items()},
        return_projection=True,
    )
    return output["projection"].reshape(1, 4, -1)


def evaluate(model: Any, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, device: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    model.eval()
    output_rows = []
    loader = make_loader(rows, tokenizer, args.eval_batch_size, False, args.seed, args.max_length)
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch.pop("targets")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**{key: value.to(device) for key, value in batch.items()})
            logits = output["logits"].float()
            probabilities = torch.softmax(logits, -1).cpu().tolist()
            embeddings = torch.nn.functional.normalize(output["pooled"].float(), dim=-1).cpu().tolist()
            for source, probs, embedding in zip(metadata, probabilities, embeddings):
                output_rows.append(
                    {
                        "variant": args.variant,
                        "seed": args.seed,
                        "fold": args.fold,
                        "sample_id": source["sample_id"],
                        "question_key": source["question_key"],
                        "metric": source["metric"],
                        "gold_label_5": source["gold_label_5"],
                        "human_distribution_5": source["human_distribution_5"],
                        "expected_human_score": source["expected_human_score"],
                        "human_entropy": source["human_entropy"],
                        "human_score_range": source["human_score_range"],
                        "language": source["language"],
                        "subject": source["subject"],
                        "scenario": source["scenario"],
                        "education_level": source["education_level"],
                        "pred_label_5": int(np.argmax(probs)) + 1,
                        "pred_score_expected": sum(label * probs[label - 1] for label in range(1, 6)),
                        **{f"prob_{label}": probs[label - 1] for label in range(1, 6)},
                        "embedding": embedding,
                    }
                )
    return output_rows, prediction_metrics(output_rows)


def save_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any, epoch: int, global_step: int, fingerprint: str) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "run_fingerprint": fingerprint,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    formal = args.mode == "groupcv"
    locked = (10, 2e-5, 0.01, 0.05, 4, 4, 32, 2048)
    observed = (args.epochs, args.learning_rate, args.weight_decay, args.warmup_ratio, args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.max_length)
    if formal and observed != locked:
        raise ValueError(f"Formal Exp44A training protocol is locked: {observed} != {locked}")
    identity = run_identity(args)
    if args.fingerprint_only:
        print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
        return

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Exp44A training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    all_rows = read_jsonl(args.out_dir / "private/data/exp44a_train_e4.jsonl")
    folds = fold_map(args.out_dir / "private/data/exp44a_groupcv_fold_assignment.csv")
    full_train_rows = [row for row in all_rows if folds[row["sample_id"]] != args.fold]
    eval_rows = [row for row in all_rows if folds[row["sample_id"]] == args.fold]
    if {row["question_key"] for row in full_train_rows} & {row["question_key"] for row in eval_rows}:
        raise RuntimeError("Exp44A question-key leakage")
    train_rows = full_train_rows
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows:
        eval_rows = eval_rows[: args.max_eval_rows]
    row_by_id = {row["sample_id"]: row for row in full_train_rows}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(
        TACOModelConfig(
            args.model_name_or_path,
            projection_dim=128,
            use_projection=args.variant in CONTRASTIVE_VARIANTS,
        )
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    planned_updates = max(1, args.epochs * updates_per_epoch)
    total_updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    current_run_dir = run_dir(args.run_root, args.variant, args.fold, args.mode)
    resume_path = current_run_dir / "resume_checkpoint.pt"
    start_epoch, global_step = 1, 0
    fingerprint = identity["run_fingerprint"]
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("run_fingerprint") != fingerprint:
            raise RuntimeError("Exp44A resume fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])

    optimizer.zero_grad(set_to_none=True)
    history = []
    started = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loader = make_loader(
            train_rows,
            tokenizer,
            args.batch_size,
            True,
            args.seed + epoch * 100 + args.fold,
            args.max_length,
        )
        triplets = (
            read_jsonl(triplet_path(args.out_dir, args.fold, epoch))
            if args.variant in CONTRASTIVE_VARIANTS
            else []
        )
        if args.max_train_rows:
            triplets = triplets[:8]
        sums: Counter[str] = Counter()
        seen = 0
        contrastive_seen = 0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("metadata")
            targets = batch.pop("targets").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{key: value.to(device) for key, value in batch.items()})["logits"]
                point, components = base_loss(logits, targets)
                contrastive = point.new_zeros(())
                diagnostics: dict[str, Any] = {}
                if batch_index <= len(triplets):
                    triplet = triplets[batch_index - 1]
                    endpoint_rows = [row_by_id[triplet[key]] for key in ("anchor_id", "positive_id", "near_id", "far_id")]
                    projections = encode_triplet(model, tokenizer, endpoint_rows, args.max_length, device)
                    if args.variant == "C3_shuffled_margin_control":
                        near_distance = triplet["shuffled_near_distance"]
                        far_distance = triplet["shuffled_far_distance"]
                    else:
                        near_distance = triplet["near_distance"]
                        far_distance = triplet["far_distance"]
                    contrastive, diagnostics = taco_loss(
                        projections,
                        torch.tensor([near_distance], device=device),
                        torch.tensor([far_distance], device=device),
                        use_margin=args.variant != "C1_balanced_plain_contrastive",
                    )
                    contrastive_seen += 1
                loss = point + 0.1 * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Exp44A loss")
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            sums.update(
                {
                    "loss": float(loss.detach()),
                    "distribution": float(components["distribution"].detach()),
                    "ordinal": float(components["ordinal"].detach()),
                    "contrastive": float(contrastive.detach()),
                }
            )
            for key, value in diagnostics.items():
                sums[key] += float(value.detach())
            seen += 1
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.time() - started
                print(
                    f"[exp44a] {args.mode}/{args.variant} seed=42 fold={args.fold} "
                    f"epoch={epoch}/{args.epochs} batch={batch_index}/{len(loader)} "
                    f"step={global_step}/{total_updates} loss={sums['loss']/seen:.4f} "
                    f"elapsed={elapsed/60:.1f}m eta={elapsed/max(global_step,1)*(total_updates-global_step)/60:.1f}m",
                    flush=True,
                )
                if args.max_updates and global_step >= args.max_updates:
                    stop = True
                    break
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "point_batches": seen,
                "contrastive_anchors": contrastive_seen,
                **{f"train_{key}": sums[key] / max(seen, 1) for key in ("loss", "distribution", "ordinal", "contrastive")},
                **{f"contrastive_{key}": sums[key] / max(contrastive_seen, 1) for key in ("triplet_accuracy", "near_margin_violation", "far_margin_violation", "mean_positive_similarity", "mean_near_similarity", "mean_far_similarity")},
            }
        )
        save_checkpoint(resume_path, model, optimizer, scheduler, epoch, global_step, fingerprint)
        atomic_json(
            current_run_dir / "run_summary.json",
            {"status": "STARTED", "variant": args.variant, "mode": args.mode, "fold": args.fold, "history": history, "run_fingerprint": fingerprint},
        )
        if stop:
            break

    predictions, metrics = evaluate(model, eval_rows, tokenizer, args, device)
    write_jsonl(current_run_dir / "heldout_predictions.jsonl", predictions)
    smoke_reload = "not_requested"
    if args.mode == "smoke":
        smoke_path = args.artifact_root / "smoke" / f"{args.variant}_fold{args.fold}.pt"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), smoke_path)
        clone = build_model(TACOModelConfig(args.model_name_or_path, use_projection=args.variant in CONTRASTIVE_VARIANTS)).to(device)
        clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True))
        smoke_reload = "PASS"
    summary = {
        "status": "COMPLETED",
        "variant": args.variant,
        "mode": args.mode,
        "seed": 42,
        "fold": args.fold,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "fixed_final_epoch": args.epochs if formal else history[-1]["epoch"],
        "global_step": global_step,
        "question_key_overlap": 0,
        "model_mode": "full_finetuning",
        "pooling": "last_non_padding_token",
        "hidden_size": model.hidden_size,
        "projection_dim": 128 if args.variant in CONTRASTIVE_VARIANTS else 0,
        "added_parameter_count": model.added_parameter_count(),
        "run_fingerprint": fingerprint,
        "run_identity": identity,
        "smoke_save_reload": smoke_reload,
        "history": history,
        "metrics": metrics,
        "nan_count": 0,
        "oom_count": 0,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    atomic_json(current_run_dir / "run_summary.json", summary)
    if resume_path.exists():
        resume_path.unlink()
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "fold": args.fold, "metrics": metrics, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
