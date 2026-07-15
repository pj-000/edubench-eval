"""Train one frozen-embedding Exp45A head and export heldout predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from thesis_exp.exp45_dopr_head.common import (
    ARTIFACT_ROOT,
    FOLDS,
    ROOT,
    RUN_ROOT,
    SEED,
    TRAINED_HEAD_VARIANTS,
    atomic_json,
    embedding_path,
    head_prediction_path,
    head_run_dir,
    load_json,
    prototype_path,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_jsonl,
)
from thesis_exp.exp45_dopr_head.modeling_exp45a_heads import (
    balanced_batch_indices,
    build_head,
    distributional_ordinal_loss,
    hard_ce_loss,
    normalize_rows,
    restored_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=TRAINED_HEAD_VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--mode", choices=("groupcv", "smoke"), default="groupcv")
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rows-per-class", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-heldout-rows", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def run_identity(args: argparse.Namespace) -> dict:
    paths = [
        embedding_path(args.out_dir, args.fold, "outer_train"),
        embedding_path(args.out_dir, args.fold, "outer_heldout"),
        prototype_path(args.out_dir, args.fold),
        args.out_dir / "configs/exp45a_head_protocol_lock.json",
    ]
    value = {
        "variant": args.variant,
        "fold": args.fold,
        "mode": args.mode,
        "seed": SEED,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "rows_per_class": args.rows_per_class,
        "max_train_rows": args.max_train_rows,
        "max_heldout_rows": args.max_heldout_rows,
        "max_updates": args.max_updates,
        "input_hashes": {str(path): sha256_file(path) for path in paths},
        "code_hashes": {
            str(path): sha256_file(path)
            for path in (
                Path("thesis_exp/exp45_dopr_head/modeling_exp45a_heads.py"),
                Path("thesis_exp/exp45_dopr_head/train_exp45a_decoupled_heads.py"),
            )
        },
    }
    value["run_fingerprint"] = stable_hash(value)
    return value


def main() -> None:
    args = parse_args()
    if args.mode == "groupcv" and (
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.rows_per_class,
        args.max_train_rows,
        args.max_heldout_rows,
        args.max_updates,
    ) != (10, 1e-3, 1e-4, 20, None, None, None):
        raise ValueError("Exp45A formal head protocol is locked")

    run_dir = head_run_dir(args.run_root, args.variant, args.fold, args.mode)
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = args.out_dir / f"private/heads/{args.mode}/{args.variant}/fold_{args.fold}/head.pt"
    prediction_path = (
        head_prediction_path(args.out_dir, args.variant, args.fold)
        if args.mode == "groupcv"
        else args.out_dir / f"private/predictions/smoke/{args.variant}/fold_{args.fold}.jsonl"
    )
    identity = run_identity(args)
    if args.skip_completed and summary_path.exists() and checkpoint_path.exists() and prediction_path.exists():
        summary = load_json(summary_path)
        if summary.get("status") == "COMPLETED" and summary.get("run_fingerprint") == identity["run_fingerprint"]:
            print(json.dumps({"status": "REUSED", "variant": args.variant, "fold": args.fold}, sort_keys=True))
            return

    import torch
    from torch.optim import AdamW

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_jsonl(embedding_path(args.out_dir, args.fold, "outer_train"))
    heldout_rows = read_jsonl(embedding_path(args.out_dir, args.fold, "outer_heldout"))
    if args.max_train_rows:
        # Smoke keeps at least one row per class, then fills deterministically.
        selected = [next(row for row in train_rows if int(row["gold_label_5"]) == label) for label in range(1, 6)]
        selected_ids = {row["sample_id"] for row in selected}
        for label in range(1, 6):
            selected.extend(row for row in train_rows if int(row["gold_label_5"]) == label and row["sample_id"] not in selected_ids)
        train_rows = selected[: args.max_train_rows]
    if args.max_heldout_rows:
        heldout_rows = heldout_rows[: args.max_heldout_rows]
    if {row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows}:
        raise RuntimeError("Exp45A head question-key leakage")

    train_h = normalize_rows(np.asarray([row["embedding"] for row in train_rows], dtype=np.float32))
    train_y = np.asarray([int(row["gold_label_5"]) for row in train_rows], dtype=np.int64)
    train_targets = np.asarray([row["human_distribution_5"] for row in train_rows], dtype=np.float32)
    heldout_h = normalize_rows(np.asarray([row["embedding"] for row in heldout_rows], dtype=np.float32))
    prototype = load_json(prototype_path(args.out_dir, args.fold))
    init = np.asarray(prototype["prototypes"], dtype=np.float32) if args.variant in {"H3_prototype_cRT_no_prior", "H4_DOPR"} else None
    head = build_head(train_h.shape[1], init, scale=16.0).to(device)
    optimizer = AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batches_per_epoch = max(math.ceil(len(train_rows) / 100), 1)
    total_updates = args.max_updates or args.epochs * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_updates, 1))
    history = []
    step = 0
    stop = False
    for epoch in range(1, args.epochs + 1):
        head.train()
        sums = {"loss": 0.0, "distribution": 0.0, "ordinal": 0.0}
        epoch_batches = 0
        for indices in balanced_batch_indices(train_y, fold=args.fold, epoch=epoch, rows_per_class=args.rows_per_class):
            labels = train_y[indices]
            counts = [int(np.sum(labels == label)) for label in range(1, 6)]
            if counts != [args.rows_per_class] * 5:
                raise RuntimeError(f"Unbalanced Exp45A batch: {counts}")
            embeddings = torch.as_tensor(train_h[indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(train_targets[indices], dtype=torch.float32, device=device)
            hard_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            logits = head(embeddings)
            if tuple(logits.shape) != (len(indices), 5):
                raise RuntimeError(f"Invalid logits shape {tuple(logits.shape)}")
            if args.variant == "H1_vanilla_cRT":
                loss = hard_ce_loss(logits, hard_labels)
                distribution = loss
                ordinal = torch.zeros((), device=device)
            else:
                loss, distribution, ordinal = distributional_ordinal_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Exp45A head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(); scheduler.step()
            step += 1; epoch_batches += 1
            sums["loss"] += float(loss.detach()); sums["distribution"] += float(distribution.detach()); sums["ordinal"] += float(ordinal.detach())
            if args.max_updates and step >= args.max_updates:
                stop = True
                break
        history.append({"epoch": epoch, "global_step": step, "learning_rate": scheduler.get_last_lr()[0], **{f"train_{key}": value / max(epoch_batches, 1) for key, value in sums.items()}, "balanced_batches": epoch_batches})
        print(f"[exp45a-head] {args.variant} fold={args.fold} epoch={epoch}/{args.epochs} step={step}/{total_updates} loss={history[-1]['train_loss']:.5f}", flush=True)
        if stop:
            break

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save({"head": head.state_dict(), "dimension": train_h.shape[1], "run_fingerprint": identity["run_fingerprint"]}, temporary)
    os.replace(temporary, checkpoint_path)
    clone = build_head(train_h.shape[1], init, scale=16.0).to(device)
    clone.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["head"])
    clone.eval()
    with torch.no_grad():
        evidence = clone(torch.as_tensor(heldout_h, dtype=torch.float32, device=device))
        final_logits = restored_logits(evidence, prototype["soft_prior"]) if args.variant == "H4_DOPR" else evidence
        probabilities = torch.softmax(final_logits, dim=-1).cpu().numpy()
    predictions = []
    for row, probability in zip(heldout_rows, probabilities):
        predictions.append(
            {
                **{key: row[key] for key in ("sample_id", "question_key", "fold", "gold_label_5", "human_distribution_5", "expected_human_score", "metric", "language", "subject")},
                "variant": args.variant,
                "pred_label_5": int(np.argmax(probability)) + 1,
                "pred_score_expected": float(probability @ np.arange(1, 6)),
                **{f"prob_{label}": float(probability[label - 1]) for label in range(1, 6)},
            }
        )
    write_jsonl(prediction_path, predictions)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "COMPLETED",
        "variant": args.variant,
        "fold": args.fold,
        "mode": args.mode,
        "fixed_final_epoch": history[-1]["epoch"],
        "global_step": step,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "run_fingerprint": identity["run_fingerprint"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "save_reload": "PASS",
        "encoder_parameters_trainable": 0,
        "question_key_overlap": 0,
        "oom_count": 0,
        "nan_count": 0,
        "dev_access_count": 0,
        "test_access_count": 0,
        "history": history,
    }
    atomic_json(summary_path, summary)
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "fold": args.fold, "rows": len(predictions)}, sort_keys=True))


if __name__ == "__main__":
    main()
