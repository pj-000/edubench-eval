"""Train one locked Exp43 GroupCV or headline run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp43_rubimor.common import (
    ARTIFACT_ROOT, METRIC_HEAD_VARIANTS, PAIR_VARIANTS, ROOT, RUN_ROOT, SEEDS,
    VARIANTS, atomic_json, prediction_metrics, read_jsonl, stable_hash, write_jsonl,
)
from thesis_exp.exp43_rubimor.losses_rubimor import pair_rank_loss, total_point_loss
from thesis_exp.exp43_rubimor.modeling_rubimor import RubiMORConfig, build_model


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
    parser.add_argument("--mode", choices=("groupcv", "headline", "smoke"), default="groupcv")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
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
    return parser.parse_args()


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, batch_size: int, shuffle: bool, seed: int, max_length: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metric_ids"] = torch.tensor([int(row["metric_id"]) for row in batch], dtype=torch.long)
        encoded["metadata"] = batch
        return encoded

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Rows(rows), batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None, num_workers=0, collate_fn=collate)


def pair_flip_ids(pairs: list[dict[str, Any]], variant: str) -> set[str]:
    if variant != "E6N":
        return set()
    flips: set[str] = set()
    for gap_bin in ("near", "far"):
        subset = sorted((pair for pair in pairs if pair["gap_bin"] == gap_bin), key=lambda row: stable_hash((row["pair_id"], 4242)))
        flips.update(row["pair_id"] for row in subset[: round(len(subset) * 0.5)])
    return flips


def added_state(model: Any) -> dict[str, Any]:
    state = {f"global_head.{key}": value.detach().cpu() for key, value in model.global_head.state_dict().items()}
    if model.use_metric_residual:
        state.update({"metric_v": model.metric_v.detach().cpu(), "metric_u": model.metric_u.detach().cpu(), "metric_bias": model.metric_bias.detach().cpu()})
    return state


def load_added_state(model: Any, state: dict[str, Any]) -> None:
    model.global_head.load_state_dict({key.split(".", 1)[1]: value for key, value in state.items() if key.startswith("global_head.")})
    if model.use_metric_residual:
        model.metric_v.data.copy_(state["metric_v"])
        model.metric_u.data.copy_(state["metric_u"])
        model.metric_bias.data.copy_(state["metric_bias"])


def evaluate(model: Any, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, device: Any, residual_ids: set[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    model.eval()
    output_rows = []
    loader = make_loader(rows, tokenizer, args.eval_batch_size, False, args.seed, args.max_length)
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            batch.pop("targets")
            metric_ids = batch.pop("metric_ids").to(device)
            enabled = torch.tensor([int(value) in residual_ids for value in metric_ids.tolist()], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{key: value.to(device) for key, value in batch.items()}, metric_ids=metric_ids, residual_enabled=enabled)["logits"].float()
            probabilities = torch.softmax(logits, -1).cpu().tolist()
            for source, probs in zip(metadata, probabilities):
                output_rows.append({
                    "variant": args.variant, "seed": args.seed, "fold": args.fold if args.mode != "headline" else "headline",
                    "sample_id": source["sample_id"], "question_key": source["question_key"], "metric": source["metric"],
                    "metric_id": source["metric_id"], "gold_label_5": source["gold_label_5"],
                    "human_distribution_5": source["human_distribution_5"], "expected_human_score": source["expected_human_score"],
                    "human_entropy": source["human_entropy"], "human_score_range": source["human_score_range"],
                    "language": source["language"], "subject": source["subject"], "scenario": source["scenario"], "education_level": source["education_level"],
                    "pred_label_5": int(np.argmax(probs)) + 1, "pred_score_expected": sum(label * probs[label - 1] for label in range(1, 6)),
                    **{f"prob_{label}": probs[label - 1] for label in range(1, 6)},
                })
    return output_rows, prediction_metrics(output_rows)


def checkpoint_save(path: Path, model: Any, optimizer: Any, scheduler: Any, epoch: int, global_step: int, best: dict[str, Any] | None = None, *, model_only: bool = False) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    payload = {"model": model.state_dict(), "epoch": epoch, "global_step": global_step, "best": best}
    if not model_only:
        payload.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    torch.save(payload, temp)
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    formal = args.mode != "smoke"
    if formal and (args.epochs, args.learning_rate, args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.max_length) != (10, 2e-5, 4, 4, 32, 2048):
        raise ValueError("Formal Exp43 protocol is locked to epochs/lr/batches/max_length = 10/2e-5/4/4/32/2048")
    if args.mode == "headline" and args.variant not in {"E0", "E3", "E5", "E6", "E6N"}:
        raise ValueError("Headline variants are locked to E0/E3/E5/E6/E6N")
    data_root = args.out_dir / "private/data"
    all_train = read_jsonl(data_root / f"exp43_train_{args.variant}.jsonl")
    if args.mode in {"groupcv", "smoke"}:
        with (data_root / "exp43_groupcv_fold_assignment.csv").open("r", encoding="utf-8", newline="") as handle:
            fold_by_id = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
        train_rows = [row for row in all_train if fold_by_id[row["sample_id"]] != args.fold]
        eval_rows = [row for row in all_train if fold_by_id[row["sample_id"]] == args.fold]
        if {r["question_key"] for r in train_rows} & {r["question_key"] for r in eval_rows}:
            raise RuntimeError("GroupCV question-key leakage")
        pairs_path = args.out_dir / f"private/pairs/exp43_train_pairs_fold{args.fold}.jsonl"
        run_dir = args.run_root / ("smoke" if args.mode == "smoke" else "groupcv") / args.variant / f"seed_{args.seed}" / f"fold_{args.fold}"
    else:
        train_rows = all_train
        eval_rows = read_jsonl(data_root / f"exp43_dev_{args.variant}.jsonl")
        pairs_path = args.out_dir / "private/pairs/exp43_train_pairs.jsonl"
        run_dir = args.run_root / "headline" / args.variant / f"seed_{args.seed}"
    pairs = read_jsonl(pairs_path) if args.variant in PAIR_VARIANTS else []
    if args.max_train_rows:
        selected = train_rows[: args.max_train_rows - 2] if pairs and args.max_train_rows >= 2 else train_rows[: args.max_train_rows]
        if pairs:
            selected_ids = {row["sample_id"] for row in selected}
            first_pair = next((pair for pair in pairs if pair["positive_id"] in {r["sample_id"] for r in train_rows} and pair["negative_id"] in {r["sample_id"] for r in train_rows}), None)
            if first_pair:
                full_by_id = {row["sample_id"]: row for row in train_rows}
                for endpoint in (first_pair["positive_id"], first_pair["negative_id"]):
                    if endpoint not in selected_ids:
                        selected.append(full_by_id[endpoint]); selected_ids.add(endpoint)
                selected = selected[: args.max_train_rows]
        train_rows = selected
    if args.max_eval_rows:
        eval_rows = eval_rows[: args.max_eval_rows]
    if args.max_train_rows:
        allowed = {row["sample_id"] for row in train_rows}
        pairs = [pair for pair in pairs if pair["positive_id"] in allowed and pair["negative_id"] in allowed]
    if args.variant in PAIR_VARIANTS and not pairs:
        raise ValueError("Pair variant has no usable pairs")

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    if not torch.cuda.is_available():
        raise RuntimeError("Exp43 training requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = json.loads((args.out_dir / "configs/exp43_metric_mapping.json").read_text(encoding="utf-8"))["metrics"]
    model = build_model(RubiMORConfig(args.model_name_or_path, len(mapping), use_metric_residual=args.variant in METRIC_HEAD_VARIANTS))
    device = torch.device("cuda")
    model.to(device)
    metric_counts = Counter(int(row["metric_id"]) for row in train_rows)
    residual_ids = {metric_id for metric_id, count in metric_counts.items() if count >= 30}
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    planned_updates = max(1, args.epochs * updates_per_epoch)
    total_updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    resume_path = run_dir / "resume_checkpoint.pt"
    start_epoch, global_step, best = 1, 0, None
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step, best = int(state["epoch"]) + 1, int(state["global_step"]), state.get("best")
    optimizer.zero_grad(set_to_none=True)
    row_by_id = {row["sample_id"]: row for row in train_rows}
    flips = pair_flip_ids(pairs, args.variant)
    flip_rate = len(flips) / len(pairs) if pairs else 0.0
    history, started, stop = [], time.time(), False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loader = make_loader(train_rows, tokenizer, args.batch_size, True, args.seed + epoch * 100 + args.fold, args.max_length)
        pair_order = sorted(pairs, key=lambda row: stable_hash((row["pair_id"], args.seed, args.fold, epoch))) if pairs else []
        sums = Counter(); seen = 0
        for batch_index, batch in enumerate(loader, 1):
            batch.pop("metadata")
            targets = batch.pop("targets").to(device)
            metric_ids = batch.pop("metric_ids").to(device)
            enabled = torch.tensor([int(value) in residual_ids for value in metric_ids.tolist()], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{key: value.to(device) for key, value in batch.items()}, metric_ids=metric_ids, residual_enabled=enabled)["logits"]
                point_loss, components = total_point_loss(args.variant, logits, targets)
                rank = point_loss.new_zeros(())
                if pairs:
                    pair = pair_order[(batch_index - 1) % len(pair_order)]
                    pos_id, neg_id = pair["positive_id"], pair["negative_id"]
                    if pair["pair_id"] in flips:
                        pos_id, neg_id = neg_id, pos_id
                    pair_batch = make_loader([row_by_id[pos_id], row_by_id[neg_id]], tokenizer, 2, False, args.seed, args.max_length)
                    encoded = next(iter(pair_batch)); encoded.pop("metadata"); encoded.pop("targets")
                    pair_metric_ids = encoded.pop("metric_ids").to(device)
                    pair_enabled = torch.tensor([int(value) in residual_ids for value in pair_metric_ids.tolist()], device=device)
                    pair_logits = model(**{key: value.to(device) for key, value in encoded.items()}, metric_ids=pair_metric_ids, residual_enabled=pair_enabled)["logits"]
                    rank = pair_rank_loss(pair_logits[:1], pair_logits[1:])
                loss = point_loss + 0.25 * rank
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Exp43 loss")
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            sums.update({"loss": float(loss.detach()), "distribution": float(components["distribution"].detach()), "ordinal": float(components["ordinal"].detach()), "rank": float(rank.detach())})
            seen += 1
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
                elapsed = time.time() - started
                print(f"[exp43] {args.mode}/{args.variant} seed={args.seed} fold={args.fold} epoch={epoch}/{args.epochs} batch={batch_index}/{len(loader)} step={global_step}/{total_updates} loss={sums['loss']/seen:.4f} elapsed={elapsed/60:.1f}m eta={elapsed/max(global_step,1)*(total_updates-global_step)/60:.1f}m", flush=True)
                if args.max_updates and global_step >= args.max_updates:
                    stop = True
                    break
        epoch_row = {"epoch": epoch, "global_step": global_step, "learning_rate": scheduler.get_last_lr()[0], **{f"train_{key}": sums[key] / max(seen, 1) for key in ("loss", "distribution", "ordinal", "rank")}}
        if args.mode == "headline":
            predictions, metrics = evaluate(model, eval_rows, tokenizer, args, device, residual_ids)
            epoch_row.update({f"dev_{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
            candidate = {"epoch": epoch, "Exact_Match": metrics["Exact_Match"], "MAE": metrics["MAE"]}
            if best is None or (candidate["Exact_Match"], -candidate["MAE"], -candidate["epoch"]) > (best["Exact_Match"], -best["MAE"], -best["epoch"]):
                best = candidate
                checkpoint_save(run_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch, global_step, best, model_only=True)
                write_jsonl(run_dir / "best_dev_predictions.jsonl", predictions)
        history.append(epoch_row)
        checkpoint_save(resume_path, model, optimizer, scheduler, epoch, global_step, best)
        atomic_json(run_dir / "run_summary.json", {"status": "STARTED", "variant": args.variant, "mode": args.mode, "seed": args.seed, "fold": args.fold, "last_completed_epoch": epoch, "global_step": global_step, "history": history})
        if stop:
            break
    if args.mode != "headline":
        predictions, final_metrics = evaluate(model, eval_rows, tokenizer, args, device, residual_ids)
        write_jsonl(run_dir / "heldout_predictions.jsonl", predictions)
    else:
        state = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        predictions = read_jsonl(run_dir / "best_dev_predictions.jsonl")
        final_metrics = prediction_metrics(predictions)
    smoke_reload = "not_requested"
    if args.mode == "smoke":
        smoke_path = args.artifact_root / "smoke" / f"{args.variant}_added_state.pt"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(added_state(model), smoke_path)
        before = added_state(model); load_added_state(model, torch.load(smoke_path, map_location="cpu", weights_only=True)); after = added_state(model)
        if any(not torch.equal(before[key], after[key]) for key in before):
            raise RuntimeError("Custom-head save/reload mismatch")
        smoke_reload = "PASS"
    residual_norms = []
    if model.use_metric_residual:
        with torch.no_grad():
            for metric_name, metric_id in sorted(mapping.items(), key=lambda item: item[1]):
                residual_norms.append({
                    "metric": metric_name, "metric_id": metric_id,
                    "enabled": metric_id in residual_ids, "train_rows": metric_counts.get(metric_id, 0),
                    "v_norm": float(model.metric_v[metric_id].float().norm().cpu()),
                    "u_norm": float(model.metric_u[metric_id].float().norm().cpu()),
                    "bias_norm": float(model.metric_bias[metric_id].float().norm().cpu()),
                    "effective_residual_norm": float((model.metric_u[metric_id].float().norm() + model.metric_bias[metric_id].float().norm()).cpu()),
                })
    summary = {
        "status": "COMPLETED", "variant": args.variant, "mode": args.mode, "seed": args.seed, "fold": args.fold,
        "train_rows": len(train_rows), "eval_rows": len(eval_rows), "fixed_final_epoch": args.epochs if formal else history[-1]["epoch"],
        "selected_epoch": best["epoch"] if best else args.epochs, "global_step": global_step,
        "question_key_overlap": len({r['question_key'] for r in train_rows} & {r['question_key'] for r in eval_rows}),
        "model_mode": "full_finetuning", "pooling": "last_non_padding_token", "hidden_size": model.hidden_size,
        "added_parameter_count": model.added_parameter_count(), "residual_enabled_metric_ids": sorted(residual_ids),
        "metric_residual_norms": residual_norms,
        "pair_count": len(pairs), "pair_flip_count": len(flips), "pair_flip_rate": flip_rate,
        "smoke_save_reload": smoke_reload, "history": history, "metrics": final_metrics,
        "nan_count": 0, "oom_count": 0, "dev_access_count": int(args.mode == "headline"), "test_access_count": 0,
    }
    atomic_json(run_dir / "run_summary.json", summary)
    if args.mode != "headline" and resume_path.exists():
        resume_path.unlink()
    if args.mode == "headline" and resume_path.exists():
        resume_path.unlink()
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "mode": args.mode, "seed": args.seed, "fold": args.fold, "metrics": final_metrics, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
