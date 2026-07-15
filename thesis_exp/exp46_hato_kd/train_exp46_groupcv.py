"""Train one fixed-fold Exp46A teacher or student run."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp46_hato_kd.common import (
    ALL_VARIANTS,
    DATA_PATH,
    FOLD_PATH,
    ROOT,
    RUN_ROOT,
    STUDENT_VARIANTS,
    TEACHER_VARIANT,
    ensure_dirs,
    read_jsonl,
    run_dir,
    sha256_file,
    split_rows,
    stable_hash,
    teacher_logit_path,
    write_json,
    write_jsonl,
    write_protocol_locks,
)
from thesis_exp.exp46_hato_kd.losses_hato import human_anchor_loss, ordinal_kd_loss, standard_kd_loss
from thesis_exp.exp46_hato_kd.modeling_hato import ModelSpec, build_model


class Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fingerprint-only", action="store_true")
    return parser.parse_args()


def effective_learning_rate(args: argparse.Namespace) -> float:
    if args.learning_rate is not None:
        return float(args.learning_rate)
    return 1e-4 if args.variant == TEACHER_VARIANT else 2e-5


def code_files() -> list[Path]:
    return [
        Path("thesis_exp/exp46_hato_kd/common.py"),
        Path("thesis_exp/exp46_hato_kd/losses_hato.py"),
        Path("thesis_exp/exp46_hato_kd/modeling_hato.py"),
        Path("thesis_exp/exp46_hato_kd/train_exp46_groupcv.py"),
    ]


def model_signature(path: Path) -> dict[str, str]:
    candidates = [path / "config.json", path / "model.safetensors.index.json", path / "model.safetensors"]
    return {str(item): sha256_file(item) for item in candidates if item.exists()}


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "variant": args.variant,
        "fold": args.fold,
        "seed": 42,
        "epochs": args.epochs,
        "learning_rate": effective_learning_rate(args),
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "max_train_rows": args.max_train_rows,
        "max_eval_rows": args.max_eval_rows,
        "max_updates": args.max_updates,
        "smoke": args.smoke,
    }
    inputs = [DATA_PATH, FOLD_PATH]
    if args.variant in STUDENT_VARIANTS:
        inputs.append(teacher_logit_path(args.fold, args.run_root))
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp46 inputs: {missing}")
    identity = {
        "config": config,
        "input_hashes": {str(path): sha256_file(path) for path in inputs},
        "model_signature": model_signature(Path(args.model_name_or_path)),
        "code_hashes": {str(path): sha256_file(path) for path in code_files()},
    }
    identity["run_fingerprint"] = stable_hash(identity)
    return identity


def make_loader(rows: list[dict[str, Any]], tokenizer: Any, batch_size: int, shuffle: bool, seed: int, max_length: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["targets"] = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32)
        encoded["metadata"] = batch
        return encoded

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Rows(rows), batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None, num_workers=0, collate_fn=collate)


def balanced_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["gold_label_5"])].append(row)
    if set(by_class) != set(range(1, 6)):
        raise RuntimeError(f"Balanced KD stream is missing classes: {sorted(by_class)}")
    rng = random.Random(seed)
    total = len(rows)
    counts = [total // 5 + int(index < total % 5) for index in range(5)]
    sampled: list[dict[str, Any]] = []
    for label, count in zip(range(1, 6), counts):
        pool = by_class[label]
        sampled.extend(pool[rng.randrange(len(pool))] for _ in range(count))
    rng.shuffle(sampled)
    return sampled


def representative_prefix(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep smoke subsets tiny while retaining every available score class."""
    if limit >= len(rows):
        return rows
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for label in range(1, 6):
        candidate = next((row for row in rows if int(row["gold_label_5"]) == label), None)
        if candidate is not None and len(selected) < limit:
            selected.append(candidate)
            selected_ids.add(candidate["sample_id"])
    for row in rows:
        if len(selected) >= limit:
            break
        if row["sample_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["sample_id"])
    return selected


def shuffled_teacher_donors(rows: list[dict[str, Any]]) -> tuple[dict[str, str], float]:
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in rows:
        groups[(int(row["gold_label_5"]), str(row.get("language", "unknown")))].append(row["sample_id"])
    mapping: dict[str, str] = {}
    for key, sample_ids in sorted(groups.items()):
        ordered = sorted(sample_ids, key=lambda value: stable_hash(("exp46-k3", key, value)))
        if len(ordered) == 1:
            mapping[ordered[0]] = ordered[0]
        else:
            for index, sample_id in enumerate(ordered):
                mapping[sample_id] = ordered[(index + 1) % len(ordered)]
    change_rate = sum(sample_id != donor for sample_id, donor in mapping.items()) / max(len(mapping), 1)
    return mapping, change_rate


def teacher_tensor(metadata: list[dict[str, Any]], logits_by_id: dict[str, list[float]], donor_by_id: dict[str, str] | None, device: Any) -> Any:
    import torch
    values = []
    for row in metadata:
        sample_id = row["sample_id"]
        donor = donor_by_id[sample_id] if donor_by_id is not None else sample_id
        if donor not in logits_by_id:
            raise KeyError(f"Missing teacher logit donor: {donor}")
        values.append(logits_by_id[donor])
    return torch.tensor(values, dtype=torch.float32, device=device)


def evaluate(model: Any, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, device: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from thesis_exp.exp43_rubimor.common import prediction_metrics

    model.eval()
    output_rows = []
    with torch.inference_mode():
        for batch in make_loader(rows, tokenizer, args.eval_batch_size, False, 42, args.max_length):
            metadata = batch.pop("metadata")
            batch.pop("targets")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{key: value.to(device) for key, value in batch.items()})["logits"].float()
            probabilities = torch.softmax(logits, -1).cpu().tolist()
            raw_logits = logits.cpu().tolist()
            for source, probs, values in zip(metadata, probabilities, raw_logits):
                output_rows.append(
                    {
                        "variant": args.variant,
                        "seed": 42,
                        "fold": args.fold,
                        "sample_id": source["sample_id"],
                        "question_key": source["question_key"],
                        "metric": source["metric"],
                        "gold_label_5": source["gold_label_5"],
                        "human_distribution_5": source["human_distribution_5"],
                        "expected_human_score": source["expected_human_score"],
                        "language": source["language"],
                        "subject": source["subject"],
                        "pred_label_5": int(np.argmax(probs)) + 1,
                        "pred_score_expected": sum(label * probs[label - 1] for label in range(1, 6)),
                        **{f"prob_{label}": probs[label - 1] for label in range(1, 6)},
                        **{f"logit_{label}": values[label - 1] for label in range(1, 6)},
                    }
                )
    return output_rows, prediction_metrics(output_rows)


def export_teacher_logits(model: Any, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace, device: Any, heldout_ids: set[str]) -> None:
    predictions, _ = evaluate(model, rows, tokenizer, args, device)
    output = [
        {
            "sample_id": row["sample_id"],
            "fold": args.fold,
            "role": "heldout" if row["sample_id"] in heldout_ids else "outer_train",
            "gold_label_5": row["gold_label_5"],
            "language": row["language"],
            "teacher_logits": [row[f"logit_{label}"] for label in range(1, 6)],
        }
        for row in predictions
    ]
    write_jsonl(teacher_logit_path(args.fold, args.run_root), output)


def save_teacher_adapter(model: Any, path: Path) -> None:
    import torch
    from peft import get_peft_model_state_dict
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": get_peft_model_state_dict(model.backbone), "classifier": model.classifier.state_dict()}, path)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    write_protocol_locks(args.out_dir)
    formal_tuple = (args.epochs, args.batch_size, args.eval_batch_size, args.gradient_accumulation, args.max_length)
    if not args.smoke and formal_tuple != (10, 4, 4, 32, 2048):
        raise ValueError("Formal Exp46 protocol is locked to epochs/batches/accumulation/max_length = 10/4/4/32/2048")
    expected_lr = 1e-4 if args.variant == TEACHER_VARIANT else 2e-5
    if not args.smoke and effective_learning_rate(args) != expected_lr:
        raise ValueError(f"Formal {args.variant} learning rate is locked to {expected_lr}")
    identity = run_identity(args)
    if args.fingerprint_only:
        print(json.dumps(identity, sort_keys=True))
        return

    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Exp46 training requires CUDA")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    all_rows = read_jsonl(DATA_PATH)
    train_rows, heldout_rows = split_rows(all_rows, args.fold)
    if args.max_train_rows:
        train_rows = representative_prefix(train_rows, args.max_train_rows)
    if args.max_eval_rows:
        heldout_rows = heldout_rows[: args.max_eval_rows]
    teacher_logits: dict[str, list[float]] = {}
    donor_by_id: dict[str, str] | None = None
    donor_change_rate = 0.0
    if args.variant in STUDENT_VARIANTS:
        logit_rows = read_jsonl(teacher_logit_path(args.fold, args.run_root))
        outer_ids = {row["sample_id"] for row in train_rows}
        leakage = [row for row in logit_rows if row["sample_id"] in outer_ids and row.get("role") != "outer_train"]
        if leakage:
            raise RuntimeError(f"Teacher-logit role leakage: {len(leakage)}")
        teacher_logits = {row["sample_id"]: [float(value) for value in row["teacher_logits"]] for row in logit_rows if row["sample_id"] in outer_ids}
        if set(teacher_logits) != outer_ids:
            raise RuntimeError(f"Teacher-logit coverage mismatch: {len(teacher_logits)} vs {len(outer_ids)}")
        if args.variant == "K3_shuffled_hato_control":
            donor_by_id, donor_change_rate = shuffled_teacher_donors(train_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_mode = args.variant == TEACHER_VARIANT
    model = build_model(ModelSpec(args.model_name_or_path, use_lora=teacher_mode, gradient_checkpointing=teacher_mode))
    device = torch.device("cuda")
    model.to(device)
    parameter_counts = model.trainable_parameter_counts()
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=effective_learning_rate(args), weight_decay=args.weight_decay)
    batches = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    planned_updates = max(1, args.epochs * updates_per_epoch)
    total_updates = min(planned_updates, args.max_updates) if args.max_updates else planned_updates
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    optimizer.zero_grad(set_to_none=True)

    history = []
    global_step = 0
    started = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        natural_loader = make_loader(train_rows, tokenizer, args.batch_size, True, 46000 + epoch * 100 + args.fold, args.max_length)
        if args.variant in {"K2_hato_kd", "K3_shuffled_hato_control"}:
            distill = balanced_rows(train_rows, 46100 + epoch * 100 + args.fold)
            distill_loader = iter(make_loader(distill, tokenizer, args.batch_size, False, 42, args.max_length))
        else:
            distill_loader = None
        sums = Counter()
        seen = 0
        for batch_index, natural in enumerate(natural_loader, 1):
            metadata = natural.pop("metadata")
            targets = natural.pop("targets").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                natural_logits = model(**{key: value.to(device) for key, value in natural.items()})["logits"]
                human, components = human_anchor_loss(natural_logits, targets)
                kd = human.new_zeros(())
                ordinal_kd = human.new_zeros(())
                if args.variant == "K1_standard_kd":
                    teacher = teacher_tensor(metadata, teacher_logits, None, device)
                    kd = standard_kd_loss(natural_logits, teacher, 2.0)
                elif args.variant in {"K2_hato_kd", "K3_shuffled_hato_control"}:
                    distilled = next(distill_loader)
                    distill_metadata = distilled.pop("metadata")
                    distilled.pop("targets")
                    distill_logits = model(**{key: value.to(device) for key, value in distilled.items()})["logits"]
                    teacher = teacher_tensor(distill_metadata, teacher_logits, donor_by_id, device)
                    kd = standard_kd_loss(distill_logits, teacher, 2.0)
                    ordinal_kd = ordinal_kd_loss(distill_logits, teacher, 2.0)
                loss = human + 0.25 * kd + (0.25 * ordinal_kd if args.variant in {"K2_hato_kd", "K3_shuffled_hato_control"} else 0.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Exp46 loss")
            window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(natural_loader) - window_start)
            (loss / window_size).backward()
            sums.update({"loss": float(loss.detach()), "human": float(human.detach()), "human_distribution": float(components["human_distribution"].detach()), "human_ordinal": float(components["human_ordinal"].detach()), "kd": float(kd.detach()), "ordinal_kd": float(ordinal_kd.detach())})
            seen += 1
            if batch_index % args.gradient_accumulation == 0 or batch_index == len(natural_loader):
                torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.time() - started
                print(f"[exp46] {args.variant} fold={args.fold} epoch={epoch}/{args.epochs} batch={batch_index}/{len(natural_loader)} step={global_step}/{total_updates} loss={sums['loss']/seen:.4f} human={sums['human']/seen:.4f} kd={sums['kd']/seen:.4f} okd={sums['ordinal_kd']/seen:.4f} elapsed={elapsed/60:.1f}m eta={elapsed/max(global_step,1)*(total_updates-global_step)/60:.1f}m", flush=True)
                if args.max_updates and global_step >= args.max_updates:
                    stop = True
                    break
        history.append({"epoch": epoch, "global_step": global_step, "learning_rate": scheduler.get_last_lr()[0], **{f"train_{key}": sums[key] / max(seen, 1) for key in ("loss", "human", "human_distribution", "human_ordinal", "kd", "ordinal_kd")}})
        if stop:
            break

    predictions, metrics = evaluate(model, heldout_rows, tokenizer, args, device)
    destination = run_dir(args.variant, args.fold, args.run_root)
    write_jsonl(destination / "heldout_predictions.jsonl", predictions)
    if teacher_mode:
        export_teacher_logits(model, all_rows if not args.smoke else train_rows + heldout_rows, tokenizer, args, device, {row["sample_id"] for row in heldout_rows})
        save_teacher_adapter(model, destination / "teacher_adapter_and_head.pt")
    summary = {
        "status": "COMPLETED",
        "variant": args.variant,
        "seed": 42,
        "fold": args.fold,
        "train_rows": len(train_rows),
        "eval_rows": len(heldout_rows),
        "fixed_final_epoch": history[-1]["epoch"],
        "global_step": global_step,
        "question_key_overlap": len({row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows}),
        "model_mode": "lora_teacher" if teacher_mode else "full_finetune_student",
        "parameter_counts": parameter_counts,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
        "teacher_logit_train_coverage": len(teacher_logits) if teacher_logits else 0,
        "shuffled_teacher_logit_donor_change_rate": donor_change_rate,
        "history": history,
        "metrics": metrics,
        "run_fingerprint": identity["run_fingerprint"],
        "run_identity": identity,
        "nan_count": 0,
        "oom_count": 0,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(destination / "run_summary.json", summary)
    print(json.dumps({"status": "COMPLETED", "variant": args.variant, "fold": args.fold, "metrics": metrics, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
