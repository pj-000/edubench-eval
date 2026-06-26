"""Train Exp16A rubric-conditioned ordinal boundary linking."""

from __future__ import annotations

import argparse
import copy
import json
import random
from contextlib import nullcontext
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from thesis_exp.src.edujudge.exp16_boundary_linking import BOUNDARY_VARIANTS, ensure_exp16_dirs
from thesis_exp.src.edujudge.exp16_boundary_linking.data import (
    BoundaryLinkingCollator,
    BoundaryLinkingDataset,
    SimpleBoundaryTokenizer,
    default_data_paths,
    load_samples,
    parse_boundary_fields,
)
from thesis_exp.src.edujudge.exp16_boundary_linking.losses import ordinal_bce_loss, parse_class_weights
from thesis_exp.src.edujudge.exp16_boundary_linking.metrics import (
    compute_metrics,
    threshold_stats,
    threshold_stats_by_metric,
)
from thesis_exp.src.edujudge.exp16_boundary_linking.model import BoundaryLinkingOrdinalModel
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_json, write_jsonl


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tokenizer(model_name_or_path: str, local_files_only: bool = False, trust_remote_code: bool = False) -> Any:
    if model_name_or_path == "__tiny_random__":
        return SimpleBoundaryTokenizer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in [
        "quality_input_ids",
        "quality_attention_mask",
        "boundary_input_ids",
        "boundary_attention_mask",
        "labels",
        "targets",
    ]:
        out[key] = out[key].to(device)
    return out


def autocast_context(args: Namespace, device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    if bool(args.fp16):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if bool(args.bf16):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def prediction_rows(outputs: dict[str, torch.Tensor], labels: torch.Tensor, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probs = outputs["probs"].detach().cpu().numpy()
    pred = outputs["pred_label"].detach().cpu().numpy()
    s = outputs["quality_score_s"].detach().cpu().numpy()
    tau = outputs["thresholds_tau"].detach().cpu().numpy()
    alpha = outputs["scale_alpha"].detach().cpu().numpy()
    gold = labels.detach().cpu().numpy()
    rows = []
    for idx, sample in enumerate(samples):
        row = {
            "sample_id": sample["sample_id"],
            "question_key": sample["question_key"],
            "metric": sample["metric"],
            "gold_label": int(gold[idx]),
            "pred_label": int(pred[idx]),
            "probs": [float(value) for value in probs[idx].tolist()],
            "quality_score_s": float(s[idx]),
            "tau1": float(tau[idx, 0]),
            "tau2": float(tau[idx, 1]),
            "tau3": float(tau[idx, 2]),
            "tau4": float(tau[idx, 3]),
            "scale_alpha": float(alpha[idx]),
            "margin_tau2": float(s[idx] - tau[idx, 1]),
            "margin_tau3": float(s[idx] - tau[idx, 2]),
        }
        row["is_low_to_high"] = bool(row["gold_label"] <= 2 and row["pred_label"] >= 4)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(
    model: BoundaryLinkingOrdinalModel,
    loader: DataLoader,
    device: torch.device,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(Namespace(fp16=False, bf16=False), device):
            outputs = model(
                quality_input_ids=batch["quality_input_ids"],
                quality_attention_mask=batch["quality_attention_mask"],
                boundary_input_ids=batch["boundary_input_ids"],
                boundary_attention_mask=batch["boundary_attention_mask"],
            )
        predictions.extend(prediction_rows(outputs, batch["labels"], batch["samples"]))
    metrics = compute_metrics(predictions, split=split)
    stats = threshold_stats(predictions, split=split)
    by_metric = threshold_stats_by_metric(predictions, split=split)
    return metrics, predictions, stats, by_metric


def score_is_better(metrics: dict[str, Any], best: float | None, save_best_by: str) -> tuple[bool, float]:
    if save_best_by == "dev_qwk":
        score = float(metrics["QWK"])
        return best is None or score > best, score
    if save_best_by == "dev_mae":
        score = float(metrics["MAE"])
        return best is None or score < best, score
    raise ValueError(f"Unsupported save_best_by: {save_best_by}")


def save_checkpoint(output_dir: Path, model: BoundaryLinkingOrdinalModel, metrics: dict[str, Any]) -> None:
    ckpt = output_dir / "checkpoint_best"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "state_dict.pt")
    write_json(ckpt / "best_metrics.json", metrics)


def build_arg_parser() -> argparse.ArgumentParser:
    paths = default_data_paths()
    parser = argparse.ArgumentParser(description="Train Exp16A boundary linking ordinal scorer.")
    parser.add_argument("--model_name_or_path", default="__tiny_random__")
    parser.add_argument("--train_path", type=Path, default=paths["train"])
    parser.add_argument("--dev_path", type=Path, default=paths["dev"])
    parser.add_argument("--test_path", type=Path, default=paths["test"])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=BOUNDARY_VARIANTS, default="qmr_meta")
    parser.add_argument("--boundary_fields", default="")
    parser.add_argument("--max_length_quality", type=int, default=2048)
    parser.add_argument("--max_length_boundary", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_weights", default="")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--freeze_encoder", type=str_to_bool, default=False)
    parser.add_argument("--eval_every_epoch", action="store_true")
    parser.add_argument("--save_best_by", choices=["dev_mae", "dev_qwk"], default="dev_mae")
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def run_training(args: Namespace) -> dict[str, Any]:
    ensure_exp16_dirs()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boundary_fields = args.boundary_fields or None
    parse_boundary_fields(args.variant, boundary_fields)

    tokenizer = make_tokenizer(
        str(args.model_name_or_path),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = BoundaryLinkingOrdinalModel.from_model_name(
        str(args.model_name_or_path),
        variant=args.variant,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    if bool(args.freeze_encoder):
        for param in model.encoder.parameters():
            param.requires_grad = False

    train_limit = int(args.max_train_samples) or None
    eval_limit = int(args.max_eval_samples) or None
    train_samples = load_samples(args.train_path, args.variant, boundary_fields, limit=train_limit)
    dev_samples = load_samples(args.dev_path, args.variant, boundary_fields, limit=eval_limit)
    test_samples = load_samples(args.test_path, args.variant, boundary_fields, limit=eval_limit)
    collator = BoundaryLinkingCollator(tokenizer, args.max_length_quality, args.max_length_boundary)
    train_loader = DataLoader(
        BoundaryLinkingDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(BoundaryLinkingDataset(dev_samples), batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)
    test_loader = DataLoader(BoundaryLinkingDataset(test_samples), batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    class_weights = parse_class_weights(args.class_weights)

    best_state: dict[str, torch.Tensor] | None = None
    best_score: float | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_step = 0
    max_steps = int(args.max_train_steps) if int(args.max_train_steps) > 0 else None
    epochs = int(np.ceil(float(args.epochs)))
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(args, device):
                outputs = model(
                    quality_input_ids=batch["quality_input_ids"],
                    quality_attention_mask=batch["quality_attention_mask"],
                    boundary_input_ids=batch["boundary_input_ids"],
                    boundary_attention_mask=batch["boundary_attention_mask"],
                )
                loss = ordinal_bce_loss(outputs["logits"], batch["targets"], batch["labels"], class_weights=class_weights)
            (loss / int(args.grad_accum_steps)).backward()
            running_loss += float(loss.detach().cpu())
            batch_count += 1
            if batch_idx % int(args.grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break
        dev_metrics, _, _, _ = evaluate(model, dev_loader, device, split="dev")
        dev_metrics = {**dev_metrics, "epoch": epoch, "global_step": global_step, "train_loss": running_loss / max(1, batch_count)}
        history.append(dev_metrics)
        better, score = score_is_better(dev_metrics, best_score, args.save_best_by)
        if better:
            best_score = score
            best_metrics = dev_metrics
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(args.output_dir, model, dev_metrics)
        if max_steps is not None and global_step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_dev_metrics, dev_predictions, dev_stats, dev_by_metric = evaluate(model, dev_loader, device, split="dev")
    final_test_metrics, test_predictions, test_stats, test_by_metric = evaluate(model, test_loader, device, split="test")
    final_dev_metrics = {**final_dev_metrics, "selected_by": args.save_best_by, "selected_score": best_score}
    final_test_metrics = {**final_test_metrics, "selected_by": args.save_best_by, "selected_score": best_score}

    write_json(args.output_dir / "metrics_dev.json", final_dev_metrics)
    write_json(args.output_dir / "metrics_test.json", final_test_metrics)
    write_json(args.output_dir / "threshold_stats_dev.json", dev_stats)
    write_json(args.output_dir / "threshold_stats_test.json", test_stats)
    write_jsonl(args.output_dir / "predictions_dev.jsonl", dev_predictions)
    write_jsonl(args.output_dir / "predictions_test.jsonl", test_predictions)
    write_csv(args.output_dir / "threshold_by_metric_dev.csv", dev_by_metric)
    write_csv(args.output_dir / "threshold_by_metric_test.csv", test_by_metric)
    write_json(args.output_dir / "training_history.json", history)
    write_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "output_dir": relpath(args.output_dir),
            "train_path": relpath(args.train_path),
            "dev_path": relpath(args.dev_path),
            "test_path": relpath(args.test_path),
            "boundary_fields_effective": parse_boundary_fields(args.variant, boundary_fields),
            "class_weights_parsed": class_weights,
            "best_metrics": best_metrics,
        },
    )
    return {
        "status": "COMPLETED",
        "output_dir": relpath(args.output_dir),
        "global_step": global_step,
        "dev_MAE": final_dev_metrics.get("MAE"),
        "dev_low_to_high": final_dev_metrics.get("low_to_high_rate"),
        "test_MAE": final_test_metrics.get("MAE"),
        "test_low_to_high": final_test_metrics.get("low_to_high_rate"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
