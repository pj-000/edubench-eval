"""Export Exp16A train/dev/test predictions from an existing best checkpoint."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.io import relpath, write_jsonl


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def load_config(run_dir: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or run_dir / "config.json"
    if not path.exists():
        raise SystemExit(f"Missing config: {relpath(path)}")
    return json.loads(path.read_text())


def load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_tokenized(tokenized: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in tokenized.items()}


def export_predictions_with_boundary_cache(
    *,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    split: str,
    device: Any,
    batch_size: int,
    max_length_quality: int,
    max_length_boundary: int,
    autocast_args: Namespace,
) -> list[dict[str, Any]]:
    import torch

    from thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking import autocast_context

    model.eval()
    boundary_by_key: dict[str, str] = {}
    for sample in samples:
        key = str(sample["boundary_key"])
        text = str(sample["boundary_text"])
        if key in boundary_by_key and boundary_by_key[key] != text:
            raise SystemExit(f"Boundary key collision or inconsistent boundary text for key={key}")
        answer = str(sample.get("answer") or "").strip()
        if answer and answer in text:
            raise SystemExit(f"Boundary cache text leaked answer for sample_id={sample.get('sample_id')}")
        boundary_by_key[key] = text

    cached: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    keys = sorted(boundary_by_key)
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            tokenized = tokenizer(
                [boundary_by_key[key] for key in batch_keys],
                padding=True,
                truncation=True,
                max_length=max_length_boundary,
                return_tensors="pt",
            )
            tokenized = move_tokenized(tokenized, device)
            with autocast_context(autocast_args, device):
                boundary_h = model.encode(tokenized["input_ids"], tokenized["attention_mask"])
                thresholds_tau, scale_alpha = model.ordered_thresholds(boundary_h)
            thresholds_tau = thresholds_tau.detach().float().cpu()
            scale_alpha = scale_alpha.detach().float().cpu()
            for idx, key in enumerate(batch_keys):
                cached[key] = (thresholds_tau[idx], scale_alpha[idx])

        predictions: list[dict[str, Any]] = []
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            quality = tokenizer(
                [sample["quality_text"] for sample in batch_samples],
                padding=True,
                truncation=True,
                max_length=max_length_quality,
                return_tensors="pt",
            )
            quality = move_tokenized(quality, device)
            with autocast_context(autocast_args, device):
                quality_h = model.encode(quality["input_ids"], quality["attention_mask"])
                quality_score_s = model.quality_head(quality_h).squeeze(-1)
            quality_score_s = quality_score_s.detach().float().cpu()
            tau = torch.stack([cached[str(sample["boundary_key"])][0] for sample in batch_samples], dim=0)
            alpha = torch.stack([cached[str(sample["boundary_key"])][1] for sample in batch_samples], dim=0)
            logits = alpha.unsqueeze(-1) * (quality_score_s.unsqueeze(-1) - tau)
            probs = torch.sigmoid(logits)
            pred_label = 1 + (probs > 0.5).sum(dim=-1)
            for idx, sample in enumerate(batch_samples):
                row = {
                    "sample_id": sample["sample_id"],
                    "question_key": sample["question_key"],
                    "boundary_key": sample["boundary_key"],
                    "metric": sample["metric"],
                    "gold_label": int(sample["label"]),
                    "pred_label": int(pred_label[idx].item()),
                    "probs": [float(value) for value in probs[idx].tolist()],
                    "quality_score_s": float(quality_score_s[idx].item()),
                    "tau1": float(tau[idx, 0].item()),
                    "tau2": float(tau[idx, 1].item()),
                    "tau3": float(tau[idx, 2].item()),
                    "tau4": float(tau[idx, 3].item()),
                    "scale_alpha": float(alpha[idx].item()),
                    "margin_tau2": float(quality_score_s[idx].item() - tau[idx, 1].item()),
                    "margin_tau3": float(quality_score_s[idx].item() - tau[idx, 2].item()),
                }
                row["is_low_to_high"] = bool(row["gold_label"] <= 2 and row["pred_label"] >= 4)
                predictions.append(row)
    del split
    return predictions


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from thesis_exp.src.edujudge.exp16_boundary_linking.data import BoundaryLinkingCollator, BoundaryLinkingDataset, load_samples
    from thesis_exp.src.edujudge.exp16_boundary_linking.model import BoundaryLinkingOrdinalModel
    from thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking import evaluate, make_tokenizer

    config = load_config(args.run_dir, args.config_path)
    checkpoint_path = args.checkpoint_path or args.run_dir / "checkpoint_best" / "state_dict.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {relpath(checkpoint_path)}")
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(args.model_name_or_path or config["model_name_or_path"])
    variant = str(args.variant or config.get("variant", "qmr"))
    boundary_fields = str(args.boundary_fields if args.boundary_fields is not None else config.get("boundary_fields", ""))
    tokenizer = make_tokenizer(
        model_name,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code or config.get("trust_remote_code", False)),
    )
    model = BoundaryLinkingOrdinalModel.from_model_name(
        model_name,
        variant=variant,
        trust_remote_code=bool(args.trust_remote_code or config.get("trust_remote_code", False)),
        local_files_only=bool(args.local_files_only),
    )
    model.load_state_dict(load_state_dict(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.eval()

    collator = BoundaryLinkingCollator(
        tokenizer,
        max_length_quality=int(args.max_length_quality or config.get("max_length_quality", 2048)),
        max_length_boundary=int(args.max_length_boundary or config.get("max_length_boundary", 768)),
    )
    eval_args = Namespace(fp16=False, bf16=False, precision=str(args.precision or config.get("precision", "fp32")))
    split_paths = {
        "train": args.train_path or Path(config["train_path"]),
        "dev": args.dev_path or Path(config["dev_path"]),
        "test": args.test_path or Path(config["test_path"]),
    }
    exported: dict[str, str] = {}
    for split in args.splits:
        path = Path(split_paths[split])
        if not path.exists():
            warn(f"missing {split} path: {relpath(path)}; skipping")
            continue
        samples = load_samples(path, variant=variant, boundary_fields=boundary_fields or None, limit=None)
        batch_size = int(args.batch_size or config.get("eval_batch_size", config.get("batch_size", 8)))
        if bool(args.use_boundary_cache):
            if variant not in {"qmr", "metric_rubric"}:
                raise SystemExit("--use_boundary_cache currently supports variant=qmr or metric_rubric only")
            predictions = export_predictions_with_boundary_cache(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                split=split,
                device=device,
                batch_size=batch_size,
                max_length_quality=int(args.max_length_quality or config.get("max_length_quality", 2048)),
                max_length_boundary=int(args.max_length_boundary or config.get("max_length_boundary", 768)),
                autocast_args=eval_args,
            )
        else:
            loader = DataLoader(
                BoundaryLinkingDataset(samples),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            _, predictions, _, _ = evaluate(model, loader, device, split=split, autocast_args=eval_args)
        output_path = output_dir / f"predictions_{split}.jsonl"
        write_jsonl(output_path, predictions)
        exported[split] = relpath(output_path)
    return {
        "variant": variant,
        "run_dir": relpath(args.run_dir),
        "use_boundary_cache": bool(args.use_boundary_cache),
        "exported": exported,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Exp16A predictions from an existing checkpoint without training.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--variant")
    parser.add_argument("--boundary_fields")
    parser.add_argument("--train_path", type=Path)
    parser.add_argument("--dev_path", type=Path)
    parser.add_argument("--test_path", type=Path)
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_length_quality", type=int, default=0)
    parser.add_argument("--max_length_boundary", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--use_boundary_cache", type=str_to_bool, default=False)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = export_predictions(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
