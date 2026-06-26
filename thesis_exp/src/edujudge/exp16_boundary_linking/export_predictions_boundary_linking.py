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
        loader = DataLoader(
            BoundaryLinkingDataset(samples),
            batch_size=int(args.batch_size or config.get("eval_batch_size", config.get("batch_size", 8))),
            shuffle=False,
            collate_fn=collator,
        )
        _, predictions, _, _ = evaluate(model, loader, device, split=split, autocast_args=eval_args)
        output_path = output_dir / f"predictions_{split}.jsonl"
        write_jsonl(output_path, predictions)
        exported[split] = relpath(output_path)
    return {"variant": variant, "run_dir": relpath(args.run_dir), "exported": exported}


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
