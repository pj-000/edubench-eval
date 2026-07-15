"""Extract fold-specific outer-train and heldout frozen E4 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp44_taco_score.modeling_exp44a_taco import TACOModelConfig, build_model
from thesis_exp.exp45_dopr_head.common import ROOT, embedding_path, encoder_dir, fold_map, read_jsonl, sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def extract(model: Any, tokenizer: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            encoded = tokenizer([row["text"] for row in batch_rows], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**{key: value.cuda() for key, value in encoded.items()})
            logits = output["logits"].float()
            probs = torch.softmax(logits, -1).cpu().numpy()
            embeddings = torch.nn.functional.normalize(output["pooled"].float(), dim=-1).cpu().numpy()
            for row, probability, embedding in zip(batch_rows, probs, embeddings):
                result.append(
                    {
                        "sample_id": row["sample_id"], "question_key": row["question_key"],
                        "fold": args.fold, "gold_label_5": row["gold_label_5"],
                        "human_distribution_5": row["human_distribution_5"],
                        "expected_human_score": row["expected_human_score"],
                        "metric": row["metric"], "language": row["language"], "subject": row["subject"],
                        "pred_label_5": int(np.argmax(probability)) + 1,
                        "pred_score_expected": float(sum(label * probability[label - 1] for label in range(1, 6))),
                        **{f"prob_{label}": float(probability[label - 1]) for label in range(1, 6)},
                        "embedding": embedding.tolist(),
                    }
                )
            print(f"[exp45a-embed] fold={args.fold} rows={min(start+args.batch_size,len(rows))}/{len(rows)}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Embedding extraction requires CUDA")
    rows = read_jsonl(args.out_dir / "private/data/exp45a_train_e4.jsonl")
    folds = fold_map(args.out_dir / "private/data/exp45a_groupcv_fold_assignment.csv")
    train_rows = [row for row in rows if folds[row["sample_id"]] != args.fold]
    heldout_rows = [row for row in rows if folds[row["sample_id"]] == args.fold]
    if {row["question_key"] for row in train_rows} & {row["question_key"] for row in heldout_rows}:
        raise RuntimeError("Exp45A embedding question-key leakage")
    outputs = {"outer_train": embedding_path(args.out_dir, args.fold, "outer_train"), "outer_heldout": embedding_path(args.out_dir, args.fold, "outer_heldout")}
    checkpoint = encoder_dir(args.out_dir, args.fold) / "final_encoder_head.pt"
    summary = encoder_dir(args.out_dir, args.fold) / "encoder_summary.json"
    if not checkpoint.exists() or not summary.exists():
        raise FileNotFoundError(f"Missing Exp45A encoder: {checkpoint}")
    metadata = json.loads(summary.read_text(encoding="utf-8"))
    manifest_path = args.out_dir / f"private/embeddings/fold_{args.fold}_manifest.json"
    if args.skip_completed and all(path.exists() for path in outputs.values()) and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            manifest.get("encoder_checkpoint_sha256") == metadata.get("checkpoint_sha256")
            and manifest.get("outer_train_rows") == len(train_rows)
            and manifest.get("outer_heldout_rows") == len(heldout_rows)
            and all(manifest.get(f"{split}_sha256") == sha256_file(path) for split, path in outputs.items())
        )
        if valid:
            print(json.dumps({"status": "EMBEDDINGS_REUSED", "fold": args.fold}, sort_keys=True)); return
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    if state["run_fingerprint"] != metadata["run_fingerprint"] or sha256_file(checkpoint) != metadata["checkpoint_sha256"]:
        raise RuntimeError("Exp45A encoder fingerprint mismatch")
    model = build_model(TACOModelConfig(args.model_name_or_path, use_projection=False)).cuda()
    model.load_state_dict(state["model"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for split, split_rows in (("outer_train", train_rows), ("outer_heldout", heldout_rows)):
        write_jsonl(outputs[split], extract(model, tokenizer, split_rows, args))
    manifest_path.write_text(
        json.dumps(
            {
                "fold": args.fold,
                "encoder_checkpoint_sha256": metadata["checkpoint_sha256"],
                "outer_train_rows": len(train_rows),
                "outer_heldout_rows": len(heldout_rows),
                **{f"{split}_sha256": sha256_file(path) for split, path in outputs.items()},
                "question_key_overlap": 0,
                "dev_access_count": 0,
                "test_access_count": 0,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "EMBEDDINGS_COMPLETE", "fold": args.fold, "outer_train": len(train_rows), "outer_heldout": len(heldout_rows), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
