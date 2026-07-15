"""Audit locked Exp44A loss scales and gradient norms before formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp44_taco_score.common import ROOT, atomic_json, fold_map, read_jsonl, stable_hash, triplet_path, write_csv
from thesis_exp.exp44_taco_score.losses_exp44a_taco import base_loss, taco_loss
from thesis_exp.exp44_taco_score.modeling_exp44a_taco import TACOModelConfig, build_model
from thesis_exp.exp44_taco_score.train_exp44a_groupcv import encode_triplet, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def gradient_norm(parameters: Any) -> float:
    values = [parameter.grad.detach().float().norm() ** 2 for parameter in parameters if parameter.grad is not None]
    if not values:
        return 0.0
    return float(sum(values).sqrt().cpu())


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Exp44A loss audit requires CUDA")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    rows = read_jsonl(args.out_dir / "private/data/exp44a_train_e4.jsonl")
    folds = fold_map(args.out_dir / "private/data/exp44a_groupcv_fold_assignment.csv")
    train_rows = [row for row in rows if folds[row["sample_id"]] != args.fold]
    selected: list[dict[str, Any]] = []
    by_label = {
        label: sorted(
            (row for row in train_rows if int(row["gold_label_5"]) == label),
            key=lambda row: stable_hash(("loss-audit", label, row["sample_id"])),
        )
        for label in range(1, 6)
    }
    cursor = {label: 0 for label in range(1, 6)}
    while len(selected) < 256:
        progressed = False
        for label in range(1, 6):
            if cursor[label] < len(by_label[label]) and len(selected) < 256:
                selected.append(by_label[label][cursor[label]])
                cursor[label] += 1
                progressed = True
        if not progressed:
            break
    triplets = read_jsonl(triplet_path(args.out_dir, args.fold, 1))[:128]
    row_by_id = {row["sample_id"]: row for row in train_rows}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(TACOModelConfig(args.model_name_or_path, use_projection=True)).cuda().train()
    device = torch.device("cuda")

    model.zero_grad(set_to_none=True)
    base_values = []
    dist_values = []
    ordinal_values = []
    loader = make_loader(selected, tokenizer, 4, False, 42, args.max_length)
    for batch in loader:
        batch.pop("metadata")
        targets = batch.pop("targets").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**{key: value.to(device) for key, value in batch.items()})["logits"]
            loss, components = base_loss(logits, targets)
        (loss / len(loader)).backward()
        base_values.append(float(loss.detach()))
        dist_values.append(float(components["distribution"].detach()))
        ordinal_values.append(float(components["ordinal"].detach()))
    base_backbone_grad = gradient_norm(model.backbone.parameters())
    logits_grad = gradient_norm(model.global_head.parameters())

    model.zero_grad(set_to_none=True)
    taco_values = []
    plain_values = []
    for triplet in triplets:
        endpoints = [row_by_id[triplet[key]] for key in ("anchor_id", "positive_id", "near_id", "far_id")]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            projections = encode_triplet(model, tokenizer, endpoints, args.max_length, device)
            near = torch.tensor([triplet["near_distance"]], device=device)
            far = torch.tensor([triplet["far_distance"]], device=device)
            taco, _ = taco_loss(projections, near, far, use_margin=True)
            plain, _ = taco_loss(projections, near, far, use_margin=False)
        (0.1 * taco / max(len(triplets), 1)).backward()
        taco_values.append(float(taco.detach()))
        plain_values.append(float(plain.detach()))
    contrastive_backbone_grad = gradient_norm(model.backbone.parameters())
    projection_grad = gradient_norm(model.projection.parameters())
    mean_dist = float(np.mean(dist_values))
    mean_ordinal = float(np.mean(ordinal_values))
    mean_base = float(np.mean(base_values))
    mean_taco = float(np.mean(taco_values))
    mean_plain = float(np.mean(plain_values))
    ratio = 0.1 * mean_taco / mean_base
    grad_ratio = contrastive_backbone_grad / base_backbone_grad if base_backbone_grad else float("inf")
    finite = all(np.isfinite(value) for value in (mean_dist, mean_ordinal, mean_taco, mean_plain, ratio, grad_ratio))
    status = "LOSS_SCALE_GO" if finite and 1e-3 <= ratio <= 1.0 and grad_ratio <= 2.0 else "LOSS_SCALE_NO_GO"
    row = {
        "fold": args.fold,
        "pointwise_rows": len(selected),
        "contrastive_anchors": len(triplets),
        "L_dist": mean_dist,
        "L_ord": mean_ordinal,
        "L_base": mean_base,
        "L_plain": mean_plain,
        "L_taco": mean_taco,
        "weighted_taco": 0.1 * mean_taco,
        "weighted_to_base_ratio": ratio,
        "base_backbone_gradient_norm": base_backbone_grad,
        "contrastive_backbone_gradient_norm": contrastive_backbone_grad,
        "contrastive_to_base_gradient_ratio": grad_ratio,
        "logits_head_gradient_norm": logits_grad,
        "projection_gradient_norm": projection_grad,
        "finite": finite,
        "status": status,
    }
    write_csv(args.out_dir / "tables/exp44a_loss_scale_audit.csv", [row])
    decision = {"status": status, **row, "dev_access_count": 0, "test_access_count": 0}
    atomic_json(args.out_dir / "decision/exp44a_loss_scale_decision.json", decision)
    print(json.dumps(decision, sort_keys=True))
    if status != "LOSS_SCALE_GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
