"""Audit initialized Exp43 auxiliary-loss scales before formal training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from thesis_exp.exp43_rubimor.common import ROOT, read_jsonl, stable_hash, write_csv, write_json
from thesis_exp.exp43_rubimor.losses_rubimor import distribution_ce, ordinal_cdf_mse, pair_rank_loss
from thesis_exp.exp43_rubimor.modeling_rubimor import RubiMORConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def stratified(rows: list[dict], count: int) -> list[dict]:
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(int(row["gold_label_5"]), int(row["metric_id"]))].append(row)
    ordered = []
    for key, values in sorted(buckets.items()):
        ordered.extend(sorted(values, key=lambda row: stable_hash((row["sample_id"], "loss_scale"))))
    selected, positions = [], {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < min(count, len(rows)):
        progressed = False
        for key in keys:
            if positions[key] < len(buckets[key]) and len(selected) < count:
                selected.append(sorted(buckets[key], key=lambda row: stable_hash((row["sample_id"], "loss_scale")))[positions[key]])
                positions[key] += 1; progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        write_json(args.out_dir / "decision/exp43_loss_scale_decision.json", {"status": "NEEDS_GPU", "test_access_count": 0})
        print(json.dumps({"status": "NEEDS_GPU"})); return
    rows = stratified(read_jsonl(args.out_dir / "private/data/exp43_train_E6.jsonl"), 256)
    pairs = sorted(read_jsonl(args.out_dir / "private/pairs/exp43_train_pairs.jsonl"), key=lambda row: stable_hash((row["pair_id"], "loss_scale")))[:128]
    by_id = {row["sample_id"]: row for row in read_jsonl(args.out_dir / "private/data/exp43_train_E6.jsonl")}
    mapping = json.loads((args.out_dir / "configs/exp43_metric_mapping.json").read_text(encoding="utf-8"))["metrics"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = build_model(RubiMORConfig(args.model_name_or_path, len(mapping), use_metric_residual=True)).cuda().eval()

    def logits_for(batch: list[dict]) -> torch.Tensor:
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=2048, return_tensors="pt")
        metric_ids = torch.tensor([row["metric_id"] for row in batch], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**{key: value.cuda() for key, value in encoded.items()}, metric_ids=metric_ids, residual_enabled=torch.ones_like(metric_ids, dtype=torch.bool))["logits"].float().cpu()

    dist_values, ord_values, dist_grads, ord_grads = [], [], [], []
    for start in range(0, len(rows), 4):
        batch = rows[start:start+4]
        logits = logits_for(batch).requires_grad_(True)
        targets = torch.tensor([row["soft_target_5"] for row in batch])
        dist, ordinal = distribution_ce(logits, targets), ordinal_cdf_mse(logits, targets)
        dist_values.append(float(dist)); ord_values.append(float(ordinal))
        dist_grads.append(float(torch.autograd.grad(dist, logits, retain_graph=True)[0].norm()))
        ord_grads.append(float((0.5 * torch.autograd.grad(ordinal, logits)[0]).norm()))
    rank_values, rank_grads = [], []
    for pair in pairs:
        logits = logits_for([by_id[pair["positive_id"]], by_id[pair["negative_id"]]]).requires_grad_(True)
        rank = pair_rank_loss(logits[:1], logits[1:])
        rank_values.append(float(rank)); rank_grads.append(float((0.25 * torch.autograd.grad(rank, logits)[0]).norm()))
    dist = float(np.mean(dist_values)); ordinal = float(np.mean(ord_values)); rank = float(np.mean(rank_values))
    weighted_ord, weighted_rank = 0.5 * ordinal, 0.25 * rank
    dist_grad, ord_grad, rank_grad = map(float, (np.mean(dist_grads), np.mean(ord_grads), np.mean(rank_grads)))
    checks = {
        "ordinal_ratio_in_range": 1e-3 <= weighted_ord / dist <= 1.0,
        "rank_ratio_in_range": 1e-3 <= weighted_rank / dist <= 1.0,
        "ordinal_gradient_guard": ord_grad <= 2 * dist_grad,
        "rank_gradient_guard": rank_grad <= 2 * dist_grad,
        "all_finite": all(np.isfinite(value) for value in (dist, ordinal, rank, dist_grad, ord_grad, rank_grad)),
    }
    status = "GO" if all(checks.values()) else "LOSS_SCALE_NO_GO"
    table = [
        {"component": "distribution", "raw_loss": dist, "coefficient": 1.0, "weighted_loss": dist, "weighted_to_distribution": 1.0, "logits_gradient_norm": dist_grad},
        {"component": "ordinal", "raw_loss": ordinal, "coefficient": 0.5, "weighted_loss": weighted_ord, "weighted_to_distribution": weighted_ord / dist, "logits_gradient_norm": ord_grad},
        {"component": "rank", "raw_loss": rank, "coefficient": 0.25, "weighted_loss": weighted_rank, "weighted_to_distribution": weighted_rank / dist, "logits_gradient_norm": rank_grad},
    ]
    write_csv(args.out_dir / "tables/exp43_loss_scale_audit.csv", table)
    write_json(args.out_dir / "configs/exp43_loss_protocol_lock.json", {"distribution": "human distribution CE", "ordinal": "CDF MSE", "ordinal_coefficient": 0.5, "rank": "RankNet softplus expected-score difference", "rank_coefficient": 0.25, "automatic_coefficient_adjustment": False})
    write_json(args.out_dir / "decision/exp43_loss_scale_decision.json", {"status": status, "checks": checks, "point_rows": len(rows), "pairs": len(pairs), "test_access_count": 0})
    print(json.dumps({"status": status, "checks": checks, "point_rows": len(rows), "pairs": len(pairs)}, sort_keys=True))


if __name__ == "__main__":
    main()

