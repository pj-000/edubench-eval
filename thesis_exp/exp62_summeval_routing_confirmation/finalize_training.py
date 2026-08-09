"""Mechanical integrity gate after all twenty Exp62 train/dev runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation import OUTPUT_ROOT, REPO_ROOT, SEEDS, VARIANTS
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import sha256_file
from thesis_exp.exp62_summeval_routing_confirmation.train import PROTOCOL_PATH


ARTIFACT_ROOT = REPO_ROOT / "thesis_exp/artifacts/exp62_summeval_routing_confirmation"


def checkpoint_paths(variant: str, seed: int) -> tuple[Path, Path, Path]:
    run = OUTPUT_ROOT / "runs" / variant / f"seed_{seed}" / "dev_summary.json"
    checkpoint = ARTIFACT_ROOT / variant / f"seed_{seed}" / "epoch10" / "checkpoint.json"
    state = ARTIFACT_ROOT / variant / f"seed_{seed}" / "epoch10" / "state_dict.pt"
    return run, checkpoint, state


def run() -> dict[str, Any]:
    protocol_hash = sha256_file(PROTOCOL_PATH)
    rows: list[dict[str, Any]] = []
    batch_orders: defaultdict[int, set[str]] = defaultdict(set)
    failures: list[str] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_path, checkpoint_path, state_path = checkpoint_paths(variant, seed)
            if not all(path.is_file() for path in (run_path, checkpoint_path, state_path)):
                failures.append(f"missing:{variant}:seed{seed}")
                continue
            run_report = json.loads(run_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checks = {
                "reports_equal": run_report == checkpoint,
                "status": checkpoint.get("status") == "EXP62_FIXED_EPOCH10_DEV_COMPLETE",
                "variant": checkpoint.get("variant") == variant,
                "seed": checkpoint.get("seed") == seed,
                "epoch": checkpoint.get("epoch") == 10,
                "protocol": checkpoint.get("protocol_sha256") == protocol_hash,
                "state": checkpoint.get("state_dict_sha256") == sha256_file(state_path),
                "test_access_zero": checkpoint.get("test_access_count") == 0,
            }
            if not all(checks.values()):
                failures.append(f"invalid:{variant}:seed{seed}:{checks}")
            batch_orders[seed].add(str(checkpoint.get("train_batch_order_sha256")))
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "primary_macro_mae": checkpoint["metrics"]["primary_macro_mae"],
                    "metrics": checkpoint["metrics"],
                    "state_dict_sha256": checkpoint["state_dict_sha256"],
                    "train_batch_order_sha256": checkpoint["train_batch_order_sha256"],
                }
            )
    for seed, values in batch_orders.items():
        if len(values) != 1:
            failures.append(f"batch_order_differs_across_arms:seed{seed}")
    checks = {
        "twenty_checkpoints": len(rows) == len(VARIANTS) * len(SEEDS),
        "no_checkpoint_failure": not failures,
        "paired_batch_order_within_seed": len(batch_orders) == len(SEEDS)
        and all(len(values) == 1 for values in batch_orders.values()),
        "test_access_count_zero": all(
            json.loads(checkpoint_paths(variant, seed)[1].read_text())["test_access_count"] == 0
            for variant in VARIANTS
            for seed in SEEDS
            if checkpoint_paths(variant, seed)[1].is_file()
        ),
    }
    result = {
        "status": (
            "EXP62_FORMAL_TRAINING_INTEGRITY_PASS_READY_TO_AUTHORIZE_TEST"
            if all(checks.values())
            else "EXP62_FORMAL_TRAINING_INTEGRITY_FAIL"
        ),
        "checks": checks,
        "failures": failures,
        "runs": rows,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "decision/formal_training_integrity.json",
    )
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}, indent=2))


if __name__ == "__main__":
    main()

