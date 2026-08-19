"""CPU audit for reusing Exp57 Consensus and Full-Routed endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp59_residual_geometry import OUTPUT_ROOT, REPO_ROOT


RUN_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp57_cbrd" / "runs"
VARIANTS = ("consensus_only", "routed_hmsa")
SEEDS = (42, 43, 44, 45, 46)


def run() -> dict[str, Any]:
    summaries: dict[int, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for seed in SEEDS:
        summaries[seed] = {}
        for variant in VARIANTS:
            path = RUN_ROOT / variant / f"seed_{seed}" / "run_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summaries[seed][variant] = json.loads(path.read_text(encoding="utf-8"))
        common = summaries[seed]["consensus_only"]
        routed = summaries[seed]["routed_hmsa"]
        prefix = f"seed_{seed}"
        checks[f"{prefix}_completed"] = all(
            row.get("status") == "COMPLETED" for row in (common, routed)
        )
        checks[f"{prefix}_zero_test_access"] = all(
            row.get("test_access_count") == 0 for row in (common, routed)
        )
        for field in (
            "train_text_hash",
            "dev_text_hash",
            "model_mode",
            "model_name_or_path",
            "checkpoint_rule",
            "inference",
            "route_loss_scale",
            "runtime_head",
        ):
            checks[f"{prefix}_{field}_equal"] = common.get(field) == routed.get(field)
        checks[f"{prefix}_initial_heads_equal"] = (
            common.get("initial_head_contract") == routed.get("initial_head_contract")
        )
        checks[f"{prefix}_optimizer_steps_equal"] = (
            common.get("clip_events") == routed.get("clip_events") == 210
        )
    train_hashes = {
        row[variant]["train_text_hash"]
        for row in summaries.values()
        for variant in VARIANTS
    }
    dev_hashes = {
        row[variant]["dev_text_hash"]
        for row in summaries.values()
        for variant in VARIANTS
    }
    checks["all_train_hashes_equal"] = len(train_hashes) == 1
    checks["all_dev_hashes_equal"] = len(dev_hashes) == 1
    report = {
        "status": "EXP59_ENDPOINT_PARITY_PASS" if all(checks.values()) else "EXP59_ENDPOINT_PARITY_FAIL",
        "checks": checks,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "train_text_hash": next(iter(train_hashes)) if len(train_hashes) == 1 else None,
        "dev_text_hash": next(iter(dev_hashes)) if len(dev_hashes) == 1 else None,
        "fixed_training_contract": {
            "protocol": "thesis_exp/configs/exp57_cbrd/stage1_protocol.json",
            "epochs": 10,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 32,
            "max_grad_norm": 1.0,
            "precision": "bf16",
            "checkpoint_rule": "highest hard-head dev Exact_rounded; ties keep earlier epoch",
        },
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "endpoint_parity.json", report)
    return report


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
