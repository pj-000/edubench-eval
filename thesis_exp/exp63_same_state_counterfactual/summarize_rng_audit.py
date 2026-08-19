"""Aggregate the 15 deterministic checkpoint-RNG audit units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp63_same_state_counterfactual import OUTPUT_ROOT, SEEDS, STAGE_EPOCHS
from thesis_exp.exp63_same_state_counterfactual.runtime import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=OUTPUT_ROOT / "rng_audit_deterministic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "decision" / "checkpoint_rng_audit_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    units: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in SEEDS:
        for stage in STAGE_EPOCHS:
            path = args.input_dir / f"seed_{seed}" / f"after_epoch_{stage}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            units.append(
                {
                    "seed": seed,
                    "stage_epoch": stage,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "status": value.get("status"),
                    "exact_checks": value.get("exact_checks"),
                    "diagnostic_differences": value.get("diagnostic_differences"),
                    "determinism_contract": value.get("determinism_contract"),
                    "optimizer_steps": value.get("optimizer_steps"),
                    "test_access_count": value.get("test_access_count"),
                }
            )
    all_exact = (
        not missing
        and len(units) == len(SEEDS) * len(STAGE_EPOCHS)
        and all(unit["status"] == "RNG_INVARIANT_EXACT" for unit in units)
        and all(unit["optimizer_steps"] == 0 for unit in units)
        and all(unit["test_access_count"] == 0 for unit in units)
    )
    result = {
        "status": (
            "EXP63_RNG_AUDIT_EXACT_PASS_RETAIN_ORIGINAL_RESULTS"
            if all_exact
            else "EXP63_RNG_AUDIT_NOT_CLOSED"
        ),
        "decision": (
            "The missing checkpoint-RNG restore is mechanically outcome-invariant "
            "under deterministic CUDA execution for all 15 frozen states; no Exp63 "
            "optimizer update rerun is required."
            if all_exact
            else "Do not retain the original Exp63 formal result without resolving "
            "the missing or changed units."
        ),
        "required_units": len(SEEDS) * len(STAGE_EPOCHS),
        "completed_units": len(units),
        "missing": missing,
        "units": units,
        "scope": (
            "This audit closes only the checkpoint-RNG protocol deviation.  It does "
            "not repair or reinterpret Exp63's arm-specific norm matching."
        ),
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
