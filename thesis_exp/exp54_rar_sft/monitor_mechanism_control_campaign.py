"""Print one compact progress snapshot for the nine formal control runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
SEEDS = (42, 43, 44)
STEP_TOTALS = {
    "R3_TOKENAVG": 996,
    "P1_FULLSEQ": 27,
    "P1_SYN_LR5E6": 27,
}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def snapshot(root: Path) -> dict[str, Any]:
    runs = []
    completed_steps = 0
    total_steps = sum(STEP_TOTALS.values()) * len(SEEDS)
    elapsed_by_seed: dict[int, float] = {}
    eta_by_seed: dict[int, float] = {}
    for seed in SEEDS:
        for arm in ARMS:
            directory = root / arm.lower() / f"seed_{seed}"
            progress = _read(directory / "progress.json")
            failure = _read(directory / "failure.json")
            expected = STEP_TOTALS[arm]
            step = 0
            status = "PENDING"
            percent = 0.0
            eta = None
            if progress is not None:
                step = min(int(progress.get("optimizer_step", 0)), expected)
                status = str(progress.get("status", "RUNNING"))
                percent = float(progress.get("completion_percent", 0.0))
                eta = progress.get("estimated_remaining_seconds")
                elapsed_by_seed[seed] = elapsed_by_seed.get(seed, 0.0) + float(
                    progress.get("elapsed_seconds", 0.0)
                )
                if eta is not None:
                    eta_by_seed[seed] = float(eta)
            if failure is not None:
                status = "FAILED"
            completed_steps += step
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "status": status,
                    "optimizer_step": step,
                    "optimizer_steps": expected,
                    "run_percent": percent,
                    "eta_seconds": eta,
                }
            )
    return {
        "status": (
            "FAILED"
            if any(row["status"] == "FAILED" for row in runs)
            else "COMPLETE"
            if (
                completed_steps == total_steps
                and all(row["status"] == "COMPLETE" for row in runs)
            )
            else "RUNNING"
        ),
        "completed_optimizer_steps": completed_steps,
        "total_optimizer_steps": total_steps,
        "optimizer_step_weighted_percent": (
            100.0 * completed_steps / total_steps
        ),
        "completed_runs": sum(row["status"] == "COMPLETE" for row in runs),
        "total_runs": len(runs),
        "parallel_campaign_eta_seconds": (
            max(eta_by_seed.values()) if eta_by_seed else None
        ),
        "runs": runs,
        "dev_accessed": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
