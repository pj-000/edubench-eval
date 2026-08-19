"""Collect three train-only R3 greedy inventories into one failure bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    ERROR_CLASSES,
    SEEDS,
    aggregate_failure_rows,
    compact_json,
    read_jsonl,
    sha256_file,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "actual_failure_bank"
)
SCHEMA_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/schemas/"
    "actual_failure_bank_row_v1.schema.json"
)


def validate_row(row: dict[str, Any]) -> None:
    required = {
        "record_id",
        "row_position",
        "gold_label",
        "metric_id",
        "language",
        "generator_arm",
        "generator_seed",
        "generator_epoch",
        "generator_adapter_sha256",
        "generation_mode",
        "rollout_seed",
        "parse_success",
        "generated_score",
        "generated_rationale",
        "forced_completion",
        "signed_error",
        "absolute_error",
        "error_class",
        "severe_low_to_high",
        "severe_high_to_low",
    }
    if set(row) != required:
        raise ValueError("failure-bank row fields differ")
    if (
        row["generator_arm"] != "R3"
        or row["generator_epoch"] != 3
        or row["generator_seed"] not in SEEDS
        or row["generation_mode"] != "greedy"
        or row["rollout_seed"] is not None
        or row["error_class"] not in ERROR_CLASSES
    ):
        raise ValueError("failure-bank row contract differs")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        b"".join(
            (compact_json(row) + "\n").encode("utf-8") for row in rows
        )
    )
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def collect(*, dry_run: bool) -> dict[str, Any]:
    sources = {
        seed: (
            OUTPUT_ROOT
            / "private"
            / f"seed{seed}"
            / "failure_rows.jsonl"
        )
        for seed in SEEDS
    }
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sources.values()
        if not path.is_file()
    ]
    if missing:
        return {
            "status": "ACTUAL_FAILURE_BANK_WAITING_FOR_GPU_GENERATION",
            "missing_private_sources": missing,
            "gpu_generation_required": True,
            "dev_accessed": False,
            "test_accessed": False,
        }
    rows = []
    source_hashes = {}
    for seed, path in sources.items():
        seed_rows = read_jsonl(path)
        if len(seed_rows) != 2654:
            raise ValueError(f"{path}: row count differs")
        if any(int(row["generator_seed"]) != seed for row in seed_rows):
            raise ValueError(f"{path}: generator seed differs")
        for row in seed_rows:
            validate_row(row)
        rows.extend(seed_rows)
        source_hashes[str(seed)] = sha256_file(path)
    rows.sort(
        key=lambda row: (
            int(row["row_position"]),
            int(row["generator_seed"]),
        )
    )
    aggregate = aggregate_failure_rows(rows)
    if dry_run:
        return {
            "status": "ACTUAL_FAILURE_BANK_COLLECT_DRY_RUN_PASS",
            "aggregate": aggregate,
            "source_hashes": source_hashes,
            "output_written": False,
            "dev_accessed": False,
            "test_accessed": False,
        }
    combined = OUTPUT_ROOT / "private" / "actual_failure_bank.jsonl"
    write_jsonl(combined, rows)
    report = {
        "schema_version": "exp54-actual-failure-bank-report-v1",
        "status": "ACTUAL_FAILURE_BANK_GREEDY_INVENTORY_COMPLETE",
        "scientific_role": (
            "Train-only diagnostic inventory; no preference pair has been "
            "selected or materialized."
        ),
        "source_hashes": source_hashes,
        "combined_private_sha256": sha256_file(combined),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "aggregate": aggregate,
        "preference_pair_construction_started": False,
        "stochastic_rollout_started": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(OUTPUT_ROOT / "aggregate_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            collect(dry_run=not args.write),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
