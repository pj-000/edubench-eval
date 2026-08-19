#!/usr/bin/env python3
"""Paired question-cluster bootstrap from frozen Exp51 test predictions.

This script performs no inference and does not select checkpoints or
hyperparameters.  It joins the already frozen prediction rows to the locked
test split only to recover the question cluster identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("b0", "exp51")
SEEDS = (42, 43, 44)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_row_deltas(
    prediction_root: Path,
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Return three-seed-mean MAE deltas and prediction artifact hashes."""
    per_seed: list[dict[str, float]] = []
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
        for arm in ARMS:
            path = prediction_root / arm / f"seed_{seed}" / "predictions_test.jsonl"
            rows = read_jsonl(path)
            keyed = {str(row["record_id"]): row for row in rows}
            if len(rows) != 2218 or len(keyed) != 2218:
                raise ValueError(f"Unexpected prediction count: {arm} seed {seed}")
            rows_by_arm[arm] = keyed
            hashes[f"{arm}/seed_{seed}"] = sha256(path)
        if set(rows_by_arm["b0"]) != set(rows_by_arm["exp51"]):
            raise ValueError(f"Prediction ID mismatch for seed {seed}")
        deltas: dict[str, float] = {}
        for record_id, b0 in rows_by_arm["b0"].items():
            exp51 = rows_by_arm["exp51"][record_id]
            human_mean = float(b0["human_mean_5"])
            deltas[record_id] = (
                abs(float(exp51["pred_label_5"]) - human_mean)
                - abs(float(b0["pred_label_5"]) - human_mean)
            )
        per_seed.append(deltas)
    record_ids = set(per_seed[0])
    if any(set(rows) != record_ids for rows in per_seed[1:]):
        raise ValueError("Seed-level record IDs differ")
    averaged = {
        record_id: float(np.mean([rows[record_id] for rows in per_seed]))
        for record_id in record_ids
    }
    return averaged, {}, hashes


def question_mapping(test_split: Path) -> tuple[dict[str, str], str]:
    rows = read_jsonl(test_split)
    if len(rows) != 2218:
        raise ValueError(f"Unexpected test count: {len(rows)}")
    mapping = {
        str(row["record_id"]): str(row.get("question_key") or row["question"])
        for row in rows
    }
    if len(mapping) != 2218:
        raise ValueError("Test record IDs are not unique")
    return mapping, sha256(test_split)


def cluster_bootstrap(
    deltas: dict[str, float],
    cluster_by_record: dict[str, str],
    *,
    resamples: int,
    rng_seed: int,
) -> dict[str, Any]:
    if set(deltas) != set(cluster_by_record):
        raise ValueError("Prediction and test record IDs differ")
    grouped: dict[str, list[float]] = {}
    for record_id, delta in deltas.items():
        grouped.setdefault(cluster_by_record[record_id], []).append(delta)
    cluster_ids = sorted(grouped)
    arrays = [np.asarray(grouped[key], dtype=float) for key in cluster_ids]
    observed = float(np.mean([value for array in arrays for value in array]))
    rng = np.random.default_rng(rng_seed)
    bootstrap = np.empty(resamples, dtype=float)
    cluster_count = len(arrays)
    for index in range(resamples):
        sampled = rng.integers(0, cluster_count, cluster_count)
        numerator = sum(float(arrays[item].sum()) for item in sampled)
        denominator = sum(int(arrays[item].size) for item in sampled)
        bootstrap[index] = numerator / denominator
    return {
        "convention": "HMSA_minus_Hard-only; negative MAE is better",
        "mean_delta_mae": observed,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "resamples": resamples,
        "rng_seed": rng_seed,
        "unit": "question cluster after averaging paired row deltas over three model seeds",
        "cluster_key": "question_key, falling back to exact question text",
        "cluster_count": cluster_count,
        "row_count": len(deltas),
        "min_cluster_size": min(len(value) for value in arrays),
        "max_cluster_size": max(len(value) for value in arrays),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=20260731)
    args = parser.parse_args()

    deltas, _, prediction_hashes = frozen_row_deltas(args.prediction_root)
    mapping, test_hash = question_mapping(args.test_split)
    report = cluster_bootstrap(
        deltas,
        mapping,
        resamples=args.resamples,
        rng_seed=args.rng_seed,
    )
    report["artifact_hashes"] = {
        "test_split_sha256": test_hash,
        "prediction_jsonl_sha256": prediction_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
