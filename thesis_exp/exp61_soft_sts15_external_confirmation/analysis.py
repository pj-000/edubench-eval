"""Frozen epoch-10 prediction analysis; it never loads raw dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp61_soft_sts15_external_confirmation import SEEDS, VARIANTS
from thesis_exp.exp61_soft_sts15_external_confirmation.metrics import (
    component_cluster_bootstrap,
    confirmation_gates,
    evaluate_hard_head,
)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty prediction artifact: {path}")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate record IDs: {path}")
    return rows


def analyze_prediction_grid(root: Path, filename: str) -> dict[str, Any]:
    grid: dict[str, dict[int, list[dict[str, Any]]]] = {}
    canonical_ids: list[str] | None = None
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    point_predictions: dict[str, dict[int, np.ndarray]] = {}
    canonical_rows: list[dict[str, Any]] | None = None
    for variant in VARIANTS:
        grid[variant], metrics[variant], point_predictions[variant] = {}, {}, {}
        for seed in SEEDS:
            rows = read_predictions(root / variant / f"seed_{seed}" / filename)
            ids = [str(row["record_id"]) for row in rows]
            if canonical_ids is None:
                canonical_ids, canonical_rows = ids, rows
            elif ids != canonical_ids:
                raise RuntimeError("prediction grid record order/set mismatch")
            probabilities = np.asarray([row["hard_head_probabilities"] for row in rows])
            means = np.asarray([row["human_mean"] for row in rows])
            labels = np.asarray([row["hard_label"] for row in rows])
            expected = probabilities @ np.arange(6)
            grid[variant][seed] = rows
            point_predictions[variant][seed] = expected
            metrics[variant][seed] = evaluate_hard_head(probabilities, means, labels)
    assert canonical_rows is not None
    bootstrap = component_cluster_bootstrap(
        point_predictions["aligned_orthogonal_only"],
        point_predictions["matched_shuffled_orthogonal_only"],
        np.asarray([row["human_mean"] for row in canonical_rows]),
        [str(row["component_sha256"]) for row in canonical_rows],
        n_resamples=10_000,
        rng_seed=610061,
    )
    gates = confirmation_gates(
        metrics["aligned_orthogonal_only"],
        metrics["matched_shuffled_orthogonal_only"],
        metrics["quantized_mean_only"],
        bootstrap,
    )
    return {
        "status": "EXP61_FIXED_EPOCH10_ANALYSIS_COMPLETE",
        "prediction_filename": filename,
        "metrics": {variant: {str(seed): values for seed, values in by_seed.items()} for variant, by_seed in metrics.items()},
        "component_cluster_bootstrap": bootstrap,
        "confirmation": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction_filename", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_prediction_grid(args.root, args.prediction_filename)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "decision": result["confirmation"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
