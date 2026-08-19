"""CPU-only confirmation analyses for the Exp57 primary comparison."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.src.edujudge.utils.io import write_csv


PRIMARY_VARIANTS = ("routed_hmsa", "consensus_only")
DEVELOPMENT_SEEDS = (42, 43, 44)
METRIC_KEYS = ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def prediction_path(variant: str, seed: int) -> Path:
    directory = run_dir(variant, seed)
    candidates = (
        directory / "predictions" / "predictions_dev.jsonl",
        directory / "predictions_dev.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No selected dev predictions below {directory}")


def common_epoch_analysis() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in DEVELOPMENT_SEEDS:
        histories = {
            variant: {
                int(row["epoch"]): row
                for row in read_json(run_dir(variant, seed) / "dev_metrics_history.json")
            }
            for variant in PRIMARY_VARIANTS
        }
        if set(histories["routed_hmsa"]) != set(range(1, 11)):
            raise RuntimeError(f"Routed-HMSA seed {seed} does not have epochs 1..10")
        if set(histories["consensus_only"]) != set(range(1, 11)):
            raise RuntimeError(f"Consensus-only seed {seed} does not have epochs 1..10")
        seed_rows: list[dict[str, Any]] = []
        for epoch in range(1, 11):
            routed = histories["routed_hmsa"][epoch]
            consensus = histories["consensus_only"][epoch]
            row = {"seed": seed, "epoch": epoch}
            for key in METRIC_KEYS:
                row[f"routed_{key}"] = float(routed[key])
                row[f"consensus_{key}"] = float(consensus[key])
                row[f"delta_{key}"] = float(routed[key]) - float(consensus[key])
            rows.append(row)
            seed_rows.append(row)
        selected = {
            variant: read_json(run_dir(variant, seed) / "selected_dev_metrics.json")
            for variant in PRIMARY_VARIANTS
        }
        selected_row = {"seed": seed}
        for key in METRIC_KEYS:
            selected_row[f"delta_{key}"] = (
                float(selected["routed_hmsa"][key])
                - float(selected["consensus_only"][key])
            )
        selected_rows.append(selected_row)
        mae_deltas = np.asarray([row["delta_MAE_human_mean"] for row in seed_rows])
        per_seed.append(
            {
                "seed": seed,
                "favorable_mae_epochs": int(np.sum(mae_deltas < 0.0)),
                "favorable_mae_fraction": float(np.mean(mae_deltas < 0.0)),
                "epoch10_delta_mae": float(mae_deltas[-1]),
                "normalized_trapezoidal_mae_curve_delta": float(
                    np.trapezoid(mae_deltas, dx=1.0) / 9.0
                ),
                "selected_checkpoint_delta_mae": selected_row["delta_MAE_human_mean"],
            }
        )
    all_mae = np.asarray([row["delta_MAE_human_mean"] for row in rows])
    report = {
        "status": "COMPLETE",
        "comparison": "routed_hmsa minus consensus_only",
        "per_seed": per_seed,
        "aggregate": {
            "favorable_seed_epoch_cells": int(np.sum(all_mae < 0.0)),
            "total_seed_epoch_cells": int(len(all_mae)),
            "favorable_seed_epoch_fraction": float(np.mean(all_mae < 0.0)),
            "mean_common_epoch_delta_mae": float(np.mean(all_mae)),
            "mean_epoch10_delta_mae": float(
                np.mean([item["epoch10_delta_mae"] for item in per_seed])
            ),
            "mean_normalized_trapezoidal_mae_curve_delta": float(
                np.mean(
                    [item["normalized_trapezoidal_mae_curve_delta"] for item in per_seed]
                )
            ),
            "mean_selected_checkpoint_delta_mae": float(
                np.mean([item["selected_checkpoint_delta_mae"] for item in per_seed])
            ),
        },
        "test_access_count": 0,
    }
    write_csv(OUTPUT_ROOT / "tables" / "stage1_primary_common_epoch.csv", rows)
    write_csv(OUTPUT_ROOT / "tables" / "stage1_primary_selected_checkpoint.csv", selected_rows)
    write_json(OUTPUT_ROOT / "audit" / "stage1_primary_common_epoch.json", report)
    return report


def _paired_row_deltas(seed: int) -> list[dict[str, Any]]:
    routed = {
        str(row["record_id"]): row
        for row in read_jsonl(prediction_path("routed_hmsa", seed))
    }
    consensus = {
        str(row["record_id"]): row
        for row in read_jsonl(prediction_path("consensus_only", seed))
    }
    if routed.keys() != consensus.keys() or len(routed) != 664:
        raise RuntimeError(f"Prediction pairing failed for seed {seed}")
    rows: list[dict[str, Any]] = []
    for record_id in sorted(routed):
        left = routed[record_id]
        right = consensus[record_id]
        human_mean = float(left["human_mean_5"])
        if human_mean != float(right["human_mean_5"]):
            raise RuntimeError(f"Human target mismatch for {record_id}")
        question_key = str(left.get("question_key") or left.get("triple_key") or record_id)
        rows.append(
            {
                "record_id": record_id,
                "question_key": question_key,
                "delta_absolute_error": (
                    abs(float(left["pred_score_5"]) - human_mean)
                    - abs(float(right["pred_score_5"]) - human_mean)
                ),
            }
        )
    return rows


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int, rng_seed: int
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_key"])].append(float(row["delta_absolute_error"]))
    keys = sorted(clusters)
    sums = np.asarray([sum(clusters[key]) for key in keys], dtype=float)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draw = rng.integers(0, len(keys), size=len(keys))
        samples[index] = float(sums[draw].sum() / counts[draw].sum())
    observed = float(sums.sum() / counts.sum())
    return {
        "question_clusters": len(keys),
        "rows": int(counts.sum()),
        "observed_delta_mae": observed,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_delta_below_zero": float(np.mean(samples < 0.0)),
        "repetitions": repetitions,
        "rng_seed": rng_seed,
    }


def primary_cluster_bootstrap(
    *, repetitions: int = 10000, rng_seed: int = 57
) -> dict[str, Any]:
    by_seed = {seed: _paired_row_deltas(seed) for seed in DEVELOPMENT_SEEDS}
    seed_reports = {
        str(seed): _cluster_bootstrap(
            rows, repetitions=repetitions, rng_seed=rng_seed + seed
        )
        for seed, rows in by_seed.items()
    }
    rows_by_seed = {
        seed: {row["record_id"]: row for row in rows}
        for seed, rows in by_seed.items()
    }
    shared_ids = set.intersection(*(set(rows) for rows in rows_by_seed.values()))
    if len(shared_ids) != 664:
        raise RuntimeError("The three seeds do not share the same 664 dev records")
    averaged_rows = []
    for record_id in sorted(shared_ids):
        seed_rows = [rows_by_seed[seed][record_id] for seed in DEVELOPMENT_SEEDS]
        averaged_rows.append(
            {
                "record_id": record_id,
                "question_key": seed_rows[0]["question_key"],
                "delta_absolute_error": float(
                    np.mean([row["delta_absolute_error"] for row in seed_rows])
                ),
            }
        )
    report = {
        "status": "COMPLETE",
        "comparison": "routed_hmsa minus consensus_only",
        "statistic": "paired hard-prediction absolute error against human_mean_5",
        "per_seed": seed_reports,
        "three_seed_mean_prediction_effect": _cluster_bootstrap(
            averaged_rows, repetitions=repetitions, rng_seed=rng_seed
        ),
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "stage1_primary_question_bootstrap.json", report)
    return report


def main() -> None:
    report = {
        "common_epoch": common_epoch_analysis(),
        "question_cluster_bootstrap": primary_cluster_bootstrap(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
