"""Frozen fixed-epoch analysis for the future Exp60 formal endpoints."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import MAPPING_AUDIT_PATH, OUTPUT_ROOT
from thesis_exp.src.edujudge.utils.io import write_csv


SEEDS = (47, 48, 49)
VARIANTS = (
    "consensus_only",
    "aligned_orthogonal_only",
    "matched_shuffled_orthogonal_only",
)
METRICS = ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")
REPETITIONS = 10_000
RNG_SEED = 60


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def prediction_path(directory: Path) -> Path:
    for candidate in (
        directory / "predictions_dev_epoch10.jsonl",
        directory / "predictions" / "predictions_dev_epoch10.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No fixed-epoch-10 predictions below {directory}")


def endpoint(variant: str, seed: int, mapping_sha: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    directory = run_dir(variant, seed)
    summary = read_json(directory / "run_summary.json")
    if summary.get("status") != "COMPLETED":
        raise RuntimeError(f"Incomplete endpoint: {directory}")
    if summary.get("selected_epoch") != 10 or summary.get("checkpoint_rule") != "fixed epoch 10 primary":
        raise RuntimeError(f"Non-frozen checkpoint: {directory}")
    if summary.get("optimizer_steps") != 210:
        raise RuntimeError(f"Wrong optimizer-step count: {directory}")
    if summary.get("mapping_sha256") != mapping_sha:
        raise RuntimeError(f"Mapping mismatch: {directory}")
    if summary.get("test_access_count") != 0:
        raise RuntimeError(f"Nonzero test access: {directory}")
    metrics = {key: float(summary["selected_metrics"][key]) for key in METRICS}
    return metrics, read_jsonl(prediction_path(directory))


def paired_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    right_by_id = {str(row["record_id"]): row for row in right}
    if len(right_by_id) != len(right):
        raise RuntimeError("Duplicate right-side prediction ids")
    rows: list[dict[str, Any]] = []
    for row in left:
        record_id = str(row["record_id"])
        peer = right_by_id.pop(record_id)
        if str(row["question_key"]) != str(peer["question_key"]):
            raise RuntimeError(f"Question-key mismatch: {record_id}")
        target = float(row["human_mean_5"])
        if target != float(peer["human_mean_5"]):
            raise RuntimeError(f"Target mismatch: {record_id}")
        rows.append(
            {
                "record_id": record_id,
                "question_key": str(row["question_key"]),
                "delta_MAE": abs(float(row["pred_label_5"]) - target)
                - abs(float(peer["pred_label_5"]) - target),
            }
        )
    if right_by_id:
        raise RuntimeError("Right-side predictions contain unmatched ids")
    return rows


def question_cluster_bootstrap(seed_rows: list[list[dict[str, Any]]]) -> dict[str, Any]:
    per_record: dict[str, list[float]] = defaultdict(list)
    question_by_record: dict[str, str] = {}
    for rows in seed_rows:
        for row in rows:
            record_id = str(row["record_id"])
            per_record[record_id].append(float(row["delta_MAE"]))
            question_by_record[record_id] = str(row["question_key"])
    if not per_record or any(len(values) != len(SEEDS) for values in per_record.values()):
        raise RuntimeError("Incomplete three-seed paired predictions")
    clusters: dict[str, list[float]] = defaultdict(list)
    for record_id, values in per_record.items():
        clusters[question_by_record[record_id]].append(float(np.mean(values)))
    keys = sorted(clusters)
    rng = np.random.default_rng(RNG_SEED)
    draws = np.empty(REPETITIONS, dtype=np.float64)
    for index in range(REPETITIONS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in sampled for value in clusters[str(key)]]
        draws[index] = float(np.mean(values))
    point = float(np.mean([value for values in clusters.values() for value in values]))
    return {
        "estimand": "item-weighted mean paired MAE difference with question-cluster resampling, conditional on three trained seeds",
        "clusters": len(keys),
        "records": len(per_record),
        "repetitions": REPETITIONS,
        "rng_seed": RNG_SEED,
        "point_estimate": point,
        "ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_delta_below_zero": float(np.mean(draws < 0.0)),
    }


def comparison(
    treatment: str,
    control: str,
    endpoints: dict[tuple[str, int], tuple[dict[str, float], list[dict[str, Any]]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paired_predictions: list[list[dict[str, Any]]] = []
    for seed in SEEDS:
        treatment_metrics, treatment_predictions = endpoints[(treatment, seed)]
        control_metrics, control_predictions = endpoints[(control, seed)]
        row: dict[str, Any] = {"seed": seed, "treatment": treatment, "control": control}
        for metric in METRICS:
            row[f"delta_{metric}"] = treatment_metrics[metric] - control_metrics[metric]
        rows.append(row)
        paired_predictions.append(paired_rows(treatment_predictions, control_predictions))
    means = {
        metric: float(np.mean([row[f"delta_{metric}"] for row in rows]))
        for metric in METRICS
    }
    return {
        "treatment": treatment,
        "control": control,
        "per_seed": rows,
        "mean_delta": means,
        "favorable_MAE_seeds": sum(row["delta_MAE_human_mean"] < 0.0 for row in rows),
        "question_cluster_bootstrap": question_cluster_bootstrap(paired_predictions),
    }


def main() -> None:
    mapping_sha = str(read_json(MAPPING_AUDIT_PATH)["mapping_sha256"])
    endpoints = {
        (variant, seed): endpoint(variant, seed, mapping_sha)
        for variant in VARIANTS
        for seed in SEEDS
    }
    primary = comparison(
        "aligned_orthogonal_only",
        "matched_shuffled_orthogonal_only",
        endpoints,
    )
    secondary = comparison("aligned_orthogonal_only", "consensus_only", endpoints)
    gates = {
        "aligned_favorable_MAE_in_3_of_3_seeds": primary["favorable_MAE_seeds"] == 3,
        "mean_delta_MAE_at_most_minus_0p005": primary["mean_delta"]["MAE_human_mean"] <= -0.005,
        "cluster_CI_upper_below_zero": primary["question_cluster_bootstrap"]["ci_95"][1] < 0.0,
        "mean_delta_Exact_at_least_minus_0p003": primary["mean_delta"]["Exact_rounded"] >= -0.003,
        "mean_delta_Kendall_at_least_minus_0p005": primary["mean_delta"]["Kendall_human_mean"] >= -0.005,
        "aligned_minus_consensus_MAE_direction_negative": secondary["mean_delta"]["MAE_human_mean"] < 0.0,
    }
    report = {
        "status": "EXP60_SEMANTIC_SPECIFICITY_GO" if all(gates.values()) else "EXP60_SEMANTIC_SPECIFICITY_NO_GO",
        "primary": primary,
        "secondary": secondary,
        "gates": gates,
        "mapping_sha256": mapping_sha,
        "checkpoint": "fixed epoch 10",
        "analysis_registration": {
            "repetitions": REPETITIONS,
            "rng_seed": RNG_SEED,
            "implemented_before_formal_results": True,
        },
        "allowed_splits": ["dev"],
        "test_access_count": 0,
    }
    table_rows = primary["per_seed"] + secondary["per_seed"]
    write_csv(OUTPUT_ROOT / "tables" / "exp60_fixed_epoch10_comparisons.csv", table_rows)
    write_json(OUTPUT_ROOT / "audit" / "exp60_fixed_epoch10_confirmation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
