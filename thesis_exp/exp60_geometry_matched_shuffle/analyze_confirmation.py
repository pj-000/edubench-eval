"""Frozen fixed-epoch analysis for the future Exp60 formal endpoints."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd.data_audit import model_rows, write_json
from thesis_exp.exp57_cbrd.metrics import compute_metrics
from thesis_exp.exp57_cbrd.train import aggregate_text_hash
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REAL_PREFLIGHT_DECISION_PATH,
    SOURCE_LOCK_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.train import (
    dataset_contract_sha256,
    verify_contract,
)
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def validate_geometry_audit(
    directory: Path, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = read_json(directory / "geometry_step_audit.json")
    if not isinstance(rows, list) or len(rows) != 210:
        raise RuntimeError(f"Geometry audit must contain 210 steps: {directory}")
    numeric_fields = (
        "aligned_normalized_orthogonality_error",
        "shuffled_normalized_orthogonality_error",
        "component_norm_relative_error",
        "preclip_total_norm_relative_error",
        "clip_coefficient_relative_error",
        "storage_component_norm_relative_error",
        "storage_preclip_total_norm_relative_error",
        "storage_clip_coefficient_relative_error",
        "storage_aligned_normalized_orthogonality_error",
        "storage_shuffled_normalized_orthogonality_error",
        "aligned_shuffled_component_cosine",
        "aligned_shuffled_component_relative_distance",
        "storage_aligned_shuffled_component_cosine",
        "storage_aligned_shuffled_component_relative_distance",
        "storage_component_activity_ratio",
        "preclip_norm",
        "postclip_norm",
    )
    for index, row in enumerate(rows, start=1):
        expected_epoch = (index - 1) // 21 + 1
        expected_microbatches = 24 if index % 21 == 0 else 32
        if (
            int(row.get("global_step", -1)) != index
            or int(row.get("epoch", -1)) != expected_epoch
            or int(row.get("microbatches_in_window", -1)) != expected_microbatches
        ):
            raise RuntimeError(f"Geometry step structure mismatch: {directory}: {index}")
        for field in numeric_fields:
            if field not in row or not math.isfinite(float(row[field])):
                raise RuntimeError(
                    f"Non-finite/missing geometry value: {directory}: {index}: {field}"
                )
    recomputed = {
        "maximum_aligned_normalized_orthogonality_error": max(
            float(row["aligned_normalized_orthogonality_error"]) for row in rows
        ),
        "maximum_shuffled_normalized_orthogonality_error": max(
            float(row["shuffled_normalized_orthogonality_error"]) for row in rows
        ),
        "maximum_component_norm_relative_error": max(
            float(row["component_norm_relative_error"]) for row in rows
        ),
        "maximum_preclip_total_norm_relative_error": max(
            float(row["preclip_total_norm_relative_error"]) for row in rows
        ),
        "maximum_clip_coefficient_relative_error": max(
            float(row["clip_coefficient_relative_error"]) for row in rows
        ),
        "maximum_storage_component_norm_relative_error": max(
            float(row["storage_component_norm_relative_error"]) for row in rows
        ),
        "maximum_storage_preclip_total_norm_relative_error": max(
            float(row["storage_preclip_total_norm_relative_error"]) for row in rows
        ),
        "maximum_storage_clip_coefficient_relative_error": max(
            float(row["storage_clip_coefficient_relative_error"]) for row in rows
        ),
        "maximum_storage_normalized_orthogonality_error": max(
            max(
                float(row["storage_aligned_normalized_orthogonality_error"]),
                float(row["storage_shuffled_normalized_orthogonality_error"]),
            )
            for row in rows
        ),
        "maximum_aligned_shuffled_component_cosine": max(
            float(row["aligned_shuffled_component_cosine"]) for row in rows
        ),
        "minimum_aligned_shuffled_component_relative_distance": min(
            float(row["aligned_shuffled_component_relative_distance"]) for row in rows
        ),
        "minimum_storage_component_activity_ratio": min(
            float(row["storage_component_activity_ratio"]) for row in rows
        ),
        "maximum_storage_aligned_shuffled_component_cosine": max(
            float(row["storage_aligned_shuffled_component_cosine"]) for row in rows
        ),
        "minimum_storage_aligned_shuffled_component_relative_distance": min(
            float(row["storage_aligned_shuffled_component_relative_distance"])
            for row in rows
        ),
        "median_storage_aligned_shuffled_component_relative_distance": statistics.median(
            float(row["storage_aligned_shuffled_component_relative_distance"])
            for row in rows
        ),
    }
    for field, value in recomputed.items():
        reported = float(summary.get(field, float("nan")))
        if not math.isfinite(reported) or reported != value:
            raise RuntimeError(
                f"Geometry summary mismatch: {directory}: {field}: {reported} != {value}"
            )
    epoch_activity = {
        str(epoch): min(
            float(row["storage_component_activity_ratio"])
            for row in rows
            if int(row["epoch"]) == epoch
        )
        for epoch in range(1, 11)
    }
    if summary.get("minimum_storage_component_activity_ratio_by_epoch") != epoch_activity:
        raise RuntimeError(f"Per-epoch treatment activity mismatch: {directory}")
    return rows


def endpoint(
    variant: str,
    seed: int,
    mapping_sha: str,
    protocol: dict[str, Any],
    frozen_dev: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    directory = run_dir(variant, seed)
    summary = read_json(directory / "run_summary.json")
    if summary.get("status") != "COMPLETED":
        raise RuntimeError(f"Incomplete endpoint: {directory}")
    if summary.get("variant") != variant or summary.get("seed") != seed:
        raise RuntimeError(f"Variant/seed self-report mismatch: {directory}")
    gpu_slot = summary.get("gpu_slot")
    if gpu_slot not in (0, 1, 2) or protocol["formal_runs"]["gpu_latin_square"][
        str(seed)
    ][f"gpu_slot_{gpu_slot}"] != variant:
        raise RuntimeError(f"GPU Latin-square mismatch: {directory}")
    bindings = protocol["formal_runs"].get("physical_gpu_bindings")
    if not isinstance(bindings, dict) or str(summary.get("cuda_visible_devices")) != str(
        bindings[f"gpu_slot_{gpu_slot}"]
    ):
        raise RuntimeError(f"Physical GPU binding mismatch: {directory}")
    if summary.get("selected_epoch") != 10 or summary.get("checkpoint_rule") != "fixed epoch 10 primary":
        raise RuntimeError(f"Non-frozen checkpoint: {directory}")
    if summary.get("optimizer_steps") != 210:
        raise RuntimeError(f"Wrong optimizer-step count: {directory}")
    if summary.get("mapping_sha256") != mapping_sha:
        raise RuntimeError(f"Mapping mismatch: {directory}")
    if summary.get("test_access_count") != 0:
        raise RuntimeError(f"Nonzero test access: {directory}")
    contract = summary.get("frozen_contract_files", {})
    if contract.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError(f"Protocol SHA mismatch: {directory}")
    if contract.get("source_lock_sha256") != sha256_file(SOURCE_LOCK_PATH):
        raise RuntimeError(f"Source-lock SHA mismatch: {directory}")
    if contract.get("real_model_preflight_decision_sha256") != sha256_file(
        REAL_PREFLIGHT_DECISION_PATH
    ):
        raise RuntimeError(f"Real-preflight decision SHA mismatch: {directory}")
    fixed = protocol["fixed_training"]
    if summary.get("train_text_hash") != fixed["train_text_hash"]:
        raise RuntimeError(f"Train hash mismatch: {directory}")
    if summary.get("dev_text_hash") != fixed["dev_text_hash"]:
        raise RuntimeError(f"Dev hash mismatch: {directory}")
    if summary.get("train_dataset_contract_sha256") != fixed[
        "train_dataset_contract_sha256"
    ]:
        raise RuntimeError(f"Train dataset-contract mismatch: {directory}")
    if summary.get("dev_dataset_contract_sha256") != fixed[
        "dev_dataset_contract_sha256"
    ]:
        raise RuntimeError(f"Dev dataset-contract mismatch: {directory}")
    for required in (
        "initial_model_snapshot_sha256",
        "model_input_manifest_sha256",
        "initial_head_contract",
    ):
        if not summary.get(required):
            raise RuntimeError(f"Missing {required}: {directory}")
    tolerance = float(protocol["implementation_gates_before_training"][
        "bf16_storage_space_relative_error_at_most"
    ])
    construction_limits = {
        "maximum_aligned_normalized_orthogonality_error": float(
            protocol["implementation_gates_before_training"][
                "normalized_orthogonality_error_at_most"
            ]
        ),
        "maximum_shuffled_normalized_orthogonality_error": float(
            protocol["implementation_gates_before_training"][
                "normalized_orthogonality_error_at_most"
            ]
        ),
        "maximum_component_norm_relative_error": float(
            protocol["implementation_gates_before_training"][
                "component_norm_relative_error_at_most"
            ]
        ),
        "maximum_preclip_total_norm_relative_error": float(
            protocol["implementation_gates_before_training"][
                "preclip_total_norm_relative_error_at_most"
            ]
        ),
        "maximum_clip_coefficient_relative_error": float(
            protocol["implementation_gates_before_training"][
                "clip_coefficient_relative_error_at_most"
            ]
        ),
    }
    if any(
        float(summary.get(field, float("inf"))) > limit
        for field, limit in construction_limits.items()
    ):
        raise RuntimeError(f"Construction-space geometry gate mismatch: {directory}")
    geometry_fields = (
        "maximum_storage_component_norm_relative_error",
        "maximum_storage_preclip_total_norm_relative_error",
        "maximum_storage_clip_coefficient_relative_error",
    )
    if any(float(summary.get(field, float("inf"))) > tolerance for field in geometry_fields):
        raise RuntimeError(f"BF16 storage geometry gate mismatch: {directory}")
    orthogonality_tolerance = float(
        protocol["implementation_gates_before_training"][
            "bf16_storage_space_normalized_orthogonality_error_at_most"
        ]
    )
    if float(
        summary.get("maximum_storage_normalized_orthogonality_error", float("inf"))
    ) > orthogonality_tolerance:
        raise RuntimeError(f"BF16 storage orthogonality gate mismatch: {directory}")
    validate_geometry_audit(directory, summary)

    predictions = read_jsonl(prediction_path(directory))
    if len(predictions) != len(frozen_dev) or len(predictions) != fixed["dev_rows"]:
        raise RuntimeError(f"Prediction row-count mismatch: {directory}")
    ids = [str(row["record_id"]) for row in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(frozen_dev):
        raise RuntimeError(f"Prediction ID-set mismatch: {directory}")
    for row in predictions:
        record_id = str(row["record_id"])
        source = frozen_dev[record_id]
        if not str(row.get("question_key") or "").strip():
            raise RuntimeError(f"Empty question key: {directory}: {record_id}")
        if str(row["question_key"]) != str(source["question_key"]):
            raise RuntimeError(f"Question-key mismatch: {directory}: {record_id}")
        if int(row["label_5"]) != int(source["label_5"]):
            raise RuntimeError(f"Target label mismatch: {directory}: {record_id}")
        if float(row["human_mean_5"]) != float(source["human_mean_5"]):
            raise RuntimeError(f"Human-mean mismatch: {directory}: {record_id}")
    recomputed = compute_metrics(predictions)
    metrics = {key: float(recomputed[key]) for key in METRICS}
    for key in METRICS:
        reported = float(summary["selected_metrics"][key])
        if not math.isfinite(metrics[key]) or not math.isfinite(reported) or not np.isclose(
            metrics[key], reported, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"Recomputed metric mismatch: {directory}: {key}: {metrics[key]} != {reported}"
            )
    return metrics, predictions, summary


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
    endpoints: dict[
        tuple[str, int], tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]
    ],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    paired_predictions: list[list[dict[str, Any]]] = []
    for seed in SEEDS:
        treatment_metrics, treatment_predictions, _ = endpoints[(treatment, seed)]
        control_metrics, control_predictions, _ = endpoints[(control, seed)]
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
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("status") != "EXP60_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS":
        raise RuntimeError("Exp60 analysis is blocked until the protocol is formally frozen")
    if not SOURCE_LOCK_PATH.is_file():
        raise FileNotFoundError(SOURCE_LOCK_PATH)
    verify_contract()
    mapping_sha = str(read_json(MAPPING_AUDIT_PATH)["mapping_sha256"])
    dev_rows = model_rows("dev")
    frozen_dev = {str(row["record_id"]): row for row in dev_rows}
    if len(frozen_dev) != len(dev_rows):
        raise RuntimeError("Frozen dev contains duplicate record IDs")
    if aggregate_text_hash(dev_rows) != protocol["fixed_training"]["dev_text_hash"]:
        raise RuntimeError("Frozen dev text hash differs from protocol")
    if dataset_contract_sha256(dev_rows) != protocol["fixed_training"][
        "dev_dataset_contract_sha256"
    ]:
        raise RuntimeError("Frozen dev dataset-contract hash differs from protocol")
    endpoints = {
        (variant, seed): endpoint(
            variant, seed, mapping_sha, protocol, frozen_dev
        )
        for variant in VARIANTS
        for seed in SEEDS
    }
    for seed in SEEDS:
        summaries = [endpoints[(variant, seed)][2] for variant in VARIANTS]
        for field in (
            "initial_model_snapshot_sha256",
            "model_input_manifest_sha256",
            "train_text_hash",
            "dev_text_hash",
            "train_dataset_contract_sha256",
            "dev_dataset_contract_sha256",
            "training_batch_id_order_sha256",
        ):
            if len({str(summary[field]) for summary in summaries}) != 1:
                raise RuntimeError(f"Same-seed three-arm {field} mismatch for seed {seed}")
        head_contracts = [summary["initial_head_contract"] for summary in summaries]
        if any(contract != head_contracts[0] for contract in head_contracts[1:]):
            raise RuntimeError(f"Same-seed initial head mismatch for seed {seed}")
        environments = [summary.get("runtime_environment") for summary in summaries]
        if any(environment != environments[0] for environment in environments[1:]):
            raise RuntimeError(f"Same-seed runtime-environment mismatch for seed {seed}")
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
        "aligned_minus_consensus_favorable_MAE_in_at_least_2_of_3_seeds": secondary[
            "favorable_MAE_seeds"
        ] >= 2,
        "aligned_minus_consensus_mean_delta_MAE_at_most_minus_0p005": secondary[
            "mean_delta"
        ]["MAE_human_mean"] <= -0.005,
    }
    report = {
        "status": "EXP60_SAMPLE_TARGET_ALIGNMENT_GO" if all(gates.values()) else "EXP60_SAMPLE_TARGET_ALIGNMENT_NO_GO",
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
