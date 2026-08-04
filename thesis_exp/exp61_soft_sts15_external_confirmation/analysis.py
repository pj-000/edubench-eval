"""Strict one-shot confirmatory analysis over the frozen Exp61 test outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    FROZEN_PROTOCOL,
    OUTPUT_ROOT,
    SEEDS,
    VARIANTS,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import (
    SOURCE_LOCK,
    sha256_file,
    verify_source_lock,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.data import read_manifest
from thesis_exp.exp61_soft_sts15_external_confirmation.evaluate_test_once import (
    ACCESS_MARKER,
    PREDICTION_ROOT,
    validate_checkpoint_grid,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.metrics import (
    component_cluster_bootstrap,
    confirmation_gates,
    evaluate_hard_head,
)


CONFIRMATION_DECISION = OUTPUT_ROOT / "final_test/confirmation_decision.json"


def frozen_test_metadata() -> list[dict[str, Any]]:
    manifest = read_manifest()
    return [
        {
            "record_id": f"softsts15:{row_id}",
            "row_id": row_id,
            "split": "test",
            "component_sha256": row["component_sha256"],
            "human_mean": float(row["human_mean"]),
            "hard_label": int(row["hard_label"]),
        }
        for row_id, row in sorted(manifest.items())
        if row["split"] == "test"
    ]


def validate_prediction_rows(
    rows: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    variant: str,
    seed: int,
    checkpoint_state_sha256: str,
) -> None:
    if len(rows) != 1578 or len(expected) != 1578:
        raise RuntimeError("confirmatory prediction grid must contain exactly 1,578 test rows")
    for index, (actual, frozen) in enumerate(zip(rows, expected)):
        checks = {
            "record_id": actual.get("record_id") == frozen["record_id"],
            "row_id": actual.get("row_id") == frozen["row_id"],
            "split": actual.get("split") == "test",
            "component": actual.get("component_sha256") == frozen["component_sha256"],
            "human_mean": math.isclose(
                float(actual.get("human_mean", math.nan)), frozen["human_mean"], abs_tol=1e-12
            ),
            "hard_label": actual.get("hard_label") == frozen["hard_label"],
            "variant": actual.get("variant") == variant,
            "seed": actual.get("seed") == seed,
            "checkpoint": actual.get("checkpoint_state_dict_sha256") == checkpoint_state_sha256,
        }
        probabilities = np.asarray(actual.get("hard_head_probabilities", []), dtype=np.float64)
        checks["six_probabilities"] = probabilities.shape == (6,)
        checks["finite_probabilities"] = bool(np.isfinite(probabilities).all())
        checks["probabilities_sum_one"] = bool(
            probabilities.shape == (6,) and np.isclose(probabilities.sum(), 1.0, atol=1e-6)
        )
        if probabilities.shape == (6,):
            expectation = float(probabilities @ np.arange(6))
            checks["expectation"] = math.isclose(
                float(actual.get("hard_head_expectation", math.nan)), expectation, abs_tol=1e-6
            )
            checks["argmax"] = actual.get("hard_head_argmax") == int(probabilities.argmax())
        if not all(checks.values()):
            raise RuntimeError(
                f"frozen test prediction metadata mismatch at row {index}: {checks}"
            )


def read_confirmatory_predictions(
    path: Path,
    expected: list[dict[str, Any]],
    *,
    variant: str,
    seed: int,
    checkpoint_state_sha256: str,
    expected_file_sha256: str,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_file_sha256:
        raise RuntimeError(f"test prediction hash mismatch: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    validate_prediction_rows(
        rows,
        expected,
        variant=variant,
        seed=seed,
        checkpoint_state_sha256=checkpoint_state_sha256,
    )
    return rows


def analyze_frozen_test_once() -> dict[str, Any]:
    if CONFIRMATION_DECISION.exists():
        raise FileExistsError("Exp61 confirmatory decision already exists")
    source_lock = verify_source_lock(require_formal_authorization=True)
    marker = json.loads(ACCESS_MARKER.read_text(encoding="utf-8"))
    if marker.get("status") != "EXP61_TEST_ACCESS_COMPLETE" or marker.get("test_access_count") != 1:
        raise RuntimeError("Exp61 one-shot test access did not complete exactly once")
    protocol_sha256 = sha256_file(FROZEN_PROTOCOL)
    source_lock_sha256 = sha256_file(SOURCE_LOCK)
    if marker.get("protocol_sha256") != protocol_sha256 or marker.get("source_lock_sha256") != source_lock_sha256:
        raise RuntimeError("test access marker differs from current frozen protocol/source lock")
    checkpoints = validate_checkpoint_grid(
        protocol_sha256=protocol_sha256,
        source_lock_sha256=source_lock_sha256,
        model_manifest_sha256=source_lock["model"]["manifest_sha256"],
        mapping_semantic_sha256=source_lock["mapping_semantic_sha256"],
    )
    if marker.get("checkpoints") != checkpoints:
        raise RuntimeError("test access checkpoint grid changed after prediction")
    expected = frozen_test_metadata()
    if len(expected) != 1578:
        raise RuntimeError("frozen manifest test size mismatch")

    metrics: dict[str, dict[int, dict[str, float]]] = {}
    point_predictions: dict[str, dict[int, np.ndarray]] = {}
    canonical_rows: list[dict[str, Any]] | None = None
    for variant in VARIANTS:
        metrics[variant], point_predictions[variant] = {}, {}
        for seed in SEEDS:
            rows = read_confirmatory_predictions(
                PREDICTION_ROOT / variant / f"seed_{seed}.jsonl",
                expected,
                variant=variant,
                seed=seed,
                checkpoint_state_sha256=checkpoints[variant][str(seed)]["state_dict_sha256"],
                expected_file_sha256=marker["prediction_sha256"][variant][str(seed)],
            )
            if canonical_rows is None:
                canonical_rows = rows
            probabilities = np.asarray([row["hard_head_probabilities"] for row in rows])
            means = np.asarray([row["human_mean"] for row in rows])
            labels = np.asarray([row["hard_label"] for row in rows])
            point_predictions[variant][seed] = probabilities @ np.arange(6)
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
        "status": "EXP61_FROZEN_TEST_CONFIRMATION_COMPLETE",
        "decision": gates["decision"],
        "metrics": {
            variant: {str(seed): values for seed, values in by_seed.items()}
            for variant, by_seed in metrics.items()
        },
        "component_cluster_bootstrap": bootstrap,
        "confirmation": gates,
        "protocol_sha256": protocol_sha256,
        "source_lock_sha256": source_lock_sha256,
        "model_manifest_sha256": source_lock["model"]["manifest_sha256"],
        "test_access_count": 1,
    }


def main() -> None:
    result = analyze_frozen_test_once()
    with CONFIRMATION_DECISION.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
