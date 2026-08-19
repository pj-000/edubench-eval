#!/usr/bin/env python3
"""Build the complete, sign-consistent Exp51 report from frozen test outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/exp51_hmsa/final_test"
LOCK_PATH = ROOT / "configs/exp51_hmsa/final_test_lock.json"
PROTOCOL_PATH = ROOT / "configs/exp51_hmsa/formal_protocol.json"
INTEGRITY_PATH = (
    ROOT
    / "outputs/exp51_hmsa/audit/six_checkpoint_dev_reload_integrity.json"
)
ARMS = ("b0", "exp51")
SEEDS = (42, 43, 44)
EXCLUDED_AGGREGATE_KEYS = {
    "n",
    "seed",
    "selected_epoch",
    "test_access_count",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_vector_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['record_id']}\t{row['pred_label_5']}" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key not in EXCLUDED_AGGREGATE_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        "mean": {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        },
        "standard_deviation": {
            key: float(np.std([row[key] for row in rows], ddof=1))
            for key in keys
        },
    }


def paired_mae_deltas(
    predictions: dict[tuple[str, int], list[dict[str, Any]]],
) -> np.ndarray:
    seed_deltas = []
    for seed in SEEDS:
        b0 = {str(row["record_id"]): row for row in predictions[("b0", seed)]}
        exp51 = {
            str(row["record_id"]): row
            for row in predictions[("exp51", seed)]
        }
        if set(b0) != set(exp51) or len(b0) != 2218:
            raise ValueError(f"Prediction ID mismatch for seed {seed}")
        seed_deltas.append(
            np.asarray(
                [
                    abs(
                        float(exp51[key]["pred_label_5"])
                        - float(b0[key]["human_mean_5"])
                    )
                    - abs(
                        float(b0[key]["pred_label_5"])
                        - float(b0[key]["human_mean_5"])
                    )
                    for key in sorted(b0)
                ],
                dtype=float,
            )
        )
    return np.stack(seed_deltas, axis=0).mean(axis=0)


def main() -> int:
    state_path = OUTPUT_ROOT / "campaign_state.json"
    frozen_summary_path = OUTPUT_ROOT / "final_test_summary.json"
    report_path = OUTPUT_ROOT / "canonical_final_test_report.json"
    sidecar_path = OUTPUT_ROOT / "canonical_final_test_report.sha256"

    state = read_json(state_path)
    lock = read_json(LOCK_PATH)
    protocol = read_json(PROTOCOL_PATH)
    integrity = read_json(INTEGRITY_PATH)
    if state.get("status") != "TEST_COMPLETED":
        raise PermissionError("Final test campaign is not complete")
    if int(state.get("test_access_count", -1)) != 1:
        raise PermissionError("Final test access count is not exactly one")
    if sha256(frozen_summary_path) != state.get("summary_sha256"):
        raise ValueError("Frozen final summary hash mismatch")
    if not integrity.get("all_six_passed"):
        raise ValueError("Six-checkpoint dev reload integrity did not pass")

    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    artifact_hashes: dict[str, Any] = {}
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = OUTPUT_ROOT / arm / f"seed_{seed}"
            metrics_path = run_dir / "test_metrics.json"
            predictions_path = run_dir / "predictions_test.jsonl"
            current_metrics = read_json(metrics_path)
            current_predictions = read_jsonl(predictions_path)
            if int(current_metrics["n"]) != 2218:
                raise ValueError(f"Unexpected row count for {arm} seed {seed}")
            if len(current_predictions) != 2218:
                raise ValueError(
                    f"Unexpected prediction count for {arm} seed {seed}"
                )
            if int(current_metrics["test_access_count"]) != 1:
                raise ValueError(f"Access count mismatch for {arm} seed {seed}")
            if (
                current_metrics["test_file_sha256"]
                != state["test_anchor"]["sha256"]
            ):
                raise ValueError(f"Test hash mismatch for {arm} seed {seed}")
            metrics[(arm, seed)] = current_metrics
            predictions[(arm, seed)] = current_predictions
            artifact_hashes[f"{arm}/seed_{seed}"] = {
                "checkpoint_sha256": current_metrics["checkpoint_sha256"],
                "metrics_json_sha256": sha256(metrics_path),
                "predictions_jsonl_sha256": sha256(predictions_path),
                "hard_prediction_vector_sha256": prediction_vector_sha256(
                    current_predictions
                ),
                "selected_epoch": int(current_metrics["selected_epoch"]),
            }

    arm_metrics = {
        arm: [numeric_metrics(metrics[(arm, seed)]) for seed in SEEDS]
        for arm in ARMS
    }
    arm_aggregates = {
        arm: aggregate(rows) for arm, rows in arm_metrics.items()
    }
    common_keys = sorted(
        set.intersection(
            *(set(numeric_metrics(value)) for value in metrics.values())
        )
    )

    per_seed = []
    delta_rows: list[dict[str, float]] = []
    for seed in SEEDS:
        b0 = numeric_metrics(metrics[("b0", seed)])
        exp51 = numeric_metrics(metrics[("exp51", seed)])
        deltas = {key: exp51[key] - b0[key] for key in common_keys}
        delta_rows.append(deltas)
        per_seed.append(
            {
                "seed": seed,
                "selected_epoch": {
                    "b0": int(metrics[("b0", seed)]["selected_epoch"]),
                    "exp51": int(
                        metrics[("exp51", seed)]["selected_epoch"]
                    ),
                },
                "b0": b0,
                "exp51": exp51,
                "delta_exp51_minus_b0": deltas,
            }
        )
    delta_aggregate = aggregate(delta_rows)

    row_deltas = paired_mae_deltas(predictions)
    rng = np.random.default_rng(int(protocol["bootstrap_rng_seed"]))
    bootstrap = np.empty(int(protocol["bootstrap_resamples"]))
    for index in range(len(bootstrap)):
        sample = rng.integers(0, len(row_deltas), len(row_deltas))
        bootstrap[index] = float(row_deltas[sample].mean())
    bootstrap_report = {
        "convention": "Exp51_minus_B0; negative MAE is better",
        "mean_delta_mae": float(row_deltas.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "resamples": len(bootstrap),
        "rng_seed": int(protocol["bootstrap_rng_seed"]),
        "unit": "test row after averaging the three seed-level deltas",
    }

    mean_delta = delta_aggregate["mean"]
    mean_b0_bias = arm_aggregates["b0"]["mean"]["Bias_human_mean"]
    mean_exp51_bias = arm_aggregates["exp51"]["mean"][
        "Bias_human_mean"
    ]
    mean_abs_bias_improvement = abs(mean_b0_bias) - abs(mean_exp51_bias)
    seed_mae_improvements = sum(
        row["MAE_human_mean"] < 0 for row in delta_rows
    )
    exact_noninferior_seeds = sum(
        round(
            metrics[("exp51", seed)]["Exact_rounded"]
            * metrics[("exp51", seed)]["n"]
        )
        - round(
            metrics[("b0", seed)]["Exact_rounded"]
            * metrics[("b0", seed)]["n"]
        )
        >= -int(protocol["exact_max_correct_drop_per_seed"])
        for seed in SEEDS
    )
    checks = {
        "finite": all(
            math.isfinite(value)
            for arm in ARMS
            for row in arm_metrics[arm]
            for value in row.values()
        ),
        "mean_mae_improvement_at_least_0p005": (
            mean_delta["MAE_human_mean"]
            <= -float(protocol["mean_mae_improvement_min"])
        ),
        "mae_improves_two_of_three": (
            seed_mae_improvements >= int(protocol["mae_improves_min_seeds"])
        ),
        "mean_exact_delta_noninferior": (
            mean_delta["Exact_rounded"]
            >= float(protocol["mean_exact_delta_min"])
        ),
        "exact_noninferior_two_of_three": (
            exact_noninferior_seeds
            >= int(protocol["exact_noninferior_min_seeds"])
        ),
        "mean_kendall_delta_noninferior": (
            mean_delta["Kendall_human_mean"]
            >= float(protocol["mean_kendall_delta_min"])
        ),
        "mean_bias_or_l2h_improves": (
            mean_abs_bias_improvement
            >= float(protocol["mean_abs_bias_improvement_min"])
            or mean_delta["L2H_count"]
            <= -float(protocol["mean_l2h_improvement_alternative"])
        ),
        "mean_l2h_does_not_increase": (
            mean_delta["L2H_count"]
            <= float(protocol["mean_l2h_delta_max"])
        ),
        "mean_label4_recall_delta_noninferior": (
            mean_delta["Recall_4_rounded"]
            >= float(protocol["mean_label4_recall_delta_min"])
        ),
        "mean_label5_recall_delta_noninferior": (
            mean_delta["Recall_5_rounded"]
            >= float(protocol["mean_label5_recall_delta_min"])
        ),
        "paired_mae_ci_upper_below_zero": (
            bootstrap_report["ci_high"]
            < float(protocol["paired_mae_ci_high_strict_max"])
        ),
    }
    passed = all(checks.values())

    report = {
        "status": "EXP51_TEST_SUCCESS" if passed else "EXP51_TEST_MIXED",
        "passed_all_frozen_gates": passed,
        "delta_convention": (
            "Every paired delta is Exp51 minus B0; negative MAE/Bias/L2H "
            "and positive Exact/Kendall are improvements."
        ),
        "source_summary_note": (
            "The frozen evaluator summary is preserved unchanged. This "
            "post-campaign report only expands metrics, normalizes delta "
            "signs, and hashes artifacts; it does not rerun inference or "
            "access the raw test split."
        ),
        "test_access_count": int(state["test_access_count"]),
        "test_anchor": state["test_anchor"],
        "checks_using_frozen_formal_protocol": checks,
        "direction_counts": {
            "MAE_human_mean_improved": f"{seed_mae_improvements}/3",
            "Exact_rounded_improved": (
                f"{sum(row['Exact_rounded'] > 0 for row in delta_rows)}/3"
            ),
            "Kendall_human_mean_improved": (
                f"{sum(row['Kendall_human_mean'] > 0 for row in delta_rows)}/3"
            ),
        },
        "per_seed": per_seed,
        "arm_aggregates": arm_aggregates,
        "paired_delta_exp51_minus_b0": delta_aggregate,
        "mean_abs_bias_improvement_b0_minus_exp51": (
            mean_abs_bias_improvement
        ),
        "paired_mae_bootstrap": bootstrap_report,
        "artifact_hashes": artifact_hashes,
        "provenance_hashes": {
            "campaign_state_sha256": sha256(state_path),
            "frozen_final_test_summary_sha256": sha256(
                frozen_summary_path
            ),
            "final_test_lock_sha256": sha256(LOCK_PATH),
            "formal_protocol_sha256": sha256(PROTOCOL_PATH),
            "six_checkpoint_dev_reload_integrity_sha256": sha256(
                INTEGRITY_PATH
            ),
        },
        "final_lock_manifest_sha256": state[
            "final_lock_manifest_sha256"
        ],
    }
    write_json(report_path, report)
    report_sha256 = sha256(report_path)
    sidecar_path.write_text(
        f"{report_sha256}  {report_path.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed_all_frozen_gates": passed,
                "test_access_count": state["test_access_count"],
                "mean": {
                    "b0_MAE": arm_aggregates["b0"]["mean"][
                        "MAE_human_mean"
                    ],
                    "exp51_MAE": arm_aggregates["exp51"]["mean"][
                        "MAE_human_mean"
                    ],
                    "delta_MAE": mean_delta["MAE_human_mean"],
                    "b0_Exact": arm_aggregates["b0"]["mean"][
                        "Exact_rounded"
                    ],
                    "exp51_Exact": arm_aggregates["exp51"]["mean"][
                        "Exact_rounded"
                    ],
                    "delta_Exact": mean_delta["Exact_rounded"],
                    "delta_Kendall": mean_delta["Kendall_human_mean"],
                    "delta_Bias": mean_delta["Bias_human_mean"],
                    "delta_L2H_count": mean_delta["L2H_count"],
                    "delta_Recall_4": mean_delta["Recall_4_rounded"],
                    "delta_Recall_5": mean_delta["Recall_5_rounded"],
                },
                "paired_mae_bootstrap": bootstrap_report,
                "checks": checks,
                "report": str(report_path),
                "report_sha256": report_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
