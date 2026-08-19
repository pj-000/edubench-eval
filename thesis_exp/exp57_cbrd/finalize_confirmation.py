"""Finalize the frozen five-seed Exp57 primary-pair confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd import CONFIG_ROOT, OUTPUT_ROOT
from thesis_exp.exp57_cbrd.confirmation_analysis import (
    METRIC_KEYS,
    _cluster_bootstrap,
    _paired_row_deltas,
    read_json,
    run_dir,
)
from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("consensus_only", "routed_hmsa")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_confirmation() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        metrics = {
            variant: read_json(run_dir(variant, seed) / "selected_dev_metrics.json")
            for variant in VARIANTS
        }
        row: dict[str, Any] = {"seed": seed}
        for key in METRIC_KEYS:
            row[f"consensus_{key}"] = float(metrics["consensus_only"][key])
            row[f"routed_{key}"] = float(metrics["routed_hmsa"][key])
            row[f"delta_{key}"] = (
                float(metrics["routed_hmsa"][key])
                - float(metrics["consensus_only"][key])
            )
        rows.append(row)
    mean_delta = {
        key: float(np.mean([row[f"delta_{key}"] for row in rows]))
        for key in METRIC_KEYS
    }
    checks = {
        "at_least_four_of_five_mae_favorable": sum(
            row["delta_MAE_human_mean"] < 0.0 for row in rows
        )
        >= 4,
        "mean_delta_mae_at_most_minus_0_005": mean_delta["MAE_human_mean"] <= -0.005,
        "mean_delta_exact_at_least_minus_0_003": mean_delta["Exact_rounded"] >= -0.003,
        "mean_delta_kendall_at_least_minus_0_005": mean_delta["Kendall_human_mean"] >= -0.005,
    }
    write_csv(OUTPUT_ROOT / "tables" / "stage1_confirmation_selected_metrics.csv", rows)
    return {"rows": rows, "mean_delta": mean_delta, "checks": checks}


def common_epoch_confirmation() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        histories = {
            variant: {
                int(row["epoch"]): row
                for row in read_json(run_dir(variant, seed) / "dev_metrics_history.json")
            }
            for variant in VARIANTS
        }
        if any(set(history) != set(range(1, 11)) for history in histories.values()):
            raise RuntimeError(f"Seed {seed} is missing a complete 10-epoch history")
        seed_rows: list[dict[str, Any]] = []
        for epoch in range(1, 11):
            row: dict[str, Any] = {"seed": seed, "epoch": epoch}
            for key in METRIC_KEYS:
                row[f"delta_{key}"] = (
                    float(histories["routed_hmsa"][epoch][key])
                    - float(histories["consensus_only"][epoch][key])
                )
            rows.append(row)
            seed_rows.append(row)
        mae = np.asarray([row["delta_MAE_human_mean"] for row in seed_rows])
        per_seed.append(
            {
                "seed": seed,
                "favorable_mae_epochs": int(np.sum(mae < 0.0)),
                "favorable_mae_fraction": float(np.mean(mae < 0.0)),
                "epoch10_delta_mae": float(mae[-1]),
                "normalized_trapezoidal_mae_curve_delta": float(
                    np.trapezoid(mae, dx=1.0) / 9.0
                ),
            }
        )
    all_mae = np.asarray([row["delta_MAE_human_mean"] for row in rows])
    write_csv(OUTPUT_ROOT / "tables" / "stage1_confirmation_common_epoch.csv", rows)
    report = {
        "per_seed": per_seed,
        "aggregate": {
            "favorable_seed_epoch_cells": int(np.sum(all_mae < 0.0)),
            "total_seed_epoch_cells": int(len(all_mae)),
            "favorable_seed_epoch_fraction": float(np.mean(all_mae < 0.0)),
            "mean_common_epoch_delta_mae": float(np.mean(all_mae)),
            "mean_epoch10_delta_mae": float(
                np.mean([row["epoch10_delta_mae"] for row in per_seed])
            ),
            "mean_normalized_trapezoidal_mae_curve_delta": float(
                np.mean([row["normalized_trapezoidal_mae_curve_delta"] for row in per_seed])
            ),
        },
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "stage1_confirmation_common_epoch.json", report)
    return report


def five_seed_cluster_bootstrap() -> dict[str, Any]:
    by_seed = {seed: _paired_row_deltas(seed) for seed in SEEDS}
    indexed = {
        seed: {row["record_id"]: row for row in rows}
        for seed, rows in by_seed.items()
    }
    shared_ids = set.intersection(*(set(rows) for rows in indexed.values()))
    if len(shared_ids) != 664:
        raise RuntimeError("Five-seed predictions do not share the same 664 records")
    averaged_rows = []
    for record_id in sorted(shared_ids):
        seed_rows = [indexed[seed][record_id] for seed in SEEDS]
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
        "per_seed": {
            str(seed): _cluster_bootstrap(rows, repetitions=10000, rng_seed=57 + seed)
            for seed, rows in by_seed.items()
        },
        "five_seed_mean_prediction_effect": _cluster_bootstrap(
            averaged_rows, repetitions=10000, rng_seed=57
        ),
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "stage1_confirmation_question_bootstrap.json", report)
    return report


def main() -> None:
    protocol_path = CONFIG_ROOT / "stage1_confirmation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP57_STAGE1_CONFIRMATION_FROZEN_BEFORE_SEEDS_45_46":
        raise RuntimeError("Confirmation protocol is not frozen")
    run_checks: dict[str, bool] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            summary = read_json(run_dir(variant, seed) / "run_summary.json")
            run_checks[f"{variant}_seed_{seed}"] = (
                summary.get("status") == "COMPLETED"
                and int(summary.get("test_access_count", -1)) == 0
                and int(summary.get("seed", -1)) == seed
                and summary.get("variant") == variant
            )
    selected = selected_confirmation()
    common_epoch = common_epoch_confirmation()
    bootstrap = five_seed_cluster_bootstrap()
    checks = {
        "all_ten_runs_complete_train_dev_only": all(run_checks.values()),
        **selected["checks"],
        "five_seed_bootstrap_ci_upper_below_zero": (
            bootstrap["five_seed_mean_prediction_effect"]["ci95_upper"] < 0.0
        ),
    }
    passed = all(checks.values())
    report = {
        "status": (
            "CONFIRMATION_PASS_INTERNAL_MECHANISM"
            if passed
            else "CONFIRMATION_NO_GO"
        ),
        "checks": checks,
        "run_checks": run_checks,
        "selected_comparison": selected,
        "common_epoch": common_epoch,
        "question_cluster_bootstrap": bootstrap,
        "confirmation_protocol_sha256": sha256_file(protocol_path),
        "interpretation": (
            "A pass confirms the primary routed-vs-consensus effect over five "
            "training seeds on the frozen EduBench development set. It does not "
            "establish external generalization, pure direction-only causality, or "
            "authorize a confirmatory claim on the historically used test set."
        ),
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "decision" / "stage1_confirmation_decision.json", report)
    write_text(
        OUTPUT_ROOT / "decision" / "stage1_confirmation_decision.md",
        "# Exp57 Stage 1 five-seed confirmation\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Mean delta MAE: `{selected['mean_delta']['MAE_human_mean']:.6f}`\n"
        f"- Mean delta Exact: `{selected['mean_delta']['Exact_rounded']:.6f}`\n"
        f"- Mean delta Kendall: `{selected['mean_delta']['Kendall_human_mean']:.6f}`\n"
        "- Historical test accessed: no\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
