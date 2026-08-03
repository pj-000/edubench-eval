"""CPU-only five-seed confirmation analysis for Exp59.

The frozen primary comparison is orthogonal_only minus the reused Exp57
consensus_only endpoint.  Parallel-only is reported as the secondary mechanism
comparison and cannot replace the primary result.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp59_residual_geometry import OUTPUT_ROOT, REPO_ROOT
from thesis_exp.src.edujudge.utils.io import write_csv


SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ("orthogonal_only", "parallel_only")
METRICS = ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")
CONSENSUS_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp57_cbrd" / "runs" / "consensus_only"
REPETITIONS = 10_000
RNG_SEED = 57


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def consensus_dir(seed: int) -> Path:
    return CONSENSUS_ROOT / f"seed_{seed}"


def predictions(directory: Path) -> Path:
    for candidate in (
        directory / "predictions_dev_best.jsonl",
        directory / "predictions" / "predictions_dev_best.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No selected dev predictions below {directory}")


def selected_metrics(directory: Path) -> dict[str, float]:
    summary = read_json(directory / "run_summary.json")
    completed_steps = summary.get("optimizer_steps", summary.get("clip_events"))
    if summary.get("status") != "COMPLETED" or completed_steps != 210:
        raise RuntimeError(f"Incomplete frozen endpoint: {directory}")
    if summary.get("test_access_count") != 0:
        raise RuntimeError(f"Nonzero test access: {directory}")
    return {key: float(summary["selected_metrics"][key]) for key in METRICS}


def metric_comparison(variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        treatment = selected_metrics(run_dir(variant, seed))
        consensus = selected_metrics(consensus_dir(seed))
        row: dict[str, Any] = {"seed": seed, "variant": variant}
        for metric in METRICS:
            row[f"treatment_{metric}"] = treatment[metric]
            row[f"consensus_{metric}"] = consensus[metric]
            row[f"delta_{metric}"] = treatment[metric] - consensus[metric]
        rows.append(row)
    mean_delta = {
        metric: float(np.mean([row[f"delta_{metric}"] for row in rows]))
        for metric in METRICS
    }
    checks = {
        "favorable_MAE_seeds_at_least_4": sum(
            row["delta_MAE_human_mean"] < 0.0 for row in rows
        )
        >= 4,
        "mean_delta_MAE_at_most_minus_0_005": mean_delta["MAE_human_mean"] <= -0.005,
        "mean_delta_Exact_at_least_minus_0_003": mean_delta["Exact_rounded"] >= -0.003,
        "mean_delta_Kendall_at_least_minus_0_005": mean_delta["Kendall_human_mean"] >= -0.005,
    }
    return rows, {"mean_delta": mean_delta, "checks_without_bootstrap": checks}


def paired_rows(variant: str, seed: int) -> list[dict[str, Any]]:
    treatment = {
        str(row["record_id"]): row
        for row in read_jsonl(predictions(run_dir(variant, seed)))
    }
    consensus = {
        str(row["record_id"]): row
        for row in read_jsonl(predictions(consensus_dir(seed)))
    }
    if treatment.keys() != consensus.keys() or len(treatment) != 664:
        raise RuntimeError(f"Prediction pairing failed for {variant} seed {seed}")
    rows: list[dict[str, Any]] = []
    for record_id in sorted(treatment):
        left = treatment[record_id]
        right = consensus[record_id]
        human_mean = float(left["human_mean_5"])
        if human_mean != float(right["human_mean_5"]):
            raise RuntimeError(f"Human target mismatch for {record_id}")
        rows.append(
            {
                "record_id": record_id,
                "question_key": str(
                    left.get("question_key") or left.get("triple_key") or record_id
                ),
                "delta_absolute_error": (
                    abs(float(left["pred_score_5"]) - human_mean)
                    - abs(float(right["pred_score_5"]) - human_mean)
                ),
            }
        )
    return rows


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, repetitions: int, rng_seed: int
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["question_key"])].append(float(row["delta_absolute_error"]))
    keys = sorted(clusters)
    sums = np.asarray([sum(clusters[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(keys), size=len(keys))
        samples[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "question_clusters": len(keys),
        "rows": int(counts.sum()),
        "observed_delta_mae": float(sums.sum() / counts.sum()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_delta_below_zero": float(np.mean(samples < 0.0)),
        "repetitions": repetitions,
        "rng_seed": rng_seed,
    }


def five_seed_bootstrap(variant: str) -> dict[str, Any]:
    by_seed = {seed: paired_rows(variant, seed) for seed in SEEDS}
    indexed = {
        seed: {row["record_id"]: row for row in rows}
        for seed, rows in by_seed.items()
    }
    shared_ids = set.intersection(*(set(rows) for rows in indexed.values()))
    if len(shared_ids) != 664:
        raise RuntimeError("Five seeds do not share the same 664 dev records")
    averaged: list[dict[str, Any]] = []
    for record_id in sorted(shared_ids):
        seed_rows = [indexed[seed][record_id] for seed in SEEDS]
        if len({row["question_key"] for row in seed_rows}) != 1:
            raise RuntimeError(f"Question-key mismatch for {record_id}")
        averaged.append(
            {
                "record_id": record_id,
                "question_key": seed_rows[0]["question_key"],
                "delta_absolute_error": float(
                    np.mean([row["delta_absolute_error"] for row in seed_rows])
                ),
            }
        )
    return {
        "comparison": f"{variant} minus consensus_only",
        "statistic": "five-seed mean paired hard-prediction absolute error against human_mean_5",
        "per_seed": {
            str(seed): cluster_bootstrap(
                rows, repetitions=REPETITIONS, rng_seed=RNG_SEED + seed
            )
            for seed, rows in by_seed.items()
        },
        "five_seed_mean_prediction_effect": cluster_bootstrap(
            averaged, repetitions=REPETITIONS, rng_seed=RNG_SEED
        ),
    }


def main() -> None:
    report: dict[str, Any] = {
        "status": "EXP59_CONFIRMATION_ANALYSIS_COMPLETE",
        "primary_comparison": "orthogonal_only minus consensus_only",
        "secondary_comparison": "parallel_only minus consensus_only",
        "seeds": list(SEEDS),
        "comparisons": {},
        "test_access_count": 0,
    }
    table_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows, comparison = metric_comparison(variant)
        table_rows.extend(rows)
        bootstrap = five_seed_bootstrap(variant)
        bootstrap_pass = (
            bootstrap["five_seed_mean_prediction_effect"]["ci95_upper"] < 0.0
        )
        checks = {
            **comparison["checks_without_bootstrap"],
            "five_seed_question_cluster_bootstrap_CI_upper_below_0": bootstrap_pass,
        }
        report["comparisons"][variant] = {
            "rows": rows,
            "mean_delta": comparison["mean_delta"],
            "bootstrap": bootstrap,
            "checks": checks,
            "passes_frozen_effect_gates": all(checks.values()),
        }
    primary_pass = report["comparisons"]["orthogonal_only"][
        "passes_frozen_effect_gates"
    ]
    secondary_pass = report["comparisons"]["parallel_only"][
        "passes_frozen_effect_gates"
    ]
    report["primary_pass"] = primary_pass
    report["secondary_pass"] = secondary_pass
    if primary_pass and secondary_pass:
        report["frozen_interpretation"] = "both_components_have_independent_value"
    elif primary_pass:
        report["frozen_interpretation"] = "non_collinear_residual_direction_supported"
    elif secondary_pass:
        report["frozen_interpretation"] = "common_aligned_or_clipping_dynamics_supported"
    else:
        report["frozen_interpretation"] = "neither_single_component_passes"
    write_csv(OUTPUT_ROOT / "tables" / "exp59_five_seed_comparison.csv", table_rows)
    write_json(OUTPUT_ROOT / "audit" / "exp59_five_seed_confirmation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
