#!/usr/bin/env python3
"""Paired bootstrap analysis for the frozen Exp33B calibration audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(
    "thesis_exp/exp33_direction_aware_aggregation/outputs/"
    "exp33b_direction_aware_aggregation_seed42"
)
COMPARISONS = (("DRGA", "rounded_human"), ("DRGA", "qwen"))


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def weighted_mae(rows: list[dict[str, Any]], indexes: list[int]) -> float:
    total = sum(rows[index]["weight"] for index in indexes)
    return sum(
        rows[index]["weight"]
        * abs(rows[index]["prediction_soft"] - rows[index]["reference"])
        for index in indexes
    ) / total


def weighted_rate(
    rows: list[dict[str, Any]], indexes: list[int], predicate: Callable[[int, int], bool]
) -> float:
    total = sum(rows[index]["weight"] for index in indexes)
    return sum(
        rows[index]["weight"]
        * predicate(rows[index]["prediction_hard"], rows[index]["reference"])
        for index in indexes
    ) / total


def weighted_qwk(rows: list[dict[str, Any]], indexes: list[int]) -> float:
    confusion = [[0.0] * 5 for _ in range(5)]
    observed = [0.0] * 5
    expected = [0.0] * 5
    total = 0.0
    for index in indexes:
        row = rows[index]
        prediction = int(row["prediction_hard"]) - 1
        reference = int(row["reference"]) - 1
        weight = float(row["weight"])
        confusion[reference][prediction] += weight
        observed[reference] += weight
        expected[prediction] += weight
        total += weight
    if total <= 0:
        return float("nan")
    numerator = 0.0
    denominator = 0.0
    for reference in range(5):
        for prediction in range(5):
            ordinal_weight = ((reference - prediction) ** 2) / 16.0
            numerator += ordinal_weight * confusion[reference][prediction]
            denominator += ordinal_weight * observed[reference] * expected[prediction] / total
    return 1.0 - numerator / denominator if denominator > 0 else float("nan")


METRICS: dict[str, tuple[Callable[[list[dict[str, Any]], list[int]], float], str]] = {
    "mae": (weighted_mae, "lower"),
    "qwk": (weighted_qwk, "higher"),
    "low_to_high": (
        lambda rows, indexes: weighted_rate(
            rows, indexes, lambda prediction, reference: reference <= 2 and prediction >= 4
        ),
        "lower",
    ),
    "high_to_low": (
        lambda rows, indexes: weighted_rate(
            rows, indexes, lambda prediction, reference: reference >= 4 and prediction <= 2
        ),
        "lower",
    ),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_methods(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        by_method[row["method"]].append(
            {
                "sample_id": row["sample_id"],
                "prediction_soft": float(row["score_prediction"]),
                "prediction_hard": int(float(row["hard_label"])),
                "reference": int(float(row["reference_point"])),
                "weight": float(row["design_weight"]),
            }
        )
    return by_method


def paired_rows(
    by_method: dict[str, list[dict[str, Any]]], method: str, baseline: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = {row["sample_id"]: row for row in by_method[method]}
    right = {row["sample_id"]: row for row in by_method[baseline]}
    identities = sorted(set(left) & set(right))
    if len(identities) != 120:
        raise ValueError(f"Expected 120 paired rows for {method} vs {baseline}, got {len(identities)}")
    return [left[sid] for sid in identities], [right[sid] for sid in identities]


def bootstrap(
    method_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    full = list(range(len(method_rows)))
    output: list[dict[str, Any]] = []
    for metric, (function, preferred) in METRICS.items():
        method_value = function(method_rows, full)
        baseline_value = function(baseline_rows, full)
        deltas: list[float] = []
        for _ in range(iterations):
            indexes = [rng.randrange(len(full)) for _ in full]
            deltas.append(function(method_rows, indexes) - function(baseline_rows, indexes))
        lower = percentile(deltas, 0.025)
        upper = percentile(deltas, 0.975)
        if preferred == "lower":
            improvement_probability = sum(value < 0 for value in deltas) / len(deltas)
        else:
            improvement_probability = sum(value > 0 for value in deltas) / len(deltas)
        output.append(
            {
                "method": "DRGA",
                "baseline": baseline_rows[0].get("method_name", ""),
                "metric": metric,
                "preferred_direction": preferred,
                "paired_rows": len(full),
                "method_value": method_value,
                "baseline_value": baseline_value,
                "delta_method_minus_baseline": method_value - baseline_value,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "bootstrap_improvement_probability": improvement_probability,
                "bootstrap_iterations": iterations,
                "seed": seed,
            }
        )
    return output


def write_plot(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    mae_rows = [row for row in rows if row["metric"] == "mae"]
    labels = [f"DRGA vs {row['baseline']}" for row in mae_rows]
    values = [float(row["delta_method_minus_baseline"]) for row in mae_rows]
    lower = [value - float(row["bootstrap_ci_lower"]) for value, row in zip(values, mae_rows)]
    upper = [float(row["bootstrap_ci_upper"]) - value for value, row in zip(values, mae_rows)]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#2f6f4e" if value < 0 else "#b54335" for value in values]
    axis.barh(labels, values, color=colors, xerr=[lower, upper], capsize=4)
    axis.axvline(0.0, color="#222222", linewidth=1)
    axis.set_xlabel("Paired MAE difference (DRGA minus baseline)")
    axis.set_title("Exp33B representative calibration: paired bootstrap 95% CI")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    private_predictions = args.out_dir / "private/exp33b_representative_crossfit_predictions.csv"
    by_method = load_methods(private_predictions)
    rows: list[dict[str, Any]] = []
    for method, baseline in COMPARISONS:
        method_rows, baseline_rows = paired_rows(by_method, method, baseline)
        for row in baseline_rows:
            row["method_name"] = baseline
        rows.extend(bootstrap(method_rows, baseline_rows, args.iterations, args.seed))
    fields = list(rows[0])
    table = args.out_dir / "tables/exp33b_paired_bootstrap.csv"
    write_csv(table, rows, fields)
    plot_created = write_plot(rows, args.out_dir / "figures/exp33b_mae_bootstrap.png")
    summary = {
        "experiment": "Exp33B paired bootstrap calibration audit",
        "iterations": args.iterations,
        "seed": args.seed,
        "paired_rows": 120,
        "comparisons": [f"{left}_vs_{right}" for left, right in COMPARISONS],
        "plot_created": plot_created,
        "dev_rows_read": 0,
        "test_access_count": 0,
    }
    destination = repo_path(args.out_dir / "decision/exp33b_statistical_audit.json")
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
