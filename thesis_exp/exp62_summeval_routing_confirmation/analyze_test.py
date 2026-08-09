"""Frozen paired seed and source-cluster analysis for Exp62 test predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from thesis_exp.exp62_summeval_routing_confirmation import SEEDS, VARIANTS


BASELINE = "direct_residual_blocked"
COMPARISONS = ("routed_hmsa", "orthogonal_only", "parallel_only")


def _prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["record_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate Exp62 test prediction record")
    return result


def paired_analysis(
    predictions: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    n_resamples: int = 10_000,
    rng_seed: int = 620062,
) -> dict[str, Any]:
    if set(predictions) != set(VARIANTS):
        raise ValueError("Exp62 test analysis requires all four arms")
    if any(set(values) != set(SEEDS) for values in predictions.values()):
        raise ValueError("Exp62 test analysis requires all five seeds per arm")
    mapped = {
        variant: {seed: _prediction_map(rows) for seed, rows in values.items()}
        for variant, values in predictions.items()
    }
    record_ids = sorted(mapped[BASELINE][SEEDS[0]])
    for variant in VARIANTS:
        for seed in SEEDS:
            if sorted(mapped[variant][seed]) != record_ids:
                raise ValueError("Exp62 paired test record IDs differ")
    comparisons: dict[str, Any] = {}
    rng = np.random.default_rng(rng_seed)
    for variant in COMPARISONS:
        seed_deltas: list[float] = []
        per_seed_record: list[np.ndarray] = []
        groups: list[str] | None = None
        for seed in SEEDS:
            deltas: list[float] = []
            current_groups: list[str] = []
            for record_id in record_ids:
                candidate = mapped[variant][seed][record_id]
                baseline = mapped[BASELINE][seed][record_id]
                if (
                    candidate["group_id"] != baseline["group_id"]
                    or candidate["human_mean"] != baseline["human_mean"]
                ):
                    raise ValueError("Exp62 paired prediction metadata mismatch")
                truth = float(candidate["human_mean"])
                deltas.append(
                    abs(float(candidate["hard_head_expectation"]) - truth)
                    - abs(float(baseline["hard_head_expectation"]) - truth)
                )
                current_groups.append(str(candidate["group_id"]))
            values = np.asarray(deltas, dtype=np.float64)
            per_seed_record.append(values)
            seed_deltas.append(float(values.mean()))
            if groups is None:
                groups = current_groups
            elif groups != current_groups:
                raise ValueError("Exp62 group order differs across seeds")
        assert groups is not None
        record_delta = np.stack(per_seed_record).mean(axis=0)
        group_indices: defaultdict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(groups):
            group_indices[group].append(index)
        group_names = sorted(group_indices)
        replicates = np.empty(n_resamples, dtype=np.float64)
        for draw in range(n_resamples):
            sampled = rng.integers(0, len(group_names), size=len(group_names))
            indices = [
                index
                for group_index in sampled
                for index in group_indices[group_names[int(group_index)]]
            ]
            replicates[draw] = float(record_delta[indices].mean())
        lower, upper = np.quantile(replicates, [0.025, 0.975])
        comparisons[f"{variant}_minus_{BASELINE}"] = {
            "seed_deltas_mae": dict(zip(map(str, SEEDS), seed_deltas)),
            "mean_seed_delta_mae": float(np.mean(seed_deltas)),
            "sample_sd_seed_delta_mae": float(np.std(seed_deltas, ddof=1)),
            "source_cluster_bootstrap": {
                "groups": len(group_names),
                "records": len(record_ids),
                "resamples": n_resamples,
                "seed": rng_seed,
                "mean_delta_mae": float(record_delta.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
            },
        }
    full = comparisons[f"routed_hmsa_minus_{BASELINE}"]
    orthogonal = comparisons[f"orthogonal_only_minus_{BASELINE}"]
    parallel = comparisons[f"parallel_only_minus_{BASELINE}"]
    gates = {
        "full_routing_confirmed": full["mean_seed_delta_mae"] < 0
        and full["source_cluster_bootstrap"]["ci95_upper"] < 0,
        "orthogonal_component_confirmed": orthogonal["mean_seed_delta_mae"] < 0
        and orthogonal["source_cluster_bootstrap"]["ci95_upper"] < 0,
        "parallel_not_reproduced": parallel["mean_seed_delta_mae"] >= 0
        or parallel["source_cluster_bootstrap"]["ci95_lower"] <= 0
        <= parallel["source_cluster_bootstrap"]["ci95_upper"],
    }
    return {"comparisons": comparisons, "interpretation_gates": gates}

