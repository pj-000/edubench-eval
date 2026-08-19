"""Frozen seed-level analysis for Exp63."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp63_same_state_counterfactual import ARMS, OUTPUT_ROOT, SEEDS, STAGE_EPOCHS
from thesis_exp.exp63_same_state_counterfactual.runtime import load_protocol, write_json


def load_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for stage in STAGE_EPOCHS:
            path = root / "counterfactual" / f"seed_{seed}" / f"after_epoch_{stage}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "COMPLETED" or value.get("test_access_count") != 0:
                raise RuntimeError(f"Invalid result: {path}")
            rows.append(value)
    return rows


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def analyze(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    protocol = load_protocol()
    rows = load_results(root)
    by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        contrasts: dict[str, list[float]] = defaultdict(list)
        for row in seed_rows:
            arms = row["arms"]
            blocked = float(arms["blocked"]["post"]["eval_hard_ce_loss"])
            parallel = float(arms["parallel_only"]["post"]["eval_hard_ce_loss"])
            for arm in ARMS:
                value = float(arms[arm]["post"]["eval_hard_ce_loss"])
                contrasts[f"{arm}_minus_blocked"].append(value - blocked)
            orth = float(arms["orthogonal_only"]["post"]["eval_hard_ce_loss"])
            contrasts["orthogonal_minus_parallel"].append(orth - parallel)
        by_seed[seed] = {
            "stage_count": len(seed_rows),
            "stage_mean_contrasts": {key: mean(values) for key, values in contrasts.items()},
            "stage_contrasts": dict(contrasts),
        }
    primary_values = [
        by_seed[seed]["stage_mean_contrasts"]["orthogonal_only_minus_blocked"]
        for seed in SEEDS
    ]
    orth_parallel_values = [
        by_seed[seed]["stage_mean_contrasts"]["orthogonal_minus_parallel"]
        for seed in SEEDS
    ]
    full_values = [
        by_seed[seed]["stage_mean_contrasts"]["full_residual_minus_blocked"]
        for seed in SEEDS
    ]
    requirements = protocol["primary_estimand"]["directional_support_all_required"]
    primary = {
        "contrast": "orthogonal_only minus blocked post-update dev hard CE",
        "seed_values": primary_values,
        "mean": mean(primary_values),
        "favorable_seeds": sum(value < 0.0 for value in primary_values),
    }
    primary["pass"] = (
        primary["mean"] <= requirements["mean_delta_at_most"]
        and primary["favorable_seeds"] >= requirements["favorable_seeds_at_least"]
    )
    orth_vs_parallel = {
        "seed_values": orth_parallel_values,
        "mean": mean(orth_parallel_values),
        "favorable_seeds": sum(value < 0.0 for value in orth_parallel_values),
        "pass_by_primary_thresholds": (
            mean(orth_parallel_values) <= requirements["mean_delta_at_most"]
            and sum(value < 0.0 for value in orth_parallel_values)
            >= requirements["favorable_seeds_at_least"]
        ),
    }
    full = {
        "seed_values": full_values,
        "mean": mean(full_values),
        "favorable_seeds": sum(value < 0.0 for value in full_values),
    }
    if primary["pass"] and orth_vs_parallel["pass_by_primary_thresholds"]:
        decision = "DIRECTION_SPECIFIC_ORTHOGONAL_SUPPORT"
    elif full["mean"] <= requirements["mean_delta_at_most"] and full["favorable_seeds"] >= 4:
        decision = "FULL_RESIDUAL_ONLY_SUPPORT"
    else:
        decision = "NO_IDENTIFIED_IMMEDIATE_SAME_STATE_DIRECTION_EFFECT"
    result = {
        "status": "COMPLETED",
        "decision": decision,
        "seed_is_primary_unit": True,
        "checkpoint_units": len(rows),
        "seed_units": len(SEEDS),
        "primary": primary,
        "orthogonal_vs_parallel": orth_vs_parallel,
        "full_residual": full,
        "by_seed": by_seed,
        "test_access_count": 0,
    }
    write_json(root / "decision" / "canonical_results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))

