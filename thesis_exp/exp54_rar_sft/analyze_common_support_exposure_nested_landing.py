"""Finite scientific-identification patch for the Exp54 residual-risk audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from thesis_exp.exp54_rar_sft.analyze_residual_risk_decomposition import (
    SEEDS,
    SCORES,
    _prediction_map,
    _probability_matrix,
    annotate_dev,
    read_json,
    read_jsonl,
    score_metrics,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "exp54-common-support-exposure-nested-landing-patch-v1"


def exposure_level(row: dict[str, Any]) -> str:
    if row["response_seen"]:
        return "E2"
    if row["question_seen"]:
        return "E1"
    return "E0"


def add_direct_adjacent_support(
    train: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], threshold: int = 20
) -> None:
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for row in train:
        counts[(row["metric_id"], row["language"])][int(row["label_5"])] += 1
    for row in rows:
        local = counts[(row["metric_id"], row["language"])]
        adjacent = [k for k in (row["label"] - 1, row["label"]) if 1 <= k <= 4]
        values = {str(k): min(local[k], local[k + 1]) for k in adjacent}
        row["direct_adjacent_support"] = values
        row["direct_adjacent_min_support"] = min(values.values())
        row["direct_adjacent_supported"] = all(value >= threshold for value in values.values())


def _stratum(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return row["metric_id"], row["language"], row["label"], row["ambiguity"]


def common_support_standardized_contrast(
    rows: Sequence[dict[str, Any]],
    outcomes: Sequence[float],
    *,
    left: str,
    right: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if len(rows) != len(outcomes):
        raise ValueError("row/outcome length differs")
    grouped: dict[tuple[str, str, int, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[_stratum(row)][exposure_level(row)].append(index)
    common = sorted(key for key, value in grouped.items() if value[left] and value[right])
    if not common:
        return {
            "identified": False,
            "reason": "no_common_support_strata",
            "common_strata": 0,
            "left_n": 0,
            "right_n": 0,
            "estimate": None,
            "ci95": [None, None],
            "p_two_sided": None,
        }
    counts = {key: len(grouped[key][left]) + len(grouped[key][right]) for key in common}
    total = sum(counts.values())
    weights = {key: counts[key] / total for key in common}
    outcome = np.asarray(outcomes, dtype=np.float64)

    def estimate(cluster_weights: dict[str, float] | None = None) -> float:
        value = 0.0
        for key in common:
            means = []
            for level in (left, right):
                indices = grouped[key][level]
                if cluster_weights is None:
                    means.append(float(outcome[indices].mean()))
                else:
                    row_weights = np.asarray([cluster_weights[rows[i]["question_key"]] for i in indices])
                    means.append(float(np.dot(outcome[indices], row_weights) / row_weights.sum()))
            value += weights[key] * (means[0] - means[1])
        return value

    point = estimate()
    questions = sorted({row["question_key"] for row in rows})
    rng = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        multipliers = rng.exponential(1.0, size=len(questions))
        sampled[replicate] = estimate(dict(zip(questions, multipliers, strict=True)))
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    less = int((sampled <= 0).sum())
    greater = int((sampled >= 0).sum())
    pvalue = min(1.0, 2 * min((less + 1) / (replicates + 1), (greater + 1) / (replicates + 1)))
    left_indices = sorted(i for key in common for i in grouped[key][left])
    right_indices = sorted(i for key in common for i in grouped[key][right])
    return {
        "identified": True,
        "left": left,
        "right": right,
        "common_strata": len(common),
        "left_n": len(left_indices),
        "right_n": len(right_indices),
        "labels_in_common_support": sorted({key[2] for key in common}),
        "estimate": float(point),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": float(pvalue),
        "weighting": "fixed_pooled_frequency_over_common_strata",
        "bootstrap": "question_cluster_exponential_multiplier",
    }


def nested_probability_transport(
    rows: Sequence[dict[str, Any]], r3: np.ndarray, p1: np.ndarray
) -> dict[str, Any]:
    levels: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all_low", lambda row: True),
        ("response_seen", lambda row: row["response_seen"]),
        ("H0_H1", lambda row: row["response_seen"] and row["ambiguity"] in {"H0", "H1"}),
        ("cumulative_supported", lambda row: row["response_seen"] and row["ambiguity"] in {"H0", "H1"} and row["supported"]),
        ("direct_adjacent_supported", lambda row: row["response_seen"] and row["ambiguity"] in {"H0", "H1"} and row["direct_adjacent_supported"]),
    ]
    delta = p1 - r3
    low_indices = [i for i, row in enumerate(rows) if row["label"] <= 2]
    result = {}
    for name, predicate in levels:
        base = [i for i in low_indices if predicate(rows[i])]
        selected = [i for i in base if -(delta[i, 3] + delta[i, 4]) > 0]
        if not selected:
            result[name] = {
                "base_low_n": len(base),
                "positive_delta_high_n": 0,
                "sum_delta_high": 0.0,
                "GLE": None,
                "MCR": None,
                "other_low_capture_rate": None,
                "identified": False,
            }
            continue
        vector = delta[selected].sum(axis=0)
        delta_high = float(-(vector[3] + vector[4]))
        delta_gold = float(sum(delta[i, rows[i]["label"] - 1] for i in selected))
        delta_other_low = float(sum(delta[i, 1 if rows[i]["label"] == 1 else 0] for i in selected))
        conservation = float(vector.sum())
        result[name] = {
            "base_low_n": len(base),
            "positive_delta_high_n": len(selected),
            "sum_delta_by_score": {str(score): float(vector[score - 1]) for score in SCORES},
            "mean_delta_by_score": {str(score): float(vector[score - 1] / len(selected)) for score in SCORES},
            "sum_delta_high": delta_high,
            "sum_delta_gold": delta_gold,
            "sum_delta_middle": float(vector[2]),
            "sum_delta_other_low": delta_other_low,
            "GLE": delta_gold / delta_high,
            "MCR": float(vector[2]) / delta_high,
            "other_low_capture_rate": delta_other_low / delta_high,
            "capture_partition_sum": (delta_gold + float(vector[2]) + delta_other_low) / delta_high,
            "probability_conservation_residual": conservation,
            "identified": True,
            "interpretation": "descriptive_only_not_mechanism_identification",
        }
    return result


def _prediction_path(root: Path, arm: str, seed: int) -> Path:
    if arm == "P1_FIELD_DPO":
        return root / "preference_dev/p1_field_dpo" / f"seed_{seed}/predictions.jsonl"
    return root / "dev_runs" / arm.lower() / f"seed{seed}/epoch3/predictions.jsonl"


def run_patch(
    private_root: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("patch protocol schema differs")
    if any("test" in part.lower() or "holdout" in part.lower() for part in private_root.parts):
        raise ValueError("test/holdout private root is forbidden")
    train_path, dev_path = private_root / "data/train.jsonl", private_root / "data/dev.jsonl"
    train, dev = read_jsonl(train_path), read_jsonl(dev_path)
    if len(train) != 2654 or len(dev) != 664:
        raise ValueError("locked train/dev counts differ")
    rows = annotate_dev(train, dev)
    add_direct_adjacent_support(train, rows)
    expected_ids = [row["record_id"] for row in rows]
    repetitions = bootstrap_replicates or int(protocol["common_support"]["replicates"])
    seed_base = int(protocol["common_support"]["seed"])
    predictions: dict[tuple[str, int], list[int]] = {}
    probabilities: dict[tuple[str, int], np.ndarray] = {}
    hashes = {"train": sha256_file(train_path), "dev": sha256_file(dev_path), "protocol": sha256_file(protocol_path)}
    for arm in ("S0", "R3", "P1_FIELD_DPO"):
        for seed in SEEDS:
            path = _prediction_path(private_root, arm, seed)
            mapping = _prediction_map(read_jsonl(path), expected_ids)
            predictions[arm, seed] = [mapping[record] for record in expected_ids]
            hashes[f"prediction:{arm}:{seed}"] = sha256_file(path)
    for arm, dirname in (("R3", "r3"), ("P1_FIELD_DPO", "p1_field_dpo")):
        for seed in SEEDS:
            path = private_root / "score_probabilities" / dirname / f"seed_{seed}/score_probabilities.jsonl"
            probabilities[arm, seed], _ = _probability_matrix(read_jsonl(path), expected_ids)
            hashes[f"probability:{arm}:{seed}"] = sha256_file(path)

    exposure = [exposure_level(row) for row in rows]
    exposure_by_label = {
        str(label): {level: sum(row["label"] == label and exposure_level(row) == level for row in rows) for level in ("E0", "E1", "E2")}
        for label in SCORES
    }
    support_by_label = {
        str(label): {
            "cumulative_supported": sum(row["label"] == label and row["supported"] for row in rows),
            "cumulative_unsupported": sum(row["label"] == label and not row["supported"] for row in rows),
            "direct_adjacent_supported": sum(row["label"] == label and row["direct_adjacent_supported"] for row in rows),
            "direct_adjacent_unsupported": sum(row["label"] == label and not row["direct_adjacent_supported"] for row in rows),
        }
        for label in SCORES
    }
    per_seed = {}
    exposure_passes = 0
    landing_stable_seeds = 0
    for seed in SEEDS:
        seed_report: dict[str, Any] = {"arms": {}}
        for arm in ("S0", "R3", "P1_FIELD_DPO"):
            pred = predictions[arm, seed]
            absolute_error = [abs(pred[i] - rows[i]["label"]) for i in range(len(rows))]
            seed_report["arms"][arm] = {
                "raw_by_exposure": {
                    level: score_metrics(
                        [rows[i]["label"] for i in range(len(rows)) if exposure[i] == level],
                        [pred[i] for i in range(len(rows)) if exposure[i] == level],
                    )
                    for level in ("E0", "E1", "E2")
                },
                "E1_minus_E2_common_support_MAE": common_support_standardized_contrast(
                    rows,
                    absolute_error,
                    left="E1",
                    right="E2",
                    replicates=repetitions,
                    seed=seed_base + seed * 19 + ("S0", "R3", "P1_FIELD_DPO").index(arm),
                ),
                "E0_minus_E2_common_support_MAE": common_support_standardized_contrast(
                    rows,
                    absolute_error,
                    left="E0",
                    right="E2",
                    replicates=repetitions,
                    seed=seed_base + seed * 23 + ("S0", "R3", "P1_FIELD_DPO").index(arm),
                ),
            }
        s0, r3, p1 = predictions["S0", seed], predictions["R3", seed], predictions["P1_FIELD_DPO", seed]
        sft_benefit = [abs(s0[i] - rows[i]["label"]) - abs(r3[i] - rows[i]["label"]) for i in range(len(rows))]
        dpo_benefit = [abs(r3[i] - rows[i]["label"]) - abs(p1[i] - rows[i]["label"]) for i in range(len(rows))]
        seed_report["paired_benefits"] = {
            "SFT_E1_minus_E2": common_support_standardized_contrast(rows, sft_benefit, left="E1", right="E2", replicates=repetitions, seed=seed_base + seed * 29),
            "DPO_E1_minus_E2": common_support_standardized_contrast(rows, dpo_benefit, left="E1", right="E2", replicates=repetitions, seed=seed_base + seed * 31),
        }
        seed_report["nested_probability_transport"] = nested_probability_transport(
            rows, probabilities["R3", seed], probabilities["P1_FIELD_DPO", seed]
        )
        exposure_result = seed_report["arms"]["R3"]["E1_minus_E2_common_support_MAE"]
        if exposure_result["identified"] and exposure_result["estimate"] >= 0.05 and exposure_result["ci95"][0] > 0:
            exposure_passes += 1
        landing = seed_report["nested_probability_transport"]["all_low"]
        if landing["identified"] and landing["sum_delta_high"] > 0 and landing["MCR"] >= 0.50 and landing["GLE"] <= 0.35:
            landing_stable_seeds += 1
        per_seed[str(seed)] = seed_report

    if exposure_passes >= 2:
        decision = "COLLECT_RESPONSE_DISJOINT_IDENTIFICATION_DATA"
    elif landing_stable_seeds == 3:
        decision = "COLLECT_LOW_BOUNDARY_IDENTIFICATION_DATA"
    else:
        decision = "WRITE_THESIS_NO_NEW_METHOD_ON_CURRENT_DATA"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "decision": decision,
        "scientific_direction_label": "D_NO_METHOD_DIRECTION_IDENTIFIED_ON_CURRENT_DATA",
        "data_scope": {
            "train_rows": len(train),
            "dev_rows": len(rows),
            "test_accessed": False,
            "new_training": False,
            "new_api_calls": False,
            "gpu_used": False,
            "input_sha256": hashes,
        },
        "inventory": {
            "exposure_counts": dict(Counter(exposure)),
            "exposure_by_label": exposure_by_label,
            "support_by_label": support_by_label,
            "E0_identifiable": Counter(exposure)["E0"] > 0,
            "low_score_exposure_identifiable": any(
                exposure_by_label[str(label)]["E1"] + exposure_by_label[str(label)]["E0"] > 0 for label in (1, 2)
            ),
        },
        "decision_evidence": {
            "R3_standardized_exposure_seed_passes": exposure_passes,
            "descriptive_landing_stable_seeds": landing_stable_seeds,
            "new_method_authorized": False,
        },
        "per_seed": per_seed,
        "interpretation": {
            "A": "Only E1-versus-E2 among labels 3/4/5 is estimable; E0 and low-score exposure are not identified.",
            "B": "Cumulative and direct-adjacent support remain positivity-limited; no causal support effect is estimated.",
            "C": "Nested low-score transport is descriptive and cannot distinguish support shortage from preference-objective mismatch.",
            "D": "No new method is authorized on the current artifacts; this does not assert that A/B/C mechanisms are ineffective.",
        },
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    report = run_patch(args.private_root, args.protocol, args.output, bootstrap_replicates=args.bootstrap_replicates)
    print(report["decision"])


if __name__ == "__main__":
    main()
