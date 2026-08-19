"""Pre-registered Exp57 pilot and three-seed development decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd import OUTPUT_ROOT, REPO_ROOT
from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


SCIENTIFIC_VARIANTS = (
    "dual_hard",
    "consensus_only",
    "routed_hmsa",
    "residual_only",
    "sign_flipped",
    "shuffled_residual",
)
SEEDS = (42, 43, 44)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def old_dir(kind: str, seed: int) -> Path:
    if kind == "hard_only":
        return REPO_ROOT / "thesis_exp" / "outputs" / "exp49_cphce" / "runs" / "b0_hard_ce" / f"seed_{seed}"
    if kind == "ordinary_hmsa":
        return REPO_ROOT / "thesis_exp" / "outputs" / "exp51_hmsa" / "runs" / "hmsa_lambda1" / f"seed_{seed}"
    raise ValueError(kind)


def metrics(directory: Path) -> dict[str, Any]:
    path = directory / "selected_dev_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def predictions(directory: Path) -> list[dict[str, Any]]:
    candidates = (
        directory / "predictions" / "predictions_dev.jsonl",
        directory / "predictions_dev.jsonl",
    )
    for path in candidates:
        if path.is_file():
            return read_jsonl(path)
    raise FileNotFoundError(f"No dev predictions below {directory}")


def prediction_agreement(left: Path, right: Path) -> float:
    left_rows = {str(row["record_id"]): int(row["pred_label_5"]) for row in predictions(left)}
    right_rows = {str(row["record_id"]): int(row["pred_label_5"]) for row in predictions(right)}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("Prediction record sets differ")
    return float(np.mean([left_rows[key] == right_rows[key] for key in sorted(left_rows)]))


def delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return float(left[key]) - float(right[key])


def pilot_decision() -> dict[str, Any]:
    seed = 42
    required = list(SCIENTIFIC_VARIANTS) + ["detached_soft"]
    summaries = {
        variant: read_json(run_dir(variant, seed) / "run_summary.json")
        for variant in required
    }
    if any(value.get("status") != "COMPLETED" for value in summaries.values()):
        raise RuntimeError("At least one pilot run is incomplete")
    current = {variant: metrics(run_dir(variant, seed)) for variant in required}
    hard = metrics(old_dir("hard_only", seed))
    hmsa = metrics(old_dir("ordinary_hmsa", seed))
    routed = current["routed_hmsa"]
    detached = current["detached_soft"]
    parity = {
        "routed_vs_ordinary_hmsa": {
            "abs_delta_mae": abs(delta(routed, hmsa, "MAE_human_mean")),
            "abs_delta_exact": abs(delta(routed, hmsa, "Exact_rounded")),
            "prediction_agreement": prediction_agreement(
                run_dir("routed_hmsa", seed), old_dir("ordinary_hmsa", seed)
            ),
        },
        "detached_vs_hard_only": {
            "abs_delta_mae": abs(delta(detached, hard, "MAE_human_mean")),
            "abs_delta_exact": abs(delta(detached, hard, "Exact_rounded")),
            "prediction_agreement": prediction_agreement(
                run_dir("detached_soft", seed), old_dir("hard_only", seed)
            ),
        },
    }
    parity_checks = {
        "routed_full_run_parity": (
            parity["routed_vs_ordinary_hmsa"]["abs_delta_mae"] <= 0.002
            and parity["routed_vs_ordinary_hmsa"]["abs_delta_exact"] <= 0.003
            and parity["routed_vs_ordinary_hmsa"]["prediction_agreement"] >= 0.99
        ),
        "detached_hard_path_parity": (
            parity["detached_vs_hard_only"]["abs_delta_mae"] <= 0.002
            and parity["detached_vs_hard_only"]["abs_delta_exact"] <= 0.003
            and parity["detached_vs_hard_only"]["prediction_agreement"] >= 0.99
        ),
    }
    catastrophic: dict[str, dict[str, Any]] = {}
    for variant in SCIENTIFIC_VARIANTS:
        mae_change = delta(current[variant], hard, "MAE_human_mean")
        exact_change = delta(current[variant], hard, "Exact_rounded")
        catastrophic[variant] = {
            "delta_mae_vs_hard_only": mae_change,
            "delta_exact_vs_hard_only": exact_change,
            "catastrophic": mae_change >= 0.02 and exact_change <= -0.02,
        }
    checks = {
        **parity_checks,
        "no_catastrophic_scientific_arm": not any(row["catastrophic"] for row in catastrophic.values()),
        "no_test_access": all(int(value.get("test_access_count", -1)) == 0 for value in summaries.values()),
    }
    passed = all(checks.values())
    return {
        "status": "PILOT_PASS_RUN_SEEDS_43_44" if passed else "PILOT_NO_GO",
        "seed": seed,
        "parity": parity,
        "catastrophic_screen": catastrophic,
        "checks": checks,
        "interpretation": "Seed 42 is an integrity/catastrophic screen only; it is not a mechanism-effect decision.",
        "test_access_count": 0,
    }


def paired_question_bootstrap(
    routed_rows: list[dict[str, Any]],
    flipped_rows: list[dict[str, Any]],
    *,
    repetitions: int = 10000,
    rng_seed: int = 57,
) -> dict[str, Any]:
    routed = {str(row["record_id"]): row for row in routed_rows}
    flipped = {str(row["record_id"]): row for row in flipped_rows}
    by_question: dict[str, list[float]] = defaultdict(list)
    for record_id in sorted(routed.keys() & flipped.keys()):
        left = routed[record_id]
        right = flipped[record_id]
        if left.get("minority_neighbor_advantage") is None:
            continue
        question = str(left.get("question_key") or left.get("triple_key") or record_id)
        by_question[question].append(
            float(left["minority_neighbor_advantage"])
            - float(right["minority_neighbor_advantage"])
        )
    cluster_values = np.asarray(
        [float(np.mean(values)) for _, values in sorted(by_question.items())],
        dtype=float,
    )
    if not len(cluster_values):
        raise RuntimeError("No disputed question clusters for sign bootstrap")
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draw = rng.integers(0, len(cluster_values), size=len(cluster_values))
        samples[index] = float(cluster_values[draw].mean())
    return {
        "question_clusters": int(len(cluster_values)),
        "observed_mean": float(cluster_values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "repetitions": repetitions,
        "rng_seed": rng_seed,
    }


def mean_delta(
    collected: dict[tuple[str, int], dict[str, Any]],
    left: str,
    right: str,
    key: str,
) -> tuple[float, list[float]]:
    values = [
        delta(collected[(left, seed)], collected[(right, seed)], key)
        for seed in SEEDS
    ]
    return float(np.mean(values)), values


def development_decision() -> dict[str, Any]:
    collected: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in SCIENTIFIC_VARIANTS:
            collected[(variant, seed)] = metrics(run_dir(variant, seed))
        collected[("hard_only", seed)] = metrics(old_dir("hard_only", seed))
    primary_mae, primary_mae_seeds = mean_delta(collected, "routed_hmsa", "consensus_only", "MAE_human_mean")
    primary_exact, _ = mean_delta(collected, "routed_hmsa", "consensus_only", "Exact_rounded")
    primary_kendall, _ = mean_delta(collected, "routed_hmsa", "consensus_only", "Kendall_human_mean")
    routed_dual_mae, routed_dual_mae_seeds = mean_delta(collected, "routed_hmsa", "dual_hard", "MAE_human_mean")
    routed_dual_exact, _ = mean_delta(collected, "routed_hmsa", "dual_hard", "Exact_rounded")
    routed_dual_kendall, _ = mean_delta(collected, "routed_hmsa", "dual_hard", "Kendall_human_mean")
    residual_mae, residual_mae_seeds = mean_delta(collected, "residual_only", "hard_only", "MAE_human_mean")
    residual_exact, _ = mean_delta(collected, "residual_only", "hard_only", "Exact_rounded")
    aligned_shuffle_mae, aligned_shuffle_seeds = mean_delta(collected, "routed_hmsa", "shuffled_residual", "MAE_human_mean")
    routed_flip_mae, _ = mean_delta(collected, "routed_hmsa", "sign_flipped", "MAE_human_mean")

    sign_rows: list[dict[str, Any]] = []
    sign_directions: dict[str, list[float]] = {state: [] for state in ("down", "up", "pooled")}
    pooled_routed: list[dict[str, Any]] = []
    pooled_flipped: list[dict[str, Any]] = []
    for seed in SEEDS:
        routed_metric = collected[("routed_hmsa", seed)]
        flipped_metric = collected[("sign_flipped", seed)]
        for state in ("down", "up", "pooled"):
            value = float(routed_metric[f"boundary_advantage_{state}_mean"]) - float(
                flipped_metric[f"boundary_advantage_{state}_mean"]
            )
            sign_directions[state].append(value)
            sign_rows.append({"seed": seed, "stratum": state, "routed_minus_flipped": value})
        for destination, rows in (
            (pooled_routed, predictions(run_dir("routed_hmsa", seed))),
            (pooled_flipped, predictions(run_dir("sign_flipped", seed))),
        ):
            destination.extend(
                [{**row, "record_id": f"seed{seed}:{row['record_id']}"} for row in rows]
            )
    sign_bootstrap = paired_question_bootstrap(pooled_routed, pooled_flipped)

    gates = {
        "primary_residual_increment": (
            primary_mae <= -0.005
            and all(value < 0.0 for value in primary_mae_seeds)
            and max(primary_mae_seeds) <= 0.002
            and primary_exact >= -0.003
            and primary_kendall >= -0.005
        ),
        "beyond_generic_dual_hard": (
            routed_dual_mae <= -0.005
            and all(value < 0.0 for value in routed_dual_mae_seeds)
            and routed_dual_exact >= -0.003
            and (routed_dual_exact >= 0.005 or routed_dual_kendall >= 0.010)
        ),
        "residual_only_sufficiency": (
            residual_mae <= -0.005
            and sum(value < 0.0 for value in residual_mae_seeds) >= 2
            and residual_exact >= -0.005
        ),
        "sign_direction": (
            all(all(value > 0.0 for value in sign_directions[state]) for state in ("down", "up", "pooled"))
            and sign_bootstrap["ci95_lower"] > 0.0
            and routed_flip_mae <= -0.005
        ),
        "aligned_beats_fixed_shuffle": (
            aligned_shuffle_mae <= -0.005
            and all(value < 0.0 for value in aligned_shuffle_seeds)
        ),
    }
    primary_passed = gates["primary_residual_increment"]
    result = {
        "status": "GO_EXTEND_PRIMARY_TO_SEEDS_45_46" if primary_passed else "CBRD_PRIMARY_NO_GO",
        "primary_comparison": {
            "mean_delta_mae": primary_mae,
            "seed_delta_mae": primary_mae_seeds,
            "mean_delta_exact": primary_exact,
            "mean_delta_kendall": primary_kendall,
        },
        "secondary_comparisons": {
            "routed_minus_dual_hard": {
                "mean_delta_mae": routed_dual_mae,
                "seed_delta_mae": routed_dual_mae_seeds,
                "mean_delta_exact": routed_dual_exact,
                "mean_delta_kendall": routed_dual_kendall,
            },
            "residual_only_minus_hard_only": {
                "mean_delta_mae": residual_mae,
                "seed_delta_mae": residual_mae_seeds,
                "mean_delta_exact": residual_exact,
            },
            "routed_minus_shuffled_residual": {
                "mean_delta_mae": aligned_shuffle_mae,
                "seed_delta_mae": aligned_shuffle_seeds,
            },
            "routed_minus_sign_flipped_mae": routed_flip_mae,
        },
        "sign_direction": {
            "per_seed_strata": sign_rows,
            "question_cluster_bootstrap": sign_bootstrap,
        },
        "gates": gates,
        "interpretation": {
            "primary_gate_controls_seed45_46": True,
            "residual_only_failure_does_not_alone_refute_joint_residual_effect": True,
            "ccf_b_mechanism_story_requires_more_than_primary_gate": True,
        },
        "test_access_count": 0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "development"), required=True)
    args = parser.parse_args()
    result = pilot_decision() if args.stage == "pilot" else development_decision()
    decision_dir = OUTPUT_ROOT / "decision"
    write_json(decision_dir / f"stage1_{args.stage}_decision.json", result)
    write_text(
        decision_dir / f"stage1_{args.stage}_decision.md",
        f"# Exp57 Stage 1 {args.stage} decision\n\n"
        f"- Status: `{result['status']}`\n"
        "- Historical test accessed: no\n",
    )
    if args.stage == "development":
        write_csv(OUTPUT_ROOT / "tables" / "stage1_sign_direction.csv", result["sign_direction"]["per_seed_strata"])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
