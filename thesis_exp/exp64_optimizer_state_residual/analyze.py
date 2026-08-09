"""Apply the frozen seed-level Exp64 decision rules without model access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp64_optimizer_state_residual import OUTPUT_ROOT, SEEDS, STAGE_EPOCHS
from thesis_exp.exp64_optimizer_state_residual.mechanics import kendall_tau_b
DIRECTIONS = ("A_to_B", "B_to_A")
PRIMARY_CANDIDATES = ("full_residual", "parallel_only", "orthogonal_only")
OUTCOME_TIE_TOLERANCE = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_dir", type=Path, default=OUTPUT_ROOT / "formal")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "decision" / "formal_decision.json"
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Exp64 cannot average an empty list")
    return sum(values) / len(values)


def _mae(predicted: list[float], observed: list[float]) -> float:
    if len(predicted) != len(observed):
        raise RuntimeError("Exp64 prediction/outcome length mismatch")
    return _mean([abs(left - right) for left, right in zip(predicted, observed)])


def _sign_accuracy(predicted: list[float], observed: list[float]) -> float:
    correct = 0
    for left, right in zip(predicted, observed):
        right = 0.0 if abs(right) <= OUTCOME_TIE_TOLERANCE else right
        correct += int((left == 0.0 and right == 0.0) or left * right > 0.0)
    return correct / len(observed)


def _load_stage(formal_dir: Path, seed: int, stage: int) -> dict[str, Any]:
    path = formal_dir / f"seed_{seed}" / f"after_epoch_{stage}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "EXP64_FORMAL_STAGE_COMPLETE":
        raise RuntimeError(f"Exp64 incomplete formal stage: {path}")
    if value.get("seed") != seed or value.get("stage") != stage:
        raise RuntimeError(f"Exp64 formal stage identity mismatch: {path}")
    if value.get("test_access_count") != 0:
        raise RuntimeError("Exp64 formal result reports test access")
    return value


def main() -> None:
    args = parse_args()
    stages = {
        (seed, stage): _load_stage(args.formal_dir, seed, stage)
        for seed in SEEDS
        for stage in STAGE_EPOCHS
    }
    seed_reports = {}
    for seed in SEEDS:
        observed: list[float] = []
        exact: list[float] = []
        raw: list[float] = []
        fixed: list[float] = []
        magnitude: list[float] = []
        full: list[float] = []
        signflip: list[float] = []
        interaction_prediction: list[float] = []
        interaction_outcome: list[float] = []
        interaction_relative_norm: list[float] = []
        for stage in STAGE_EPOCHS:
            item = stages[(seed, stage)]
            for arm in PRIMARY_CANDIDATES:
                arm_item = item["arms"][arm]
                for direction in DIRECTIONS:
                    observed.append(float(arm_item["outcomes"][direction]))
                    predictors = arm_item["predictions"][direction]
                    exact.append(float(predictors["q_exact"]))
                    raw.append(float(predictors["q_raw_validation"]))
                    fixed.append(float(predictors["q_fixed_denominator"]))
                    magnitude.append(float(arm_item["magnitude_only_score"]))
            for direction in DIRECTIONS:
                full.append(float(item["arms"]["full_residual"]["outcomes"][direction]))
                signflip.append(
                    float(item["arms"]["sign_flipped_residual"]["outcomes"][direction])
                )
                interaction_prediction.append(
                    float(item["nonadditivity"]["predictions"][direction])
                )
                interaction_outcome.append(float(item["nonadditivity"]["outcomes"][direction]))
            interaction_relative_norm.append(
                float(item["nonadditivity"]["norm"])
                / max(
                    float(item["arms"]["full_residual"]["exact_attributable_norm"]),
                    1e-30,
                )
            )
        mean_interaction_prediction = _mean(interaction_prediction)
        mean_interaction_outcome = _mean(interaction_outcome)
        seed_reports[str(seed)] = {
            "observations": len(observed),
            "full_mean_outcome": _mean(full),
            "signflip_mean_outcome": _mean(signflip),
            "full_more_favorable_than_signflip": _mean(full) < _mean(signflip),
            "mae_exact": _mae(exact, observed),
            "mae_raw_validation": _mae(raw, observed),
            "mae_fixed_denominator": _mae(fixed, observed),
            "exact_beats_raw": _mae(exact, observed) < _mae(raw, observed),
            "exact_beats_fixed": _mae(exact, observed) < _mae(fixed, observed),
            "tau_exact": kendall_tau_b(
                exact, observed, outcome_tie_tolerance=OUTCOME_TIE_TOLERANCE
            ),
            "tau_magnitude": kendall_tau_b(
                magnitude, observed, outcome_tie_tolerance=OUTCOME_TIE_TOLERANCE
            ),
            "exact_tau_beats_magnitude": kendall_tau_b(
                exact, observed, outcome_tie_tolerance=OUTCOME_TIE_TOLERANCE
            )
            > kendall_tau_b(
                magnitude, observed, outcome_tie_tolerance=OUTCOME_TIE_TOLERANCE
            ),
            "exact_sign_accuracy": _sign_accuracy(exact, observed),
            "magnitude_sign_accuracy": _sign_accuracy(magnitude, observed),
            "mean_interaction_relative_norm": _mean(interaction_relative_norm),
            "mean_interaction_prediction": mean_interaction_prediction,
            "mean_interaction_outcome": mean_interaction_outcome,
            "interaction_sign_correct": mean_interaction_prediction * mean_interaction_outcome
            > 0.0,
        }

    stage_reports = {}
    for stage in STAGE_EPOCHS:
        full = []
        observed = []
        exact = []
        raw = []
        fixed = []
        for seed in SEEDS:
            item = stages[(seed, stage)]
            for direction in DIRECTIONS:
                full.append(float(item["arms"]["full_residual"]["outcomes"][direction]))
            for arm in PRIMARY_CANDIDATES:
                for direction in DIRECTIONS:
                    arm_item = item["arms"][arm]
                    observed.append(float(arm_item["outcomes"][direction]))
                    predictors = arm_item["predictions"][direction]
                    exact.append(float(predictors["q_exact"]))
                    raw.append(float(predictors["q_raw_validation"]))
                    fixed.append(float(predictors["q_fixed_denominator"]))
        stage_reports[str(stage)] = {
            "full_mean_outcome": _mean(full),
            "mae_exact": _mae(exact, observed),
            "mae_raw_validation": _mae(raw, observed),
            "mae_fixed_denominator": _mae(fixed, observed),
            "full_favorable": _mean(full) < 0.0,
            "exact_beats_both": _mae(exact, observed) < _mae(raw, observed)
            and _mae(exact, observed) < _mae(fixed, observed),
        }

    seed_values = list(seed_reports.values())
    full_seed_passes = sum(value["full_mean_outcome"] < 0.0 for value in seed_values)
    exact_raw_passes = sum(value["exact_beats_raw"] for value in seed_values)
    exact_fixed_passes = sum(value["exact_beats_fixed"] for value in seed_values)
    directional_passes = sum(value["exact_tau_beats_magnitude"] for value in seed_values)
    signflip_passes = sum(value["full_more_favorable_than_signflip"] for value in seed_values)
    stage_full_passes = sum(value["full_favorable"] for value in stage_reports.values())
    stage_coordinate_passes = sum(value["exact_beats_both"] for value in stage_reports.values())
    overall_full = _mean([value["full_mean_outcome"] for value in seed_values])
    overall_signflip = _mean([value["signflip_mean_outcome"] for value in seed_values])
    gates = {
        "full_local_effect": full_seed_passes >= 4 and overall_full < 0.0,
        "exact_beats_raw": exact_raw_passes >= 4,
        "exact_beats_fixed": exact_fixed_passes >= 4,
        "exact_direction_beats_magnitude": directional_passes >= 4,
        "full_beats_signflip": signflip_passes >= 4 and overall_full < overall_signflip,
        "full_not_single_stage": stage_full_passes >= 2,
        "coordinate_result_not_single_stage": stage_coordinate_passes >= 2,
        "test_access_zero": True,
    }
    nonadditivity_seed_passes = sum(
        value["mean_interaction_relative_norm"] > 1e-3
        and value["interaction_sign_correct"]
        for value in seed_values
    )
    result = {
        "status": "EXP64_GO" if all(gates.values()) else "EXP64_NO_GO_STOP",
        "seed_reports": seed_reports,
        "stage_reports": stage_reports,
        "aggregate": {
            "full_seed_passes": full_seed_passes,
            "exact_raw_seed_passes": exact_raw_passes,
            "exact_fixed_seed_passes": exact_fixed_passes,
            "directional_seed_passes": directional_passes,
            "signflip_seed_passes": signflip_passes,
            "stage_full_passes": stage_full_passes,
            "stage_coordinate_passes": stage_coordinate_passes,
            "overall_full_mean": overall_full,
            "overall_signflip_mean": overall_signflip,
        },
        "primary_gates": gates,
        "nonadditivity": {
            "conditional_claim_authorized": nonadditivity_seed_passes >= 4,
            "seed_passes": nonadditivity_seed_passes,
            "primary_result_does_not_depend_on_this": True,
        },
        "stopping_rule": "Stop after this frozen five-seed run regardless of outcome.",
        "test_access_count": 0,
    }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
