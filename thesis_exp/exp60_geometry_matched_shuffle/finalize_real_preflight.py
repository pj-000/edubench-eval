"""Aggregate the three seed-specific real-model no-update preflights."""

from __future__ import annotations

import json
import math
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import (
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REAL_PREFLIGHT_DECISION_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.contract import (
    verify_preflight_source_lock,
)


SEEDS = (47, 48, 49)


def read_json(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def treatment_separation_by_seed(
    observations: dict[int, dict[str, Any]],
    *,
    cosine_max: float,
    distance_min: float,
    activity_min: float,
) -> dict[str, bool]:
    return {
        str(seed): (
            len(observations[seed]["storage_component_cosines"]) == 2
            and len(observations[seed]["storage_component_relative_distances"]) == 2
            and len(observations[seed]["storage_component_activity_ratios"]) == 2
            and all(
                math.isfinite(float(value))
                for key in (
                    "storage_component_cosines",
                    "storage_component_relative_distances",
                    "storage_component_activity_ratios",
                )
                for value in observations[seed][key]
            )
            and all(
                float(value) >= activity_min
                for value in observations[seed]["storage_component_activity_ratios"]
            )
            and any(
                float(cosine) <= cosine_max and float(distance) >= distance_min
                for cosine, distance in zip(
                    observations[seed]["storage_component_cosines"],
                    observations[seed]["storage_component_relative_distances"],
                )
            )
        )
        for seed in SEEDS
    }


def run() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    current_binding = verify_preflight_source_lock()
    reports = {
        seed: read_json(
            OUTPUT_ROOT
            / "real_model_preflight"
            / f"seed_{seed}"
            / "real_model_no_update_preflight.json"
        )
        for seed in SEEDS
    }
    if any(int(report.get("seed", -1)) != seed for seed, report in reports.items()):
        raise RuntimeError("Exp60 real-preflight seed/report path mismatch")
    observations = {
        seed: report["treatment_separation_observation"]
        for seed, report in reports.items()
    }
    cosines = [
        float(value)
        for observation in observations.values()
        for value in observation["storage_component_cosines"]
    ]
    distances = [
        float(value)
        for observation in observations.values()
        for value in observation["storage_component_relative_distances"]
    ]
    activity_ratios = [
        float(value)
        for observation in observations.values()
        for value in observation["storage_component_activity_ratios"]
    ]
    gates = protocol["implementation_gates_before_training"]
    cosine_max = float(gates["preflight_treatment_component_cosine_at_most"])
    distance_min = float(
        gates["preflight_treatment_component_relative_distance_at_least"]
    )
    activity_min = float(
        gates["preflight_treatment_component_activity_ratio_at_least"]
    )
    finite_treatment_values = all(
        math.isfinite(value) for value in (*cosines, *distances, *activity_ratios)
    )
    per_seed_separation = treatment_separation_by_seed(
        observations,
        cosine_max=cosine_max,
        distance_min=distance_min,
        activity_min=activity_min,
    )
    gpu_slots = {int(report["gpu_slot"]) for report in reports.values()}
    visible_devices = {
        str(report["gpu_identity"]["cuda_visible_devices"])
        for report in reports.values()
    }
    physical_identities = {
        str(report["gpu_identity"]["identity"])
        for report in reports.values()
    }
    stable_gpu_uuids = {
        str(report["gpu_identity"].get("stable_gpu_uuid") or "")
        for report in reports.values()
    }
    checks = {
        "all_three_seed_reports_pass": all(
            report.get("status") == "EXP60_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS"
            for report in reports.values()
        ),
        "all_six_windows_present": len(cosines) == len(distances) == len(activity_ratios) == 6,
        "all_treatment_values_finite": finite_treatment_values,
        "each_seed_has_nondegenerate_and_separated_treatment": all(
            per_seed_separation.values()
        ),
        "preflight_gpu_slots_cover_0_1_2": gpu_slots == {0, 1, 2},
        "preflight_gpu_schedule_matches_protocol": all(
            int(report["gpu_slot"])
            == int(protocol["formal_runs"]["real_preflight_gpu_schedule"][str(seed)])
            for seed, report in reports.items()
        ),
        "three_distinct_cuda_visible_devices": len(visible_devices) == 3,
        "three_distinct_physical_gpu_identities": len(physical_identities) == 3,
        "all_reports_have_stable_gpu_uuid": "" not in stable_gpu_uuids,
        "three_distinct_stable_gpu_uuids": len(stable_gpu_uuids) == 3,
        "mig_is_not_used": all(
            str(report["gpu_identity"].get("mig_mode") or "").lower() != "enabled"
            and not str(report["gpu_identity"].get("stable_gpu_uuid") or "").lower().startswith("mig-")
            for report in reports.values()
        ),
        "mapping_sha_identical": len(
            {str(report["mapping_sha256"]) for report in reports.values()}
        ) == 1,
        "model_file_manifest_identical": len(
            {
                str(report["model_input_manifest"]["manifest_sha256"])
                for report in reports.values()
            }
        ) == 1,
        "git_commit_identical": len(
            {str(report["environment"]["git_commit"]) for report in reports.values()}
        ) == 1,
        "preflight_source_lock_identical_and_current": all(
            report.get("preflight_contract_binding") == current_binding
            for report in reports.values()
        ),
        "all_runs_started_from_clean_tracked_git": all(
            bool(report["environment"]["git_tracked_diff_empty"])
            for report in reports.values()
        ),
        "no_optimizer_steps": all(
            int(report.get("optimizer_steps", -1)) == 0 for report in reports.values()
        ),
        "no_test_access": all(
            int(report.get("test_access_count", -1)) == 0 for report in reports.values()
        ),
    }
    decision = {
        "status": "EXP60_REAL_MODEL_PREFLIGHT_ALL_SEEDS_PASS"
        if all(checks.values())
        else "EXP60_REAL_MODEL_PREFLIGHT_NOT_AUTHORIZED",
        "seeds": list(SEEDS),
        "mapping_sha256": reports[47]["mapping_sha256"],
        "model_input_manifest_sha256": reports[47]["model_input_manifest"][
            "manifest_sha256"
        ],
        "storage_component_cosines": cosines,
        "storage_component_relative_distances": distances,
        "storage_component_activity_ratios": activity_ratios,
        "per_seed_treatment_separation": per_seed_separation,
        "treatment_thresholds": {
            "cosine_at_most": cosine_max,
            "relative_distance_at_least": distance_min,
            "activity_ratio_at_least": activity_min,
        },
        "preflight_contract_binding": current_binding,
        "preflight_git_commit": reports[47]["environment"]["git_commit"],
        "preflight_gpu_bindings": {
            f"gpu_slot_{int(report['gpu_slot'])}": report["gpu_identity"]
            for report in reports.values()
        },
        "checks": checks,
        "protocol_status_at_preflight": protocol["status"],
        "next_action": (
            "independent review, then one protocol-status freeze and formal source lock"
            if all(checks.values())
            else "do not freeze protocol or start formal training"
        ),
        "test_access_count": 0,
    }
    write_json(REAL_PREFLIGHT_DECISION_PATH, decision)
    return decision


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
