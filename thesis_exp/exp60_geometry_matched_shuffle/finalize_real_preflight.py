"""Aggregate the three seed-specific real-model no-update preflights."""

from __future__ import annotations

import json
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import (
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REAL_PREFLIGHT_DECISION_PATH,
)


SEEDS = (47, 48, 49)


def read_json(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
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
    cosines = [
        float(cosine)
        for report in reports.values()
        for cosine in report["treatment_separation_observation"][
            "storage_component_cosines"
        ]
    ]
    cosine_stop = float(protocol["implementation_gates_before_training"][
        "preflight_treatment_separation_stop_if_all_window_cosines_above"
    ])
    checks = {
        "all_three_seed_reports_pass": all(
            report.get("status") == "EXP60_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS"
            for report in reports.values()
        ),
        "all_six_windows_present": len(cosines) == 6,
        "treatment_not_near_duplicate_in_all_windows": not all(
            cosine > cosine_stop for cosine in cosines
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
        "cosine_stop_threshold": cosine_stop,
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
