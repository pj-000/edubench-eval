from __future__ import annotations

import json
from pathlib import Path

from thesis_exp.exp54_rar_sft.monitor_mechanism_control_campaign import (
    snapshot,
)
from thesis_exp.exp54_rar_sft.train_mechanism_controls_formal import (
    ARMS,
    PREFERENCE_PAIR_COUNT,
    PREFERENCE_STEPS,
    SEEDS,
    _chunks,
)


def test_formal_run_matrix_is_exact() -> None:
    assert ARMS == ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
    assert SEEDS == (42, 43, 44)
    assert len(ARMS) * len(SEEDS) == 9


def test_preference_groups_match_frozen_budget() -> None:
    groups = _chunks(list(range(PREFERENCE_PAIR_COUNT)), 32)
    assert len(groups) == PREFERENCE_STEPS == 27
    assert [len(group) for group in groups[:-1]] == [32] * 26
    assert len(groups[-1]) == 6


def test_campaign_monitor_reports_exact_percentage(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    progress = root / "r3_tokenavg/seed_42/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "optimizer_step": 498,
                "completion_percent": 50.0,
                "elapsed_seconds": 10.0,
                "estimated_remaining_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    value = snapshot(root)
    assert value["completed_optimizer_steps"] == 498
    assert value["total_optimizer_steps"] == 3150
    assert value["optimizer_step_weighted_percent"] == 100.0 * 498 / 3150
    assert value["total_runs"] == 9
    assert value["dev_accessed"] is False
    assert value["test_accessed"] is False


def test_campaign_failure_overrides_full_step_count(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    for seed in SEEDS:
        for arm, steps in (
            ("R3_TOKENAVG", 996),
            ("P1_FULLSEQ", 27),
            ("P1_SYN_LR5E6", 27),
        ):
            directory = root / arm.lower() / f"seed_{seed}"
            directory.mkdir(parents=True)
            (directory / "progress.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "optimizer_step": steps,
                        "completion_percent": 100.0,
                        "elapsed_seconds": 1.0,
                        "estimated_remaining_seconds": 0.0,
                    }
                ),
                encoding="utf-8",
            )
    failed = root / "p1_syn_lr5e6/seed_44/failure.json"
    failed.write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
    value = snapshot(root)
    assert value["completed_optimizer_steps"] == value["total_optimizer_steps"]
    assert value["status"] == "FAILED"


def test_plan_forbids_dev_and_test() -> None:
    path = (
        Path(__file__).parents[1]
        / "exp54_rar_sft/configs/mechanism_control_formal_plan_v1.json"
    )
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["full_gpu_training_allowed"] is True
    assert plan["dev_allowed"] is False
    assert plan["test_allowed"] is False
    assert plan["model_selection_allowed"] is False
    assert plan["candidate_artifacts_promoted_to_formal_inputs"] is True
    assert plan["gpu_assignments_by_arm_and_seed"]["R3_TOKENAVG"]["42"][
        "name"
    ] == "NVIDIA GeForce RTX 3090"
    assert plan["gpu_assignments_by_arm_and_seed"]["P1_FULLSEQ"]["42"][
        "name"
    ] == "NVIDIA RTX A6000"
    assert plan["execution_device_override"][
        "must_be_disclosed_as_execution_hardware_difference"
    ] is True
