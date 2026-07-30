from __future__ import annotations

import json
from pathlib import Path

from thesis_exp.exp54_rar_sft.collect_mechanism_control_test_results import (
    CONTRASTS,
)
from thesis_exp.exp54_rar_sft.run_mechanism_control_test_inference_vllm import (
    ARMS,
    SEEDS,
)


def test_mechanism_test_grid_and_contrasts_are_exact() -> None:
    assert len(ARMS) * len(SEEDS) == 9
    assert CONTRASTS == (
        ("C1_BLOCK_BALANCED_SFT", "R3_TOKENAVG", "P0_R3_SFT"),
        ("C2_FIELD_LOCAL_DPO", "P1_FULLSEQ", "P1_FIELD_DPO"),
        (
            "C3_ACTUAL_ERROR_NEGATIVES",
            "P1_SYN_LR5E6",
            "P1_FIELD_DPO",
        ),
    )


def test_mechanism_test_plan_reuses_old_predictions() -> None:
    path = (
        Path(__file__).parents[1]
        / "exp54_rar_sft/configs/mechanism_control_test_plan_v1.json"
    )
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["old_predictions_regenerated"] is False
    assert plan["test_adaptive_tuning_or_retraining"] is False
    assert plan["new_checkpoint_max_test_invocations"] == 1
    assert plan["generation"]["gpu_memory_utilization"] == 0.9
    assert {
        value["name"]
        for value in plan["gpu_assignments_by_arm"].values()
    } == {"NVIDIA GeForce RTX 3090"}
    assert plan["execution_hardware_difference"]["must_be_disclosed"] is True
    assert set(plan["bindings"]) == {
        "runner_sha256",
        "collector_sha256",
        "worker_sha256",
        "runtime_lock_sha256",
        "checkpoint_lock_sha256",
        "checkpoint_audit_report_sha256",
    }
