from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_mechanism_control_smoke import audit_results
from thesis_exp.exp54_rar_sft.train_mechanism_control_smoke import (
    ARMS,
    _validate_candidate_config,
    _validate_plan,
)


PLAN_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_smoke_plan_v1.json"
)
MECHANISM_CONFIG_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_controls_candidate_v1.json"
)
RUNNER_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/train_mechanism_control_smoke.py"
)
LOSS_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/mechanism_control_losses.py"
)
CANDIDATE_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "mechanism_controls_candidate"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_smoke_plan_binds_exact_runner_candidate_and_three_arms() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    _validate_plan(plan)
    assert tuple(plan["arms"]) == ARMS
    assert plan["source_rows"] == {
        "R3_TOKENAVG": 8,
        "P1_FULLSEQ": 32,
        "P1_SYN_LR5E6": 32,
    }
    bindings = plan["implementation_bindings"]
    assert bindings["runner_sha256"] == _sha256(RUNNER_PATH)
    assert bindings["control_loss_sha256"] == _sha256(LOSS_PATH)
    assert bindings["candidate_config_sha256"] == _sha256(
        MECHANISM_CONFIG_PATH
    )
    assert bindings["candidate_lock_sha256"] == _sha256(
        CANDIDATE_ROOT / "candidate_lock.json"
    )
    assert bindings["candidate_report_sha256"] == _sha256(
        CANDIDATE_ROOT / "candidate_report.json"
    )
    assert bindings["candidate_audit_sha256"] == _sha256(
        CANDIDATE_ROOT / "audit_report.json"
    )


def test_smoke_plan_is_one_step_train_only_and_not_formal_training() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert plan["optimizer_steps_per_arm"] == 1
    assert plan["gpu_smoke_allowed"] is True
    assert plan["full_training_allowed"] is False
    assert plan["dev_allowed"] is False
    assert plan["test_allowed"] is False
    assert plan["model_selection_allowed"] is False
    assert {
        value["name"] for value in plan["gpu_assignments"].values()
    } == {"NVIDIA RTX A6000"}
    assert len(
        {
            value["uuid"] for value in plan["gpu_assignments"].values()
        }
    ) == 3


def test_smoke_plan_tamper_hard_fails() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["full_training_allowed"] = True
    with pytest.raises(PermissionError, match="plan differs"):
        _validate_plan(plan)


def test_candidate_still_forbids_full_training_and_eval() -> None:
    config = json.loads(MECHANISM_CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_candidate_config(config)
    config["authorization"]["test_inference_allowed"] = True
    with pytest.raises(PermissionError, match="candidate differs"):
        _validate_candidate_config(config)


def test_runner_contains_no_split_specific_dev_or_test_input() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "dev.jsonl",
        "test.jsonl",
        "/dev/",
        "/test/",
        "run_sorc_dpo_dev",
        "run_sorc_dpo_test",
    ):
        assert forbidden not in source


def test_smoke_result_auditor_requires_positive_unique_updates(
    tmp_path: Path,
) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    results = {}
    paths = {}
    for index, arm in enumerate(ARMS, start=1):
        result_dir = tmp_path / arm.lower()
        adapter_dir = result_dir / "adapter"
        adapter_dir.mkdir(parents=True)
        adapter_path = adapter_dir / "adapter_model.safetensors"
        adapter_path.write_bytes(f"adapter-{arm}".encode("ascii"))
        adapter_hash = _sha256(adapter_path)
        result = {
            "status": "MECHANISM_CONTROL_TRAIN_ONLY_SMOKE_COMPLETE",
            "arm": arm,
            "seed": 42,
            "source_row_count": plan["source_rows"][arm],
            "training": {
                "losses": [0.5 + index],
                "gradient_norm_before_clip": 1.0 + index,
                "parameter_update_l2": 0.01 * index,
                "optimizer_steps": 1,
                "micro_batches": 4,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "adapter_model_sha256": adapter_hash,
                "device_identity": {
                    "uuid": plan["gpu_assignments"][arm]["uuid"],
                    "name": plan["gpu_assignments"][arm]["name"],
                },
            },
            "elapsed_seconds": 3.0,
            "dev_accessed": False,
            "test_accessed": False,
            "full_training_started": False,
        }
        result_path = result_dir / "result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        results[arm] = result
        paths[arm] = result_path
    audit = audit_results(
        plan=plan,
        result_by_arm=results,
        result_path_by_arm=paths,
    )
    assert (
        audit["status"]
        == "MECHANISM_CONTROL_SMOKE_PASS_FULL_TRAINING_NOT_STARTED"
    )

    results["P1_FULLSEQ"]["training"]["parameter_update_l2"] = 0.0
    with pytest.raises(ValueError, match="gradient/update"):
        audit_results(
            plan=plan,
            result_by_arm=results,
            result_path_by_arm=paths,
        )
