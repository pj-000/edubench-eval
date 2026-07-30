from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft import REPO_ROOT
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
