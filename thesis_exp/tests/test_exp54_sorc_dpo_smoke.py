from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.audit_sorc_dpo_smoke_package import (
    _derive_expected,
)
from thesis_exp.exp54_rar_sft.build_sorc_dpo_smoke_package import (
    SCORE_TYPES,
    _selected_score_keys,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    load_and_validate_smoke_rows,
    verify_gpu_execution_authorization,
)


def _plan() -> dict:
    path = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/sorc_dpo_smoke_plan_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _source_row(record_id: str, pair_type: str, *, task: str = "score") -> dict:
    return {
        "record_id": record_id,
        "pair_type": pair_type,
        "pair_task": task,
    }


def test_smoke_plan_is_one_train_only_step_per_arm() -> None:
    plan = _plan()
    assert plan["source_split"] == "train"
    assert plan["execution_contract"]["gpu_smoke_allowed"] is False
    assert plan["execution_contract"]["forward_backward_allowed"] is False
    assert plan["execution_contract"]["fixed_right_padding_to"] == 2048
    expected = {
        "P1_FIELD_DPO": (32, 32, 0),
        "P2_SORC_SCORE": (32, 32, 0),
        "P3_JOINT_SORC": (91, 32, 59),
        "P1_SYN_SEED42": (32, 32, 0),
    }
    for arm, (pairs, score, rationale) in expected.items():
        value = plan["arms"][arm]
        assert value["pair_count"] == pairs
        assert value["score_pairs"] == score
        assert value["rationale_pairs"] == rationale
        assert value["accumulation_group_pair_count"] == pairs
        assert value["optimizer_steps"] == 1


def test_score_smoke_selection_is_deterministic_under_input_shuffle() -> None:
    plan = _plan()
    rows = []
    for pair_type in SCORE_TYPES:
        rows.extend(
            _source_row(f"{pair_type}-{index:03d}", pair_type)
            for index in range(30)
        )
    forward = _selected_score_keys(rows, plan)
    reversed_order = _selected_score_keys(list(reversed(rows)), plan)
    assert forward == reversed_order
    counts = {pair_type: 0 for pair_type in SCORE_TYPES}
    for _record_id, pair_type in forward:
        counts[pair_type] += 1
    assert counts == {
        "adjacent_score": 16,
        "severe_l2h": 8,
        "h2l_guard": 8,
    }


def test_independent_derivation_matches_same_record_score_controls() -> None:
    plan = _plan()
    score_rows = []
    for pair_type in SCORE_TYPES:
        score_rows.extend(
            _source_row(f"{pair_type}-{index:03d}", pair_type)
            for index in range(30)
        )
    p1 = [dict(row, source="p1") for row in score_rows]
    p2 = [dict(row, source="p2") for row in score_rows]
    syn = [dict(row, source="syn") for row in score_rows]
    p3_score = [dict(row, source="p3-score") for row in score_rows]
    rationale = [
        _source_row(
            f"rationale-{index:03d}",
            "rationale_alignment",
            task="rationale",
        )
        for index in range(80)
    ]
    expected = _derive_expected(
        {
            "P1_FIELD_DPO": p1,
            "P2_SORC_SCORE": p2,
            "P1_SYN_SEED42": syn,
            "P3_JOINT_SORC": [*p3_score, *rationale],
        },
        plan,
    )
    vectors = {
        arm: [
            (row["record_id"], row["pair_type"])
            for row in values[:32]
        ]
        for arm, values in expected.items()
    }
    assert vectors["P1_FIELD_DPO"] == vectors["P2_SORC_SCORE"]
    assert vectors["P1_FIELD_DPO"] == vectors["P1_SYN_SEED42"]
    assert vectors["P1_FIELD_DPO"] == vectors["P3_JOINT_SORC"]
    assert len(expected["P3_JOINT_SORC"]) == 91


def _materialized_row() -> dict:
    return {
        "pair_id": "pair-1",
        "pair_task": "score",
        "pair_type": "adjacent_score",
        "active_field": "score",
        "chosen_input_ids": [1, 2, 3, 4],
        "chosen_field_token_positions": [2],
        "rejected_input_ids": [1, 2, 5, 4],
        "rejected_field_token_positions": [2],
        "odpo_offset": 0.0,
        "objective_weight": 1.0,
        "cutoff_len": 2048,
    }


def test_cpu_validate_loads_exact_private_subset(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    subset = private / "p1_field_dpo.jsonl"
    subset.write_text(
        json.dumps(_materialized_row(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "smoke_plan.json"
    plan = _plan()
    plan["arms"]["P1_FIELD_DPO"]["pair_count"] = 1
    plan["arms"]["P1_FIELD_DPO"]["accumulation_group_pair_count"] = 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    lock_path = tmp_path / "smoke_lock.json"
    lock = {
        "status": (
            "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_NOT_FROZEN_EXECUTION_FORBIDDEN"
        ),
        "gpu_smoke_allowed": False,
        "source_hashes": {"smoke_plan": sha256_file(plan_path)},
        "private_subset_hashes": {
            "P1_FIELD_DPO": sha256_file(subset),
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    rows, _lock, _plan_value = load_and_validate_smoke_rows(
        arm="P1_FIELD_DPO",
        smoke_lock_path=lock_path,
        smoke_plan_path=plan_path,
        private_subset_dir=private,
    )
    assert rows == [_materialized_row()]


def test_gpu_execute_requires_external_authorization(tmp_path: Path) -> None:
    missing = tmp_path / "missing_authorization.json"
    with pytest.raises(FileNotFoundError):
        verify_gpu_execution_authorization(
            path=missing,
            arm="P1_FIELD_DPO",
            smoke_lock_path=tmp_path / "smoke_lock.json",
            smoke_plan_path=tmp_path / "smoke_plan.json",
            training_config_path=tmp_path / "training_config.json",
        )


def test_smoke_candidate_runtime_source_closure_is_exact() -> None:
    path = (
        REPO_ROOT
        / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
        "preference_smoke/smoke_implementation_candidate_lock.json"
    )
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "thesis_package_init",
        "exp54_package_init",
        "failure_bank_io",
        "split_guard",
        "training_contract",
        "sorc_dpo_loss_collator",
        "smoke_runner",
    }
    assert set(lock["runtime_source_closure"]) == expected
    assert lock["runtime_source_expected_name_set"] == sorted(expected)
    assert lock["runtime_source_file_count"] == 7
    assert lock["gpu_smoke_allowed"] is False
    assert lock["forward_backward_executed"] is False
