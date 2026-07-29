from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

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
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK,
    DEFAULT_QUALIFICATION_CANDIDATE_LOCK,
    DEFAULT_SMOKE_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    FINAL_QUALIFICATION_SCHEMA_VERSION,
    FINAL_QUALIFICATION_STATUS,
    MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES,
    SMOKE_ROOT,
    _canonical_sha256,
    _seed42_checkpoint,
    load_and_validate_smoke_rows,
    read_json,
    verify_authorized_cuda_device,
    verify_gpu_execution_authorization,
    verify_p3_qualification_lock,
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
            checkpoint_lock_path=tmp_path / "checkpoint_lock.json",
            base_training_configuration_path=tmp_path / "base_config.json",
            frozen_training_lock_path=tmp_path / "frozen_lock.json",
            implementation_candidate_lock_path=(
                tmp_path / "implementation_lock.json"
            ),
        )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_authorization(
    *,
    arm: str = "P1_FIELD_DPO",
    qualification_path: Path | None = None,
) -> dict:
    base = read_json(DEFAULT_BASE_TRAINING_CONFIGURATION)
    value = {
        "schema_version": "exp54-sorc-dpo-gpu-smoke-authorization-v1",
        "status": "SORC_DPO_GPU_SMOKE_AUTHORIZED",
        "gpu_smoke_allowed": True,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "arm": arm,
        "seed": 42,
        "optimizer_steps": 1,
        "physical_micro_batch_pairs": 1,
        "smoke_lock_sha256": sha256_file(DEFAULT_SMOKE_LOCK),
        "smoke_plan_sha256": sha256_file(DEFAULT_SMOKE_PLAN),
        "training_config_sha256": sha256_file(DEFAULT_TRAINING_CONFIG),
        "checkpoint_lock_sha256": sha256_file(DEFAULT_CHECKPOINT_LOCK),
        "base_training_configuration_sha256": sha256_file(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        "base_model_snapshot_identity_sha256": _canonical_sha256(
            base["model"]
        ),
        "preference_training_frozen_lock_sha256": sha256_file(
            DEFAULT_FROZEN_TRAINING_LOCK
        ),
        "smoke_implementation_candidate_lock_sha256": sha256_file(
            DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
        ),
        "runner_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/train_sorc_dpo_smoke.py"
        ),
        "cuda_device_name": "NVIDIA RTX A6000",
        "cuda_device_uuid": "GPU-test-authorized-a6000",
        "minimum_free_memory_bytes_before_load": (
            MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES
        ),
        "output_dir": str(
            SMOKE_ROOT / "runs" / f"unit-test-{arm.lower()}-not-created"
        ),
    }
    if qualification_path is not None:
        value.update(
            {
                "rationale_qualification_lock_path": str(
                    qualification_path
                ),
                "rationale_qualification_lock_sha256": sha256_file(
                    qualification_path
                ),
                "rationale_qualification_candidate_lock_sha256": sha256_file(
                    DEFAULT_QUALIFICATION_CANDIDATE_LOCK
                ),
            }
        )
    return value


def _verify_authorization(path: Path, **overrides: Path) -> dict:
    arguments = {
        "path": path,
        "arm": "P1_FIELD_DPO",
        "smoke_lock_path": DEFAULT_SMOKE_LOCK,
        "smoke_plan_path": DEFAULT_SMOKE_PLAN,
        "training_config_path": DEFAULT_TRAINING_CONFIG,
        "checkpoint_lock_path": DEFAULT_CHECKPOINT_LOCK,
        "base_training_configuration_path": (
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        "frozen_training_lock_path": DEFAULT_FROZEN_TRAINING_LOCK,
        "implementation_candidate_lock_path": (
            DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
        ),
    }
    arguments.update(overrides)
    return verify_gpu_execution_authorization(**arguments)


def test_exact_current_identity_files_pass_authorization(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    authorization = _valid_authorization()
    _write_json(authorization_path, authorization)
    assert _verify_authorization(authorization_path) == authorization


def test_alternate_checkpoint_lock_hard_fails_authorization(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, _valid_authorization())
    alternate = tmp_path / "alternate_checkpoint_lock.json"
    shutil.copyfile(DEFAULT_CHECKPOINT_LOCK, alternate)
    alternate.write_text(
        alternate.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="checkpoint_lock_sha256"):
        _verify_authorization(
            authorization_path,
            checkpoint_lock_path=alternate,
        )


def test_modified_base_configuration_hard_fails_authorization(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, _valid_authorization())
    alternate = tmp_path / "base_configuration.json"
    shutil.copyfile(DEFAULT_BASE_TRAINING_CONFIGURATION, alternate)
    alternate.write_text(
        alternate.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PermissionError,
        match="base_training_configuration_sha256",
    ):
        _verify_authorization(
            authorization_path,
            base_training_configuration_path=alternate,
        )


def test_checkpoint_relative_path_cannot_escape_formal_runs(
    tmp_path: Path,
) -> None:
    formal_root = tmp_path / "formal_runs"
    formal_root.mkdir()
    original = read_json(DEFAULT_CHECKPOINT_LOCK)
    selected = [
        value
        for value in original["checkpoint_bindings"]
        if value["arm"] == "R3" and value["seed"] == 42
    ][0]
    bad_lock = {
        "status": "SFT_DEV_CHECKPOINT_SELECTION_FROZEN",
        "test_accessed": False,
        "checkpoint_bindings": [
            dict(selected, checkpoint_relative_path="../alternate")
        ],
    }
    path = tmp_path / "checkpoint_lock.json"
    _write_json(path, bad_lock)
    with pytest.raises(PermissionError, match="escapes"):
        _seed42_checkpoint(path, formal_runs_root=formal_root)


def _final_qualification() -> dict:
    candidate = read_json(DEFAULT_QUALIFICATION_CANDIDATE_LOCK)
    return {
        "schema_version": FINAL_QUALIFICATION_SCHEMA_VERSION,
        "status": FINAL_QUALIFICATION_STATUS,
        "rationale_blind_qualification_completed": True,
        "p3_preference_training_allowed": True,
        "evaluator_family_qualification_completed": True,
        "evaluator_family_count": 2,
        "evaluator_family_results": {
            "evaluator-family-a": {
                "qualification_completed": True,
                "r3_wins": 61,
                "r2_wins": 40,
                "ties": 19,
            },
            "evaluator-family-b": {
                "qualification_completed": True,
                "r3_wins": 58,
                "r2_wins": 42,
                "ties": 20,
            },
        },
        "source_hashes": candidate["source_hashes"],
        "dev_accessed": False,
        "test_accessed": False,
    }


def test_current_candidate_qualification_lock_cannot_enable_p3() -> None:
    with pytest.raises(PermissionError):
        verify_p3_qualification_lock(
            qualification_path=DEFAULT_QUALIFICATION_CANDIDATE_LOCK,
            qualification_candidate_lock_path=(
                DEFAULT_QUALIFICATION_CANDIDATE_LOCK
            ),
        )


def test_p3_preference_false_qualification_hard_fails(
    tmp_path: Path,
) -> None:
    value = _final_qualification()
    value["p3_preference_training_allowed"] = False
    path = tmp_path / "qualification.json"
    _write_json(path, value)
    with pytest.raises(
        PermissionError,
        match="p3_preference_training_allowed",
    ):
        verify_p3_qualification_lock(
            qualification_path=path,
            qualification_candidate_lock_path=(
                DEFAULT_QUALIFICATION_CANDIDATE_LOCK
            ),
        )


def test_deprecated_p3_training_allowed_alias_is_never_accepted(
    tmp_path: Path,
) -> None:
    value = _final_qualification()
    value.pop("p3_preference_training_allowed")
    value["p3_training_allowed"] = True
    path = tmp_path / "qualification.json"
    _write_json(path, value)
    with pytest.raises(PermissionError):
        verify_p3_qualification_lock(
            qualification_path=path,
            qualification_candidate_lock_path=(
                DEFAULT_QUALIFICATION_CANDIDATE_LOCK
            ),
        )


def test_exact_final_qualification_contract_passes(tmp_path: Path) -> None:
    value = _final_qualification()
    path = tmp_path / "qualification.json"
    _write_json(path, value)
    assert verify_p3_qualification_lock(
        qualification_path=path,
        qualification_candidate_lock_path=(
            DEFAULT_QUALIFICATION_CANDIDATE_LOCK
        ),
    ) == value


def test_final_qualification_source_hash_mismatch_hard_fails(
    tmp_path: Path,
) -> None:
    value = _final_qualification()
    value["source_hashes"]["pair_protocol"] = "0" * 64
    path = tmp_path / "qualification.json"
    _write_json(path, value)
    with pytest.raises(PermissionError, match="source hashes"):
        verify_p3_qualification_lock(
            qualification_path=path,
            qualification_candidate_lock_path=(
                DEFAULT_QUALIFICATION_CANDIDATE_LOCK
            ),
        )


def test_p3_qualification_hash_change_hard_fails_authorization(
    tmp_path: Path,
) -> None:
    qualification_path = tmp_path / "qualification.json"
    _write_json(qualification_path, _final_qualification())
    authorization_path = tmp_path / "authorization.json"
    _write_json(
        authorization_path,
        _valid_authorization(
            arm="P3_JOINT_SORC",
            qualification_path=qualification_path,
        ),
    )
    qualification_path.write_text(
        qualification_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        PermissionError,
        match="qualification lock differs",
    ):
        _verify_authorization(
            authorization_path,
            arm="P3_JOINT_SORC",
        )


class _FakeCuda:
    def __init__(
        self,
        *,
        name: str = "NVIDIA RTX A6000",
        uuid: str = "GPU-test-authorized-a6000",
        free_bytes: int = MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES,
    ) -> None:
        self.name = name
        self.uuid = uuid
        self.free_bytes = free_bytes

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_name(self, _index: int) -> str:
        return self.name

    def get_device_properties(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(uuid=self.uuid)

    def mem_get_info(self, _index: int) -> tuple[int, int]:
        return self.free_bytes, 48 * 1024**3


def test_authorized_idle_a6000_identity_passes() -> None:
    authorization = _valid_authorization()
    identity = verify_authorized_cuda_device(
        torch_module=SimpleNamespace(cuda=_FakeCuda()),
        authorization=authorization,
    )
    assert identity["name"] == "NVIDIA RTX A6000"
    assert identity["uuid"] == authorization["cuda_device_uuid"]


@pytest.mark.parametrize(
    ("cuda", "message"),
    [
        (_FakeCuda(name="NVIDIA GeForce RTX 3090"), "name"),
        (_FakeCuda(uuid="GPU-other-a6000"), "UUID"),
        (_FakeCuda(free_bytes=MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES - 1), "idle"),
    ],
)
def test_wrong_or_busy_cuda_device_hard_fails(
    cuda: _FakeCuda,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        verify_authorized_cuda_device(
            torch_module=SimpleNamespace(cuda=cuda),
            authorization=_valid_authorization(),
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
