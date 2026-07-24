import copy
import json
from pathlib import Path

import pytest

import thesis_exp.exp54_rar_sft.audit_smoke_training_package as auditor
import thesis_exp.exp54_rar_sft.build_smoke_training_package as builder
import thesis_exp.exp54_rar_sft.train_rar_sft_smoke as smoke_runner
from thesis_exp.exp54_rar_sft.authorization_guard import (
    closure_sha256,
    runtime_source_closure,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.build_smoke_training_package import (
    build_smoke_training_package,
)
from thesis_exp.exp54_rar_sft.audit_smoke_training_package import (
    audit_smoke_training_package,
)
from thesis_exp.exp54_rar_sft.freeze_training_configuration import (
    build_training_configuration_frozen_lock,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    smoke_runtime_source_closure,
    verify_smoke_authorization,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    SMOKE_ARMS,
    SMOKE_EVENTS_PER_ARM,
    selector_digest,
    smoke_manifest_path,
    smoke_prompt_cache_path,
    validate_smoke_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/smoke_training_plan.json"
)
REAL_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
REAL_CANDIDATE_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "training_configuration_candidate_report.json"
)
REAL_CANDIDATE_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_candidate_lock.json"
)
REAL_MATERIALIZED_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
REAL_TRAINING_CONFIGURATION_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_frozen_lock.json"
)
REAL_SMOKE_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "smoke_training_package_report.json"
)
REAL_SMOKE_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "smoke_training_package_frozen_lock.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _synthetic_full_inputs(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "train-data"
    prompt_rows = [
        {
            "prompt_cache_id": f"prompt-{position}",
            "record_id": f"record-{position}",
            "prompt_token_ids_sha256": f"prompt-hash-{position}",
        }
        for position in range(2_654)
    ]
    prompt_path = data_dir / "shared_prompt_cache.jsonl"
    _write_jsonl(prompt_path, prompt_rows)
    manifest_hashes = {}
    for arm in SMOKE_ARMS:
        rows = []
        for epoch_index in range(3):
            for position in range(2_654):
                active = position % 2 == 0
                rows.append(
                    {
                        "arm": arm,
                        "seed": 42,
                        "epoch_index": epoch_index,
                        "epoch_number": epoch_index + 1,
                        "row_position": position,
                        "base_event_id": f"event-{epoch_index}-{position}",
                        "record_id": f"record-{position}",
                        "prompt_cache_id": f"prompt-{position}",
                        "prompt_token_ids_sha256": f"prompt-hash-{position}",
                        "score_target": position % 5 + 1,
                        "score_loss_active": True,
                        "cutoff_len": 2_048,
                        "packing": False,
                        "truncated": False,
                        "rationale_active": (
                            False
                            if arm == "S0"
                            else True
                            if arm == "R1"
                            else active
                        ),
                        "sequence_token_count": 100 + position % 7,
                        "score_token_positions": [10],
                        "rationale_token_positions": (
                            [11, 12]
                            if (
                                arm == "R1"
                                or (arm in {"R2", "R3"} and active)
                            )
                            else []
                        ),
                    }
                )
        path = data_dir / f"training_manifest_{arm.lower()}_seed42.jsonl"
        _write_jsonl(path, rows)
        manifest_hashes[arm] = sha256_file(path)
    materialized_path = root / "materialized-lock.json"
    materialized = {
        "status": "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED",
        "manifest_frozen": True,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "private_artifact_hashes": {
            "shared_prompt_cache": sha256_file(prompt_path),
            "manifests_by_seed": {"seed42": manifest_hashes},
        },
    }
    _write_json(materialized_path, materialized)
    configuration_path = root / "configuration-frozen-lock.json"
    configuration = {
        "status": "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "review_gate": {
            "verdict": "TRAINING_CONFIGURATION_PASS",
            "reviewed_commit": (
                "d2762a427b8bed957459b04ba239a988ab3acea5"
            ),
            "candidate_report_sha256": (
                "27186421a67aa5c0b324672e567280859295e3066201fbca5b97ae2cd07dc0aa"
            ),
            "candidate_lock_sha256": (
                "a69b7c4594d86d8dbffa432550d0d69a0d39bd67785c89dc8045abf676e4ee0a"
            ),
        },
        "configuration_frozen": True,
        "smoke_package_build_allowed": True,
        "materialized_manifest_frozen_lock_sha256": sha256_file(
            materialized_path
        ),
        "configuration_sha256": sha256_file(REAL_CONFIG),
        "runtime_source_closure": runtime_source_closure(),
        "runtime_source_closure_sha256": closure_sha256(
            runtime_source_closure()
        ),
        "runtime_source_closure_file_count": 16,
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(configuration_path, configuration)
    return data_dir, materialized_path, configuration_path


def test_reviewed_training_configuration_freezes_exact_hash_chain() -> None:
    frozen = build_training_configuration_frozen_lock(
        config_path=REAL_CONFIG,
        frozen_manifest_lock_path=REAL_MATERIALIZED_FROZEN_LOCK,
        candidate_report_path=REAL_CANDIDATE_REPORT,
        candidate_lock_path=REAL_CANDIDATE_LOCK,
    )
    assert frozen["configuration_frozen"] is True
    assert frozen["smoke_package_build_allowed"] is True
    assert frozen["runtime_source_closure_file_count"] == 16
    assert frozen["review_gate"]["reviewed_commit"] == (
        "d2762a427b8bed957459b04ba239a988ab3acea5"
    )
    assert frozen["trust_anchor_install_allowed"] is False
    assert frozen["forward_backward_allowed"] is False
    assert frozen["smoke_training_allowed"] is False
    assert frozen["formal_training_allowed"] is False


def test_smoke_plan_and_selector_are_fixed() -> None:
    plan = json.loads(REAL_PLAN.read_text(encoding="utf-8"))
    validate_smoke_plan(plan)
    assert selector_digest("event-a") == selector_digest("event-a")
    assert selector_digest("event-a") != selector_digest("event-b")
    assert len(smoke_runtime_source_closure()) == 20
    changed = copy.deepcopy(plan)
    changed["events_per_arm"] = 10
    with pytest.raises(ValueError, match="events_per_arm"):
        validate_smoke_plan(changed)


def test_smoke_builder_freezes_same_eight_events_across_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, materialized_path, configuration_path = (
        _synthetic_full_inputs(tmp_path)
    )
    monkeypatch.setattr(builder, "_require_absent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auditor,
        "_require_absent",
        lambda *_args, **_kwargs: None,
    )
    private_dir = tmp_path / "private-smoke"
    report_path = tmp_path / "smoke-report.json"
    lock_path = tmp_path / "smoke-lock.json"
    report, lock = build_smoke_training_package(
        full_data_dir=data_dir,
        private_smoke_dir=private_dir,
        smoke_plan_path=REAL_PLAN,
        training_configuration_frozen_lock_path=configuration_path,
        materialized_manifest_frozen_lock_path=materialized_path,
        report_path=report_path,
        frozen_lock_path=lock_path,
    )
    assert report["events_per_arm"] == SMOKE_EVENTS_PER_ARM
    assert report["selected_active_events"] == 4
    assert report["selected_inactive_events"] == 4
    assert report["same_event_vector_and_order_across_arms"] is True
    assert report["fixed_padded_token_budget_equal_across_arms"] is True
    assert report["rationale_active_events_by_arm"] == {
        "S0": 0,
        "R1": 8,
        "R2": 4,
        "R3": 4,
    }
    assert report["score_target_histogram"] == {
        "1": 2,
        "2": 2,
        "3": 1,
        "4": 1,
        "5": 2,
    }
    assert lock["candidate_report_sha256"] == sha256_file(report_path)
    assert lock["smoke_training_allowed"] is False
    assert lock["forward_backward_allowed"] is False
    vectors = []
    for arm in SMOKE_ARMS:
        rows = [
            json.loads(line)
            for line in smoke_manifest_path(private_dir, arm)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        vectors.append([row["base_event_id"] for row in rows])
    assert all(vector == vectors[0] for vector in vectors[1:])
    prompt_rows = smoke_prompt_cache_path(private_dir).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(prompt_rows) == SMOKE_EVENTS_PER_ARM
    audit = audit_smoke_training_package(
        full_data_dir=data_dir,
        private_smoke_dir=private_dir,
        smoke_plan_path=REAL_PLAN,
        training_configuration_frozen_lock_path=configuration_path,
        materialized_manifest_frozen_lock_path=materialized_path,
        report_path=report_path,
        frozen_lock_path=lock_path,
    )
    assert audit["private_rows_exactly_reconstructed"] is True
    assert audit["prompt_rows_exactly_reconstructed"] is True
    assert audit["forward_backward_allowed"] is False


def _synthetic_smoke_authorization_files(
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    configuration_lock_path = root / "configuration-frozen.json"
    configuration_lock = {
        "status": "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "configuration_sha256": sha256_file(REAL_CONFIG),
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(configuration_lock_path, configuration_lock)
    closure = smoke_runtime_source_closure()
    smoke_lock_path = root / "smoke-package.json"
    smoke_lock = {
        "status": "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "smoke_plan_sha256": sha256_file(REAL_PLAN),
        "materialized_manifest_frozen_lock_sha256": "m" * 64,
        "configuration_sha256": sha256_file(REAL_CONFIG),
        "smoke_runtime_source_closure": closure,
        "smoke_runtime_source_closure_sha256": closure_sha256(closure),
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(smoke_lock_path, smoke_lock)
    authorization_path = root / "smoke-authorization.json"
    authorization = {
        "schema_version": "exp54-smoke-execution-authorization-v1",
        "authorization_mode": "smoke",
        "review_verdict": "SMOKE_PACKAGE_PASS",
        "reviewed_smoke_package_commit": "a" * 40,
        "smoke_package_frozen_lock_sha256": sha256_file(smoke_lock_path),
        "training_configuration_frozen_lock_sha256": sha256_file(
            configuration_lock_path
        ),
        "smoke_plan_sha256": sha256_file(REAL_PLAN),
        "materialized_manifest_frozen_lock_sha256": "m" * 64,
        "smoke_runtime_source_closure_sha256": closure_sha256(closure),
        "smoke_schema_version": "exp54-smoke-training-v1",
        "allowed_arms": list(SMOKE_ARMS),
        "allowed_seed": 42,
        "optimizer_steps_per_arm": 1,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "hyperparameter_selection_allowed": False,
    }
    _write_json(authorization_path, authorization)
    trusted_digest_path = root / "trusted-smoke-digest"
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    return (
        configuration_lock_path,
        smoke_lock_path,
        authorization_path,
        trusted_digest_path,
    )


def test_smoke_authorization_binds_reviewed_package_and_one_step(
    tmp_path: Path,
) -> None:
    (
        configuration_lock_path,
        smoke_lock_path,
        authorization_path,
        trusted_digest_path,
    ) = _synthetic_smoke_authorization_files(tmp_path)
    accepted = verify_smoke_authorization(
        authorization_path=authorization_path,
        config_path=REAL_CONFIG,
        smoke_package_lock_path=smoke_lock_path,
        training_configuration_frozen_lock_path=configuration_lock_path,
        arm="R3",
        trusted_digest_path=trusted_digest_path,
        require_root_owned_digest=False,
    )
    assert accepted["optimizer_steps_per_arm"] == 1
    assert accepted["formal_training_allowed"] is False
    equivalent_config = tmp_path / "equivalent-config.json"
    equivalent_config.write_text(
        json.dumps(json.loads(REAL_CONFIG.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="bind this config"):
        verify_smoke_authorization(
            authorization_path=authorization_path,
            config_path=equivalent_config,
            smoke_package_lock_path=smoke_lock_path,
            training_configuration_frozen_lock_path=configuration_lock_path,
            arm="R3",
            trusted_digest_path=trusted_digest_path,
            require_root_owned_digest=False,
        )
    forged = json.loads(authorization_path.read_text(encoding="utf-8"))
    forged["optimizer_steps_per_arm"] = 2
    _write_json(authorization_path, forged)
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    with pytest.raises(PermissionError, match="step budget"):
        verify_smoke_authorization(
            authorization_path=authorization_path,
            config_path=REAL_CONFIG,
            smoke_package_lock_path=smoke_lock_path,
            training_configuration_frozen_lock_path=configuration_lock_path,
            arm="R3",
            trusted_digest_path=trusted_digest_path,
            require_root_owned_digest=False,
        )


def test_smoke_runner_rejects_before_model_load_forward_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    prompt = tmp_path / "prompt.jsonl"
    manifest.write_text("", encoding="utf-8")
    prompt.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        smoke_runner,
        "_verify_frozen_smoke_package",
        lambda **_kwargs: ({}, manifest, prompt),
    )
    monkeypatch.setattr(
        smoke_runner,
        "verify_smoke_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("smoke execution is not authorized")
        ),
    )
    model_load_reached = False

    def _forbidden_model_verification(_config):
        nonlocal model_load_reached
        model_load_reached = True
        raise AssertionError("model verification must not be reached")

    monkeypatch.setattr(
        smoke_runner,
        "verify_model_snapshot",
        _forbidden_model_verification,
    )
    output_dir = tmp_path / "output"
    with pytest.raises(PermissionError, match="not authorized"):
        smoke_runner.run_smoke_training(
            arm="R3",
            config_path=REAL_CONFIG,
            training_configuration_frozen_lock_path=tmp_path / "config-lock",
            smoke_plan_path=REAL_PLAN,
            smoke_package_lock_path=tmp_path / "smoke-lock",
            smoke_authorization_path=None,
            private_smoke_dir=tmp_path / "private",
            output_dir=output_dir,
        )
    assert model_load_reached is False
    assert output_dir.exists() is False


def test_adapter_artifact_audit_rejects_base_weights(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    hashes = smoke_runner._adapter_artifact_hashes(tmp_path)
    assert set(hashes) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    (tmp_path / "model.safetensors").write_bytes(b"base")
    with pytest.raises(RuntimeError, match="base weights"):
        smoke_runner._adapter_artifact_hashes(tmp_path)


def test_real_public_smoke_locks_are_frozen_but_not_authorized() -> None:
    report = json.loads(REAL_SMOKE_REPORT.read_text(encoding="utf-8"))
    lock = json.loads(REAL_SMOKE_FROZEN_LOCK.read_text(encoding="utf-8"))
    configuration = json.loads(
        REAL_TRAINING_CONFIGURATION_FROZEN_LOCK.read_text(encoding="utf-8")
    )
    assert lock["candidate_report_sha256"] == sha256_file(REAL_SMOKE_REPORT)
    assert lock["training_configuration_frozen_lock_sha256"] == sha256_file(
        REAL_TRAINING_CONFIGURATION_FROZEN_LOCK
    )
    assert report["score_target_histogram"] == {
        "1": 2,
        "2": 2,
        "3": 1,
        "4": 1,
        "5": 2,
    }
    assert report["rationale_active_events_by_arm"] == {
        "S0": 0,
        "R1": 4,
        "R2": 4,
        "R3": 4,
    }
    assert report["smoke_runtime_source_closure_file_count"] == 20
    assert configuration["runtime_source_closure_file_count"] == 16
    for artifact in (report, lock, configuration):
        assert artifact["trust_anchor_install_allowed"] is False
        assert artifact["forward_backward_allowed"] is False
        assert artifact["smoke_training_allowed"] is False
        assert artifact["formal_training_allowed"] is False
        assert artifact["dev_accessed"] is False
        assert artifact["test_accessed"] is False
        assert artifact["training_used"] is False
