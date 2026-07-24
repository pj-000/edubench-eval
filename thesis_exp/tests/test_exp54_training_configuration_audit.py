import copy
import hashlib
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import file_sha256
from thesis_exp.exp54_rar_sft.audit_training_configuration import (
    validate_tensor_shard_mapping,
    validate_runtime,
)


CONFIG = json.loads(
    Path(
        "thesis_exp/exp54_rar_sft/configs/"
        "training_configuration_candidate.json"
    ).read_text(encoding="utf-8")
)


def _observed_runtime() -> dict:
    runtime = copy.deepcopy(CONFIG["runtime"])
    runtime.pop("required_gpu_name")
    runtime.pop("visible_gpu_count")
    runtime.pop("bf16_required")
    driver = runtime.pop("driver")
    runtime["cuda_available"] = True
    runtime["bf16_supported"] = True
    runtime["gpu_inventory"] = [
        {
            "name": "NVIDIA RTX A6000",
            "driver": driver,
            "memory_mib": 49_140,
        }
        for _ in range(4)
    ]
    return runtime


def test_runtime_candidate_accepts_four_locked_a6000_gpus() -> None:
    checks = validate_runtime(_observed_runtime(), CONFIG["runtime"])
    assert checks["all_versions_match"] is True
    assert checks["matching_rtx_a6000_count"] == 4
    assert checks["single_visible_gpu_required_at_training"] is True


def test_runtime_version_drift_hard_fails() -> None:
    observed = _observed_runtime()
    observed["transformers"] = "different"
    with pytest.raises(ValueError, match="runtime differs"):
        validate_runtime(observed, CONFIG["runtime"])


def test_runtime_rejects_insufficient_matched_hardware() -> None:
    observed = _observed_runtime()
    observed["gpu_inventory"] = observed["gpu_inventory"][:3]
    with pytest.raises(ValueError, match="fewer than four"):
        validate_runtime(observed, CONFIG["runtime"])


def test_per_tensor_shard_mapping_accepts_exact_index() -> None:
    result = validate_tensor_shard_mapping(
        weight_map={"x": "a.safetensors", "y": "b.safetensors"},
        tensor_names_by_shard={
            "a.safetensors": {"x"},
            "b.safetensors": {"y"},
        },
        expected_shards={"a.safetensors", "b.safetensors"},
    )
    assert all(value is True or value == 0 for value in result.values())


def test_per_tensor_shard_mapping_rejects_swapped_index() -> None:
    with pytest.raises(ValueError, match="per-tensor shard mapping"):
        validate_tensor_shard_mapping(
            weight_map={"x": "b.safetensors", "y": "a.safetensors"},
            tensor_names_by_shard={
                "a.safetensors": {"x"},
                "b.safetensors": {"y"},
            },
            expected_shards={"a.safetensors", "b.safetensors"},
        )


def test_per_tensor_shard_mapping_rejects_duplicate_tensor_names() -> None:
    with pytest.raises(ValueError, match="duplicated across shards"):
        validate_tensor_shard_mapping(
            weight_map={"x": "a.safetensors", "y": "b.safetensors"},
            tensor_names_by_shard={
                "a.safetensors": {"x"},
                "b.safetensors": {"x", "y"},
            },
            expected_shards={"a.safetensors", "b.safetensors"},
        )


def test_formal_configuration_report_is_aggregate_and_not_authorized() -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    report_path = base / "audit/training_configuration_candidate_report.json"
    lock_path = base / "protocol/training_configuration_candidate_lock.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert report["status"] == (
        "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED"
    )
    assert lock["status"] == report["status"]
    assert lock["candidate_report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert report["manifest_freeze"]["private_artifacts_rehashed_equal"] is True
    assert report["model_snapshot"]["all_file_sizes_and_hashes_match"] is True
    assert report["model_snapshot"]["base_parameters"] == 4_022_468_096
    assert report["model_snapshot"]["lora_trainable_parameters"] == 33_030_144
    assert report["model_snapshot"]["regular_file_count"] == 16
    assert report["model_snapshot"]["forbidden_adapter_artifact_count"] == 0
    assert report["model_snapshot"]["historical_adapter_artifacts_absent"] is True
    assert report["model_snapshot"]["pre_lora_base_model_adapter_free"] is True
    assert report["model_snapshot"]["per_tensor_shard_mapping_match"] is True
    assert (
        report["model_snapshot"]["duplicate_tensor_names_across_shards"] == 0
    )
    assert report["model_snapshot"]["unindexed_tensor_count"] == 0
    assert report["model_snapshot"]["missing_indexed_tensor_count"] == 0
    assert set(
        report["model_snapshot"]["lora_target_occurrences"].values()
    ) == {36}
    assert all(report["runtime_checks"].values())
    assert report["data_and_step_plan"]["physical_manifest_passes"] == 1
    assert report["data_and_step_plan"]["logical_epochs_in_manifest"] == 3
    assert report["data_and_step_plan"]["total_optimizer_steps"] == 996
    generation = report["generation_train_target_coverage"]
    assert generation["generation_cap_covers_all_frozen_train_targets"] is True
    assert not any(
        generation["events_exceeding_max_new_tokens_by_arm"].values()
    )
    assert generation["max_materialized_assistant_tokens_by_arm"] == {
        "R1": 157,
        "R2": 151,
        "R3": 151,
        "S0": 11,
    }
    assert report["runtime_source_closure_complete"] is True
    assert len(report["runtime_source_closure"]) == 15
    authorization = report["authorization_gate"]
    assert authorization["authorization_authenticity_verified"] is True
    assert authorization["forged_lock_pair_rejected"] is True
    assert authorization["untrusted_arbitrary_lock_paths_rejected"] is True
    assert authorization["external_digest_required"] is True
    assert report["audit_source_hashes"]["configuration_auditor"] == file_sha256(
        Path("thesis_exp/exp54_rar_sft/audit_training_configuration.py")
    )
    assert report["audit_source_hashes"]["manifest_freezer"] == file_sha256(
        Path("thesis_exp/exp54_rar_sft/freeze_materialized_manifests.py")
    )
    for artifact in (report, lock):
        assert artifact["smoke_training_allowed"] is False
        assert artifact["formal_training_allowed"] is False
        assert artifact["dev_accessed"] is False
        assert artifact["test_accessed"] is False
        assert artifact["training_used"] is False
