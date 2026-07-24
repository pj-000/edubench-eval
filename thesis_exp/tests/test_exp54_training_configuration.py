import copy
import hashlib
import json
from pathlib import Path

import pytest

import thesis_exp.exp54_rar_sft.train_rar_sft as train_module
from thesis_exp.exp54_rar_sft.authorization_guard import (
    EXPECTED_RUNTIME_SOURCE_NAMES,
    RUNTIME_SOURCE_PATHS,
    closure_sha256,
    read_trusted_authorization_digest,
    runtime_source_closure,
    sha256_file,
    verify_external_authorization,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    optimizer_group_plan,
    require_adapter_free_base_model,
    require_training_authorization,
    run_training,
    validate_model_directory_listing,
    validate_training_configuration,
)


CONFIG_PATH = Path(
    "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
FROZEN_LOCK_PATH = Path(
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
TRAINING_CONFIG_LOCK_PATH = Path(
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_candidate_lock.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_candidate_training_configuration_has_exact_step_plan() -> None:
    plan = validate_training_configuration(_config())
    assert plan == {
        "micro_batches_per_logical_epoch": 1_327,
        "optimizer_steps_per_logical_epoch": 332,
        "full_accumulation_groups_per_logical_epoch": 331,
        "remainder_micro_batches_per_logical_epoch": 3,
        "total_optimizer_steps": 996,
    }
    groups = optimizer_group_plan(
        rows_per_epoch=2_654,
        micro_batch_size=2,
        gradient_accumulation_steps=4,
    )
    assert groups == [4] * 331 + [3]


def test_runtime_source_closure_contains_every_imported_package_init() -> None:
    assert set(RUNTIME_SOURCE_PATHS) == EXPECTED_RUNTIME_SOURCE_NAMES
    assert {
        "thesis_package_init",
        "thesis_src_package_init",
        "edujudge_package_init",
        "exp02_package_init",
        "utils_package_init",
        "exp54_package_init",
    }.issubset(RUNTIME_SOURCE_PATHS)
    assert len(runtime_source_closure()) == 16


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", "physical_manifest_passes"), 3),
        (("data", "shuffle"), True),
        (("optimization", "gradient_accumulation_steps"), 8),
        (("optimization", "warmup_steps"), 49),
        (("loss", "rationale_block_weight_when_active"), 0.5),
        (("lora", "target_modules"), ["q_proj", "v_proj"]),
        (("precision_and_memory", "tf32"), True),
        (("checkpointing", "evaluation_during_training"), True),
        (("authorization", "formal_training_allowed"), True),
    ],
)
def test_configuration_tampering_hard_fails(path: tuple[str, ...], value) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="training config differs"):
        validate_training_configuration(config)


def test_training_refuses_without_future_authorization_lock() -> None:
    with pytest.raises(PermissionError, match="not authorized"):
        require_training_authorization(
            authorization_lock_path=None,
            config_path=CONFIG_PATH,
            frozen_lock_path=FROZEN_LOCK_PATH,
            training_config_lock_path=TRAINING_CONFIG_LOCK_PATH,
            arm="R3",
            seed=42,
        )


def _synthetic_configuration_lock() -> dict:
    closure = runtime_source_closure()
    return {
        "status": "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED",
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "manifest_frozen_lock_sha256": sha256_file(FROZEN_LOCK_PATH),
        "candidate_report_sha256": "a" * 64,
        "runtime_source_closure": closure,
        "runtime_source_closure_sha256": closure_sha256(closure),
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def _synthetic_formal_authorization(
    training_config_lock_path: Path,
) -> dict:
    closure = runtime_source_closure()
    return {
        "schema_version": "exp54-external-authorization-v1",
        "authorization_mode": "formal",
        "reviewed_training_configuration_commit": "b" * 40,
        "review_verdict": "TRAINING_CONFIGURATION_PASS",
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "frozen_manifest_lock_sha256": sha256_file(FROZEN_LOCK_PATH),
        "training_configuration_lock_sha256": sha256_file(
            training_config_lock_path
        ),
        "candidate_report_sha256": "a" * 64,
        "runtime_source_closure_sha256": closure_sha256(closure),
        "allowed_arms": ["S0", "R1", "R2", "R3"],
        "allowed_seeds": [42, 43, 44],
        "sealed_data_state": {
            "dev_accessed": False,
            "test_accessed": False,
            "training_used": False,
        },
    }


def test_future_authorization_must_bind_every_reviewed_layer(
    tmp_path: Path,
) -> None:
    training_config_lock_path = tmp_path / "configuration-lock.json"
    training_config_lock_path.write_text(
        json.dumps(_synthetic_configuration_lock()),
        encoding="utf-8",
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(_synthetic_formal_authorization(training_config_lock_path)),
        encoding="utf-8",
    )
    trusted_digest_path = tmp_path / "trusted-digest"
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    loaded = verify_external_authorization(
        authorization_lock_path=authorization_path,
        config_path=CONFIG_PATH,
        frozen_lock_path=FROZEN_LOCK_PATH,
        training_config_lock_path=training_config_lock_path,
        arm="R3",
        seed=42,
        trusted_digest_path=trusted_digest_path,
        require_root_owned_digest=False,
    )
    assert loaded["review_verdict"] == "TRAINING_CONFIGURATION_PASS"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configuration_sha256", "0" * 64),
        ("frozen_manifest_lock_sha256", "0" * 64),
        ("training_configuration_lock_sha256", "0" * 64),
        ("runtime_source_closure_sha256", "0" * 64),
        ("allowed_arms", ["S0"]),
        ("allowed_seeds", [43, 44]),
        (
            "sealed_data_state",
            {
                "dev_accessed": True,
                "test_accessed": False,
                "training_used": False,
            },
        ),
    ],
)
def test_future_authorization_tampering_hard_fails(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    training_config_lock_path = tmp_path / "configuration-lock.json"
    training_config_lock_path.write_text(
        json.dumps(_synthetic_configuration_lock()),
        encoding="utf-8",
    )
    authorization = _synthetic_formal_authorization(
        training_config_lock_path
    )
    authorization[field] = value
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )
    trusted_digest_path = tmp_path / "trusted-digest"
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    with pytest.raises(PermissionError):
        verify_external_authorization(
            authorization_lock_path=authorization_path,
            config_path=CONFIG_PATH,
            frozen_lock_path=FROZEN_LOCK_PATH,
            training_config_lock_path=training_config_lock_path,
            arm="R3",
            seed=42,
            trusted_digest_path=trusted_digest_path,
            require_root_owned_digest=False,
        )


def test_self_consistent_forged_lock_pair_lacks_external_authenticity(
    tmp_path: Path,
) -> None:
    training_config_lock_path = tmp_path / "forged-config-lock.json"
    training_config_lock_path.write_text(
        json.dumps(_synthetic_configuration_lock()),
        encoding="utf-8",
    )
    authorization_path = tmp_path / "forged-authorization.json"
    authorization_path.write_text(
        json.dumps(_synthetic_formal_authorization(training_config_lock_path)),
        encoding="utf-8",
    )
    trusted_digest_path = tmp_path / "trusted-digest"
    trusted_digest_path.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(PermissionError, match="trusted external.*digest"):
        verify_external_authorization(
            authorization_lock_path=authorization_path,
            config_path=CONFIG_PATH,
            frozen_lock_path=FROZEN_LOCK_PATH,
            training_config_lock_path=training_config_lock_path,
            arm="R3",
            seed=42,
            trusted_digest_path=trusted_digest_path,
            require_root_owned_digest=False,
        )


@pytest.mark.parametrize(
    "closure_name",
    [
        "manifest_io_and_split_guard",
        "exp54_package_init",
        "prompt_cleaning",
        "utils_package_init",
    ],
)
def test_runtime_dependency_change_invalidates_old_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closure_name: str,
) -> None:
    dependency = tmp_path / f"{closure_name}.py"
    dependency.write_text("original\n", encoding="utf-8")
    monkeypatch.setitem(RUNTIME_SOURCE_PATHS, closure_name, dependency)
    training_config_lock_path = tmp_path / "configuration-lock.json"
    training_config_lock_path.write_text(
        json.dumps(_synthetic_configuration_lock()),
        encoding="utf-8",
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(_synthetic_formal_authorization(training_config_lock_path)),
        encoding="utf-8",
    )
    trusted_digest_path = tmp_path / "trusted-digest"
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    dependency.write_text("changed\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="source closure"):
        verify_external_authorization(
            authorization_lock_path=authorization_path,
            config_path=CONFIG_PATH,
            frozen_lock_path=FROZEN_LOCK_PATH,
            training_config_lock_path=training_config_lock_path,
            arm="R3",
            seed=42,
            trusted_digest_path=trusted_digest_path,
            require_root_owned_digest=False,
        )


def test_trusted_digest_rejects_writable_or_non_root_anchor(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted-digest"
    trusted.write_text("0" * 64 + "\n", encoding="ascii")
    trusted.chmod(0o666)
    with pytest.raises(PermissionError, match="writable"):
        read_trusted_authorization_digest(
            trusted,
            require_root_owned=False,
        )
    trusted.chmod(0o400)
    with pytest.raises(PermissionError, match="owned by root"):
        read_trusted_authorization_digest(
            trusted,
            require_root_owned=True,
        )


def test_trusted_digest_rejects_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_text("0" * 64 + "\n", encoding="ascii")
    link = tmp_path / "trusted-digest"
    link.symlink_to(target)
    with pytest.raises(PermissionError, match="non-symlink"):
        read_trusted_authorization_digest(
            link,
            require_root_owned=False,
        )


def test_model_directory_rejects_added_adapter_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    expected = {"config.json": 2}
    payload = json.dumps(
        sorted(expected.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    listing_hash = hashlib.sha256(payload).hexdigest()
    validate_model_directory_listing(
        tmp_path,
        expected_regular_files=expected,
        expected_listing_sha256=listing_hash,
    )
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter artifacts"):
        validate_model_directory_listing(
            tmp_path,
            expected_regular_files=expected,
            expected_listing_sha256=listing_hash,
        )


@pytest.mark.parametrize(
    "artifact",
    ["adapter_model.safetensors", "adapter_model.bin"],
)
def test_model_directory_rejects_nested_adapter_weights(
    tmp_path: Path,
    artifact: str,
) -> None:
    nested = tmp_path / "historical"
    nested.mkdir()
    (nested / artifact).write_bytes(b"adapter")
    with pytest.raises(ValueError, match="adapter artifacts"):
        validate_model_directory_listing(
            tmp_path,
            expected_regular_files={},
            expected_listing_sha256="0" * 64,
        )


def test_model_directory_rejects_symlink_root_with_exact_listing(
    tmp_path: Path,
) -> None:
    real_model = tmp_path / "real-model"
    real_model.mkdir()
    (real_model / "config.json").write_text("{}", encoding="utf-8")
    expected = {"config.json": 2}
    payload = json.dumps(
        sorted(expected.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    listing_hash = hashlib.sha256(payload).hexdigest()
    model_link = tmp_path / "model-link"
    model_link.symlink_to(real_model, target_is_directory=True)
    with pytest.raises(ValueError, match="root must not be a symlink"):
        validate_model_directory_listing(
            model_link,
            expected_regular_files=expected,
            expected_listing_sha256=listing_hash,
        )


def test_model_directory_rejects_non_directory_root(
    tmp_path: Path,
) -> None:
    model_file = tmp_path / "not-a-directory"
    model_file.write_text("model", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a directory"):
        validate_model_directory_listing(
            model_file,
            expected_regular_files={},
            expected_listing_sha256="0" * 64,
        )


class _MockBaseModel:
    def __init__(
        self,
        *,
        hf_loaded: bool = False,
        peft_config=None,
        parameter_names: tuple[str, ...] = (),
    ) -> None:
        self._hf_peft_config_loaded = hf_loaded
        self.peft_config = peft_config
        self._parameter_names = parameter_names

    def named_parameters(self):
        return [(name, object()) for name in self._parameter_names]


def test_clean_base_model_passes_pre_lora_guard() -> None:
    require_adapter_free_base_model(_MockBaseModel())


@pytest.mark.parametrize(
    "model",
    [
        _MockBaseModel(hf_loaded=True),
        _MockBaseModel(peft_config={"default": {}}),
        _MockBaseModel(parameter_names=("model.layers.0.q_proj.lora_A",)),
        _MockBaseModel(parameter_names=("model.layers.0.adapter.weight",)),
    ],
)
def test_pre_lora_guard_rejects_existing_adapter_state(model) -> None:
    with pytest.raises(RuntimeError, match="base model"):
        require_adapter_free_base_model(model)


def test_failed_external_authorization_precedes_model_load_and_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "training_manifest_r3_seed42.jsonl"
    prompt_cache = data_dir / "shared_prompt_cache.jsonl"
    manifest.write_text("", encoding="utf-8")
    prompt_cache.write_text("", encoding="utf-8")
    frozen_lock = tmp_path / "frozen.json"
    frozen_lock.write_text(
        json.dumps(
            {
                "status": (
                    "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
                ),
                "manifest_frozen": True,
                "smoke_training_allowed": False,
                "formal_training_allowed": False,
                "dev_accessed": False,
                "test_accessed": False,
                "training_used": False,
                "private_artifact_hashes": {
                    "manifests_by_seed": {
                        "seed42": {"R3": sha256_file(manifest)}
                    },
                    "shared_prompt_cache": sha256_file(prompt_cache),
                },
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "forged-authorization.json"
    authorization.write_text("{}", encoding="utf-8")
    config_lock = tmp_path / "forged-config-lock.json"
    config_lock.write_text("{}", encoding="utf-8")
    model_load_called = False

    def _forbidden_model_load(_config):
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("model loading must not be reached")

    monkeypatch.setattr(
        train_module,
        "verify_model_snapshot",
        _forbidden_model_load,
    )
    output_dir = tmp_path / "output"
    with pytest.raises(PermissionError, match="trusted external.*digest"):
        run_training(
            config_path=CONFIG_PATH,
            frozen_lock_path=frozen_lock,
            training_config_lock_path=config_lock,
            authorization_lock_path=authorization,
            data_dir=data_dir,
            output_dir=output_dir,
            arm="R3",
            seed=42,
        )
    assert model_load_called is False
    assert not output_dir.exists()
