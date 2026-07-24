import copy
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.train_rar_sft import (
    TRAINING_SOURCE_PATHS,
    optimizer_group_plan,
    require_training_authorization,
    validate_training_configuration,
)
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import file_sha256


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


def _synthetic_formal_authorization() -> dict:
    return {
        "status": "EXP54_FORMAL_TRAINING_AUTHORIZED",
        "config_sha256": file_sha256(CONFIG_PATH),
        "frozen_manifest_lock_sha256": file_sha256(FROZEN_LOCK_PATH),
        "training_configuration_lock_sha256": file_sha256(
            TRAINING_CONFIG_LOCK_PATH
        ),
        "training_source_hashes": {
            name: file_sha256(path)
            for name, path in TRAINING_SOURCE_PATHS.items()
        },
        "allowed_arms": ["S0", "R1", "R2", "R3"],
        "allowed_seeds": [42, 43, 44],
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def test_future_authorization_must_bind_every_reviewed_layer(
    tmp_path: Path,
) -> None:
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(_synthetic_formal_authorization()),
        encoding="utf-8",
    )
    loaded = require_training_authorization(
        authorization_lock_path=authorization_path,
        config_path=CONFIG_PATH,
        frozen_lock_path=FROZEN_LOCK_PATH,
        training_config_lock_path=TRAINING_CONFIG_LOCK_PATH,
        arm="R3",
        seed=42,
    )
    assert loaded["status"] == "EXP54_FORMAL_TRAINING_AUTHORIZED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_sha256", "0" * 64),
        ("frozen_manifest_lock_sha256", "0" * 64),
        ("training_configuration_lock_sha256", "0" * 64),
        ("training_source_hashes", {}),
        ("allowed_arms", ["S0"]),
        ("allowed_seeds", [43, 44]),
        ("dev_accessed", True),
    ],
)
def test_future_authorization_tampering_hard_fails(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    authorization = _synthetic_formal_authorization()
    authorization[field] = value
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(authorization),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        require_training_authorization(
            authorization_lock_path=authorization_path,
            config_path=CONFIG_PATH,
            frozen_lock_path=FROZEN_LOCK_PATH,
            training_config_lock_path=TRAINING_CONFIG_LOCK_PATH,
            arm="R3",
            seed=42,
        )
