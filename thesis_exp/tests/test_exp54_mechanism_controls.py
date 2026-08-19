from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_mechanism_control_manifests import (
    FULLSEQ_SCHEMA_VERSION,
    attach_full_completion_metadata,
)
from thesis_exp.exp54_rar_sft.audit_mechanism_control_manifests import (
    _audit_fullseq_rows,
    _audit_p1_syn_match,
)


CONFIG_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_controls_candidate_v1.json"
)
CONTRACT_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/THESIS_SCIENTIFIC_CONTRACT_V1.md"
)
MATERIALIZED_LOCK_PATH = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
PREFERENCE_LOCK_PATH = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_training_candidate/preference_training_frozen_lock.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p1_row() -> dict[str, object]:
    return {
        "pair_id": "pair-a",
        "record_id": "record-a",
        "pair_task": "score",
        "pair_type": "adjacent_score",
        "active_field": "score",
        "odpo_offset": 0.0,
        "objective_weight": 1.0,
        "chosen_input_ids": [10, 11, 12, 13, 14],
        "chosen_field_token_positions": [2],
        "rejected_input_ids": [10, 11, 15, 16, 17, 18],
        "rejected_field_token_positions": [2],
        "cutoff_len": 8,
    }


def _prompt_cache() -> dict[str, dict[str, object]]:
    return {
        "record-a": {
            "record_id": "record-a",
            "prompt_token_ids": [10, 11],
        }
    }


def test_candidate_freezes_exact_nine_run_matrix_and_contract_hash() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["status"] == "CPU_IMPLEMENTATION_CANDIDATE_GPU_NOT_AUTHORIZED"
    assert config["scientific_contract"]["sha256"] == _sha256(CONTRACT_PATH)
    assert config["run_count"] == 9
    assert [
        (row["arm"], row["seed"]) for row in config["run_matrix"]
    ] == [
        ("R3_TOKENAVG", 42),
        ("R3_TOKENAVG", 43),
        ("R3_TOKENAVG", 44),
        ("P1_FULLSEQ", 42),
        ("P1_FULLSEQ", 43),
        ("P1_FULLSEQ", 44),
        ("P1_SYN_LR5E6", 42),
        ("P1_SYN_LR5E6", 43),
        ("P1_SYN_LR5E6", 44),
    ]
    assert config["controls"]["P1_SYN_LR5E6"][
        "old_seed42_lr5e7_checkpoint_reusable"
    ] is False
    assert config["authorization"] == {
        "cpu_unit_tests_allowed": True,
        "private_train_manifest_audit_allowed": True,
        "gpu_smoke_allowed": False,
        "full_gpu_training_allowed": False,
        "dev_inference_allowed": False,
        "test_inference_allowed": False,
    }


def test_new_controls_do_not_modify_frozen_loss_or_manifest_builder() -> None:
    materialized = json.loads(
        MATERIALIZED_LOCK_PATH.read_text(encoding="utf-8")
    )
    preference = json.loads(
        PREFERENCE_LOCK_PATH.read_text(encoding="utf-8")
    )
    assert _sha256(
        REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
    ) == materialized["source_hashes"]["block_loss_and_collator"]
    assert _sha256(
        REPO_ROOT / "thesis_exp/exp54_rar_sft/sorc_dpo_loss.py"
    ) == preference["source_hashes"]["loss_and_collator"]
    assert _sha256(
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/"
        "build_sorc_dpo_training_manifests.py"
    ) == preference["source_hashes"]["manifest_builder"]


def test_fullseq_overlay_preserves_sequences_and_adds_complete_suffix() -> None:
    source = _p1_row()
    output = attach_full_completion_metadata([source], _prompt_cache())

    assert len(output) == 1
    row = output[0]
    assert row["mechanism_control_schema_version"] == FULLSEQ_SCHEMA_VERSION
    assert row["chosen_input_ids"] == source["chosen_input_ids"]
    assert row["rejected_input_ids"] == source["rejected_input_ids"]
    assert row["chosen_prompt_token_count"] == 2
    assert row["chosen_completion_token_positions"] == [2, 3, 4]
    assert row["rejected_prompt_token_count"] == 2
    assert row["rejected_completion_token_positions"] == [2, 3, 4, 5]
    assert "chosen_completion_token_positions" not in source


def test_independent_auditor_rejects_changed_frozen_p1_field() -> None:
    sources = [dict(_p1_row()) for _ in range(838)]
    controls = attach_full_completion_metadata(sources, _prompt_cache())
    _audit_fullseq_rows(sources, controls, _prompt_cache())
    controls[0]["objective_weight"] = 2.0
    with pytest.raises(ValueError, match="frozen P1 field changed"):
        _audit_fullseq_rows(sources, controls, _prompt_cache())


def test_independent_auditor_rejects_p1_syn_record_substitution() -> None:
    actual = _p1_row()
    synthetic = dict(actual)
    _audit_p1_syn_match([actual] * 838, [synthetic] * 838)
    changed = [dict(synthetic) for _ in range(838)]
    changed[0]["record_id"] = "record-b"
    with pytest.raises(ValueError, match="matched field changed"):
        _audit_p1_syn_match([actual] * 838, changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.update(pair_task="rationale"),
            "non-score",
        ),
        (
            lambda row: row.update(odpo_offset=0.2),
            "ODPO offset",
        ),
        (
            lambda row: row.update(chosen_input_ids=[99, 11, 12]),
            "prompt prefix",
        ),
        (
            lambda row: row.update(chosen_field_token_positions=[1]),
            "escapes completion",
        ),
    ],
)
def test_fullseq_overlay_tamper_hard_fails(mutation, message: str) -> None:
    row = _p1_row()
    mutation(row)
    with pytest.raises(ValueError, match=message):
        attach_full_completion_metadata([row], _prompt_cache())


def test_contract_scientific_boundaries_are_explicit() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    for required in (
        "Total new full training runs: **9**",
        "R3-BLOCK minus R3-TOKENAVG",
        "P1-FIELD minus P1-FULLSEQ",
        "P1-ACTUAL minus P1-SYN",
        "post-hoc mechanism analyses on a frozen benchmark test",
        "untouched-test preregistered confirmatory evidence",
    ):
        assert required in contract
