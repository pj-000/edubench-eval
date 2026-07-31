from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_label2_identification_protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
)


def _protocol() -> dict:
    return json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


def test_frozen_protocol_passes() -> None:
    validate_protocol(_protocol(), repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backbone_training_allowed", True),
        ("adapter_training_allowed", True),
        ("new_preference_pair_construction_allowed", True),
        ("test_predictions_must_not_be_read", False),
    ],
)
def test_training_and_test_boundaries_fail_closed(field: str, value: bool) -> None:
    protocol = _protocol()
    protocol["evidence_scope"][field] = value
    with pytest.raises(ValueError):
        validate_protocol(protocol, repo_root=REPO_ROOT)


def test_test_artifact_cannot_enter_locked_inputs(tmp_path: Path) -> None:
    protocol = _protocol()
    protocol["locked_inputs"]["forbidden"] = {
        "path": "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
        "sorc_dpo_one_time_test_v1/final_results/final_results.json",
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="test artifact"):
        validate_protocol(protocol, repo_root=REPO_ROOT)


def test_attribution_order_cannot_change() -> None:
    protocol = _protocol()
    protocol["primary_exclusive_attribution"]["order"][0:2] = reversed(
        protocol["primary_exclusive_attribution"]["order"][0:2]
    )
    with pytest.raises(ValueError, match="order changed"):
        validate_protocol(protocol, repo_root=REPO_ROOT)


def test_single_token_score_assumption_is_rejected() -> None:
    protocol = _protocol()
    protocol["score_probability_contract"][
        "single_token_assumption_allowed"
    ] = True
    with pytest.raises(ValueError, match="single-token"):
        validate_protocol(protocol, repo_root=REPO_ROOT)


def test_row_cannot_score_its_own_calibrator() -> None:
    protocol = _protocol()
    protocol["calibration_protocol"][
        "no_record_scores_its_own_calibrator"
    ] = False
    with pytest.raises(ValueError, match="leakage guard"):
        validate_protocol(protocol, repo_root=REPO_ROOT)


def test_locked_input_hash_mismatch_is_rejected() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["locked_inputs"]["dev"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_protocol(protocol, repo_root=REPO_ROOT)
