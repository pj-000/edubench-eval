from __future__ import annotations

import copy
import json

import pytest

from thesis_exp.exp54_rar_sft.audit_llm_judge_spec_pilot_protocol import (
    DEFAULT_CONFIG,
    audit_protocol,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_protocol_passes() -> None:
    audit_protocol(_config())


@pytest.mark.parametrize("split", ["dev", "test", "all"])
def test_non_train_sampling_fails(split: str) -> None:
    config = _config()
    config["sampling"]["split"] = split
    with pytest.raises(ValueError, match="not train-only"):
        audit_protocol(config)


def test_evaluation_leak_to_proposer_fails() -> None:
    config = _config()
    config["sampling"]["evaluation_visible_to_clarification_proposer"] = True
    with pytest.raises(ValueError, match="leak"):
        audit_protocol(config)


def test_cross_condition_judge_reuse_fails() -> None:
    config = _config()
    config["roles"]["cross_condition_judge_reuse_allowed"] = True
    with pytest.raises(ValueError, match="reuse"):
        audit_protocol(config)


def test_policy_changing_clarification_fails() -> None:
    config = _config()
    config["clarification_contract"]["new_criterion_allowed"] = True
    with pytest.raises(ValueError, match="policy-changing"):
        audit_protocol(config)


def test_rating_budget_tamper_fails() -> None:
    config = _config()
    config["judge_execution"]["formal_rating_decisions"] = 159
    with pytest.raises(ValueError, match="budget"):
        audit_protocol(config)


def test_evaluation_composition_tamper_fails() -> None:
    config = _config()
    config["sampling"]["clear_anchor_evaluation_items_per_group"] = 0
    with pytest.raises(ValueError, match="clear-anchor"):
        audit_protocol(config)


def test_method_claim_authorization_fails() -> None:
    config = copy.deepcopy(_config())
    config["execution_boundary"]["CCF_A_method_claim_allowed"] = True
    with pytest.raises(ValueError, match="authorization"):
        audit_protocol(config)
