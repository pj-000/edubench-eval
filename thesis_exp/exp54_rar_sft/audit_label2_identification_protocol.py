"""Fail-closed validation for the frozen Label-2 identification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/label2_identification_audit_v1.json"
)

EXPECTED_SCHEMA = "exp54-label2-identification-audit-v1"
EXPECTED_STATUS = "PROTOCOL_FROZEN_NOT_EXECUTED"
EXPECTED_ARMS = ("P1_FIELD_DPO", "R3")
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_ATTRIBUTION_ORDER = (
    "measurement_ambiguous",
    "rubric_incomplete",
    "decoder_failure",
    "prior_recoverable",
    "calibration_recoverable",
    "support_deficient",
    "preference_coverage_deficient",
    "residual",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate_protocol(protocol: dict[str, Any], *, repo_root: Path) -> None:
    if protocol.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unexpected protocol schema")
    if protocol.get("status") != EXPECTED_STATUS:
        raise ValueError("protocol is not frozen and unexecuted")

    estimand = protocol["estimand"]
    if estimand.get("target") != "observed_consensus_label_5":
        raise ValueError("audit target drifted away from observed label_5")
    if (
        estimand.get("primary_arm"),
        estimand.get("paired_comparator_arm"),
    ) != EXPECTED_ARMS:
        raise ValueError("primary/comparator arms changed")
    if tuple(estimand.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("formal seeds changed")
    if estimand.get("primary_split") != "dev":
        raise ValueError("primary split must remain dev")
    if estimand.get("dependence_unit") != "question_key":
        raise ValueError("question clustering was removed")

    scope = protocol["evidence_scope"]
    if scope.get("allowed_splits") != ["train", "dev"]:
        raise ValueError("only train/dev may be used")
    forbidden = set(scope.get("forbidden_splits", []))
    if not {"test", "one_time_test", "holdout_2"}.issubset(forbidden):
        raise ValueError("test exclusions are incomplete")
    for field in (
        "backbone_training_allowed",
        "adapter_training_allowed",
        "new_preference_pair_construction_allowed",
    ):
        if scope.get(field) is not False:
            raise ValueError(f"{field} must be false")
    if scope.get("test_predictions_must_not_be_read") is not True:
        raise ValueError("test read prohibition is missing")

    locked_inputs = protocol["locked_inputs"]
    for name, item in locked_inputs.items():
        relative = str(item.get("path", ""))
        if not relative or Path(relative).is_absolute():
            raise ValueError(f"{name}: path must be a nonempty repo-relative path")
        lowered_parts = {part.lower() for part in Path(relative).parts}
        if "test" in lowered_parts or "sorc_dpo_one_time_test_v1" in lowered_parts:
            raise ValueError(f"{name}: test artifact is forbidden")
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if _sha256(path) != item.get("sha256"):
            raise ValueError(f"{name}: locked input hash mismatch")

    attribution = protocol["primary_exclusive_attribution"]
    order = tuple(attribution.get("order", []))
    if order != EXPECTED_ATTRIBUTION_ORDER:
        raise ValueError("exclusive attribution order changed")
    definitions = attribution.get("definitions", {})
    if set(definitions) != set(EXPECTED_ATTRIBUTION_ORDER):
        raise ValueError("attribution definitions are incomplete or extra")
    flags = tuple(protocol.get("independent_sensitivity_flags", []))
    if flags != EXPECTED_ATTRIBUTION_ORDER[:-1]:
        raise ValueError("sensitivity flags must mirror non-residual mechanisms")

    score_contract = protocol["score_probability_contract"]
    if score_contract.get("candidate_values") != [1, 2, 3, 4, 5]:
        raise ValueError("canonical score candidate set changed")
    if score_contract.get("single_token_assumption_allowed") is not False:
        raise ValueError("single-token score assumption is forbidden")
    if score_contract.get("rationale_tokens_used") is not False:
        raise ValueError("score probabilities must not depend on rationale tokens")

    calibration = protocol["calibration_protocol"]
    if calibration.get("data") != "dev only":
        raise ValueError("calibration data scope changed")
    if calibration.get("outer_grouping") != "question_key":
        raise ValueError("calibration must remain question-group cross-fitted")
    if calibration.get("no_record_scores_its_own_calibrator") is not True:
        raise ValueError("calibration leakage guard is disabled")

    statistics = protocol["statistics"]
    if statistics.get("bootstrap_cluster") != "question_key":
        raise ValueError("bootstrap must remain question-clustered")
    if statistics.get("bootstrap_replicates") != 10000:
        raise ValueError("bootstrap replicate count changed")
    gate = statistics["primary_gate"]
    if gate.get("exclusive_fraction_at_least") != 0.6:
        raise ValueError("dominance fraction changed")
    if gate.get("cluster_ci_lower_strictly_greater_than") != 0.5:
        raise ValueError("dominance confidence bound changed")
    if gate.get("minimum_seeds_passing") != 2:
        raise ValueError("seed replication requirement changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = _load_object(args.protocol)
    validate_protocol(protocol, repo_root=REPO_ROOT)
    print("LABEL2_IDENTIFICATION_PROTOCOL_PASS")


if __name__ == "__main__":
    main()
