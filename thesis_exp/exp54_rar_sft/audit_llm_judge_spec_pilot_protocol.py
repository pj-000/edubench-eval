"""Fail-closed audit for the LLM-judge specification-intervention pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp54_rar_sft import REPO_ROOT


DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/llm_judge_spec_intervention_pilot_v1.json"
)


def audit_protocol(config: dict) -> None:
    if config["schema_version"] != "exp54-llm-judge-spec-intervention-pilot-v1":
        raise ValueError("unexpected schema version")
    if config["status"] != "PROTOCOL_FROZEN_NOT_EXECUTED":
        raise ValueError("protocol is not in the frozen pre-execution state")
    population = config["population"]
    if population != {
        "type": "LLM_judge_instances",
        "runtime_model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "not_human_raters": True,
        "not_posterior_samples": True,
    }:
        raise ValueError("judge population identity differs")

    sampling = config["sampling"]
    if sampling["split"] != "train_only":
        raise ValueError("pilot sampling is not train-only")
    if sampling["target_adjacent_boundaries"] != ["1-2", "2-3", "3-4", "4-5"]:
        raise ValueError("all four adjacent boundaries are not frozen")
    if sampling["rubric_boundary_groups"] != 4:
        raise ValueError("rubric-boundary group count differs")
    if sampling["evaluation_items_total"] != 16:
        raise ValueError("evaluation item count differs")
    if sampling["recurrent_crossing_evaluation_items_per_group"] != 3:
        raise ValueError("recurrent crossing evaluation count differs")
    if sampling["clear_anchor_evaluation_items_per_group"] != 1:
        raise ValueError("clear-anchor evaluation count differs")
    if sampling["minimum_recurrent_crossing_records_per_group"] != 5:
        raise ValueError("minimum recurrent crossing pool differs")
    if (
        sampling["recurrent_crossing_evaluation_items_per_group"]
        + sampling["clear_anchor_evaluation_items_per_group"]
        != sampling["evaluation_items_per_group"]
    ):
        raise ValueError("evaluation item composition does not close")
    if sampling["evaluation_visible_to_clarification_proposer"]:
        raise ValueError("evaluation items leak to the clarification proposer")
    if sampling["dev_access_allowed"] or sampling["test_access_allowed"]:
        raise ValueError("pilot permits dev/test access")

    roles = config["roles"]
    if roles["fidelity_auditor_instances"] != 2:
        raise ValueError("two fidelity auditors are required")
    if roles["original_condition_judge_instances"] != 5:
        raise ValueError("original panel size differs")
    if roles["clarified_condition_judge_instances"] != 5:
        raise ValueError("clarified panel size differs")
    if not roles["fresh_context_per_role_instance"]:
        raise ValueError("fresh role contexts are not required")
    if roles["cross_condition_judge_reuse_allowed"]:
        raise ValueError("cross-condition judge reuse is allowed")
    if roles["row_level_output_exchange_allowed"]:
        raise ValueError("row-level output exchange is allowed")

    execution = config["judge_execution"]
    expected = (
        sampling["evaluation_items_total"]
        * len(execution["conditions"])
        * execution["judge_instances_per_condition"]
    )
    if execution["formal_rating_decisions"] != expected or expected != 160:
        raise ValueError("formal rating-decision budget differs")
    if not execution["condition_blinding"] or execution["other_judge_outputs_visible"]:
        raise ValueError("judge panel blinding differs")

    contract = config["clarification_contract"]
    required_false = (
        "new_criterion_allowed",
        "criterion_removal_allowed",
        "weight_change_allowed",
        "new_score_cap_allowed",
        "evaluation_answer_access_allowed",
        "answer_specific_rule_allowed",
    )
    if any(contract[field] for field in required_false):
        raise ValueError("clarification contract permits a policy-changing edit")
    if not contract["clause_level_original_text_provenance_required"]:
        raise ValueError("clarification provenance is not required")
    if not contract["both_fidelity_auditors_must_pass"]:
        raise ValueError("dual-auditor fidelity is not required")

    gate = config["go_gate"]
    if gate["minimum_dual_auditor_fidelity_pass_rate"] != 0.8:
        raise ValueError("fidelity gate differs")
    if gate["minimum_relative_disagreement_reduction"] != 0.25:
        raise ValueError("disagreement-reduction gate differs")
    if gate["minimum_groups_favoring_clarification"] != 2:
        raise ValueError("clarification heterogeneity gate differs")
    if gate["minimum_groups_favoring_replication_or_direct"] != 1:
        raise ValueError("replication heterogeneity gate differs")
    if not gate["effect_must_extend_beyond_boundary_2_3"]:
        raise ValueError("pilot is allowed to collapse to Label-2")

    boundary = config["execution_boundary"]
    forbidden = (
        "model_training_allowed",
        "new_loss_implementation_allowed",
        "new_DPO_pairs_allowed",
        "dev_access_allowed",
        "test_access_allowed",
        "human_validity_claim_allowed",
        "CCF_A_method_claim_allowed",
    )
    if any(boundary[field] for field in forbidden):
        raise ValueError("protocol exceeds its scientific authorization")
    if boundary["gpu_required"]:
        raise ValueError("protocol incorrectly requires a GPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit_protocol(config)
    print("LLM_JUDGE_SPEC_PILOT_PROTOCOL_PASS")


if __name__ == "__main__":
    main()
