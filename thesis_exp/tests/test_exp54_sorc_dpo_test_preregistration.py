from __future__ import annotations

import json
from pathlib import Path

from thesis_exp.exp54_rar_sft import REPO_ROOT


CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_one_time_test_preregistration_v1.json"
)
PREFERENCE_AUDIT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/training_audit_report.json"
)
COLD_START_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_training_candidate/candidate_report.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_preregistration_binds_all_twelve_exact_adapters() -> None:
    config = _read(CONFIG)
    preference = _read(PREFERENCE_AUDIT)
    cold_start = _read(COLD_START_REPORT)["checkpoint_bindings"]

    assert set(config["arms"]) == {
        "P0_R3_SFT",
        "P1_FIELD_DPO",
        "P2_SORC_SCORE",
        "P3_JOINT_SORC",
    }
    for seed in (42, 43, 44):
        field = f"seed_{seed}_adapter_model_sha256"
        assert config["arms"]["P0_R3_SFT"][field] == cold_start[str(seed)][
            "adapter_model_sha256"
        ]
        for arm in ("P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC"):
            assert config["arms"][arm][field] == preference["runs"][
                f"{arm}/seed_{seed}"
            ]["adapter_model_sha256"]


def test_preregistration_is_fail_closed_and_component_identified() -> None:
    config = _read(CONFIG)
    assert config["status"] == "CANDIDATE_NOT_AUTHORIZED"
    assert config["test_execution_allowed"] is False
    assert config["test_accessed"] is False
    assert config["test_source_binding"]["content_read_during_preregistration"] is False
    assert config["test_source_binding"]["dirty_worktree_copy_forbidden"] is True
    assert config["execution_rules"]["all_12_arm_seed_runs_required"] is True
    assert (
        config["execution_rules"][
            "no_partial_result_inspection_before_all_12_runs_complete"
        ]
        is True
    )
    assert [row["id"] for row in config["confirmatory_contrasts"]] == [
        "H1_FIELD_DPO",
        "H2_ORDINAL_OFFSET",
        "H3_RATIONALE_BLOCK",
    ]
    h3 = config["confirmatory_contrasts"][2]
    assert h3["not_FLOP_matched"] is True
    assert h3["pair_counts"] == {"P2": 838, "P3": 2438}
    assert [
        (row["baseline_arm"], row["treatment_arm"])
        for row in config["confirmatory_contrasts"]
    ] == [
        ("P0_R3_SFT", "P1_FIELD_DPO"),
        ("P1_FIELD_DPO", "P2_SORC_SCORE"),
        ("P2_SORC_SCORE", "P3_JOINT_SORC"),
    ]
    assert all(
        row["all_inference_uses_endpoint_specific_benefit_delta"] is True
        for row in config["confirmatory_contrasts"]
    )


def test_statistical_decisions_are_frozen_before_test() -> None:
    config = _read(CONFIG)
    protocol = config["statistical_protocol"]
    assert protocol["primary_multiplicity"] == {
        "family": (
            "six_tests_equal_to_three_component_contrasts_times_two_"
            "co_primary_endpoints"
        ),
        "procedure": "Holm_Bonferroni_step_down",
        "familywise_alpha": 0.05,
        "selection_after_results_forbidden": True,
        "unadjusted_percentile_95_intervals_reported_as_descriptive": True,
    }
    assert protocol["undefined_replicate_rule"][
        "minimum_valid_replicates_for_any_interval_or_p_value"
    ] == 9500
    assert protocol["smallest_effect_size_of_interest"] == {
        "MAE": 0.01,
        "L2H_rate": 0.05,
        "Exact": 0.01,
        "Kendall_tau_b": 0.01,
        "absolute_Signed_Bias": 0.02,
        "Label_5_Recall": 0.02,
        "H2L_rate": 0.02,
        "QWK": 0.01,
    }
    assert "within_each_seed" in protocol["point_estimate_algorithm"]
    assert "same_sampled_index_vector" in protocol["bootstrap_algorithm"]
    assert config["primary_endpoints"] == [
        "MAE_against_label_5",
        "low_to_high_rate_for_gold_label_1_or_2_predicted_as_4_or_5",
    ]
    assert config["descriptive_primary_counts"] == [
        "low_to_high_count_for_gold_label_1_or_2_predicted_as_4_or_5"
    ]
    assert "no_operational_failure" in protocol["strong_component_support"]
    assert (
        "no_forced_completion_diagnostic_harm"
        in protocol["directional_component_support"]
    )


def test_preregistration_does_not_promise_unavailable_calibration() -> None:
    config = _read(CONFIG)
    assert config["probability_calibration_metrics"]["included"] is False
    assert config["rationale_claim"]["internal_reasoning_claim_allowed"] is False
    assert config["rationale_claim"][
        "test_score_results_cannot_override_the_dev_rationale_audit"
    ] is True
