from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.analyze_sorc_dpo_lr5e6_mechanism import (
    paired_arm_delta,
    summarize_group,
)


def _row(
    arm: str,
    pair_id: str,
    beta_contrast: float,
    *,
    pair_type: str = "severe_l2h",
) -> dict:
    offset = 0.0 if arm == "P1_FIELD_DPO" else 0.75
    return {
        "arm": arm,
        "pair_id": pair_id,
        "pair_task": "score",
        "pair_type": pair_type,
        "pair_source": "actual_controlled",
        "gold_label": 2,
        "rejected_score": 4,
        "metric_id": "metric-a",
        "language": "zh",
        "odpo_offset": offset,
        "beta_scaled_contrast": beta_contrast,
        "risk_adjusted_margin": beta_contrast - offset,
        "reference_raw_margin": -1.0,
        "policy_raw_margin": -1.0 + 10.0 * beta_contrast,
        "chosen_logp_change": 6.0 * beta_contrast,
        "rejected_logp_change": -4.0 * beta_contrast,
        "contrast_positive": beta_contrast > 0.0,
        "offset_satisfied": beta_contrast > offset,
        "policy_prefers_chosen": beta_contrast > 0.1,
        "chosen_logp_increased": beta_contrast > 0.0,
        "rejected_logp_decreased": beta_contrast > 0.0,
    }


def test_group_summary_reports_required_mechanism_statistics() -> None:
    report = summarize_group(
        [
            _row("P2_SORC_SCORE", "a", 0.05),
            _row("P2_SORC_SCORE", "b", 0.15),
        ]
    )
    assert report["count"] == 2
    assert report["numeric"]["beta_scaled_contrast"]["mean"] == pytest.approx(
        0.10
    )
    assert report["numeric"]["risk_adjusted_margin"]["median"] == pytest.approx(
        -0.65
    )
    assert report["rates"]["contrast_positive"] == 1.0
    assert report["rates"]["offset_satisfied"] == 0.0


def test_paired_delta_uses_exact_shared_pair_identity() -> None:
    rows = [
        _row("P1_FIELD_DPO", "a", 0.02),
        _row("P1_FIELD_DPO", "b", 0.04),
        _row("P2_SORC_SCORE", "a", 0.05),
        _row("P2_SORC_SCORE", "b", 0.03),
    ]
    report = paired_arm_delta(
        rows,
        left_arm="P1_FIELD_DPO",
        right_arm="P2_SORC_SCORE",
    )
    assert report["count"] == 2
    assert report["delta"]["beta_scaled_contrast"]["mean"] == pytest.approx(
        0.01
    )
    assert report["beta_scaled_contrast_delta_direction"] == {
        "positive_rate": 0.5,
        "zero_rate": 0.0,
        "negative_rate": 0.5,
    }


def test_paired_delta_rejects_inventory_or_metadata_changes() -> None:
    missing = [
        _row("P1_FIELD_DPO", "a", 0.02),
        _row("P2_SORC_SCORE", "b", 0.05),
    ]
    with pytest.raises(ValueError, match="inventories differ"):
        paired_arm_delta(
            missing,
            left_arm="P1_FIELD_DPO",
            right_arm="P2_SORC_SCORE",
        )

    changed = [
        _row("P1_FIELD_DPO", "a", 0.02),
        {
            **_row("P2_SORC_SCORE", "a", 0.05),
            "rejected_score": 5,
        },
    ]
    with pytest.raises(ValueError, match="metadata differs"):
        paired_arm_delta(
            changed,
            left_arm="P1_FIELD_DPO",
            right_arm="P2_SORC_SCORE",
        )
