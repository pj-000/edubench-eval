from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.diagnose_sorc_dpo_train_signal import (
    _delta_cosine,
    _quantile,
    _tensor_update_report,
    grouped_summaries,
    memory_efficient_field_mean_logps,
    summarize_rows,
)


def _row(value: float, *, pair_type: str = "adjacent_score") -> dict:
    return {
        "pair_task": "score",
        "pair_type": pair_type,
        "pair_source": "actual_controlled",
        "gold_label": 2,
        "rejected_score": 4,
        "metric_id": "metric-a",
        "language": "zh",
        "reference_chosen_logp": -2.0,
        "reference_rejected_logp": -1.0,
        "policy_chosen_logp": -2.0 + value,
        "policy_rejected_logp": -1.0 - value,
        "reference_raw_margin": -1.0,
        "policy_raw_margin": -1.0 + 2.0 * value,
        "chosen_logp_change": value,
        "rejected_logp_change": -value,
        "preference_contrast": 2.0 * value,
        "beta_scaled_contrast": 0.2 * value,
        "risk_adjusted_margin": 0.2 * value - 0.75,
        "reference_prefers_chosen": False,
        "policy_prefers_chosen": value > 0.5,
        "contrast_positive": value > 0.0,
        "offset_satisfied": 0.2 * value > 0.75,
        "chosen_logp_increased": value > 0.0,
        "rejected_logp_decreased": value > 0.0,
        "reference_predicted_score": 4,
        "policy_predicted_score": 2 if value > 0.5 else 4,
    }


def test_quantile_uses_deterministic_linear_interpolation() -> None:
    assert _quantile([0.0, 10.0], 0.10) == pytest.approx(1.0)
    assert _quantile([0.0, 10.0], 0.90) == pytest.approx(9.0)
    assert _quantile([3.0], 0.50) == 3.0


def test_summary_reports_margin_and_score_corrections() -> None:
    report = summarize_rows([_row(0.25), _row(1.0)])
    assert report["count"] == 2
    assert report["numeric"]["preference_contrast"]["mean"] == pytest.approx(
        1.25
    )
    assert report["rates"]["policy_prefers_chosen"] == pytest.approx(0.5)
    assert report["score_decision"]["rejected_to_gold_flip_rate"] == pytest.approx(
        0.5
    )


def test_grouped_summary_contains_required_strata() -> None:
    rows = [_row(0.25), _row(1.0, pair_type="severe_l2h")]
    report = grouped_summaries(rows)
    assert report["overall"]["count"] == 2
    assert set(report["by"]["pair_type"]) == {
        "adjacent_score",
        "severe_l2h",
    }
    assert report["by"]["gold_label"]["2"]["count"] == 2


def test_adapter_update_norm_and_cosine_are_exact() -> None:
    torch = pytest.importorskip("torch")
    reference = {
        "model.layers.0.lora_A.weight": torch.tensor([1.0, 2.0]),
        "model.layers.0.lora_B.weight": torch.tensor([3.0]),
    }
    policy = {
        "model.layers.0.lora_A.weight": torch.tensor([2.0, 4.0]),
        "model.layers.0.lora_B.weight": torch.tensor([2.0]),
    }
    report, delta = _tensor_update_report(reference, policy)
    assert report["global"]["delta_l2"] == pytest.approx(6.0**0.5)
    assert report["global"]["changed_parameter_fraction"] == 1.0
    assert set(report["by_component"]) == {"lora_A", "lora_B"}
    assert set(report["by_layer_and_component"]) == {"0:lora_A", "0:lora_B"}
    assert _delta_cosine(delta, delta) == pytest.approx(1.0)


def test_zero_update_cosine_hard_fails() -> None:
    torch = pytest.importorskip("torch")
    zero = {"x": torch.zeros(2)}
    with pytest.raises(ValueError, match="zero adapter update"):
        _delta_cosine(zero, zero)


def test_memory_efficient_logps_match_training_implementation() -> None:
    torch = pytest.importorskip("torch")
    from thesis_exp.exp54_rar_sft.sorc_dpo_loss import field_mean_logps

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 6, 11, generator=generator)
    input_ids = torch.randint(0, 11, (3, 6), generator=generator)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[0, 2] = True
    mask[1, [1, 3, 4]] = True
    mask[2, [2, 5]] = True
    expected = field_mean_logps(logits, input_ids, mask)[
        "per_sequence_logp"
    ]
    actual = memory_efficient_field_mean_logps(logits, input_ids, mask)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
