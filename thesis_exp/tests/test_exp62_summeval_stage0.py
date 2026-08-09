import pytest
import numpy as np

from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    audit_raw_annotations,
    empirical_target,
    expand_records,
    normalized_text,
    quantized_mean_label,
    require_all,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import render_input
from thesis_exp.exp62_summeval_routing_confirmation.metrics import evaluate_hard_head
from thesis_exp.exp62_summeval_routing_confirmation.analyze_test import paired_analysis
from thesis_exp.exp62_summeval_routing_confirmation import SEEDS, VARIANTS


def annotation(group: str, model: str, summary: str = "A summary.") -> dict:
    return {
        "id": group,
        "model_id": model,
        "filepath": f"cnndm/dailymail/stories/{group}.story",
        "decoded": summary,
        "expert_annotations": [
            {"coherence": 3, "fluency": 5, "consistency": 5, "relevance": 4},
            {"coherence": 4, "fluency": 5, "consistency": 5, "relevance": 4},
            {"coherence": 4, "fluency": 4, "consistency": 5, "relevance": 5},
        ],
    }


def test_quantized_mean_and_empirical_target() -> None:
    assert quantized_mean_label((3, 4, 4)) == 4
    assert empirical_target((3, 4, 4)) == (0.0, 0.0, 1 / 3, 2 / 3, 0.0)


def test_quantized_mean_can_differ_from_every_rating() -> None:
    assert quantized_mean_label((1, 3, 5)) == 3
    assert quantized_mean_label((1, 2, 5)) == 3


def test_score_contract_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        empirical_target((1, 2))
    with pytest.raises(ValueError, match="exactly three"):
        quantized_mean_label((1, 2, 6))


def test_expansion_uses_only_source_free_dimensions() -> None:
    rows = expand_records([annotation("g1", "m1")], expected_raw_rows=None)
    assert {row["dimension"] for row in rows} == {"coherence", "fluency"}
    assert all(len(row["soft_target"]) == 5 for row in rows)
    assert all(row["group_id"] == "g1" for row in rows)


def test_normalization_is_case_and_whitespace_stable() -> None:
    assert normalized_text(" A  Summary\nHere ") == "a summary here"


def test_gate_helper_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="bad"):
        require_all({"good": True, "bad": False}, "test")


def test_raw_audit_rejects_summary_crossing_source_groups() -> None:
    rows = []
    for group in ("g1", "g2"):
        for index in range(16):
            rows.append(annotation(group, f"m{index}", summary="same"))
    with pytest.raises(RuntimeError, match="no_summary_crosses_source_group"):
        audit_raw_annotations(rows)


def test_input_template_is_dimension_explicit_and_source_free() -> None:
    coherence = render_input("summary text", "coherence")
    fluency = render_input("summary text", "fluency")
    assert "Candidate summary:\nsummary text" in coherence
    assert "well structured" in coherence
    assert "grammar" in fluency
    assert "source article" not in coherence.lower()
    assert "reference" not in fluency.lower()


def test_metrics_use_expected_scores_and_macro_dimensions() -> None:
    probabilities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    result = evaluate_hard_head(
        probabilities,
        np.asarray([1.0, 5.0, 3.0, 4.0]),
        np.asarray([1, 5, 3, 4]),
        ["coherence", "coherence", "fluency", "fluency"],
    )
    assert result["primary_macro_mae"] == 0.0
    assert result["macro"]["exact_match"] == 1.0


def test_paired_test_analysis_preserves_seed_and_group_pairing() -> None:
    predictions = {}
    offsets = {
        "direct_residual_blocked": 0.4,
        "routed_hmsa": 0.1,
        "orthogonal_only": 0.2,
        "parallel_only": 0.5,
    }
    for variant in VARIANTS:
        predictions[variant] = {}
        for seed in SEEDS:
            rows = []
            for group_index in range(3):
                for record_index in range(2):
                    truth = 3.0
                    rows.append(
                        {
                            "record_id": f"g{group_index}:r{record_index}",
                            "group_id": f"g{group_index}",
                            "human_mean": truth,
                            "hard_head_expectation": truth + offsets[variant],
                        }
                    )
            predictions[variant][seed] = rows
    result = paired_analysis(predictions, n_resamples=100, rng_seed=1)
    assert result["interpretation_gates"]["full_routing_confirmed"] is True
    assert result["interpretation_gates"]["orthogonal_component_confirmed"] is True
    assert result["interpretation_gates"]["parallel_not_reproduced"] is True
