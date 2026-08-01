from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.analyze_llm_judge_spec_pilot import (
    exact_cluster_bootstrap_lower,
    pairwise_ordinal_disagreement,
)


def test_pairwise_ordinal_disagreement() -> None:
    assert pairwise_ordinal_disagreement([3, 3, 3, 3, 3]) == 0
    assert pairwise_ordinal_disagreement([1, 5]) == 1
    assert pairwise_ordinal_disagreement([3, 3, 4, 3, 3]) == pytest.approx(0.1)


def test_pairwise_disagreement_requires_two_scores() -> None:
    with pytest.raises(ValueError, match="at least two"):
        pairwise_ordinal_disagreement([3])


def test_exact_cluster_bootstrap_lower_is_deterministic() -> None:
    value = exact_cluster_bootstrap_lower([-0.0125, 0.0, 0.0375, 0.0])
    assert value == pytest.approx(-0.00625)
