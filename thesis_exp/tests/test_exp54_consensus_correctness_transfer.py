from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.audit_consensus_correctness_transfer import (
    crosses_boundary,
    human_class,
    panel_summary,
)


@pytest.mark.parametrize("scores, expected", [([3, 3, 3], "H0"), ([2, 2, 3], "H1"), ([1, 3, 3], "H2")])
def test_human_class(scores: list[int], expected: str) -> None:
    assert human_class(scores) == expected


def test_panel_consensus_error_and_severity() -> None:
    result = panel_summary([5, 5, 5, 5, 4], 2)
    assert result["agreement_class"] == "P4"
    assert result["consensus_error"] is True
    assert result["severe_consensus_error"] is True
    assert result["severe_direction"] == "L2H"


def test_tied_panel_has_no_mode() -> None:
    result = panel_summary([1, 1, 2, 2, 3], 2)
    assert result["mode"] is None
    assert result["agreement_class"] == "Pweak"
    assert result["consensus_error"] is False


def test_boundary_crossing_is_direction_agnostic() -> None:
    assert crosses_boundary(2, 4, 2)
    assert crosses_boundary(4, 2, 3)
    assert not crosses_boundary(4, 5, 3)


def test_correct_mode_is_not_evidence_that_an_error_is_human_ambiguous() -> None:
    result = panel_summary([3, 3, 3, 3, 3], 3)
    assert result["correct"] is True
    assert result["consensus_error"] is False
