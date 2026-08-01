from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.collect_sorc_dpo_rationale_audit import (
    collapse_orientations,
    preference_metrics,
)


def _key(
    presentation_id: str,
    orientation: int,
    *,
    swapped: bool,
) -> dict:
    arms = (
        ("P2_SORC_SCORE", "P3_JOINT_SORC")
        if swapped
        else ("P3_JOINT_SORC", "P2_SORC_SCORE")
    )
    return {
        "presentation_id": presentation_id,
        "pair_id": "pair-a",
        "orientation": orientation,
        "record_id": "record-a",
        "seed": 42,
        "label_5": 2,
        "candidate_a_arm": arms[0],
        "candidate_b_arm": arms[1],
        "candidate_a_forced": arms[0] == "P2_SORC_SCORE",
        "candidate_b_forced": arms[1] == "P2_SORC_SCORE",
    }


def _judgment(presentation_id: str, preference: str) -> dict:
    return {
        "presentation_id": presentation_id,
        "score_rationale_consistency": preference,
        "overall_scoring_justification_usefulness": preference,
        "overall_preference": preference,
        "brief_reason": "short blind reason",
    }


def test_orientation_collapse_maps_candidate_letters_to_p3() -> None:
    keys = [
        _key("a", 0, swapped=False),
        _key("b", 1, swapped=True),
    ]
    judgments = [_judgment("a", "A"), _judgment("b", "B")]
    rows = collapse_orientations(
        keys,
        judgments,
        stage="score_visible",
        enforce_full_shape=False,
    )
    assert len(rows) == 1
    assert rows[0]["verdicts"]["overall_preference"] == "win"
    assert rows[0]["orientation_conflict"] is False
    assert rows[0]["forced_stratum"] == "P2_only_forced"


def test_orientation_conflict_collapses_to_tie() -> None:
    keys = [
        _key("a", 0, swapped=False),
        _key("b", 1, swapped=True),
    ]
    judgments = [_judgment("a", "A"), _judgment("b", "A")]
    rows = collapse_orientations(
        keys,
        judgments,
        stage="score_visible",
        enforce_full_shape=False,
    )
    assert rows[0]["verdicts"]["overall_preference"] == "tie"
    assert rows[0]["orientation_conflict"] is True


def test_preference_metrics_keep_ties() -> None:
    report = preference_metrics(["win", "tie", "loss", "win"])
    assert report["wins"] == 2
    assert report["ties"] == 1
    assert report["losses"] == 1
    assert report["tie_adjusted_preference"] == pytest.approx(0.625)
    assert report["net_preference"] == pytest.approx(0.25)
