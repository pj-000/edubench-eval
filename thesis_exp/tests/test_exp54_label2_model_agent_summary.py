from __future__ import annotations

import copy

import pytest

from thesis_exp.exp54_rar_sft.summarize_label2_model_agent_audit import (
    cohen_kappa,
    validate_completed_packet,
)


def _row() -> dict[str, str]:
    return {
        "case_number": "1",
        "presentation_id": "L2-A-test",
        "question": "q",
        "answer": "a",
        "rubric": "r",
        "score_2_uniquely_defensible": "",
        "score_3_also_defensible": "",
        "decisive_criterion_absent": "",
        "present_criterion_too_vague": "",
        "boundary_evidence_span": "",
        "reviewer_notes": "",
    }


def _fourteen(row: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for index in range(14):
        item = copy.deepcopy(row)
        item["case_number"] = str(index + 1)
        item["presentation_id"] = f"L2-A-{index}"
        rows.append(item)
    return rows


def test_completed_packet_validation_accepts_only_response_edits() -> None:
    source = _fourteen(_row())
    completed = copy.deepcopy(source)
    for row in completed:
        for field in (
            "score_2_uniquely_defensible",
            "score_3_also_defensible",
            "decisive_criterion_absent",
            "present_criterion_too_vague",
        ):
            row[field] = "NO"
        row["boundary_evidence_span"] = "visible evidence"
    validate_completed_packet(source, completed, allowed={"YES", "NO", "UNCERTAIN"})


def test_completed_packet_rejects_changed_blinded_content() -> None:
    source = _fourteen(_row())
    completed = copy.deepcopy(source)
    for row in completed:
        for field in (
            "score_2_uniquely_defensible",
            "score_3_also_defensible",
            "decisive_criterion_absent",
            "present_criterion_too_vague",
        ):
            row[field] = "NO"
        row["boundary_evidence_span"] = "visible evidence"
    completed[0]["answer"] = "changed"
    with pytest.raises(ValueError, match="changed a blinded source field"):
        validate_completed_packet(source, completed, allowed={"YES", "NO", "UNCERTAIN"})


def test_completed_packet_rejects_invalid_response() -> None:
    source = _fourteen(_row())
    completed = copy.deepcopy(source)
    for row in completed:
        for field in (
            "score_2_uniquely_defensible",
            "score_3_also_defensible",
            "decisive_criterion_absent",
            "present_criterion_too_vague",
        ):
            row[field] = "NO"
        row["boundary_evidence_span"] = "visible evidence"
    completed[0]["score_2_uniquely_defensible"] = "MAYBE"
    with pytest.raises(ValueError, match="invalid categorical response"):
        validate_completed_packet(source, completed, allowed={"YES", "NO", "UNCERTAIN"})


def test_cohen_kappa_known_values() -> None:
    assert cohen_kappa(["YES", "NO"], ["YES", "NO"]) == 1.0
    assert cohen_kappa(["YES", "NO"], ["NO", "YES"]) == -1.0
