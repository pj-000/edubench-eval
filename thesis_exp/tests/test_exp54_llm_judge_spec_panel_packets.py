from __future__ import annotations

import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.build_llm_judge_spec_panel_packets import (
    build_panel_packets,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    groups = []
    clarifications = []
    for lower in range(1, 5):
        group_id = f"RB{lower}-{lower}{lower + 1}"
        rubric = [f"rubric-{lower}"]
        groups.append(
            {
                "group_id": group_id,
                "target_adjacent_boundary": [lower, lower + 1],
                "original_rubric": rubric,
                "evaluation_items": [
                    {
                        "presentation_id": f"{group_id}-E{index}",
                        "language": "en",
                        "metric": "m",
                        "question": f"q-{index}",
                        "answer": f"a-{index}",
                    }
                    for index in range(1, 5)
                ],
            }
        )
        clarifications.append(
            {
                "group_id": group_id,
                "target_adjacent_boundary": [lower, lower + 1],
                "original_rubric": rubric,
                "clarification": {"boundary_rule": "rule", "operational_guidance": ["g"]},
            }
        )
    evaluation = tmp_path / "evaluation.json"
    clarification = tmp_path / "clarification.json"
    audit_a = tmp_path / "audit_a.json"
    audit_b = tmp_path / "audit_b.json"
    _write(evaluation, {"groups": groups})
    _write(clarification, {"groups": clarifications})
    for path, auditor in ((audit_a, "A"), (audit_b, "B")):
        _write(
            path,
            {
                "auditor": auditor,
                "groups": [
                    {"group_id": group["group_id"], "decision": "PASS"}
                    for group in groups
                ],
            },
        )
    return evaluation, clarification, audit_a, audit_b


def test_builds_ten_blinded_packets_with_unique_orders(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "panels"
    report = build_panel_packets(
        evaluation_path=inputs[0],
        clarification_path=inputs[1],
        audit_a_path=inputs[2],
        audit_b_path=inputs[3],
        output_dir=output,
    )
    assert len(report["packet_hashes"]) == 10
    assert report["unique_order_vectors"] == 10
    original = json.loads((output / "judge_original-J1.json").read_text())
    clarified = json.loads((output / "judge_clarified-J1.json").read_text())
    assert all(
        set(item["scoring_specification"]) == {"original_rubric"}
        for item in original["items"]
    )
    assert all(
        set(item["scoring_specification"])
        == {"original_rubric", "policy_preserving_clarification"}
        for item in clarified["items"]
    )
    assert "condition" not in original and "condition" not in clarified


def test_rejects_any_fidelity_failure(tmp_path: Path) -> None:
    evaluation, clarification, audit_a, audit_b = _inputs(tmp_path)
    audit = json.loads(audit_b.read_text())
    audit["groups"][0]["decision"] = "FAIL"
    _write(audit_b, audit)
    with pytest.raises(ValueError, match="all groups must pass"):
        build_panel_packets(
            evaluation_path=evaluation,
            clarification_path=clarification,
            audit_a_path=audit_a,
            audit_b_path=audit_b,
            output_dir=tmp_path / "panels",
        )
