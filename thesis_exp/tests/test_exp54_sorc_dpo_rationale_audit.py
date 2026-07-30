from __future__ import annotations

import json
from pathlib import Path

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_sorc_dpo_rationale_audit import (
    _prediction_path,
)


def test_prediction_paths_are_arm_and_seed_specific() -> None:
    root = Path("/tmp/frozen-dev")
    assert _prediction_path(root, "P2_SORC_SCORE", 42) == (
        root / "p2_sorc_score/seed_42/predictions.jsonl"
    )
    assert _prediction_path(root, "P3_JOINT_SORC", 44) == (
        root / "p3_joint_sorc/seed_44/predictions.jsonl"
    )


def test_sorc_dpo_rationale_schemas_are_fail_closed() -> None:
    schema_root = REPO_ROOT / "thesis_exp/exp54_rar_sft/schemas"
    expected = {
        "sorc_dpo_rationale_audit_score_blind_v1.schema.json": {
            "rubric_alignment",
            "answer_grounding",
            "specificity",
            "unsupported_claims_control",
            "completeness",
            "repetition_control",
            "overall_preference",
        },
        "sorc_dpo_rationale_audit_score_visible_v1.schema.json": {
            "score_rationale_consistency",
            "overall_scoring_justification_usefulness",
            "overall_preference",
        },
    }
    for filename, verdict_fields in expected.items():
        schema = json.loads((schema_root / filename).read_text())
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {
            "presentation_id",
            "brief_reason",
            *verdict_fields,
        }
        for field in verdict_fields:
            assert schema["properties"][field]["enum"] == ["A", "B", "tie"]
