from __future__ import annotations

import pytest

from thesis_exp.exp54_rar_sft.audit_llm_judge_spec_pilot_packets import (
    FORBIDDEN_VISIBLE_KEYS,
    _walk_keys,
)


def test_hidden_fields_are_detected_recursively() -> None:
    visible = {"groups": [{"items": [{"generated_scores": [2, 3, 2]}]}]}
    assert _walk_keys(visible) & FORBIDDEN_VISIBLE_KEYS == {"generated_scores"}


@pytest.mark.parametrize(
    "field",
    [
        "record_id",
        "gold_label",
        "label_5",
        "human_scores",
        "generated_score",
        "error_class",
        "selection_role",
    ],
)
def test_all_decisive_hidden_fields_are_forbidden(field: str) -> None:
    assert field in FORBIDDEN_VISIBLE_KEYS


def test_safe_visible_fields_are_not_forbidden() -> None:
    assert not ({"presentation_id", "question", "answer", "metric", "language"} & FORBIDDEN_VISIBLE_KEYS)
