from __future__ import annotations

from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import audit_rows, reject_eval_path


def train_row() -> dict:
    return {
        "record_id": "r1",
        "question": " Q  ",
        "answer": "A\nB",
        "metric_id": "IFTC",
        "metric_canonical": "Instruction Following & Task Completion",
        "generator_model": "model-x",
        "human_1_5": 2,
        "human_2_5": 3,
        "human_3_5": 3,
        "label_5": 3,
        "language": "en",
        "rubric": ["5: complete", "1: incomplete"],
    }


def reason(score: int, text: str = "specific reason") -> dict:
    return {
        "question": "Q",
        "response": "A B",
        "principle": "指令遵循与任务完成",
        "model": "model-x",
        "score": score,
        "reason": text,
    }


def test_exact_bilingual_alignment_keeps_rater_and_aggregate_gates_separate() -> None:
    public, private = audit_rows([train_row()], {1: [reason(2)], 2: [reason(3)], 3: [reason(3)]})
    row = public[0]
    assert row["all_three_aligned"] is True
    assert row["aggregate_consistent_reason_count"] == 2
    assert row["human_1_status"] == "aligned_rater_only"
    assert row["human_2_status"] == "aligned_aggregate_consistent"
    assert private[0]["field_gates"] == {"score": 1, "rubric_checks": 0, "evidence": 0, "rationale": 0}


def test_ambiguous_or_fuzzy_match_never_becomes_eligible() -> None:
    duplicate = reason(2)
    fuzzy = {**reason(3), "response": "A B extra"}
    public, _ = audit_rows([train_row()], {1: [duplicate, dict(duplicate)], 2: [fuzzy], 3: []})
    row = public[0]
    assert row["human_1_status"] == "ambiguous"
    assert row["human_2_status"] == "unmatched"
    assert row["any_aligned"] is False


@pytest.mark.parametrize(
    ("metric_id", "canonical", "legacy_chinese"),
    [
        ("DKA", "Domain Knowledge Accuracy", "领域知识专业性"),
        ("RTC", "Role & Tone Consistency", "角色与口吻一致性"),
    ],
)
def test_legacy_metric_names_map_to_current_canonical_ids(
    metric_id: str, canonical: str, legacy_chinese: str
) -> None:
    row = train_row()
    row.update({"metric_id": metric_id, "metric_canonical": canonical})
    source = reason(2)
    source["principle"] = legacy_chinese
    public, _ = audit_rows([row], {1: [source], 2: [], 3: []})
    assert public[0]["human_1_status"] == "aligned_rater_only"


@pytest.mark.parametrize("name", ["dev.jsonl", "test.jsonl"])
def test_rar0_refuses_evaluation_splits(name: str) -> None:
    with pytest.raises(PermissionError):
        reject_eval_path(Path("thesis_exp/data/splits/paper_like_triple_seed42") / name)
