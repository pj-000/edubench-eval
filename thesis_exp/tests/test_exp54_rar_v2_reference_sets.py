from __future__ import annotations

from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_rar0_alignment import text_sha256
from thesis_exp.exp54_rar_sft.build_rar_v2_reference_sets import (
    build_reference_sets,
    contains_explicit_score_report,
    normalize_reason,
    redact_explicit_scores,
    summarize,
)


def aligned_row(record_id: str, reasons: dict | None = None) -> dict:
    return {
        "record_id": record_id,
        "question_key": f"q-{record_id}",
        "score": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "human_reasons": reasons or {},
    }


def reason(score: int, text: str, aggregate_consistent: bool) -> dict:
    return {
        "source_score": score,
        "aggregate_score_consistent": aggregate_consistent,
        "reason": text,
        "reason_sha256": text_sha256(text),
    }


@pytest.mark.parametrize(
    "source",
    [
        "内容存在明显遗漏，因此只能给到 3 分。",
        "论证完整，根据评分标准，给予10分。",
        "Score: 4. The response omits one required step.",
        "I gave this response 2 points because the evidence is incomplete.",
        "表达积极且具有引导性，因此给出了8分的评分。",
        "专业知识基本准确，所以给8分。",
        "地理知识运用准确，符合小学要求，得分7分。",
    ],
)
def test_redacts_explicit_rater_score_reports(source: str) -> None:
    cleaned, events = redact_explicit_scores(source)
    assert events
    assert contains_explicit_score_report(cleaned) is False


@pytest.mark.parametrize(
    "source",
    [
        "建议每天学习30-45分钟，并完成3个练习。",
        "答案中的“Score”值“1”与正确回答的事实矛盾。",
        "TotalPossiblePoints设置为5分不合理。",
        "评分细节中解释完整性0分与实际内容不符。",
    ],
)
def test_preserves_answer_evidence_numbers(source: str) -> None:
    cleaned, events = redact_explicit_scores(source)
    assert cleaned == normalize_reason(source)
    assert events == []


def test_builds_all_rater_and_consistent_sets_without_copying_rows() -> None:
    rows = [
        aligned_row(
            "r1",
            {
                "human_1": reason(2, "理由一。", True),
                "human_2": reason(2, "理由二，所以给予2分。", True),
                "human_3": reason(3, "不同意见。", False),
            },
        ),
        aligned_row("r2"),
    ]
    inventory, all_rater, consistent, audit = build_reference_sets(rows)
    assert len(inventory) == len(all_rater) == len(consistent) == 2
    assert all_rater[0]["reference_count"] == 3
    assert consistent[0]["reference_count"] == 2
    assert [ref["rater_id"] for ref in consistent[0]["references"]] == [
        "human_1",
        "human_2",
    ]
    assert consistent[1]["rationale_mask"] == 0
    assert len(audit) == 1
    assert "2分" not in consistent[0]["references"][1]["reason"]


def test_rejects_changed_aligned_reason_hash() -> None:
    rows = [
        aligned_row(
            "r1",
            {
                "human_1": {
                    **reason(2, "原始理由。", True),
                    "reason": "被修改的理由。",
                }
            },
        )
    ]
    with pytest.raises(ValueError, match="hash mismatch"):
        build_reference_sets(rows)


def test_real_inventory_matches_frozen_counts() -> None:
    path = Path(
        "thesis_exp/outputs/exp54_rar_sft/rar0_alignment/private/"
        "aligned_reason_candidates.jsonl"
    )
    if not path.exists():
        pytest.skip("RAR-0A private alignment output is unavailable")
    import json

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inventory, all_rater, consistent, audit = build_reference_sets(rows)
    summary = summarize(inventory, all_rater, consistent, audit)
    assert summary["status"] == "REFERENCE_SETS_READY"
    assert summary["counts"]["train_rows"] == 2654
    assert summary["counts"]["reason_covered_rows"] == 1612
    assert summary["counts"]["no_reason_rows"] == 1042
    assert summary["counts"]["all_rater_references"] == 4836
    assert summary["counts"]["label_consistent_references"] == 3934
    assert summary["counts"]["dissenting_references"] == 902
