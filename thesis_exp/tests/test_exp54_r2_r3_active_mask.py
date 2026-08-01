from __future__ import annotations

from thesis_exp.exp54_rar_sft.build_r2_r3_active_mask import (
    build_mask_rows,
    summarize_mask,
)


def references() -> list[dict]:
    return [
        {
            "record_id": "a",
            "normalized_qa_key": "qa-a",
            "label_5": 1,
            "metric_id": "IFTC",
            "language": "zh",
            "references": [
                {
                    "reference_id": "a:1",
                    "reason": "理由甲",
                    "clean_reason_sha256": "hash-a",
                }
            ],
        },
        {
            "record_id": "b",
            "normalized_qa_key": "qa-b",
            "label_5": 1,
            "metric_id": "IFTC",
            "language": "zh",
            "references": [
                {
                    "reference_id": "b:1",
                    "reason": "理由乙",
                    "clean_reason_sha256": "hash-b",
                }
            ],
        },
        {
            "record_id": "c",
            "normalized_qa_key": "qa-c",
            "label_5": 2,
            "metric_id": "IFTC",
            "language": "zh",
            "references": [
                {
                    "reference_id": "c:1",
                    "reason": "理由丙",
                    "clean_reason_sha256": "hash-c",
                }
            ],
        },
    ]


def donor_rows() -> list[dict]:
    return [
        {
            "recipient_reference_id": "a:1",
            "active": True,
            "inactive_reason": "",
        },
        {
            "recipient_reference_id": "b:1",
            "active": True,
            "inactive_reason": "",
        },
        {
            "recipient_reference_id": "c:1",
            "active": False,
            "inactive_reason": "no_legal_strict_permutation",
        },
    ]


def test_builds_one_canonical_mask_for_both_arms() -> None:
    mask = build_mask_rows(references(), donor_rows())
    assert [row["reference_id"] for row in mask] == ["a:1", "b:1", "c:1"]
    assert [row["rationale_active"] for row in mask] == [True, True, False]
    assert mask[2]["inactive_reason"] == "no_legal_strict_permutation"


def test_summarizes_low_score_sample_and_reference_coverage() -> None:
    summary = summarize_mask(build_mask_rows(references(), donor_rows()))
    assert summary["source_references"] == 3
    assert summary["active_references"] == 2
    assert summary["inactive_references"] == 1
    assert summary["coverage_by_label"][0]["active_reason_samples"] == 2
    assert summary["coverage_by_label"][1]["fully_inactive_samples"] == 1
