from __future__ import annotations

from thesis_exp.exp54_rar_sft.audit_manifest_schedule_compatibility import (
    frequency_audit,
    schedule_index,
)


def reference_row(
    sample_id: str,
    reference_ids: list[str],
) -> dict:
    return {
        "record_id": sample_id,
        "normalized_qa_key": f"qa-{sample_id}",
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "references": [
            {
                "reference_id": reference_id,
                "rater_id": f"human_{index + 1}",
                "reason": reference_id,
                "clean_reason_sha256": f"hash-{reference_id}",
            }
            for index, reference_id in enumerate(reference_ids)
        ],
    }


def mapping_row(recipient: str, donor: str) -> dict:
    return {
        "recipient_reference_id": recipient,
        "donor_reference_id": donor,
        "active": True,
    }


def test_schedule_index_is_deterministic_and_rotates() -> None:
    first = [schedule_index(42, "sample-a", epoch, 3) for epoch in range(3)]
    second = [schedule_index(42, "sample-a", epoch, 3) for epoch in range(3)]
    assert first == second
    assert sorted(first) == [0, 1, 2]


def test_cross_reference_count_permutation_changes_three_epoch_frequency() -> None:
    rows = [
        reference_row("a", ["a1", "a2"]),
        reference_row("b", ["b1", "b2", "b3"]),
    ]
    mapping = [
        mapping_row("a1", "b1"),
        mapping_row("b1", "a1"),
        mapping_row("a2", "b2"),
        mapping_row("b2", "a2"),
        {
            "recipient_reference_id": "b3",
            "donor_reference_id": "",
            "active": False,
        },
    ]
    report = frequency_audit(rows, mapping, epochs=3, seeds=(42,))
    assert report["all_seeds_frequency_identical"] is False
    assert report["seed_reports"][0]["absolute_frequency_delta_l1"] > 0


def test_equal_reference_count_full_cycle_preserves_frequency() -> None:
    rows = [
        reference_row("a", ["a1", "a2"]),
        reference_row("b", ["b1", "b2"]),
    ]
    mapping = [
        mapping_row("a1", "b1"),
        mapping_row("b1", "a1"),
        mapping_row("a2", "b2"),
        mapping_row("b2", "a2"),
    ]
    report = frequency_audit(rows, mapping, epochs=6, seeds=(42, 43, 44))
    assert report["all_seeds_frequency_identical"] is True
