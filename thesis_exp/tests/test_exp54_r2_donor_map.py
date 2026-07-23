from __future__ import annotations

from thesis_exp.exp54_rar_sft.build_r2_donor_map import (
    build_donor_map,
    match_stratum,
    validate_map,
)


def item(record_id: str, reference_id: str, reason: str) -> dict:
    return {
        "record_id": record_id,
        "reference_id": reference_id,
        "reference_index": 0,
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "reason": reason,
        "reason_sha256": reference_id,
    }


def row(record_id: str, reasons: list[str], *, label: int = 2) -> dict:
    return {
        "record_id": record_id,
        "label_5": label,
        "metric_id": "IFTC",
        "language": "zh",
        "references": [
            {
                "reference_id": f"{record_id}:human_{index + 1}",
                "reason": reason,
                "clean_reason_sha256": f"hash-{record_id}-{index}",
            }
            for index, reason in enumerate(reasons)
        ],
    }


def test_equal_two_sample_stratum_has_full_derangement() -> None:
    items = [
        item("a", "a:1", "短"),
        item("a", "a:2", "较短"),
        item("b", "b:1", "短文"),
        item("b", "b:2", "一段稍长文本"),
    ]
    result = match_stratum(items, len)
    assert all(row["active"] for row in result)
    assert all(
        row["recipient_record_id"] != row["donor_record_id"] for row in result
    )
    assert {row["recipient_reference_id"] for row in result} == {
        row["donor_reference_id"] for row in result
    }


def test_single_sample_stratum_deactivates_all_references() -> None:
    items = [
        item("a", "a:1", "理由一"),
        item("a", "a:2", "理由二"),
    ]
    result = match_stratum(items, len)
    assert all(not row["active"] for row in result)
    assert all(not row["donor_reference_id"] for row in result)


def test_unequal_two_sample_counts_keep_maximal_balanced_subset() -> None:
    items = [
        item("a", "a:1", "a"),
        item("a", "a:2", "aa"),
        item("a", "a:3", "aaa"),
        item("b", "b:1", "b"),
        item("b", "b:2", "bb"),
    ]
    result = match_stratum(items, len)
    active = [row for row in result if row["active"]]
    inactive = [row for row in result if not row["active"]]
    assert len(active) == 4
    assert len(inactive) == 1
    assert {row["recipient_reference_id"] for row in active} == {
        row["donor_reference_id"] for row in active
    }


def test_matching_is_deterministic_and_stratum_preserving() -> None:
    rows = [
        row("a", ["a", "aaaa"], label=2),
        row("b", ["bb", "bbbbbb"], label=2),
        row("c", ["ccc", "cccccccc"], label=3),
        row("d", ["ddd", "ddddddddd"], label=3),
    ]
    first = build_donor_map(rows, len)
    second = build_donor_map(rows, len)
    assert first == second
    report = validate_map(rows, first)
    assert all(report["checks"].values())
    assert report["counts"]["active_references"] == 8
    assert report["row_coverage_by_label"] == [
        {
            "label_5": 2,
            "source_reason_rows": 2,
            "active_reason_rows": 2,
            "fully_deactivated_rows": 0,
            "active_row_coverage": 1.0,
            "active_references": 4,
            "inactive_references": 0,
        },
        {
            "label_5": 3,
            "source_reason_rows": 2,
            "active_reason_rows": 2,
            "fully_deactivated_rows": 0,
            "active_row_coverage": 1.0,
            "active_references": 4,
            "inactive_references": 0,
        },
    ]


def test_length_cost_prefers_nearby_donors_when_feasible() -> None:
    items = [
        item("a", "a:1", "x"),
        item("b", "b:1", "yy"),
        item("c", "c:1", "z" * 50),
    ]
    result = match_stratum(items, len)
    assert all(row["active"] for row in result)
    assert max(int(row["absolute_length_difference"]) for row in result) >= 48
