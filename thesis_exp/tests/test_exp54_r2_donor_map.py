from __future__ import annotations

import itertools
import random

from thesis_exp.exp54_rar_sft.build_r2_donor_map import (
    build_donor_map,
    match_stratum,
    validate_map,
)


def item(
    record_id: str,
    reference_id: str,
    reason: str,
    *,
    qa_key: str | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "normalized_qa_key": qa_key or f"qa-{record_id}",
        "reference_id": reference_id,
        "reference_index": 0,
        "label_5": 2,
        "metric_id": "IFTC",
        "language": "zh",
        "reason": reason,
        "reason_sha256": reference_id,
    }


def row(
    record_id: str,
    reasons: list[str],
    *,
    label: int = 2,
    qa_key: str | None = None,
    metric_id: str = "IFTC",
    language: str = "zh",
) -> dict:
    return {
        "record_id": record_id,
        "normalized_qa_key": qa_key or f"qa-{record_id}",
        "label_5": label,
        "metric_id": metric_id,
        "language": language,
        "references": [
            {
                "reference_id": f"{record_id}:human_{index + 1}",
                "reason": reason,
                "clean_reason_sha256": f"hash-{record_id}-{index}",
            }
            for index, reason in enumerate(reasons)
        ],
    }


def brute_force_objective(items: list[dict]) -> tuple[int, int]:
    """Solve the strict active-subset permutation objective for n <= 8."""
    ordered = sorted(items, key=lambda value: value["reference_id"])
    lengths = [len(value["reason"]) for value in ordered]
    best_active = -1
    best_length_cost = 10**18
    for active_count in range(len(ordered) + 1):
        for active_indices in itertools.combinations(range(len(ordered)), active_count):
            for donor_indices in itertools.permutations(active_indices):
                legal = all(
                    recipient_index != donor_index
                    and ordered[recipient_index]["record_id"]
                    != ordered[donor_index]["record_id"]
                    and ordered[recipient_index]["normalized_qa_key"]
                    != ordered[donor_index]["normalized_qa_key"]
                    for recipient_index, donor_index in zip(
                        active_indices,
                        donor_indices,
                    )
                )
                if not legal:
                    continue
                length_cost = sum(
                    abs(lengths[recipient_index] - lengths[donor_index])
                    for recipient_index, donor_index in zip(
                        active_indices,
                        donor_indices,
                    )
                )
                if active_count > best_active:
                    best_active = active_count
                    best_length_cost = length_cost
                elif active_count == best_active:
                    best_length_cost = min(best_length_cost, length_cost)
    return best_active, best_length_cost


def mapping_objective(mapping: list[dict]) -> tuple[int, int]:
    active = [entry for entry in mapping if entry["active"]]
    return len(active), sum(
        int(entry["absolute_length_difference"]) for entry in active
    )


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


def test_different_ids_with_same_normalized_content_cannot_donate() -> None:
    items = [
        item("a", "a:1", "理由一", qa_key="duplicate-content"),
        item("b", "b:1", "理由二", qa_key="duplicate-content"),
    ]
    result = match_stratum(items, len)
    assert all(not entry["active"] for entry in result)


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


def test_input_order_does_not_change_mapping() -> None:
    items = [
        item("a", "a:1", "a"),
        item("a", "a:2", "aaaa"),
        item("b", "b:1", "bb"),
        item("c", "c:1", "cccccc"),
        item("d", "d:1", "ddd"),
    ]
    expected = match_stratum(items, len)
    randomizer = random.Random(20260723)
    for _ in range(20):
        shuffled = list(items)
        randomizer.shuffle(shuffled)
        assert match_stratum(shuffled, len) == expected


def test_random_small_strata_match_brute_force_oracle() -> None:
    randomizer = random.Random(20260723)
    for case_index in range(40):
        size = randomizer.randint(1, 7)
        items = []
        for reference_index in range(size):
            sample_index = randomizer.randint(0, max(0, size // 2))
            qa_variant = randomizer.randint(0, max(0, size // 3))
            items.append(
                item(
                    f"sample-{sample_index}",
                    f"case-{case_index}:ref-{reference_index}",
                    "x" * randomizer.randint(1, 30),
                    qa_key=f"qa-{qa_variant}",
                )
            )
        expected = brute_force_objective(items)
        actual = mapping_objective(match_stratum(items, len))
        assert actual == expected, (
            f"case {case_index} disagreed with strict oracle: "
            f"actual={actual}, expected={expected}"
        )


def test_length_cost_prefers_nearby_donors_when_feasible() -> None:
    items = [
        item("a", "a:1", "x"),
        item("b", "b:1", "yy"),
        item("c", "c:1", "z" * 50),
    ]
    result = match_stratum(items, len)
    assert all(row["active"] for row in result)
    assert max(int(row["absolute_length_difference"]) for row in result) >= 48
