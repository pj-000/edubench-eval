from __future__ import annotations

from collections import Counter

from thesis_exp.exp55_within_label_shuffle.build_targets import (
    mapping_sha256,
    shuffle_train_rows,
    target_key,
)


def row(row_id: str, label: int, target: list[float]) -> dict:
    return {
        "record_id": row_id,
        "id": row_id,
        "label": label - 1,
        "label_5": label,
        "text": f"text-{row_id}",
        "soft_target_5": target,
    }


def test_shuffle_is_deterministic_and_preserves_label_multisets() -> None:
    rows = [
        row("a", 2, [0, 1, 0, 0, 0]),
        row("b", 2, [1 / 3, 2 / 3, 0, 0, 0]),
        row("c", 2, [0, 2 / 3, 1 / 3, 0, 0]),
        row("d", 3, [0, 0, 1, 0, 0]),
        row("e", 3, [0, 1 / 3, 2 / 3, 0, 0]),
    ]
    shuffled_a, mapping_a = shuffle_train_rows(rows)
    shuffled_b, mapping_b = shuffle_train_rows(rows)
    assert mapping_a == mapping_b
    assert mapping_sha256(mapping_a) == mapping_sha256(mapping_b)
    for label in (2, 3):
        before = Counter(target_key(item["soft_target_5"]) for item in rows if item["label_5"] == label)
        after = Counter(
            target_key(item["soft_target_5"]) for item in shuffled_a if item["label_5"] == label
        )
        assert before == after
    assert [item["record_id"] for item in shuffled_a] == [item["record_id"] for item in rows]
    assert all(
        max(range(5), key=item["soft_target_5"].__getitem__) + 1 == item["label_5"]
        for item in shuffled_a
    )


def test_shuffle_seed_changes_mapping() -> None:
    rows = [row(str(index), 4, [0, 0, 0, 1, 0]) for index in range(20)]
    _, mapping_a = shuffle_train_rows(rows, shuffle_seed=1)
    _, mapping_b = shuffle_train_rows(rows, shuffle_seed=2)
    assert mapping_sha256(mapping_a) != mapping_sha256(mapping_b)
