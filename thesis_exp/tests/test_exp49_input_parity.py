from __future__ import annotations

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split


def test_hard_and_soft_arms_share_the_same_text() -> None:
    rows = load_split("train")
    hard_hash = aggregate_text_hash(rows)
    soft_hash = aggregate_text_hash(rows)
    assert hard_hash == soft_hash
    assert all("Rubric:" not in row["text"] for row in rows)
    assert all("Evaluation Dimension:" in row["text"] for row in rows)
