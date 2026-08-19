from __future__ import annotations

from thesis_exp.exp54_rar_sft.build_llm_judge_spec_pilot_packets import (
    clear_anchor,
    crossing_count,
    rubric_key,
    select_groups,
)


def _train(record_id: str, rubric: str, label: int, human: list[int]) -> dict:
    return {
        "record_id": record_id,
        "rubric": [rubric],
        "label_5": label,
        "human_1_5": human[0],
        "human_2_5": human[1],
        "human_3_5": human[2],
    }


def _events(record_id: str, gold: int, predictions: list[int]) -> list[dict]:
    return [
        {"record_id": record_id, "gold_label": gold, "generated_score": prediction}
        for prediction in predictions
    ]


def test_crossing_count_uses_each_seed_event() -> None:
    assert crossing_count(_events("r", 2, [2, 3, 4]), 2) == 2
    assert crossing_count(_events("r", 4, [3, 4, 5]), 3) == 1


def test_clear_anchor_requires_human_and_model_agreement() -> None:
    row = _train("r", "rubric", 4, [4, 4, 4])
    assert clear_anchor(row, _events("r", 4, [4, 4, 4])) is True
    assert clear_anchor(row, _events("r", 4, [4, 3, 4])) is False
    disputed = _train("r", "rubric", 4, [3, 4, 4])
    assert clear_anchor(disputed, _events("r", 4, [4, 4, 4])) is False


def test_rubric_key_is_structural_and_deterministic() -> None:
    first = _train("r1", "same", 1, [1, 1, 1])
    second = _train("r2", "same", 5, [5, 5, 5])
    assert rubric_key(first) == rubric_key(second)


def test_select_groups_closes_all_boundaries_with_unique_rubrics() -> None:
    train = []
    bank = []
    for lower in range(1, 5):
        rubric = f"rubric-{lower}"
        for index in range(5):
            record_id = f"b{lower}-cross-{index}"
            train.append(_train(record_id, rubric, lower, [lower] * 3))
            bank.extend(_events(record_id, lower, [lower + 1] * 3))
        anchor_id = f"b{lower}-anchor"
        train.append(_train(anchor_id, rubric, lower + 1, [lower + 1] * 3))
        bank.extend(_events(anchor_id, lower + 1, [lower + 1] * 3))
    selected = select_groups(train, bank)
    assert [group["lower"] for group in selected] == [1, 2, 3, 4]
    assert len({group["rubric_key"] for group in selected}) == 4
    assert all(len(group["development_ids"]) == 2 for group in selected)
    assert all(len(group["crossing_evaluation_ids"]) == 3 for group in selected)
    assert all(group["anchor_evaluation_id"] for group in selected)
