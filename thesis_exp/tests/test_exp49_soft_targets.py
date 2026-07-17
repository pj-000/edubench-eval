from __future__ import annotations

from thesis_exp.exp49_cphce.build_targets import human_distribution, load_split


def test_known_human_patterns_map_to_expected_distribution() -> None:
    assert human_distribution({"human_1": 4, "human_2": 5, "human_3": 5, "label_5": 5}) == [0.0, 0.0, 0.0, 1 / 3, 2 / 3]
    assert human_distribution({"human_1": 1, "human_2": 1, "human_3": 2, "label_5": 1}) == [2 / 3, 1 / 3, 0.0, 0.0, 0.0]


def test_all_locked_train_and_dev_targets_are_valid() -> None:
    assert len(load_split("train")) == 2654
    assert len(load_split("dev")) == 664
