from thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset import (
    empirical_distribution,
    normalized_sentence,
    unique_mode,
)


def test_empirical_distribution_uses_six_ordered_bins() -> None:
    assert empirical_distribution((0, 1, 1, 3, 5)) == [0.2, 0.4, 0.0, 0.2, 0.0, 0.2]


def test_unique_mode_marks_ties() -> None:
    assert unique_mode((3, 3, 4, 4, 5)) == -1
    assert unique_mode((3, 4, 4, 4, 5)) == 4


def test_sentence_normalization_is_case_and_whitespace_stable() -> None:
    assert normalized_sentence("  A  Sentence\nHere ") == "a sentence here"

