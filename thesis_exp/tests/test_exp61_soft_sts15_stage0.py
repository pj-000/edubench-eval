import pandas as pd
import pytest

from thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset import (
    empirical_distribution,
    normalized_sentence,
    parse_dataset_lines,
    quantized_mean_target,
    require_all,
    serialize_manifest,
    unique_mode,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.audit_token_lengths import (
    choose_max_length,
    render_input,
)


def test_empirical_distribution_uses_six_ordered_bins() -> None:
    assert empirical_distribution((0, 1, 1, 3, 5)) == [0.2, 0.4, 0.0, 0.2, 0.0, 0.2]


def test_unique_mode_marks_ties() -> None:
    assert unique_mode((3, 3, 4, 4, 5)) == -1
    assert unique_mode((3, 4, 4, 4, 5)) == 4


def test_sentence_normalization_is_case_and_whitespace_stable() -> None:
    assert normalized_sentence("  A  Sentence\nHere ") == "a sentence here"


def valid_line(scores: str = "0 1 2 3 4") -> str:
    return f"2.0\t5\torigin\t{scores}\tSentence one.\tSentence two."


def test_dataset_parser_is_fail_closed_on_score_count() -> None:
    with pytest.raises(RuntimeError, match="published scores instead of 5"):
        parse_dataset_lines([valid_line("0 1 2 3")])


def test_dataset_parser_is_fail_closed_on_label_range() -> None:
    with pytest.raises(RuntimeError, match="outside"):
        parse_dataset_lines([valid_line("0 1 2 3 6")])


def test_dataset_parser_is_fail_closed_on_non_integer_score() -> None:
    with pytest.raises(RuntimeError, match="non-integer"):
        parse_dataset_lines([valid_line("0 1 2 3 x")])


def test_dataset_parser_is_fail_closed_on_row_count() -> None:
    with pytest.raises(RuntimeError, match="rows instead of 2"):
        parse_dataset_lines([valid_line()], expected_rows=2)


def test_quantized_mean_is_not_assumed_to_be_a_mode() -> None:
    scores = (0, 0, 3, 3, 4)
    assert quantized_mean_target(scores) == 2
    assert 2 not in scores
    assert unique_mode(scores) == -1


def test_gate_helper_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="bad_gate"):
        require_all({"good_gate": True, "bad_gate": False}, "test")


def test_manifest_serialization_is_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {
                "row_id": 0,
                "sentence1_normalized": "a",
                "sentence2_normalized": "b",
                "scores": (0, 1, 1, 2, 2),
                "component": "a",
                "frozen_split": "train",
                "origin": "synthetic",
                "mode": -1,
                "tercile": 1,
            }
        ]
    )
    assert serialize_manifest(frame) == serialize_manifest(frame.copy())


def test_tokenizer_max_length_selection_is_fail_closed() -> None:
    assert choose_max_length(143) == 256
    with pytest.raises(RuntimeError, match="exceeds"):
        choose_max_length(1025)


def test_input_template_contains_only_the_two_sentences_and_task() -> None:
    rendered = render_input("first", "second")
    assert "Sentence 1:\nfirst" in rendered
    assert "Sentence 2:\nsecond" in rendered
    assert "semantic similarity" in rendered
    assert "origin" not in rendered.lower()
