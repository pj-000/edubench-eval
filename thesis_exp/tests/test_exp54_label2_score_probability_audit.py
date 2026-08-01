from __future__ import annotations

import copy

import pytest

from thesis_exp.exp54_rar_sft.audit_label2_score_probabilities import (
    EXPECTED_ROW_KEYS,
    normalized_probabilities,
    validate_probability_row,
)


def _fixture() -> tuple[dict, dict, dict]:
    dev = {
        "record_id": "r1",
        "question_key": "q1",
        "metric_id": "m1",
        "language": "en",
        "label_5": 2,
        "human_mean_5": 2.0,
        "human_1_5": 1.0,
        "human_2_5": 2.0,
        "human_3_5": 3.0,
    }
    prediction = {
        "prediction": {"score": 3},
        "forced_completion": False,
        "prompt_token_ids_sha256": "a" * 64,
    }
    logprobs = [-3.0, -1.0, -0.5, -2.0, -4.0]
    probabilities = normalized_probabilities(logprobs)
    row = {
        "row_position": 0,
        "record_id": "r1",
        "question_key": "q1",
        "metric_id": "m1",
        "language": "en",
        "label_5": 2,
        "human_mean_5": 2.0,
        "human_1_5": 1.0,
        "human_2_5": 2.0,
        "human_3_5": 3.0,
        "parsed_score": 3,
        "forced_completion": False,
        "prompt_token_ids_sha256": "a" * 64,
        "canonical_score_option_logprobs": {
            str(score): logprobs[score - 1] for score in range(1, 6)
        },
        "canonical_score_option_probabilities": {
            str(score): probabilities[score - 1] for score in range(1, 6)
        },
        "forced_choice_score": 3,
    }
    assert set(row) == EXPECTED_ROW_KEYS
    return row, dev, prediction


def test_probability_row_rebuilds_from_sources() -> None:
    row, dev, prediction = _fixture()
    validate_probability_row(
        row,
        dev_row=dev,
        prediction_row=prediction,
        row_position=0,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update(record_id="other"), "record_id differs"),
        (
            lambda row: row["canonical_score_option_probabilities"].update(
                {"2": 0.99}
            ),
            "normalization differs",
        ),
        (lambda row: row.update(forced_choice_score=2), "forced choice differs"),
        (lambda row: row.update(extra="bad"), "row keys differ"),
    ],
)
def test_probability_row_tampering_fails(mutation, match: str) -> None:
    row, dev, prediction = _fixture()
    tampered = copy.deepcopy(row)
    mutation(tampered)
    with pytest.raises(ValueError, match=match):
        validate_probability_row(
            tampered,
            dev_row=dev,
            prediction_row=prediction,
            row_position=0,
        )
