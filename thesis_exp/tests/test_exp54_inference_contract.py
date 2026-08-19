import pytest

from thesis_exp.exp54_rar_sft.inference_contract import (
    GENERATION_KWARGS,
    parse_review_json,
)


def test_strict_parser_accepts_exact_contract() -> None:
    assert parse_review_json(' {"score":3,"rationale":"Grounded."} \n') == {
        "score": 3,
        "rationale": "Grounded.",
    }
    assert parse_review_json('{"score":1,"rationale":""}')["rationale"] == ""


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"score":3}',
        '{"score":3,"rationale":"x","extra":1}',
        '{"score":true,"rationale":"x"}',
        '{"score":3.0,"rationale":"x"}',
        '{"score":0,"rationale":"x"}',
        '{"score":6,"rationale":"x"}',
        '{"score":3,"rationale":null}',
        '{"score":3,"rationale":"x"} trailing',
        '{"score":3,"rationale":"x"} {"score":4,"rationale":"y"}',
        '{"score":1,"score":5,"rationale":"x"}',
        '{"score":1,"rationale":"x","rationale":"y"}',
        '{"score":1,"rationale":"x","meta":{"a":1,"a":2}}',
    ],
)
def test_strict_parser_rejects_non_contract_outputs(text: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_review_json(text)


def test_generation_contract_is_deterministic_and_bounded() -> None:
    assert GENERATION_KWARGS == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
        "use_cache": True,
    }
