import json

import pytest

from thesis_exp.exp54_rar_sft.training_contract import (
    build_prompt_cache_row,
    materialize_sequence,
    prompt_messages,
    serialize_target,
    tokenize_target,
)


class CharacterTokenizer:
    """Small tokenizer double whose token boundaries are individual characters."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert enable_thinking is False
        rendered = "".join(
            (
                f"<{message['role']}>\n"
                f"{message['content']}"
                f"<end>\n"
            )
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>\n"
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered


class WholeStringTokenizer(CharacterTokenizer):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [999],
            "offset_mapping": [(0, len(text))],
        }


class RepeatedOffsetTokenizer(CharacterTokenizer):
    """Mimic byte-level subpieces sharing one Unicode character offset."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        encoded = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        emoji_index = text.find("😀")
        if emoji_index >= 0:
            encoded["input_ids"].insert(emoji_index + 1, 123456)
            encoded["offset_mapping"].insert(
                emoji_index + 1,
                (emoji_index, emoji_index + 1),
            )
        return encoded


def _row() -> dict:
    return {
        "record_id": "row-1",
        "question": "What is 2 + 2?",
        "answer": "It is four.",
        "metric_canonical": "correctness",
        "rubric": ["5: fully correct", "1: incorrect"],
        "label_5": 5,
        "references": [{"reason": "PRIVATE-REASON-MUST-NOT-LEAK"}],
    }


def test_target_serialization_round_trips_and_marks_exact_fields() -> None:
    rationale = '答案含有 "四"，路径为 C:\\tmp。'
    target, spans = serialize_target(4, rationale)

    assert json.loads(target) == {"score": 4, "rationale": rationale + " "}
    assert target[slice(*spans["score"])] == "4"
    assert json.loads(
        '"' + target[slice(*spans["rationale"])] + '"'
    ) == rationale
    assert target == (
        '{"score":4,"rationale":"答案含有 \\"四\\"，路径为 C:\\\\tmp。 "}'
    )


def test_prompt_uses_only_declared_inference_fields() -> None:
    rendered = json.dumps(prompt_messages(_row()), ensure_ascii=False)

    assert "What is 2 + 2?" in rendered
    assert "It is four." in rendered
    assert "correctness" in rendered
    assert "5: fully correct" in rendered
    assert "PRIVATE-REASON-MUST-NOT-LEAK" not in rendered


def test_score_and_rationale_masks_exclude_json_punctuation() -> None:
    tokenizer = CharacterTokenizer()
    target = tokenize_target(
        tokenizer,
        score=3,
        rationale="简短理由。",
        rationale_active=True,
    )

    score_text = "".join(
        chr(target["target_token_ids"][index])
        for index in target["score_token_positions_in_target"]
    )
    rationale_text = "".join(
        chr(target["target_token_ids"][index])
        for index in target["rationale_token_positions_in_target"]
    )
    supervised = set(target["score_token_positions_in_target"]) | set(
        target["rationale_token_positions_in_target"]
    )

    assert score_text == "3"
    assert rationale_text == "简短理由。"
    assert all(
        chr(token_id) not in "{}:,\""
        for index, token_id in enumerate(target["target_token_ids"])
        if index in supervised
    )


def test_inactive_rationale_keeps_score_supervision_only() -> None:
    target = tokenize_target(
        CharacterTokenizer(),
        score=2,
        rationale="",
        rationale_active=False,
    )

    assert target["score_token_positions_in_target"]
    assert target["rationale_token_positions_in_target"] == []
    with pytest.raises(ValueError, match="inactive rationale"):
        tokenize_target(
            CharacterTokenizer(),
            score=2,
            rationale="must not be present",
            rationale_active=False,
        )


def test_tokenizer_boundary_crossing_is_a_hard_failure() -> None:
    with pytest.raises(ValueError, match="crosses tokenizer boundaries"):
        tokenize_target(
            WholeStringTokenizer(),
            score=1,
            rationale="reason",
            rationale_active=True,
        )


def test_repeated_unicode_offsets_are_allowed_when_field_coverage_is_exact() -> None:
    target = tokenize_target(
        RepeatedOffsetTokenizer(),
        score=1,
        rationale="emoji 😀 rationale",
        rationale_active=True,
    )

    assert len(target["rationale_token_positions_in_target"]) > len(
        "emoji 😀 rationale"
    )


def test_materialized_masks_exclude_prompt_and_assistant_suffix() -> None:
    tokenizer = CharacterTokenizer()
    prompt = build_prompt_cache_row(tokenizer, _row(), row_position=0)
    target = tokenize_target(
        tokenizer,
        score=5,
        rationale="Because the answer is correct.",
        rationale_active=True,
    )
    sequence = materialize_sequence(tokenizer, prompt, target)
    target_start = prompt["prompt_token_count"]
    target_end = target_start + len(target["target_token_ids"])

    assert sequence["assistant_suffix_token_ids"]
    assert all(
        target_start <= position < target_end
        for position in (
            sequence["score_token_positions"]
            + sequence["rationale_token_positions"]
        )
    )
    assert len(sequence["score_token_positions"]) == 1
    assert sequence["rationale_token_positions"]
