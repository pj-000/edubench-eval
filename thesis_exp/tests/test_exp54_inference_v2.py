from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    REPO_ROOT,
    _accepted_completion,
    _missing_utf8_continuation,
    _tokenize_closure_candidates,
    compile_v2_grammar,
    constrained_greedy_decode,
    file_sha256,
    load_v2_protocol,
    prepare_v2_runtime,
)


MODEL_PATH = Path(
    "/home/share/models/modelscope/Qwen/Qwen3-4B-Instruct-2507"
)


@pytest.fixture(scope="module")
def tokenizer_and_protocol():
    pytest.importorskip("xgrammar")
    if not MODEL_PATH.is_dir():
        pytest.skip("locked Qwen tokenizer is available only on server")
    from transformers import AutoTokenizer

    protocol = load_v2_protocol()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.pad_token_id = protocol["model_and_tokenizer"]["pad_token_id"]
    tokenizer.padding_side = "left"
    return tokenizer, protocol


def _matcher(tokenizer, protocol):
    import xgrammar as xgr

    _, compiled = compile_v2_grammar(
        tokenizer,
        model_vocab_size=protocol["model_and_tokenizer"][
            "model_vocab_size"
        ],
        protocol=protocol,
    )
    return xgr.GrammarMatcher(
        compiled,
        override_stop_tokens=[
            protocol["model_and_tokenizer"]["eos_token_id"]
        ],
        terminate_without_stop_token=True,
    )


def _accepts(tokenizer, protocol, text: str) -> bool:
    matcher = _matcher(tokenizer, protocol)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return all(matcher.accept_token(value) for value in token_ids) and bool(
        matcher.is_completed()
    )


def test_protocol_candidate_is_fail_closed_and_hash_bound():
    protocol = load_v2_protocol()
    grammar_path = REPO_ROOT / protocol["grammar"]["path"]
    assert file_sha256(grammar_path) == protocol["grammar"]["sha256"]
    assert protocol["status"].endswith("NOT_AUTHORIZED")
    assert protocol["selection_and_access"] == {
        "decoder_variant_selection_from_dev_allowed": False,
        "diagnostic_threshold_selection_from_dev_allowed": False,
        "train_only_and_synthetic_tests_required": True,
        "independent_review_required_before_dev": True,
        "formal_v2_dev_allowed": False,
        "formal_test_allowed": False,
    }
    assert protocol["application"]["arms"] == ["S0", "R1", "R2", "R3"]
    assert protocol["application"]["seeds"] == [42, 43, 44]
    assert protocol["application"]["logical_epochs"] == [1, 2, 3]
    assert protocol["generation"]["max_new_tokens"] == 256


@pytest.mark.parametrize(
    "text",
    [
        '{"score":1,"rationale":""}',
        '{"score":5,"rationale":"中文理由"}',
        '{"score":2,"rationale":"English \\"quote\\" and \\\\ path"}',
        '{"score":3,"rationale":"line\\nnext\\ttab"}',
        '{"score":4,"rationale":"emoji 😀，全角ＡＢＣ mixed"}',
        '{"score":1,"rationale":"unicode \\u4e2d"}',
    ],
)
def test_grammar_accepts_exact_legal_language(
    tokenizer_and_protocol,
    text,
):
    tokenizer, protocol = tokenizer_and_protocol
    assert _accepts(tokenizer, protocol, text)
    parsed = parse_review_json(text)
    assert set(parsed) == {"score", "rationale"}


@pytest.mark.parametrize(
    "text",
    [
        '{"score":0,"rationale":""}',
        '{"score":6,"rationale":""}',
        '{"score":"3","rationale":""}',
        '{"rationale":"","score":3}',
        '{"score":3}',
        '{"score":3,"rationale":"","extra":1}',
        '{"score":3,"score":4,"rationale":""}',
        '{"score":3,"rationale":null}',
        '{"score":3,"rationale":"bad\nnewline"}',
        ' {"score":3,"rationale":""}',
        '{"score":3,"rationale":""} ',
        '{"score":3,"rationale":""}{"score":4,"rationale":""}',
        '{"score":3,"rationale":""}trailing',
    ],
)
def test_grammar_rejects_outside_language(
    tokenizer_and_protocol,
    text,
):
    tokenizer, protocol = tokenizer_and_protocol
    assert not _accepts(tokenizer, protocol, text)


def test_shortest_completion_covers_all_escape_prefix_states(
    tokenizer_and_protocol,
):
    tokenizer, protocol = tokenizer_and_protocol
    candidates = _tokenize_closure_candidates(tokenizer)
    base = '{"score":1,"rationale":"abc'
    for tail in ("", "\\", "\\u", "\\uA", "\\uAB", "\\uABC"):
        matcher = _matcher(tokenizer, protocol)
        prefix = base + tail
        assert all(
            matcher.accept_token(value)
            for value in tokenizer.encode(prefix, add_special_tokens=False)
        )
        completion = _accepted_completion(matcher, candidates)
        assert completion is not None
        assert all(matcher.accept_token(value) for value in completion)
        assert matcher.is_completed()


def test_completion_repairs_only_a_trailing_utf8_prefix(
    tokenizer_and_protocol,
):
    tokenizer, protocol = tokenizer_and_protocol
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=protocol["model_and_tokenizer"][
            "model_vocab_size"
        ],
        protocol=protocol,
    )
    candidates = _tokenize_closure_candidates(tokenizer)
    decoded_vocab = runtime[4]
    single_byte_token_ids = runtime[5]
    partial_byte = b"\xe4"
    partial_token_id = next(
        index
        for index, value in enumerate(decoded_vocab)
        if value == partial_byte
    )
    prefix = '{"score":1,"rationale":"abc'
    matcher = _matcher(tokenizer, protocol)
    assert all(
        matcher.accept_token(value)
        for value in tokenizer.encode(prefix, add_special_tokens=False)
    )
    assert matcher.accept_token(partial_token_id)
    assert _accepted_completion(matcher, candidates) is None
    completion = _accepted_completion(
        matcher,
        candidates,
        generated_bytes=prefix.encode("utf-8") + partial_byte,
        single_byte_token_ids=single_byte_token_ids,
    )
    assert completion is not None
    assert all(matcher.accept_token(value) for value in completion)
    assert matcher.is_completed()
    completed_bytes = (
        prefix.encode("utf-8")
        + partial_byte
        + b"".join(decoded_vocab[value] for value in completion)
    )
    completed_text = completed_bytes.decode("utf-8")
    combined_token_ids = (
        tokenizer.encode(prefix, add_special_tokens=False)
        + [partial_token_id]
        + list(completion)
    )
    assert tokenizer.decode(
        combined_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == completed_text
    assert parse_review_json(completed_text)["score"] == 1
    assert _missing_utf8_continuation(completed_bytes) is None


class _ScriptedModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        scripted_tokens: list[int],
        illegal_token_id: int,
        continuation_token_id: int | None = None,
    ):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.scripted_tokens = scripted_tokens
        self.illegal_token_id = illegal_token_id
        self.continuation_token_id = continuation_token_id
        self.call_index = 0

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
    ):
        import torch

        batch_size = int(input_ids.shape[0])
        logits = torch.full(
            (batch_size, 1, self.config.vocab_size),
            -20.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        if self.call_index < len(self.scripted_tokens):
            preferred = self.scripted_tokens[self.call_index]
        elif self.continuation_token_id is not None:
            preferred = self.continuation_token_id
        else:
            preferred = self.scripted_tokens[-1]
        logits[:, :, preferred] = 5.0
        logits[:, :, self.illegal_token_id] = 10.0
        self.call_index += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_adversarial_illegal_top1_is_blocked_and_deterministic(
    tokenizer_and_protocol,
):
    tokenizer, protocol = tokenizer_and_protocol
    text = '{"score":3,"rationale":"中文 \\"x\\" 😀"}'
    target = tokenizer.encode(text, add_special_tokens=False)
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=protocol["model_and_tokenizer"][
            "model_vocab_size"
        ],
        protocol=protocol,
    )
    outputs = []
    for _ in range(2):
        model = _ScriptedModel(
            vocab_size=protocol["model_and_tokenizer"]["model_vocab_size"],
            scripted_tokens=target,
            illegal_token_id=protocol["model_and_tokenizer"][
                "eos_token_id"
            ],
        )
        result = constrained_greedy_decode(
            model,
            tokenizer,
            [[1]],
            protocol=protocol,
            runtime=runtime,
            device="cpu",
        )[0]
        outputs.append(result)
    assert outputs[0].text == text
    assert outputs[0].token_ids == outputs[1].token_ids
    assert outputs[0].diagnostics.unconstrained_top1_blocked_steps == len(
        target
    )
    assert not outputs[0].diagnostics.forced_completion
    assert parse_review_json(outputs[0].text)["score"] == 3


def test_budget_completion_closes_at_boundary_without_semantic_repair(
    tokenizer_and_protocol,
):
    tokenizer, protocol = tokenizer_and_protocol
    test_protocol = copy.deepcopy(protocol)
    test_protocol["generation"]["max_new_tokens"] = 32
    prefix = '{"score":2,"rationale":"'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    continuation = tokenizer.encode("a", add_special_tokens=False)
    assert len(continuation) == 1
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=test_protocol["model_and_tokenizer"][
            "model_vocab_size"
        ],
        protocol=test_protocol,
    )
    model = _ScriptedModel(
        vocab_size=test_protocol["model_and_tokenizer"]["model_vocab_size"],
        scripted_tokens=prefix_tokens,
        continuation_token_id=continuation[0],
        illegal_token_id=test_protocol["model_and_tokenizer"][
            "eos_token_id"
        ],
    )
    result = constrained_greedy_decode(
        model,
        tokenizer,
        [[1]],
        protocol=test_protocol,
        runtime=runtime,
        device="cpu",
    )[0]
    parsed = parse_review_json(result.text)
    assert parsed["score"] == 2
    assert set(parsed["rationale"]) == {"a"}
    assert result.diagnostics.generated_token_count == 32
    assert result.diagnostics.forced_completion
    assert result.diagnostics.completion_at_max_token_boundary


def test_root_completion_makes_second_object_unreachable(
    tokenizer_and_protocol,
):
    tokenizer, protocol = tokenizer_and_protocol
    matcher = _matcher(tokenizer, protocol)
    first = '{"score":1,"rationale":""}'
    assert all(
        matcher.accept_token(value)
        for value in tokenizer.encode(first, add_special_tokens=False)
    )
    assert matcher.is_completed()
    second = tokenizer.encode(
        '{"score":2,"rationale":""}',
        add_special_tokens=False,
    )
    assert not matcher.accept_token(second[0])


def test_public_v1_evidence_contains_no_row_level_or_raw_content():
    report_path = (
        REPO_ROOT
        / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
        "dev_execution_attempt_v1_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["privacy"] == {
        "labels_public": False,
        "private_evidence_retained": True,
        "questions_answers_rubrics_public": False,
        "raw_outputs_public": False,
        "row_level_identifiers_public": False,
    }
    stack = [report]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            assert not {
                "record_id",
                "reference_id",
                "question",
                "answer",
                "raw_output",
                "rationale_text",
                "label_5",
            }.intersection(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    assert report["test_accessed"] is False
