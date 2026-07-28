"""Tokenizer-aware constrained decoder for the Exp54 V2 output protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "inference_protocol_v2_candidate.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v2_protocol(path: Path = DEFAULT_PROTOCOL_CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "RAR_SFT_INFERENCE_PROTOCOL_V2":
        raise ValueError("unexpected V2 inference protocol identifier")
    grammar = value["grammar"]
    grammar_path = REPO_ROOT / grammar["path"]
    if file_sha256(grammar_path) != grammar["sha256"]:
        raise ValueError("V2 grammar hash differs from the protocol lock")
    if value["generation"] != {
        "search": "greedy_argmax_after_grammar_mask",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "length_penalty": None,
        "posthoc_repair_allowed": False,
        "unconstrained_fallback_allowed": False,
    }:
        raise ValueError("V2 generation settings differ from the frozen candidate")
    return value


@dataclass(frozen=True)
class DecodingDiagnostics:
    generated_token_count: int
    grammar_intervention_steps: int
    active_generation_steps: int
    unconstrained_top1_blocked_steps: int
    removed_probability_mass_mean: float
    forced_completion: bool
    forced_completion_token_count: int
    completion_at_max_token_boundary: bool


@dataclass(frozen=True)
class DecodedSequence:
    token_ids: list[int]
    text: str
    diagnostics: DecodingDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids,
            "text": self.text,
            "diagnostics": asdict(self.diagnostics),
        }


def _closure_candidate_texts() -> tuple[str, ...]:
    """Return semantic-free suffixes covering every JSON escape-prefix state."""
    candidates = {
        '"}',
        'n"}',
        't"}',
        'r"}',
        'b"}',
        'f"}',
        '/"}',
        '\\"}',
        'u0000"}',
        '0000"}',
        '000"}',
        '00"}',
        '0"}',
    }
    for score in range(1, 6):
        minimal = f'{{"score":{score},"rationale":""}}'
        candidates.update(
            minimal[offset:] for offset in range(len(minimal))
        )
    return tuple(sorted(candidates))


def _tokenize_closure_candidates(tokenizer: Any) -> tuple[tuple[int, ...], ...]:
    candidates: set[tuple[int, ...]] = set()
    for text in _closure_candidate_texts():
        token_ids = tuple(
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        )
        if token_ids and tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) == text:
            candidates.add(token_ids)
    if not candidates:
        raise ValueError("tokenizer produced no exact structural completion")
    return tuple(sorted(candidates, key=lambda value: (len(value), value)))


def _accepted_completion(
    matcher: Any,
    candidates: Sequence[Sequence[int]],
    *,
    generated_bytes: bytes = b"",
    single_byte_token_ids: dict[int, int] | None = None,
) -> tuple[int, ...] | None:
    """Find the deterministic shortest candidate that completes this matcher."""
    accepted: list[tuple[int, ...]] = []
    for candidate in candidates:
        probe = matcher.fork()
        if all(probe.accept_token(int(token_id)) for token_id in candidate):
            if probe.is_completed():
                accepted.append(tuple(int(value) for value in candidate))
    missing_utf8 = _missing_utf8_continuation(generated_bytes)
    if missing_utf8 is not None and single_byte_token_ids is not None:
        byte_prefix = tuple(
            single_byte_token_ids[value] for value in missing_utf8
        )
        for candidate in candidates:
            fallback = byte_prefix + tuple(candidate)
            probe = matcher.fork()
            if all(probe.accept_token(token_id) for token_id in fallback):
                if probe.is_completed():
                    accepted.append(fallback)
    return min(accepted, key=lambda value: (len(value), value)) if accepted else None


def _missing_utf8_continuation(value: bytes) -> bytes | None:
    """Return deterministic bytes completing a trailing valid UTF-8 prefix."""
    for prefix_length in range(1, min(3, len(value)) + 1):
        prefix = value[-prefix_length:]
        lead = prefix[0]
        if 0xC2 <= lead <= 0xDF:
            expected = 2
        elif 0xE0 <= lead <= 0xEF:
            expected = 3
        elif 0xF0 <= lead <= 0xF4:
            expected = 4
        else:
            continue
        if len(prefix) >= expected:
            continue
        if any(not 0x80 <= item <= 0xBF for item in prefix[1:]):
            continue
        first_missing = 0x80
        if len(prefix) == 1:
            if lead == 0xE0:
                first_missing = 0xA0
            elif lead == 0xF0:
                first_missing = 0x90
        missing = bytes(
            [first_missing]
            + [0x80] * (expected - len(prefix) - 1)
        )
        try:
            (prefix + missing).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        return missing
    return None


def _token_byte_tables(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    eos_token_id: int,
) -> tuple[tuple[bytes, ...], dict[int, int]]:
    import xgrammar as xgr

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=model_vocab_size,
        stop_token_ids=[eos_token_id],
    )
    decoded_vocab = tuple(
        bytes(value) for value in tokenizer_info.decoded_vocab
    )
    single_byte_token_ids: dict[int, int] = {}
    for token_id, value in enumerate(decoded_vocab):
        if len(value) == 1:
            single_byte_token_ids.setdefault(value[0], token_id)
    if set(single_byte_token_ids) != set(range(256)):
        raise ValueError("tokenizer lacks a complete single-byte fallback")
    return decoded_vocab, single_byte_token_ids


def compile_v2_grammar(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    protocol: dict[str, Any],
) -> tuple[Any, Any]:
    import importlib.metadata

    import xgrammar as xgr

    backend = protocol["backend"]
    if importlib.metadata.version("xgrammar") != backend["version"]:
        raise ValueError("installed XGrammar version differs from protocol lock")
    dependency = backend["dependency_wheels"][0]
    if (
        importlib.metadata.version(dependency["package"])
        != dependency["version"]
    ):
        raise ValueError("installed XGrammar dependency differs from protocol lock")
    model_lock = protocol["model_and_tokenizer"]
    if model_vocab_size != int(model_lock["model_vocab_size"]):
        raise ValueError("model vocabulary size differs from protocol lock")
    if int(tokenizer.pad_token_id) != int(model_lock["pad_token_id"]):
        raise ValueError("pad token differs from protocol lock")
    if int(tokenizer.eos_token_id) != int(model_lock["eos_token_id"]):
        raise ValueError("EOS token differs from protocol lock")
    grammar_path = REPO_ROOT / protocol["grammar"]["path"]
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=model_vocab_size,
        stop_token_ids=[int(model_lock["eos_token_id"])],
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar(
        grammar_path.read_text(encoding="utf-8"),
        root_rule_name=protocol["grammar"]["root_rule"],
    )
    return xgr, compiled


def prepare_v2_runtime(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    protocol: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    Any,
    Any,
    tuple[tuple[int, ...], ...],
    tuple[bytes, ...],
    dict[int, int],
]:
    resolved_protocol = protocol or load_v2_protocol()
    xgr, compiled_grammar = compile_v2_grammar(
        tokenizer,
        model_vocab_size=model_vocab_size,
        protocol=resolved_protocol,
    )
    decoded_vocab, single_byte_token_ids = _token_byte_tables(
        tokenizer,
        model_vocab_size=model_vocab_size,
        eos_token_id=int(
            resolved_protocol["model_and_tokenizer"]["eos_token_id"]
        ),
    )
    return (
        resolved_protocol,
        xgr,
        compiled_grammar,
        _tokenize_closure_candidates(tokenizer),
        decoded_vocab,
        single_byte_token_ids,
    )


def constrained_greedy_decode(
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    *,
    protocol: dict[str, Any] | None = None,
    runtime: (
        tuple[
            dict[str, Any],
            Any,
            Any,
            tuple[tuple[int, ...], ...],
            tuple[bytes, ...],
            dict[int, int],
        ]
        | None
    ) = None,
    device: str = "cuda:0",
) -> list[DecodedSequence]:
    """Decode a batch under one shared grammar, with no post-hoc repair."""
    import torch

    if not prompt_token_ids:
        return []
    if runtime is None:
        runtime = prepare_v2_runtime(
            tokenizer,
            model_vocab_size=int(model.config.vocab_size),
            protocol=protocol,
        )
    (
        resolved_protocol,
        xgr,
        compiled_grammar,
        closure_candidates,
        decoded_vocab,
        single_byte_token_ids,
    ) = runtime
    if protocol is not None and protocol != resolved_protocol:
        raise ValueError("explicit protocol differs from prepared V2 runtime")
    max_new_tokens = int(
        resolved_protocol["generation"]["max_new_tokens"]
    )
    model_vocab_size = int(model.config.vocab_size)
    if model_vocab_size != int(
        resolved_protocol["model_and_tokenizer"]["model_vocab_size"]
    ):
        raise ValueError("model vocabulary differs from prepared V2 runtime")
    matchers = [
        xgr.GrammarMatcher(
            compiled_grammar,
            override_stop_tokens=[
                int(resolved_protocol["model_and_tokenizer"]["eos_token_id"])
            ],
            terminate_without_stop_token=True,
        )
        for _ in prompt_token_ids
    ]
    encoded = tokenizer.pad(
        {"input_ids": [list(values) for values in prompt_token_ids]},
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size = int(input_ids.shape[0])
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    generated_bytes = [bytearray() for _ in range(batch_size)]
    completed = [False] * batch_size
    forced_paths: list[list[int]] = [[] for _ in range(batch_size)]
    forced_completion = [False] * batch_size
    forced_token_counts = [0] * batch_size
    grammar_intervention_steps = [0] * batch_size
    unconstrained_blocked_steps = [0] * batch_size
    removed_mass_sums = [0.0] * batch_size
    active_steps = [0] * batch_size
    bitmask = xgr.allocate_token_bitmask(batch_size, model_vocab_size)
    past_key_values = None
    next_input_ids = input_ids

    for step_index in range(max_new_tokens):
        with torch.inference_mode():
            output = model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        raw_logits = output.logits[:, -1, :]
        if int(raw_logits.shape[-1]) != model_vocab_size:
            raise ValueError("model logit vocabulary differs from config")
        past_key_values = output.past_key_values
        xgr.reset_token_bitmask(bitmask)
        active_indices = [
            index for index, done in enumerate(completed) if not done
        ]
        for index in active_indices:
            matchers[index].fill_next_token_bitmask(bitmask, index)
        masked_logits = raw_logits.clone()
        xgr.apply_token_bitmask_inplace(
            masked_logits,
            bitmask.to(raw_logits.device),
            vocab_size=model_vocab_size,
        )
        raw_top = torch.argmax(raw_logits, dim=-1)
        masked_top = torch.argmax(masked_logits, dim=-1)
        next_tokens = torch.full(
            (batch_size,),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=raw_logits.device,
        )

        for index in active_indices:
            remaining = max_new_tokens - step_index
            completion = _accepted_completion(
                matchers[index],
                closure_candidates,
                generated_bytes=bytes(generated_bytes[index]),
                single_byte_token_ids=single_byte_token_ids,
            )
            if completion is None:
                raise RuntimeError(
                    "no tokenizer-valid structural completion from matcher state"
                )
            if len(completion) > remaining:
                raise RuntimeError("structural completion exceeds token budget")
            if not forced_paths[index] and len(completion) == remaining:
                forced_paths[index] = list(completion)
                forced_completion[index] = True
            chosen = (
                forced_paths[index].pop(0)
                if forced_paths[index]
                else int(masked_top[index].item())
            )
            if not bool(
                torch.isfinite(masked_logits[index, chosen]).item()
            ):
                raise RuntimeError("decoder selected a grammar-masked token")
            next_tokens[index] = chosen
            active_steps[index] += 1
            if chosen != int(raw_top[index].item()):
                grammar_intervention_steps[index] += 1
            if not bool(
                torch.isfinite(
                    masked_logits[index, int(raw_top[index].item())]
                ).item()
            ):
                unconstrained_blocked_steps[index] += 1
            raw_lse = torch.logsumexp(
                raw_logits[index].float(),
                dim=-1,
            )
            masked_lse = torch.logsumexp(
                masked_logits[index].float(),
                dim=-1,
            )
            legal_mass = torch.exp(masked_lse - raw_lse).clamp(0.0, 1.0)
            removed_mass_sums[index] += float((1.0 - legal_mass).item())
            if forced_completion[index]:
                forced_token_counts[index] += 1
            if not matchers[index].accept_token(chosen):
                raise RuntimeError("XGrammar rejected its own selected token")
            generated[index].append(chosen)
            generated_bytes[index].extend(decoded_vocab[chosen])
            if matchers[index].is_completed():
                if forced_paths[index]:
                    raise RuntimeError(
                        "matcher completed before forced path was exhausted"
                    )
                completed[index] = True

        if all(completed):
            break
        appended_attention = torch.tensor(
            [[0 if done else 1] for done in completed],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat(
            (attention_mask, appended_attention),
            dim=1,
        )
        next_input_ids = next_tokens.unsqueeze(1)

    if not all(completed):
        unfinished = [
            index for index, done in enumerate(completed) if not done
        ]
        raise RuntimeError(
            f"grammar decoding did not complete within budget: {unfinished}"
        )

    results = []
    for index, token_ids in enumerate(generated):
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        results.append(
            DecodedSequence(
                token_ids=token_ids,
                text=text,
                diagnostics=DecodingDiagnostics(
                    generated_token_count=len(token_ids),
                    grammar_intervention_steps=(
                        grammar_intervention_steps[index]
                    ),
                    active_generation_steps=active_steps[index],
                    unconstrained_top1_blocked_steps=(
                        unconstrained_blocked_steps[index]
                    ),
                    removed_probability_mass_mean=(
                        removed_mass_sums[index] / active_steps[index]
                    ),
                    forced_completion=forced_completion[index],
                    forced_completion_token_count=(
                        forced_token_counts[index]
                    ),
                    completion_at_max_token_boundary=(
                        len(token_ids) == max_new_tokens
                    ),
                ),
            )
        )
    return results
