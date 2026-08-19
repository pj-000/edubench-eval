"""Frozen candidate prompt, target serialization, and token-mask contract."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import (
    clean_answer_text,
    clean_question_text,
)


CONTRACT_VERSION = "exp54-rar-v2-materialized-v1"
SYSTEM_MESSAGE = (
    "You are an educational response evaluator. Judge the answer using the "
    "requested evaluation metric and five-level rubric. Return one integer "
    "score from 1 to 5 and one concise visible rationale grounded in the "
    "answer and rubric. Do not reveal hidden reasoning."
)
OUTPUT_INSTRUCTION = (
    'Return only valid JSON with exactly this schema: '
    '{"score":<integer 1-5>,"rationale":"<concise rationale>"}'
)
CHAT_TEMPLATE_KWARGS = {
    "enable_thinking": False,
}
# A nonempty rationale receives one unsupervised trailing space inside its JSON
# string. This prevents BPE tokens such as `."` from crossing the boundary
# between supervised rationale content and unsupervised fixed JSON syntax.
RATIONALE_BOUNDARY_PADDING = " "
DEFAULT_RUBRIC_REGISTRY = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/canonical_rubric_registry.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    return sha256_bytes(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    )


def require_canonical_positions(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain integers")
    if value != sorted(set(value)):
        raise ValueError(f"{field} must be unique and strictly increasing")
    return list(value)


@lru_cache(maxsize=4)
def load_rubric_registry(path: Path = DEFAULT_RUBRIC_REGISTRY) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("status") != "EXP54_CANONICAL_RUBRIC_REGISTRY_V1":
        raise ValueError("canonical rubric registry has an invalid status")
    if registry.get("schema_version") != "exp54-rubric-registry-v1":
        raise ValueError("canonical rubric registry has an invalid schema")
    if registry.get("level_order") != [5, 4, 3, 2, 1]:
        raise ValueError("canonical rubric registry must contain levels 5..1")
    entries = registry.get("entries")
    if not isinstance(entries, dict) or len(entries) != 24:
        raise ValueError("canonical rubric registry must contain 24 entries")
    metrics: set[str] = set()
    for key, entry in entries.items():
        metric_id = str(entry.get("metric_id") or "")
        language = str(entry.get("language") or "")
        if key != f"{metric_id}|{language}":
            raise ValueError(f"rubric registry key mismatch: {key}")
        if language not in {"en", "zh"}:
            raise ValueError(f"rubric registry language is unsupported: {key}")
        digest = str(entry.get("rendered_rubric_sha256") or "")
        if len(digest) != 64:
            raise ValueError(f"rubric registry hash is invalid: {key}")
        metrics.add(metric_id)
    if len(metrics) != 12:
        raise ValueError("canonical rubric registry must cover 12 metrics")
    return registry


def rubric_text(
    value: Any,
    *,
    metric_id: str,
    language: str,
    registry: dict[str, Any] | None = None,
) -> str:
    if not isinstance(value, list):
        raise ValueError("rubric must be a structured five-item list")
    if len(value) != 5:
        raise ValueError("rubric must contain exactly five levels")
    lines = [str(item).strip() for item in value]
    if any(not line for line in lines):
        raise ValueError("rubric levels must all be nonempty")
    registry = registry or load_rubric_registry()
    entry = registry["entries"].get(f"{metric_id}|{language}")
    if entry is None:
        raise ValueError(
            f"rubric registry has no entry for {metric_id}|{language}"
        )
    rendered = "\n".join(lines)
    if sha256_bytes(rendered.encode("utf-8")) != entry[
        "rendered_rubric_sha256"
    ]:
        raise ValueError(
            f"rubric differs from canonical registry for {metric_id}|{language}"
        )
    return rendered


def prompt_input_fields(row: dict[str, Any]) -> dict[str, str]:
    question = clean_question_text(row.get("question"))
    answer = clean_answer_text(row.get("answer"))
    metric = str(
        row.get("metric_canonical")
        or row.get("metric_raw")
        or row.get("metric_id")
        or ""
    ).strip()
    metric_id = str(row.get("metric_id") or "").strip()
    language = str(row.get("language") or "").strip()
    rubric = rubric_text(
        row.get("rubric"),
        metric_id=metric_id,
        language=language,
    )
    if not question or not answer or not metric or not metric_id or not language:
        raise ValueError(f"{row.get('record_id')}: incomplete prompt input")
    return {
        "question": question,
        "answer": answer,
        "metric": metric,
        "rubric": rubric,
    }


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    fields = prompt_input_fields(row)
    user = "\n\n".join(
        [
            f"Question:\n{fields['question']}",
            f"Answer:\n{fields['answer']}",
            f"Evaluation Metric:\n{fields['metric']}",
            f"Five-level Rubric:\n{fields['rubric']}",
            OUTPUT_INSTRUCTION,
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def serialize_target(score: int, rationale: str) -> tuple[str, dict[str, tuple[int, int]]]:
    if score not in range(1, 6):
        raise ValueError("score must be in 1-5")
    if not isinstance(rationale, str):
        raise TypeError("rationale must be a string")
    score_text = str(score)
    serialized_rationale_content = json.dumps(
        rationale,
        ensure_ascii=False,
    )[1:-1]
    materialized_rationale = (
        rationale + RATIONALE_BOUNDARY_PADDING if rationale else ""
    )
    rationale_json = json.dumps(materialized_rationale, ensure_ascii=False)
    prefix = '{"score":'
    middle = ',"rationale":'
    target = prefix + score_text + middle + rationale_json + "}"
    score_start = len(prefix)
    rationale_start = len(prefix) + len(score_text) + len(middle) + 1
    rationale_end = rationale_start + len(serialized_rationale_content)
    spans = {
        "score": (score_start, score_start + len(score_text)),
        "rationale": (rationale_start, rationale_end),
    }
    if json.loads(target) != {
        "score": score,
        "rationale": materialized_rationale,
    }:
        raise AssertionError("target serialization does not round-trip")
    return target, spans


def _span_token_positions(
    target: str,
    offsets: list[tuple[int, int]],
    span: tuple[int, int],
    *,
    field: str,
) -> list[int]:
    span_start, span_end = span
    if span_start == span_end:
        return []
    crossing = [
        (index, start, end)
        for index, (start, end) in enumerate(offsets)
        if end > start
        and start < span_end
        and end > span_start
        and not (span_start <= start and end <= span_end)
    ]
    if crossing:
        raise ValueError(f"{field} span crosses tokenizer boundaries: {crossing[:3]}")
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and span_start <= start and end <= span_end
    ]
    covered = []
    for position in positions:
        start, end = offsets[position]
        covered.extend(range(start, end))
    expected_coverage = set(range(span_start, span_end))
    actual_coverage = set(covered)
    if actual_coverage != expected_coverage:
        missing = sorted(expected_coverage - actual_coverage)
        extra = sorted(actual_coverage - expected_coverage)
        raise ValueError(
            f"{field} span is not exactly token-covered: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    if not target[span_start:span_end]:
        raise AssertionError(f"{field} nonempty span resolved to empty text")
    return positions


def tokenize_target(
    tokenizer: Any,
    *,
    score: int,
    rationale: str,
    rationale_active: bool,
) -> dict[str, Any]:
    if rationale_active and not rationale:
        raise ValueError("active rationale must be nonempty")
    if not rationale_active and rationale:
        raise ValueError("inactive rationale target must be empty")
    target, spans = serialize_target(score, rationale)
    encoded = tokenizer(
        target,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    score_positions = _span_token_positions(
        target,
        offsets,
        spans["score"],
        field="score",
    )
    rationale_positions = (
        _span_token_positions(
            target,
            offsets,
            spans["rationale"],
            field="rationale",
        )
        if rationale_active
        else []
    )
    if not score_positions:
        raise ValueError("score span has no supervised token")
    if rationale_active and not rationale_positions:
        raise ValueError("active rationale span has no supervised token")
    if set(score_positions) & set(rationale_positions):
        raise AssertionError("score and rationale token masks overlap")
    return {
        "target_text": target,
        "target_bytes_sha256": sha256_bytes(target.encode("utf-8")),
        "target_token_ids": token_ids,
        "target_token_ids_sha256": token_ids_sha256(token_ids),
        "score_token_positions_in_target": score_positions,
        "rationale_token_positions_in_target": rationale_positions,
        "score_token_ids": [token_ids[index] for index in score_positions],
        "rationale_token_ids": [
            token_ids[index] for index in rationale_positions
        ],
    }


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        **CHAT_TEMPLATE_KWARGS,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_cache_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    row_position: int,
) -> dict[str, Any]:
    record_id = str(row.get("record_id") or "")
    if not record_id:
        raise ValueError(f"row {row_position}: missing record_id")
    fields = prompt_input_fields(row)
    messages = prompt_messages(row)
    rendered = str(
        apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    prompt_ids = list(
        apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if not prompt_ids:
        raise ValueError(f"{record_id}: empty prompt tokenization")
    if int(row["label_5"]) not in range(1, 6):
        raise ValueError(f"{record_id}: score outside 1-5")
    return {
        "prompt_cache_id": f"prompt-{record_id}",
        "row_position": row_position,
        "record_id": record_id,
        "messages": messages,
        "input_fields_sha256": sha256_bytes(canonical_json_bytes(fields)),
        "prompt_bytes_sha256": sha256_bytes(rendered.encode("utf-8")),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
        "prompt_token_count": len(prompt_ids),
    }


def materialize_sequence(
    tokenizer: Any,
    prompt_cache_row: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    assistant_message = {
        "role": "assistant",
        "content": str(target["target_text"]),
    }
    full_ids = list(
        apply_chat_template(
            tokenizer,
            [*prompt_cache_row["messages"], assistant_message],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prompt_ids = list(prompt_cache_row["prompt_token_ids"])
    target_ids = list(target["target_token_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("full sequence does not preserve prompt token prefix")
    target_start = len(prompt_ids)
    if full_ids[target_start : target_start + len(target_ids)] != target_ids:
        raise ValueError("assistant target tokens do not align inside chat template")
    suffix_ids = full_ids[target_start + len(target_ids) :]
    if not suffix_ids:
        raise ValueError("chat template produced no assistant closing suffix")
    score_positions = [
        target_start + index
        for index in target["score_token_positions_in_target"]
    ]
    rationale_positions = [
        target_start + index
        for index in target["rationale_token_positions_in_target"]
    ]
    supervised = set(score_positions) | set(rationale_positions)
    if any(position < target_start for position in supervised):
        raise AssertionError("prompt token received task supervision")
    if any(position >= target_start + len(target_ids) for position in supervised):
        raise AssertionError("assistant suffix token received task supervision")
    return {
        "sequence_token_count": len(full_ids),
        "full_token_ids_sha256": token_ids_sha256(full_ids),
        "assistant_suffix_token_ids": suffix_ids,
        "assistant_suffix_token_ids_sha256": token_ids_sha256(suffix_ids),
        "score_token_positions": score_positions,
        "rationale_token_positions": rationale_positions,
        "score_mask_sha256": token_ids_sha256(score_positions),
        "rationale_mask_sha256": token_ids_sha256(rationale_positions),
    }


def load_locked_tokenizer(
    tokenizer_path: Path,
    tokenizer_report: dict[str, Any],
) -> Any:
    """Load and verify the exact tokenizer shared by builder and auditor."""
    from transformers import AutoTokenizer

    tokenizer_lock = tokenizer_report["tokenizer_lock"]
    if tokenizer_lock.get("status") != "QWEN_TOKENIZER_REVISION_LOCKED":
        raise ValueError("tokenizer report is not formally locked")
    expected_files = {
        str(item["path"]): str(item["sha256"])
        for item in tokenizer_lock["tokenizer_files"]
    }
    for filename, expected_hash in expected_files.items():
        path = tokenizer_path / filename
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_bytes(path.read_bytes()) != expected_hash:
            raise ValueError(f"tokenizer file differs from lock: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.__class__.__name__ != tokenizer_lock["tokenizer_class"]:
        raise ValueError("tokenizer class differs from lock")
    if len(tokenizer) != int(tokenizer_lock["vocab_size"]):
        raise ValueError("tokenizer vocabulary size differs from lock")
    return tokenizer


def tokenizer_boundary_probes(tokenizer: Any) -> dict[str, Any]:
    """Exercise difficult JSON-string characters without publishing their text."""
    probes = {
        "quote": 'contains "quoted" text.',
        "backslash": r"path C:\tmp\answer.",
        "newline": "first line\nsecond line.",
        "emoji": "emoji 😀 rationale.",
        "chinese_punctuation": "理由包含中文标点：“正确”。",
        "mixed_unicode": "结论 correct ✅; path=C:\\资料。",
    }
    output: dict[str, Any] = {}
    for name, rationale in probes.items():
        target = tokenize_target(
            tokenizer,
            score=3,
            rationale=rationale,
            rationale_active=True,
        )
        output[name] = {
            "target_token_count": len(target["target_token_ids"]),
            "target_token_ids_sha256": target["target_token_ids_sha256"],
            "score_supervised_tokens": len(
                target["score_token_positions_in_target"]
            ),
            "rationale_supervised_tokens": len(
                target["rationale_token_positions_in_target"]
            ),
            "boundary_validation_passed": True,
        }
    return output
