"""Frozen-candidate deterministic generation and strict JSON parser contract."""

from __future__ import annotations

import json
from typing import Any


GENERATION_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 256,
    "use_cache": True,
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_review_json(text: str) -> dict[str, Any]:
    """Parse exactly one JSON object with score and visible rationale."""
    if not isinstance(text, str):
        raise TypeError("generated review must be text")
    stripped = text.strip()
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("generated review is not valid JSON") from exc
    if stripped[end:].strip():
        raise ValueError("generated review contains trailing non-whitespace")
    if not isinstance(value, dict):
        raise ValueError("generated review must be one JSON object")
    if set(value) != {"score", "rationale"}:
        raise ValueError("generated review must contain exactly score and rationale")
    score = value["score"]
    if not isinstance(score, int) or isinstance(score, bool) or score not in range(1, 6):
        raise ValueError("generated score must be an integer from 1 to 5")
    if not isinstance(value["rationale"], str):
        raise ValueError("generated rationale must be a string")
    return {"score": score, "rationale": value["rationale"]}
