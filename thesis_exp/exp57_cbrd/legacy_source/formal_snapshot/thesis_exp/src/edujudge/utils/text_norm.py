"""Text normalization helpers used for IDs and leakage checks."""

from __future__ import annotations

import json
import re
import unicodedata


_WS_RE = re.compile(r"\s+")


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", stringify(value))
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def detect_language_from_text(value: object) -> str:
    text = stringify(value)
    if not text:
        return "unknown"
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    alpha = sum(1 for ch in text if ch.isalpha())
    if cjk >= 3 and cjk / max(1, len(text)) > 0.03:
        return "zh"
    if alpha:
        return "en"
    return "unknown"


def truncate_text(value: object, max_len: int = 500) -> str:
    text = stringify(value)
    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
