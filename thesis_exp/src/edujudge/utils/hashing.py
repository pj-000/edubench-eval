"""Stable hash helpers."""

from __future__ import annotations

import hashlib


def sha1_text(*parts: object) -> str:
    """Return a stable SHA1 hex digest for normalized string parts."""
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part if part is not None else "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()

