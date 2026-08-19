"""Test-only data surface, imported solely by the one-shot evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation.data import (
    _load_rows_for_frozen_split,
)


def load_sealed_test_rows(source_repo: Path) -> list[dict[str, Any]]:
    rows = _load_rows_for_frozen_split(source_repo, "test")
    if len(rows) != 1578 or any(row["split"] != "test" for row in rows):
        raise RuntimeError("Exp61 sealed test contract mismatch")
    return rows
