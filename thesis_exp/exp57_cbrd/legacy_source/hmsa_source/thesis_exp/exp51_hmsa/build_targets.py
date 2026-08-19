"""Reuse the locked Exp49 rows and expose hard plus three-human targets."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp49_cphce.build_targets import load_split as load_exp49_split


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "hard_target_5": [float(index == int(row["label_5"]) - 1) for index in range(5)]}


def load_split(split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise PermissionError("Exp51 scout/formal paths must not load test")
    return [convert_row(row) for row in load_exp49_split(split)]
