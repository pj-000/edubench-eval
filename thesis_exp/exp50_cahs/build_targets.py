"""Build the fixed consensus-anchored human target for Exp50."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp49_cphce.build_targets import load_split as load_exp49_split
from thesis_exp.exp50_cahs import ALPHA


def cahs_target(label_5: int, human_soft: list[float], alpha: float = ALPHA) -> list[float]:
    if alpha != ALPHA:
        raise ValueError(f"Exp50 locks alpha={ALPHA}; received {alpha}")
    label_index = int(label_5) - 1
    if label_index not in range(5):
        raise ValueError(f"label_5 outside 1-5: {label_5}")
    if len(human_soft) != 5 or abs(sum(human_soft) - 1.0) > 1e-9:
        raise ValueError(f"invalid human distribution: {human_soft}")
    hard = [float(index == label_index) for index in range(5)]
    target = [(1.0 - alpha) * hard[index] + alpha * float(human_soft[index]) for index in range(5)]
    validate_cahs_target(target, label_5)
    return target


def validate_cahs_target(target: list[float], label_5: int) -> None:
    if len(target) != 5 or abs(sum(target) - 1.0) > 1e-9:
        raise ValueError(f"invalid CAHS target: {target}")
    allowed = (0.0, 1.0 / 6.0, 5.0 / 6.0, 1.0)
    if any(not any(abs(value - candidate) <= 1e-9 for candidate in allowed) for value in target):
        raise ValueError(f"unexpected CAHS mass: {target}")
    if max(range(5), key=target.__getitem__) + 1 != int(label_5):
        raise ValueError(f"CAHS mode differs from label_5={label_5}: {target}")


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "cahs_target_5": cahs_target(int(row["label_5"]), list(row["soft_target_5"]))}


def load_split(split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise PermissionError("Exp50 scout/formal paths must not load test")
    return [convert_row(row) for row in load_exp49_split(split)]
