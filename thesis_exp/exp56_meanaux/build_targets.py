"""Reuse locked rows and expose the continuous three-rater mean target."""

from __future__ import annotations

from typing import Any

from thesis_exp.exp49_cphce.build_targets import load_split as load_exp49_split


def validate_human_mean(value: float) -> float:
    value = float(value)
    if not 1.0 <= value <= 5.0:
        raise ValueError(f"human_mean_5 outside 1-5: {value}")
    scaled = (value - 1.0) * 3.0
    if abs(scaled - round(scaled)) > 1e-8:
        raise ValueError(f"human_mean_5 is not on the three-rater 1/3 grid: {value}")
    return value


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "hard_target_5": [
            float(index == int(row["label_5"]) - 1) for index in range(5)
        ],
        "mean_aux_target": validate_human_mean(float(row["human_mean_5"])),
    }


def load_split(split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise PermissionError("Exp56 MeanAux is a train/dev-only post-hoc control")
    if split not in {"train", "dev"}:
        raise ValueError(f"Unsupported split: {split}")
    return [convert_row(row) for row in load_exp49_split(split)]
