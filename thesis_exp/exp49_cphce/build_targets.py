"""Build the two Exp49 targets while reusing the exact Exp02 prompt builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp49_cphce import EXPECTED_ROWS, split_path
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import make_prompt


LABELS = (1, 2, 3, 4, 5)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def human_scores(row: dict[str, Any]) -> tuple[int, int, int]:
    values: list[int] = []
    for index in (1, 2, 3):
        raw = row.get(f"human_{index}_5", row.get(f"human_{index}"))
        if raw is None:
            raise ValueError(f"Missing human_{index} for {row.get('record_id')}")
        value = int(round(float(raw)))
        if value not in LABELS:
            raise ValueError(f"human_{index} outside 1-5: {value}")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def human_distribution(row: dict[str, Any]) -> list[float]:
    scores = human_scores(row)
    target = [scores.count(label) / 3.0 for label in LABELS]
    validate_soft_target(target, int(row["label_5"]))
    return target


def validate_soft_target(target: Iterable[float], label_5: int) -> None:
    values = list(target)
    allowed = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    if len(values) != 5:
        raise ValueError(f"Expected five target values, got {len(values)}")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"Soft target does not sum to one: {values}")
    if any(not any(abs(value - candidate) <= 1e-9 for candidate in allowed) for value in values):
        raise ValueError(f"Unexpected soft-target mass: {values}")
    if max(range(5), key=values.__getitem__) + 1 != int(label_5):
        raise ValueError(f"Soft target mode differs from label_5={label_5}: {values}")


def convert_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    label_5 = int(row["label_5"])
    scores = human_scores(row)
    computed_mean = sum(scores) / 3.0
    stored_mean = float(row["human_mean_5"])
    if abs(computed_mean - stored_mean) > 1e-9:
        raise ValueError(f"human_mean_5 mismatch for {row.get('record_id')}")
    target = human_distribution(row)
    return {
        "id": row.get("record_id"),
        "record_id": row.get("record_id"),
        "split": split,
        "text": make_prompt(row),
        "label": label_5 - 1,
        "label_5": label_5,
        "human_mean_5": stored_mean,
        "human_1_5": scores[0],
        "human_2_5": scores[1],
        "human_3_5": scores[2],
        "soft_target_5": target,
        "triple_key": row.get("triple_key"),
        "question_key": row.get("question_key"),
        "answer_key": row.get("answer_key"),
        "metric_canonical": row.get("metric_canonical"),
        "scenario_canonical": row.get("scenario_canonical"),
        "subject_canonical": row.get("subject_canonical"),
        "language": row.get("language"),
        "generator_model": row.get("generator_model"),
    }


def load_split(split: str, *, allow_test: bool = False) -> list[dict[str, Any]]:
    if split == "test" and not allow_test:
        raise PermissionError("Exp49 train/dev path must not read the test split")
    source = read_jsonl(split_path(split))
    expected = EXPECTED_ROWS[split]
    if len(source) != expected:
        raise ValueError(f"{split} row count {len(source)} != {expected}")
    return [convert_row(row, split) for row in source]


def aggregate_text_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("record_id") or row.get("id")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["text"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def row_text_hashes(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("record_id") or row.get("id")): hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
        for row in rows
    }
