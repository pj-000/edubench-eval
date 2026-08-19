"""Shared train-only contracts for the Exp54 actual failure bank."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT


TRAIN_PATH = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
)
EXPECTED_TRAIN_ROWS = 2654
LABELS = (1, 2, 3, 4, 5)
SEEDS = (42, 43, 44)
ERROR_CLASSES = (
    "correct",
    "adjacent_overestimate",
    "adjacent_underestimate",
    "severe_low_to_high",
    "severe_high_to_low",
    "other_overestimate",
    "other_underestimate",
    "invalid_output",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def load_train_rows(path: Path = TRAIN_PATH) -> list[dict[str, Any]]:
    normalized = "/" + path.resolve().as_posix().lower().strip("/") + "/"
    if "/dev." in normalized or "/test." in normalized:
        raise ValueError(f"failure bank forbids evaluation split: {path}")
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Exp54 train row count differs")
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if any(not value for value in record_ids):
        raise ValueError("Exp54 train contains an empty record ID")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Exp54 train contains duplicate record IDs")
    if any(int(row["label_5"]) not in LABELS for row in rows):
        raise ValueError("Exp54 train label is outside 1-5")
    return rows


def classify_error(
    gold: int,
    predicted: int | None,
    *,
    parse_success: bool,
) -> dict[str, Any]:
    if gold not in LABELS:
        raise ValueError("gold score must be in 1-5")
    if not parse_success:
        if predicted is not None:
            raise ValueError("failed parse cannot carry a generated score")
        return {
            "signed_error": 0,
            "absolute_error": 0,
            "error_class": "invalid_output",
            "severe_low_to_high": False,
            "severe_high_to_low": False,
        }
    if predicted not in LABELS:
        raise ValueError("parsed generated score must be in 1-5")
    signed = int(predicted - gold)
    absolute = abs(signed)
    severe_l2h = gold <= 2 and predicted >= 4
    severe_h2l = gold >= 4 and predicted <= 2
    if signed == 0:
        error_class = "correct"
    elif severe_l2h:
        error_class = "severe_low_to_high"
    elif severe_h2l:
        error_class = "severe_high_to_low"
    elif signed == 1:
        error_class = "adjacent_overestimate"
    elif signed == -1:
        error_class = "adjacent_underestimate"
    elif signed > 0:
        error_class = "other_overestimate"
    else:
        error_class = "other_underestimate"
    return {
        "signed_error": signed,
        "absolute_error": absolute,
        "error_class": error_class,
        "severe_low_to_high": severe_l2h,
        "severe_high_to_low": severe_h2l,
    }


def score_leakage(rationale: str, score: int) -> bool:
    compact = re.sub(r"\s+", "", rationale.lower())
    return (
        f'"score":{score}' in compact
        or f"score{score}" in compact
        or f"评分{score}" in compact
        or f"{score}分" in compact
    )


def any_explicit_score_leakage(rationale: str) -> bool:
    """Detect an explicit 1-5 score mention, independent of target score."""
    return any(score_leakage(rationale, score) for score in LABELS)


def make_failure_row(
    *,
    source: dict[str, Any],
    row_position: int,
    generator_seed: int,
    adapter_sha256: str,
    generation_mode: str,
    rollout_seed: int | None,
    prediction: dict[str, Any] | None,
    forced_completion: bool,
) -> dict[str, Any]:
    if generator_seed not in SEEDS:
        raise ValueError("generator seed differs")
    if generation_mode not in {"greedy", "stochastic"}:
        raise ValueError("generation mode differs")
    parse_success = prediction is not None
    generated_score = (
        int(prediction["score"]) if prediction is not None else None
    )
    generated_rationale = (
        str(prediction["rationale"]) if prediction is not None else None
    )
    diagnostics = classify_error(
        int(source["label_5"]),
        generated_score,
        parse_success=parse_success,
    )
    return {
        "record_id": str(source["record_id"]),
        "row_position": row_position,
        "gold_label": int(source["label_5"]),
        "metric_id": str(source["metric_id"]),
        "language": str(source["language"]),
        "generator_arm": "R3",
        "generator_seed": generator_seed,
        "generator_epoch": 3,
        "generator_adapter_sha256": adapter_sha256,
        "generation_mode": generation_mode,
        "rollout_seed": rollout_seed,
        "parse_success": parse_success,
        "generated_score": generated_score,
        "generated_rationale": generated_rationale,
        "forced_completion": forced_completion,
        **diagnostics,
    }


def aggregate_failure_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    expected = EXPECTED_TRAIN_ROWS * len(SEEDS)
    if len(materialized) != expected:
        raise ValueError(
            f"failure bank requires {expected} greedy rows, got "
            f"{len(materialized)}"
        )
    unique = {
        (str(row["record_id"]), int(row["generator_seed"]))
        for row in materialized
    }
    if len(unique) != expected:
        raise ValueError("failure bank record×seed keys are duplicated")
    class_counts = Counter(str(row["error_class"]) for row in materialized)
    seed_class = defaultdict(Counter)
    label_class = defaultdict(Counter)
    metric_class = defaultdict(Counter)
    language_class = defaultdict(Counter)
    for row in materialized:
        seed_class[int(row["generator_seed"])][str(row["error_class"])] += 1
        label_class[int(row["gold_label"])][str(row["error_class"])] += 1
        metric_class[str(row["metric_id"])][str(row["error_class"])] += 1
        language_class[str(row["language"])][str(row["error_class"])] += 1

    def complete(counter: Counter[str]) -> dict[str, int]:
        return {name: int(counter[name]) for name in ERROR_CLASSES}

    actual_l2h_records = {
        str(row["record_id"])
        for row in materialized
        if bool(row["severe_low_to_high"])
    }
    actual_h2l_records = {
        str(row["record_id"])
        for row in materialized
        if bool(row["severe_high_to_low"])
    }
    correct_records = {
        str(row["record_id"])
        for row in materialized
        if row["error_class"] == "correct"
    }
    return {
        "rows": len(materialized),
        "unique_record_seed_keys": len(unique),
        "error_class_counts": complete(class_counts),
        "by_seed": {
            str(seed): complete(seed_class[seed]) for seed in SEEDS
        },
        "by_gold_label": {
            str(label): complete(label_class[label]) for label in LABELS
        },
        "by_metric": {
            metric: complete(metric_class[metric])
            for metric in sorted(metric_class)
        },
        "by_language": {
            language: complete(language_class[language])
            for language in sorted(language_class)
        },
        "records_with_any_correct_generation": len(correct_records),
        "records_with_actual_severe_l2h": len(actual_l2h_records),
        "records_with_actual_severe_h2l": len(actual_h2l_records),
        "forced_completion_count": sum(
            bool(row["forced_completion"]) for row in materialized
        ),
        "empty_rationale_count": sum(
            row["parse_success"]
            and not str(row["generated_rationale"]).strip()
            for row in materialized
        ),
        "explicit_score_leakage_count": sum(
            row["parse_success"]
            and score_leakage(
                str(row["generated_rationale"]),
                int(row["generated_score"]),
            )
            for row in materialized
        ),
    }
