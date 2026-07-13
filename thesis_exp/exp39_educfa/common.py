"""Shared train-only helpers for Exp39A EduCFA."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
ALL_PATH = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
PROMPT_DIR = Path("thesis_exp/exp39_educfa/prompts")
SCHEMA_DIR = Path("thesis_exp/exp39_educfa/schemas")

OPERATORS = (
    "remove_required_key_point",
    "introduce_factual_or_rubric_contradiction",
    "remove_reasoning_or_supporting_evidence",
    "violate_explicit_task_or_format_constraint",
    "introduce_scope_drift_or_irrelevant_content",
)
TARGET_COUNTS = {1: 40, 2: 120, 3: 80}
VARIANTS = (
    "v0h_human_soft",
    "v1_matched_real_low_oversampling",
    "v2_unverified_counterfactual",
    "v3_generic_corruption",
    "v4_educfa",
    "v5_shuffled_counterfactual",
)


def reject_eval_path(path: Path) -> None:
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    if path.name.lower() in {"dev.jsonl", "test.jsonl", "dev.json", "test.json"}:
        raise ValueError(f"Exp39A forbids evaluation path: {path}")
    if "/paper_like_triple_seed42/dev." in normalized or "/paper_like_triple_seed42/test." in normalized:
        raise ValueError(f"Exp39A forbids paper-like dev/test: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    reject_eval_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id") or row.get("record_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def score_value(row: dict[str, Any], key: str) -> int:
    value = row.get(f"{key}_5", row.get(key))
    return int(float(value))


def human_distribution(row: dict[str, Any]) -> list[float]:
    scores = [score_value(row, key) for key in ("human_1", "human_2", "human_3")]
    return [scores.count(label) / 3.0 for label in range(1, 6)]


def half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def normalize_distribution(values: Iterable[float]) -> list[float]:
    output = [float(value) for value in values]
    total = sum(output)
    if len(output) != 5 or min(output) < 0 or total <= 0:
        raise ValueError(f"Invalid distribution: {output}")
    return [value / total for value in output]


def interval_distribution(minimum: int, maximum: int, target: int) -> list[float]:
    if not 1 <= minimum <= target <= maximum <= 5:
        raise ValueError(f"Invalid interval/target: {minimum}, {target}, {maximum}")
    return normalize_distribution(
        math.exp(-abs(score - target)) if minimum <= score <= maximum else 0.0
        for score in range(1, 6)
    )


def text_edit_ratio(source: str, candidate: str) -> float:
    return 1.0 - SequenceMatcher(None, source, candidate, autojunk=False).ratio()


def length_ratio(source: str, candidate: str) -> float:
    return len(candidate) / max(len(source), 1)


def qwk(gold: Iterable[int], pred: Iterable[int]) -> float:
    left, right = list(gold), list(pred)
    if not left:
        return float("nan")
    observed = np.zeros((5, 5), dtype=float)
    for a, b in zip(left, right):
        observed[int(a) - 1, int(b) - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5), dtype=float)
    denominator = float((weights * expected).sum())
    return float("nan") if denominator == 0 else 1.0 - float((weights * observed).sum()) / denominator


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    if len(gold) < 2:
        return float("nan")
    table = np.zeros((5, 5), dtype=int)
    for a, b in zip(gold, pred):
        table[int(a) - 1, int(b) - 1] += 1
    concordant = discordant = 0
    for i in range(5):
        for j in range(5):
            count = int(table[i, j])
            concordant += count * int(table[i + 1 :, j + 1 :].sum())
            discordant += count * int(table[i + 1 :, :j].sum())
    tied_gold = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=1))
    tied_pred = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(axis=0))
    tied_both = sum(int(n) * (int(n) - 1) // 2 for n in table.reshape(-1))
    denominator = math.sqrt(
        (concordant + discordant + tied_gold - tied_both)
        * (concordant + discordant + tied_pred - tied_both)
    )
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def score_bin(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 2, 0, np.where(values == 3, 1, 2))


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    gold = np.asarray([int(row["gold_label_5"]) for row in rows])
    pred = np.asarray([int(row["pred_label_5"]) for row in rows])
    expected = np.asarray([float(row["pred_score_expected"]) for row in rows])
    low, high = gold <= 2, gold >= 4
    result: dict[str, Any] = {
        "n": len(rows),
        "MAE": float(np.mean(np.abs(pred - gold))),
        "expected_score_MAE": float(np.mean(np.abs(expected - gold))),
        "Signed_Bias": float(np.mean(pred - gold)),
        "Exact_Match": float(np.mean(pred == gold)),
        "Kendall_tau": kendall_tau_b(gold.tolist(), pred.tolist()),
        "QWK": qwk(gold.tolist(), pred.tolist()),
        "Bin_Agreement": float(np.mean(score_bin(gold) == score_bin(pred))),
        "low_n": int(low.sum()),
        "low_to_high_count": int(np.sum(low & (pred >= 4))),
        "low_to_high_rate": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "high_n": int(high.sum()),
        "high_to_low_count": int(np.sum(high & (pred <= 2))),
        "high_to_low_rate": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
    }
    counts = Counter(pred.tolist())
    for label in range(1, 6):
        mask = gold == label
        result[f"label{label}_recall"] = float(np.mean(pred[mask] == label)) if mask.any() else float("nan")
        result[f"pred_count_{label}"] = counts[label]
    return result


def human_distribution_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"human_CE": float("nan"), "human_Brier": float("nan"), "human_RPS": float("nan")}
    targets = np.asarray([row["human_distribution_5"] for row in rows], dtype=float)
    probs = np.asarray([[row[f"prob_{k}"] for k in range(1, 6)] for row in rows], dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    return {
        "human_CE": float(np.mean(-np.sum(targets * np.log(probs), axis=1))),
        "human_Brier": float(np.mean(np.sum((probs - targets) ** 2, axis=1))),
        "human_RPS": float(np.mean(np.sum((np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(targets, axis=1)[:, :-1]) ** 2, axis=1) / 4.0)),
    }
