"""Shared locked protocol, paths, and deterministic helpers for Exp43."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
DEV_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl")
TEST_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl")
PROCESSED_PATH = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
RUN_ROOT = Path("thesis_exp/runs/exp43_rubimor")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp43_rubimor")
EXPECTED_FOLD_HASH = "e1602c8fc03876bea9948132162e513623f4255d64883854644d03eba34daa24"

VARIANTS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E6N")
SEEDS = (42, 43, 44)
RUBRIC_VARIANTS = frozenset(("E1", "E3", "E4", "E5", "E6", "E6N"))
SOFT_VARIANTS = frozenset(("E2", "E3", "E4", "E5", "E6", "E6N"))
ORDINAL_VARIANTS = frozenset(("E4", "E5", "E6", "E6N"))
METRIC_HEAD_VARIANTS = frozenset(("E5", "E6", "E6N"))
PAIR_VARIANTS = frozenset(("E6", "E6N"))


def ensure_dirs(root: Path = ROOT) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "state",
        "private/data", "private/pairs", "private/predictions", "private/checkpoints",
        "private/resume", "logs_private",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    values = list(rows)
    fields = list(fieldnames or [])
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    write_json(tmp, value)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def canonical_metric(row: dict[str, Any]) -> str:
    return str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric") or "").strip()


def raw_rubric(row: dict[str, Any]) -> str:
    value = row.get("rubric")
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def triple_hash(row: dict[str, Any]) -> str:
    return stable_hash((normalize_text(row.get("question")), normalize_text(row.get("answer")), canonical_metric(row)))


def human_stats(row: dict[str, Any]) -> dict[str, Any]:
    scores = []
    for index in (1, 2, 3):
        raw = row.get(f"human_{index}_5", row.get(f"human_{index}"))
        score = int(float(raw))
        if score not in range(1, 6) or float(raw) != score:
            raise ValueError(f"Invalid human_{index} for {sample_id(row)}: {raw}")
        scores.append(score)
    distribution = [scores.count(label) / 3.0 for label in range(1, 6)]
    expected = sum(label * distribution[label - 1] for label in range(1, 6))
    return {
        "human_scores": scores,
        "human_distribution_5": distribution,
        "expected_human_score": expected,
        "human_entropy": -sum(value * math.log(value) for value in distribution if value > 0),
        "human_score_range": max(scores) - min(scores),
        "gold_label_5": int(math.floor(expected + 0.5)),
    }


def qwk(gold: list[int], pred: list[int]) -> float:
    observed = np.zeros((5, 5), dtype=float)
    for left, right in zip(gold, pred):
        observed[left - 1, right - 1] += 1
    expected = np.outer(observed.sum(1), observed.sum(0)) / max(observed.sum(), 1)
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5), dtype=float)
    denominator = float((weights * expected).sum())
    return float("nan") if denominator == 0 else 1.0 - float((weights * observed).sum()) / denominator


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    table = np.zeros((5, 5), dtype=int)
    for left, right in zip(gold, pred):
        table[left - 1, right - 1] += 1
    concordant = discordant = 0
    for i in range(5):
        for j in range(5):
            n = int(table[i, j])
            concordant += n * int(table[i + 1 :, j + 1 :].sum())
            discordant += n * int(table[i + 1 :, :j].sum())
    tied_gold = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(1))
    tied_pred = sum(int(n) * (int(n) - 1) // 2 for n in table.sum(0))
    tied_both = sum(int(n) * (int(n) - 1) // 2 for n in table.reshape(-1))
    denominator = math.sqrt((concordant + discordant + tied_gold - tied_both) * (concordant + discordant + tied_pred - tied_both))
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = np.asarray([int(row["gold_label_5"]) for row in rows])
    pred = np.asarray([int(row["pred_label_5"]) for row in rows])
    expected = np.asarray([float(row["pred_score_expected"]) for row in rows])
    targets = np.asarray([row["human_distribution_5"] for row in rows], dtype=float)
    probs = np.asarray([[row[f"prob_{label}"] for label in range(1, 6)] for row in rows], dtype=float)
    low, high = gold <= 2, gold >= 4
    result: dict[str, Any] = {
        "n": len(rows), "MAE": float(np.mean(np.abs(pred - gold))),
        "QWK": qwk(gold.tolist(), pred.tolist()), "Exact_Match": float(np.mean(pred == gold)),
        "Kendall_tau": kendall_tau_b(gold.tolist(), pred.tolist()),
        "Signed_Bias": float(np.mean(pred - gold)), "abs_Signed_Bias": float(abs(np.mean(pred - gold))),
        "Bin_Agreement": float(np.mean(np.where(gold <= 2, 0, np.where(gold == 3, 1, 2)) == np.where(pred <= 2, 0, np.where(pred == 3, 1, 2)))),
        "expected_score_MAE": float(np.mean(np.abs(expected - np.sum(targets * np.arange(1, 6), axis=1)))),
        "human_CE": float(np.mean(-np.sum(targets * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))),
        "human_Brier": float(np.mean(np.sum((probs - targets) ** 2, axis=1))),
        "human_RPS": float(np.mean(np.sum((np.cumsum(probs, 1)[:, :-1] - np.cumsum(targets, 1)[:, :-1]) ** 2, axis=1) / 4.0)),
        "low_n": int(low.sum()), "low_to_high_count": int(np.sum(low & (pred >= 4))),
        "low_to_high_rate": float(np.mean(pred[low] >= 4)) if low.any() else float("nan"),
        "high_n": int(high.sum()), "high_to_low_count": int(np.sum(high & (pred <= 2))),
        "high_to_low_rate": float(np.mean(pred[high] <= 2)) if high.any() else float("nan"),
    }
    counts = Counter(pred.tolist())
    for label in range(1, 6):
        mask = gold == label
        result[f"label{label}_recall"] = float(np.mean(pred[mask] == label)) if mask.any() else float("nan")
        result[f"pred_count_{label}"] = counts[label]
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    return float(np.nanmean(data)), float(np.nanstd(data, ddof=1)) if len(data) > 1 else 0.0
