"""Shared train-only helpers for Exp38A HAILS-Score."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path("thesis_exp/exp38_hails_score/outputs/exp38a_hails_score_seed42")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
EXP37_ROOT = Path(
    "thesis_exp/exp37_failure_evidence_qualification/outputs/"
    "exp37a_r1_model_reviewed_qualification_seed42"
)
PACKET_PATH = EXP37_ROOT / "private_packets/exp37a_r1_reviewer_a_packets.jsonl"
EXP36_MANIFEST = Path(
    "thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/tables/"
    "exp36a_resolved_teacher_input_manifest.csv"
)
R0_ROOT = Path(
    "thesis_exp/exp37_failure_evidence_qualification/outputs/"
    "exp37a_failure_evidence_qualification_seed42"
)
EXPECTED_VIEWS = {"low_tail_all": 76, "boundary_view": 60, "high_control_view": 60}
VARIANTS = (
    "v0_original_hard",
    "v0h_human_empirical",
    "v1_qwen_hard",
    "v2_qwen_interval_only",
    "v3_naive_interval",
    "v4_hails",
    "v5_shuffled_interval",
)


def reject_eval_path(path: Path) -> None:
    name = path.name.lower()
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    if name in {"dev.jsonl", "test.jsonl", "dev.json", "test.json"} or "/dev/" in normalized or "/test/" in normalized:
        raise ValueError(f"Exp38A is train-only and forbids evaluation path: {path}")


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


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def one_hot(score: int) -> list[float]:
    return [1.0 if index == int(score) else 0.0 for index in range(1, 6)]


def normalize_distribution(values: Iterable[float]) -> list[float]:
    output = [float(value) for value in values]
    total = sum(output)
    if len(output) != 5 or min(output) < 0 or total <= 0:
        raise ValueError(f"Invalid distribution: {output}")
    return [value / total for value in output]


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
    denominator = math.sqrt((concordant + discordant + tied_gold - tied_both) * (concordant + discordant + tied_pred - tied_both))
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


def resolve_final_reference(exp37_root: Path = EXP37_ROOT, explicit: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    if explicit is not None:
        candidates = [explicit]
    else:
        packet_ids = {sample_id(row) for row in read_jsonl(exp37_root / "private_packets/exp37a_r1_reviewer_a_packets.jsonl")}
        candidates = []
        for path in (exp37_root / "private_reference").glob("*.jsonl"):
            rows = read_jsonl(path)
            ids = {sample_id(row) for row in rows}
            if len(rows) == 196 and ids == packet_ids and all(row.get("reference_type") for row in rows):
                candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve the 196-row final reference ({len(candidates)} candidates). "
            "Pass --final-reference explicitly."
        )
    path = candidates[0]
    rows = read_jsonl(path)
    if len(rows) != 196 or len({sample_id(row) for row in rows}) != 196:
        raise ValueError("Final reference must contain 196 unique sample IDs")
    return path, rows


def frozen_view_map() -> dict[str, str]:
    output: dict[str, str] = {}
    for view, count in EXPECTED_VIEWS.items():
        path = R0_ROOT / "annotation_templates" / f"exp37a_{view}_reviewer_a_template.jsonl"
        rows = read_jsonl(path)
        if len(rows) != count:
            raise ValueError(f"Frozen view {view} expected {count}, found {len(rows)}")
        for row in rows:
            sid = sample_id(row)
            if sid in output:
                raise ValueError(f"Duplicate frozen ID across views: {sid}")
            output[sid] = view
    if len(output) != 196:
        raise ValueError("Frozen qualification view union must have 196 IDs")
    return output


def qwen_interval_distribution(minimum: int, center: int, maximum: int) -> list[float]:
    values = [math.exp(-abs(score - center)) if minimum <= score <= maximum else 0.0 for score in range(1, 6)]
    return normalize_distribution(values)


def input_hash(row: dict[str, Any]) -> str:
    return stable_hash({
        "question": row.get("question"),
        "answer": row.get("answer"),
        "metric_canonical": row.get("metric_canonical"),
    })
