"""Locked paths and deterministic helpers for Exp44A."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp43_rubimor.common import (
    prediction_metrics,
    read_csv,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

ROOT = Path("thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42")
RUN_ROOT = Path("thesis_exp/runs/exp44_taco_score")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp44_taco_score")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
EXP43_ROOT = Path("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered")
EXP43_FOLD_PATH = EXP43_ROOT / "private/data/exp43_groupcv_fold_assignment.csv"
EXP43_EXPECTED_FOLD_HASH = "e1602c8fc03876bea9948132162e513623f4255d64883854644d03eba34daa24"
EXP43_EXPECTED_E4_HASH = "30e12d1d94762ac2a0ff7484ab50ab22a302f056773bd890156d61fdf395f1e8"
VARIANTS = (
    "C0_E4_baseline",
    "C1_balanced_plain_contrastive",
    "C2_TACO",
    "C3_shuffled_margin_control",
)
CONTRASTIVE_VARIANTS = frozenset(VARIANTS[1:])
FOLDS = tuple(range(5))
SEED = 42

FORBIDDEN_PATHS = (
    Path("thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"),
    Path("thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl"),
)


def ensure_dirs(root: Path = ROOT) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "state",
        "private/data", "private/triplets", "private/predictions",
        "private/checkpoints", "logs_private",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def assert_train_only(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_PATHS:
        if resolved == forbidden.resolve():
            raise RuntimeError(f"Exp44A forbids dev/test access: {path}")


def fold_map(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    result = {str(row["sample_id"]): int(row["fold"]) for row in rows}
    if len(result) != 2654 or set(result.values()) != set(FOLDS):
        raise RuntimeError(f"Invalid Exp44A fold assignment: rows={len(result)}")
    return result


def distribution_cdf(distribution: list[float]) -> list[float]:
    if len(distribution) != 5:
        raise ValueError("Expected a five-class distribution")
    total = 0.0
    result = []
    for value in distribution[:4]:
        total += float(value)
        result.append(total)
    return result


def ordinal_distribution_distance(left: list[float], right: list[float]) -> float:
    return float(sum(abs(a - b) for a, b in zip(distribution_cdf(left), distribution_cdf(right))))


def stable_seed(*parts: Any) -> int:
    return int(stable_hash(parts)[:16], 16) % (2**63 - 1)


def stratum(label: int) -> str:
    if label <= 2:
        return "low"
    if label == 3:
        return "mid"
    return "high"


def triplet_path(root: Path, fold: int, epoch: int) -> Path:
    return root / f"private/triplets/fold_{fold}/epoch_{epoch}_triplets.jsonl"


def run_dir(run_root: Path, variant: str, fold: int, mode: str = "groupcv") -> Path:
    return run_root / mode / variant / "seed_42" / f"fold_{fold}"


def group_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["question_key"])].append(row)
    return groups


__all__ = [
    "ARTIFACT_ROOT", "CONTRASTIVE_VARIANTS", "EXP43_EXPECTED_E4_HASH",
    "EXP43_EXPECTED_FOLD_HASH", "EXP43_FOLD_PATH", "EXP43_ROOT", "FOLDS",
    "ROOT", "RUN_ROOT", "SEED", "TRAIN_PATH", "VARIANTS", "assert_train_only",
    "atomic_json", "ensure_dirs", "fold_map", "group_by_question",
    "ordinal_distribution_distance", "prediction_metrics", "read_csv",
    "read_jsonl", "run_dir", "sha256_file", "stable_hash", "stable_seed",
    "stratum", "triplet_path", "write_csv", "write_json", "write_jsonl",
]

