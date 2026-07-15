"""Locked paths and deterministic helpers for Exp45A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.exp43_rubimor.common import (
    atomic_json,
    prediction_metrics,
    read_csv,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

ROOT = Path("thesis_exp/exp45_dopr_head/outputs/exp45a_dopr_seed42")
RUN_ROOT = Path("thesis_exp/runs/exp45_dopr_head")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp45_dopr_head")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
EXP43_ROOT = Path("thesis_exp/exp43_rubimor/outputs/exp43_rubimor_preregistered")
EXP44_ROOT = Path("thesis_exp/exp44_taco_score/outputs/exp44a_taco_seed42")
EXP44_RUN_ROOT = Path("thesis_exp/runs/exp44_taco_score")
EXPECTED_FOLD_HASH = "e1602c8fc03876bea9948132162e513623f4255d64883854644d03eba34daa24"
EXPECTED_E4_HASH = "30e12d1d94762ac2a0ff7484ab50ab22a302f056773bd890156d61fdf395f1e8"
SEED = 42
FOLDS = tuple(range(5))
HEAD_VARIANTS = (
    "H0_E4_natural_head",
    "H1_vanilla_cRT",
    "H2_distributional_ordinal_cRT",
    "H3_prototype_cRT_no_prior",
    "H4_DOPR",
)
TRAINED_HEAD_VARIANTS = HEAD_VARIANTS[1:]
FORBIDDEN_PATHS = (
    Path("thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"),
    Path("thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl"),
)


def ensure_dirs(root: Path = ROOT) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "state",
        "private/data", "private/encoders", "private/embeddings",
        "private/prototypes", "private/heads", "private/predictions",
        "logs_private",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def assert_train_only(path: Path) -> None:
    resolved = path.resolve()
    if any(resolved == forbidden.resolve() for forbidden in FORBIDDEN_PATHS):
        raise RuntimeError(f"Exp45A forbids dev/test access: {path}")


def fold_map(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    result = {str(row["sample_id"]): int(row["fold"]) for row in rows}
    if len(result) != 2654 or set(result.values()) != set(FOLDS):
        raise RuntimeError(f"Invalid Exp45A fold assignment: rows={len(result)}")
    return result


def encoder_dir(root: Path, fold: int, mode: str = "groupcv") -> Path:
    return root / f"private/encoders/{mode}/fold_{fold}"


def embedding_path(root: Path, fold: int, split: str) -> Path:
    if split not in {"outer_train", "outer_heldout"}:
        raise ValueError(split)
    return root / f"private/embeddings/fold_{fold}_{split}.jsonl"


def prototype_path(root: Path, fold: int) -> Path:
    return root / f"private/prototypes/fold_{fold}.json"


def head_prediction_path(root: Path, variant: str, fold: int) -> Path:
    return root / f"private/predictions/{variant}/fold_{fold}.jsonl"


def head_run_dir(run_root: Path, variant: str, fold: int, mode: str = "groupcv") -> Path:
    return run_root / mode / variant / "seed_42" / f"fold_{fold}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ARTIFACT_ROOT", "EXPECTED_E4_HASH", "EXPECTED_FOLD_HASH", "EXP43_ROOT",
    "EXP44_ROOT", "EXP44_RUN_ROOT", "FOLDS", "FORBIDDEN_PATHS", "HEAD_VARIANTS",
    "ROOT", "RUN_ROOT", "SEED", "TRAINED_HEAD_VARIANTS", "TRAIN_PATH",
    "assert_train_only", "atomic_json", "embedding_path", "encoder_dir", "ensure_dirs",
    "fold_map", "head_prediction_path", "head_run_dir", "load_json", "prediction_metrics",
    "prototype_path", "read_csv", "read_jsonl", "sha256_file", "stable_hash",
    "write_csv", "write_json", "write_jsonl",
]
