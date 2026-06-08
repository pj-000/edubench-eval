"""Compute train-split class weights for Exp5 L1."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import (
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    EXP04_A4_DATASET_DIR,
    EXP05_TABLES_DIR,
    LABELS,
    ensure_exp05_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json


def clip(value: float, w_min: float, w_max: float) -> float:
    return min(w_max, max(w_min, value))


def compute_class_weights(
    train_path: Path = EXP04_A4_DATASET_DIR / "train.jsonl",
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
) -> list[dict[str, Any]]:
    rows = read_jsonl(train_path)
    counts = Counter(int(row["label_5"]) for row in rows)
    invalid = sorted(label for label in counts if label not in LABELS)
    if invalid:
        raise ValueError(f"Unexpected label_5 values in train split: {invalid}")
    missing = [label for label in LABELS if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Cannot compute class weights; missing train labels: {missing}")

    n_total = len(rows)
    out = []
    for label in LABELS:
        count = counts[label]
        raw = n_total / (len(LABELS) * count)
        out.append(
            {
                "label_5": label,
                "train_count": count,
                "raw_weight": raw,
                "clipped_weight": clip(raw, w_min, w_max),
                "w_min": w_min,
                "w_max": w_max,
                "notes": "computed from train split only",
            }
        )
    return out


def write_class_weights(
    train_path: Path = EXP04_A4_DATASET_DIR / "train.jsonl",
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
) -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    rows = compute_class_weights(train_path=train_path, w_min=w_min, w_max=w_max)
    write_csv(
        EXP05_TABLES_DIR / "class_weights.csv",
        rows,
        fieldnames=[
            "label_5",
            "train_count",
            "raw_weight",
            "clipped_weight",
            "w_min",
            "w_max",
            "notes",
        ],
    )
    write_json(
        EXP05_TABLES_DIR / "class_weights.json",
        {
            "source_split": relpath(train_path),
            "w_min": w_min,
            "w_max": w_max,
            "weights": {str(row["label_5"]): row["clipped_weight"] for row in rows},
            "rows": rows,
        },
    )
    return rows


def load_class_weight_vector(path: Path = EXP05_TABLES_DIR / "class_weights.json") -> list[float]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    weights = data["weights"]
    vector = [0.0] * (max(LABELS) + 1)
    for label in LABELS:
        vector[label] = float(weights[str(label)])
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Exp5 L1 class weights.")
    parser.add_argument("--train_path", type=Path, default=EXP04_A4_DATASET_DIR / "train.jsonl")
    parser.add_argument("--w_min", type=float, default=DEFAULT_W_MIN)
    parser.add_argument("--w_max", type=float, default=DEFAULT_W_MAX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_class_weights(train_path=args.train_path, w_min=args.w_min, w_max=args.w_max)
    print(f"Exp5 class weights written: {[(row['label_5'], row['clipped_weight']) for row in rows]}")
    print(f"Output: {relpath(EXP05_TABLES_DIR / 'class_weights.csv')}")


if __name__ == "__main__":
    main()
