"""Dataset, class-weight, and safety helpers for Exp9."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    EXPECTED_SPLIT_ROWS,
    EXP09_DATASET_DIR,
    EXP09_TABLES_DIR,
    LABELS,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json


CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / f"{split}.jsonl")


def load_splits(data_dir: Path = EXP09_DATASET_DIR) -> dict[str, list[dict[str, Any]]]:
    return {split: read_split(data_dir, split) for split in ["train", "dev", "test"]}


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit else rows


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("record_id") or row.get("id"))


def split_record_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {record_key(row) for row in rows}


def label_counts(rows: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter(int(row["label_5"]) for row in rows)
    invalid = sorted(label for label in counts if label not in LABELS)
    if invalid:
        raise ValueError(f"Unexpected label_5 values: {invalid}")
    return counts


def compute_pointwise_class_weights(
    train_rows: list[dict[str, Any]],
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
) -> list[dict[str, Any]]:
    counts = label_counts(train_rows)
    missing = [label for label in LABELS if counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Cannot compute class weights; missing labels: {missing}")
    n_total = len(train_rows)
    rows = []
    for label in LABELS:
        raw = n_total / (len(LABELS) * counts[label])
        clipped = min(w_max, max(w_min, raw))
        rows.append(
            {
                "label_5": label,
                "train_count": counts[label],
                "raw_weight": raw,
                "clipped_weight": clipped,
                "w_min": w_min,
                "w_max": w_max,
                "notes": "QD-B1-style clipped inverse-frequency weight from QD-S0 train only",
            }
        )
    return rows


def class_weight_vector(weight_rows: list[dict[str, Any]]) -> list[float]:
    vector = [0.0] * 6
    for row in weight_rows:
        vector[int(row["label_5"])] = float(row["clipped_weight"])
    return vector


def write_pointwise_class_weights(
    train_rows: list[dict[str, Any]],
    output_dir: Path = EXP09_TABLES_DIR,
    w_min: float = DEFAULT_W_MIN,
    w_max: float = DEFAULT_W_MAX,
) -> list[dict[str, Any]]:
    rows = compute_pointwise_class_weights(train_rows, w_min=w_min, w_max=w_max)
    write_csv(output_dir / "pointwise_class_weights.csv", rows)
    write_json(
        output_dir / "pointwise_class_weights.json",
        {
            "formula": "clip(N / (5 * N_c), w_min, w_max)",
            "source": "QD-S0_human_only train split only",
            "weights": {str(row["label_5"]): row["clipped_weight"] for row in rows},
            "rows": rows,
        },
    )
    return rows


def tracked_weight_files() -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS)


def exp0_to_exp8_tracked_output_changes() -> list[str]:
    paths = [
        "thesis_exp/outputs/exp00_data",
        "thesis_exp/outputs/exp01_audit",
        "thesis_exp/outputs/exp02_ce_baseline",
        "thesis_exp/outputs/exp03_input_ablation",
        "thesis_exp/outputs/exp04_objectives",
        "thesis_exp/outputs/exp05_low_score_loss",
        "thesis_exp/outputs/exp06_question_disjoint_baselines",
        "thesis_exp/outputs/exp06_synthetic_low_score",
        "thesis_exp/outputs/exp07_rank_consistent_ordinal",
        "thesis_exp/outputs/exp07_calibration",
        "thesis_exp/outputs/exp08_edurisk_loss",
    ]
    try:
        result = subprocess.run(["git", "diff", "--name-only", "--", *paths], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if path.strip())


def dataset_sanity_rows(data_dir: Path = EXP09_DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "") -> None:
        rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})

    for split, expected in EXPECTED_SPLIT_ROWS.items():
        path = data_dir / f"{split}.jsonl"
        split_rows = read_split(data_dir, split) if path.exists() else []
        labels = [int(row.get("label_5", -1)) for row in split_rows if row.get("label_5") is not None]
        add(f"{split} jsonl exists", path.exists(), relpath(path))
        add(f"{split} row count = {expected}", len(split_rows) == expected, len(split_rows))
        add(f"{split} rows are human only", all(row.get("source_type") == "human" for row in split_rows))
        add(f"{split} labels are 1..5", len(labels) == len(split_rows) and all(1 <= value <= 5 for value in labels))
        add(f"{split} A4 text exists", all(bool(str(row.get("text") or "").strip()) for row in split_rows))
        add(f"{split} has no synthetic rows", not any(row.get("source_type") == "synthetic" for row in split_rows))
    weights = tracked_weight_files()
    add("no checkpoint/weights tracked", not weights, ", ".join(weights))
    changed = exp0_to_exp8_tracked_output_changes()
    add("no tracked Exp0-Exp8 output modifications", not changed, ", ".join(changed))
    return rows
