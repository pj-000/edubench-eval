"""Dataset and class-weight helpers for Exp8 EduRisk."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp08_edurisk import (
    DEFAULT_CLASS_BALANCE_BETA,
    EXPECTED_SPLIT_ROWS,
    EXP08_DATASET_DIR,
    EXP08_RUN_ID,
    EXP08_TABLES_DIR,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_R1_RUN_DIR,
    QD_R1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
)
from thesis_exp.src.edujudge.exp08_edurisk.losses import (
    effective_number_weights_from_counts,
    weight_vector_from_rows,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv


CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / f"{split}.jsonl")


def load_splits(data_dir: Path = EXP08_DATASET_DIR) -> dict[str, list[dict[str, Any]]]:
    return {split: read_split(data_dir, split) for split in ["train", "dev", "test"]}


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit else rows


def label_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts = {label: 0 for label in range(1, 6)}
    for row in rows:
        label = int(row["label_5"])
        if label not in counts:
            raise ValueError(f"label_5 must be in 1..5, got {label}")
        counts[label] += 1
    return counts


def class_balanced_weight_rows(
    train_rows: list[dict[str, Any]],
    beta: float = DEFAULT_CLASS_BALANCE_BETA,
) -> list[dict[str, Any]]:
    return effective_number_weights_from_counts(label_counts(train_rows), beta=beta)


def class_weight_vector(
    train_rows: list[dict[str, Any]],
    beta: float = DEFAULT_CLASS_BALANCE_BETA,
) -> list[float]:
    rows = class_balanced_weight_rows(train_rows, beta=beta)
    return [float(value) for value in weight_vector_from_rows(rows).tolist()]


def write_class_balanced_weights(
    train_rows: list[dict[str, Any]],
    beta: float,
    output_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows = class_balanced_weight_rows(train_rows, beta=beta)
    destinations = output_dirs or [EXP08_TABLES_DIR]
    for output_dir in destinations:
        write_csv(output_dir / "class_balanced_weights.csv", rows)
    return rows


def tracked_weight_files() -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS)


def exp0_to_exp7_tracked_output_changes() -> list[str]:
    paths = [f"thesis_exp/outputs/exp0{idx}" for idx in range(8)]
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--", *paths],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if path.strip())


def baseline_run_exists(run_id: str) -> bool:
    if run_id == QD_R1_RUN_ID:
        return QD_R1_RUN_DIR.exists()
    return (QD_BASELINE_RUNS_DIR / run_id).exists()


def dataset_sanity_rows(data_dir: Path = EXP08_DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "") -> None:
        rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})

    add("Exp8 run id locked", EXP08_RUN_ID == "QD-ER1_EduRisk_human_only", EXP08_RUN_ID)
    for split, expected in EXPECTED_SPLIT_ROWS.items():
        path = data_dir / f"{split}.jsonl"
        split_rows = read_split(data_dir, split) if path.exists() else []
        labels = [int(row.get("label_5", -1)) for row in split_rows if row.get("label_5") is not None]
        add(f"{split} jsonl exists", path.exists(), relpath(path))
        add(f"{split} row count = {expected}", len(split_rows) == expected, len(split_rows))
        add(f"{split} rows are human only", all(row.get("source_type") == "human" for row in split_rows))
        add(
            f"{split} label provenance is human_score",
            all(row.get("label_provenance") == "human_score" for row in split_rows),
        )
        add(f"{split} labels are 1..5", len(labels) == len(split_rows) and all(1 <= value <= 5 for value in labels))
        add(f"{split} A4 text exists", all(bool(str(row.get("text") or "").strip()) for row in split_rows))
        add(
            f"{split} template_name is A4",
            all(row.get("template_name") == "A4_question_answer_metric_rubric_metadata" for row in split_rows),
        )
        add(f"{split} has no synthetic rows", not any(row.get("source_type") == "synthetic" for row in split_rows))

    if (data_dir / "train.jsonl").exists():
        train_rows = read_split(data_dir, "train")
        counts = label_counts(train_rows)
        add("train label 1 count = 58", counts.get(1) == 58, counts)
        add("train label 2 count = 53", counts.get(2) == 53, counts)
        add("train label 3 count = 297", counts.get(3) == 297, counts)
        add("train label 4 count = 1163", counts.get(4) == 1163, counts)
        add("train label 5 count = 1755", counts.get(5) == 1755, counts)

    add("QD-B0 baseline run available", baseline_run_exists(QD_B0_RUN_ID), QD_B0_RUN_ID)
    add("QD-B1 baseline run available", baseline_run_exists(QD_B1_RUN_ID), QD_B1_RUN_ID)
    add("QD-R1 baseline run available", baseline_run_exists(QD_R1_RUN_ID), QD_R1_RUN_ID)
    weights = tracked_weight_files()
    add("no checkpoint/weights tracked", not weights, ", ".join(weights))
    changed = exp0_to_exp7_tracked_output_changes()
    add("no tracked Exp0-Exp7 output modifications", not changed, ", ".join(changed))
    return rows
