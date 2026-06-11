"""Dataset and repository safety checks for Exp7."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXPECTED_SPLIT_ROWS,
    EXP07_DATASET_DIR,
    QD_BASELINE_TABLES_DIR,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath


CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / f"{split}.jsonl")


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return rows[:limit] if limit else rows


def tracked_weight_files() -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True)
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS)


def exp0_to_exp6_tracked_output_changes() -> list[str]:
    paths = [f"thesis_exp/outputs/exp0{idx}" for idx in range(7)]
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


def dataset_sanity_rows(data_dir: Path = EXP07_DATASET_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "") -> None:
        rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})

    for split, expected in EXPECTED_SPLIT_ROWS.items():
        path = data_dir / f"{split}.jsonl"
        split_rows = read_split(data_dir, split) if path.exists() else []
        labels = [int(row.get("label_5", -1)) for row in split_rows if row.get("label_5") is not None]
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
        add(
            f"{split} has no synthetic rows",
            not any(row.get("source_type") == "synthetic" for row in split_rows),
        )

    add(
        "QD-B0/QD-B1 baseline comparison files exist",
        (QD_BASELINE_TABLES_DIR / "qd_baseline_metrics_summary.csv").exists()
        and (QD_BASELINE_TABLES_DIR / "qd_baseline_test_comparison.csv").exists(),
        relpath(QD_BASELINE_TABLES_DIR),
    )
    weights = tracked_weight_files()
    add("no checkpoint/weights tracked", not weights, ", ".join(weights))
    changed = exp0_to_exp6_tracked_output_changes()
    add("no tracked Exp0-Exp6 output modifications", not changed, ", ".join(changed))
    return rows
