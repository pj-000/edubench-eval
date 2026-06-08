"""Validate the fixed A4 dataset reference used by Exp5 L1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04.build_exp04_dataset import build_exp04_dataset
from thesis_exp.src.edujudge.exp05 import (
    EXP04_A4_DATASET_DIR,
    EXP05_OUTPUT_DIR,
    EXP05_TABLES_DIR,
    EXPECTED_SPLIT_ROWS,
    ensure_exp05_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


REQUIRED_FIELDS = {
    "id",
    "record_id",
    "text",
    "label_5",
    "human_mean_5",
    "metric_canonical",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
}


def validate_split(path: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    missing_fields: set[str] = set()
    bad_labels = 0
    for row in rows:
        missing_fields.update(REQUIRED_FIELDS - set(row))
        try:
            label = int(row["label_5"])
            if label < 1 or label > 5:
                bad_labels += 1
        except Exception:
            bad_labels += 1
    expected = EXPECTED_SPLIT_ROWS[split]
    clean_path = str(path).lower()
    sample_path = "synthetic" in clean_path or "sampled" in clean_path or "sample_" in clean_path
    status = "PASS" if len(rows) == expected and not missing_fields and bad_labels == 0 and not sample_path else "FAIL"
    return {
        "split": split,
        "path": relpath(path),
        "rows": len(rows),
        "expected_rows": expected,
        "missing_fields": ",".join(sorted(missing_fields)),
        "bad_labels": bad_labels,
        "synthetic_or_sample_path": sample_path,
        "status": status,
    }


def ensure_exp05_dataset() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    if not all((EXP04_A4_DATASET_DIR / f"{split}.jsonl").exists() for split in EXPECTED_SPLIT_ROWS):
        build_exp04_dataset(force=False)
    rows = [validate_split(EXP04_A4_DATASET_DIR / f"{split}.jsonl", split) for split in ["train", "dev", "test"]]
    write_csv(EXP05_TABLES_DIR / "dataset_reference_sanity.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    lines = [
        "# Exp5 A4 Dataset Reference",
        "",
        f"Overall status: **{overall}**",
        "",
        "Exp5 L1 reuses the fixed Exp4 A4 dataset. It changes only the loss weighting.",
        "",
        "| split | rows | expected | status | path |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['expected_rows']} | "
            f"{row['status']} | `{row['path']}` |"
        )
    write_text(EXP05_OUTPUT_DIR / "dataset_reference.md", "\n".join(lines))
    if overall != "PASS":
        raise RuntimeError(f"Exp5 dataset reference failed; see {relpath(EXP05_TABLES_DIR / 'dataset_reference_sanity.csv')}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Exp5 A4 dataset reference.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = ensure_exp05_dataset()
    print(f"Exp5 dataset reference ready: {[row['status'] for row in rows]}")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / 'dataset_reference.md')}")


if __name__ == "__main__":
    main()
