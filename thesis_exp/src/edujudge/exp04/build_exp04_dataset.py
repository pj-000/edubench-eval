"""Prepare the fixed A4 dataset used by Exp4.

Exp4 compares training objectives, not input templates. The input text is
therefore copied from the completed Exp3 A4 dataset into an Exp4-owned dataset
directory. This keeps Exp3 outputs immutable while giving Exp4 a stable input
path for server training.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04 import (
    A4_FIXED_DATASET_DIR,
    EXPECTED_SPLIT_ROWS,
    EXP03_A4_DATASET_DIR,
    EXP04_TABLES_DIR,
    ensure_exp04_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


REQUIRED_FIELDS = {
    "id",
    "record_id",
    "triple_key",
    "text",
    "label",
    "label_5",
    "human_mean_5",
    "metric_canonical",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
}


def validate_rows(path: Path, split: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    missing_fields: set[str] = set()
    bad_labels = 0
    bad_means = 0
    for row in rows:
        missing_fields.update(REQUIRED_FIELDS - set(row))
        try:
            label_5 = int(row["label_5"])
            if label_5 < 1 or label_5 > 5:
                bad_labels += 1
        except Exception:
            bad_labels += 1
        try:
            human_mean = float(row["human_mean_5"])
            if human_mean < 1.0 or human_mean > 5.0:
                bad_means += 1
        except Exception:
            bad_means += 1
    expected = EXPECTED_SPLIT_ROWS[split]
    status = "PASS" if len(rows) == expected and not missing_fields and bad_labels == 0 and bad_means == 0 else "FAIL"
    return {
        "split": split,
        "path": relpath(path),
        "rows": len(rows),
        "expected_rows": expected,
        "missing_fields": ",".join(sorted(missing_fields)),
        "bad_labels": bad_labels,
        "bad_human_mean_5": bad_means,
        "status": status,
    }


def build_exp04_dataset(force: bool = False) -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    if not EXP03_A4_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Exp3 A4 dataset not found: {relpath(EXP03_A4_DATASET_DIR)}. "
            "Run or sync Exp3 A4 outputs before Exp4."
        )

    A4_FIXED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        src = EXP03_A4_DATASET_DIR / f"{split}.jsonl"
        dst = A4_FIXED_DATASET_DIR / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing Exp3 A4 split: {relpath(src)}")
        if force or not dst.exists():
            shutil.copy2(src, dst)
        rows.append(validate_rows(dst, split))

    write_csv(EXP04_TABLES_DIR / "dataset_sanity.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    lines = [
        "# Exp4 Fixed A4 Dataset",
        "",
        f"Overall status: **{overall}**",
        "",
        "Exp4 fixes the A4 input template from Exp3 and changes only the modeling objective.",
        "",
        "| split | rows | expected | status | path |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['expected_rows']} | "
            f"{row['status']} | `{row['path']}` |"
        )
    write_text(A4_FIXED_DATASET_DIR / "dataset_card.md", "\n".join(lines))
    if overall != "PASS":
        raise RuntimeError(f"Exp4 dataset sanity failed; see {relpath(EXP04_TABLES_DIR / 'dataset_sanity.csv')}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy and validate the fixed Exp4 A4 dataset.")
    parser.add_argument("--force", action="store_true", help="Overwrite the Exp4 dataset copy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_exp04_dataset(force=args.force)
    print(f"Exp4 fixed A4 dataset ready: {[row['status'] for row in rows]}")
    print(f"Output: {relpath(A4_FIXED_DATASET_DIR)}")


if __name__ == "__main__":
    main()
