"""Create locked question-key-disjoint five-fold assignments for Exp36A."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import ROOT, TRAIN_PATH, read_jsonl, sample_id, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError(f"Exp36A requires exactly 2654 train rows, found {len(rows)}")
    ids = [sample_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate train sample IDs")
    labels = np.asarray([
        f"{int(row['label_5'])}|{row.get('language','unknown')}|{row.get('metric_group','unknown')}"
        for row in rows
    ])
    groups = np.asarray([str(row["question_key"]) for row in rows])
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    assignments: dict[int, int] = {}
    for fold, (_, heldout) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        for index in heldout:
            if int(index) in assignments:
                raise RuntimeError("OOF row assigned twice")
            assignments[int(index)] = fold
    if len(assignments) != len(rows):
        raise RuntimeError("OOF assignment is incomplete")
    group_folds: dict[str, set[int]] = {}
    assignment_rows = []
    for index, row in enumerate(rows):
        fold = assignments[index]
        group_folds.setdefault(str(row["question_key"]), set()).add(fold)
        assignment_rows.append({
            "sample_id": sample_id(row),
            "question_key": row["question_key"],
            "fold": fold,
            "rounded_label": int(row["label_5"]),
            "language": row.get("language", "unknown"),
            "metric_family": row.get("metric_group", "unknown"),
        })
    leaking = [key for key, folds in group_folds.items() if len(folds) != 1]
    if leaking:
        raise RuntimeError(f"Question-key fold leakage: {len(leaking)} groups")
    balance = []
    for fold in range(args.folds):
        subset = [row for row in assignment_rows if row["fold"] == fold]
        labels_count = Counter(row["rounded_label"] for row in subset)
        languages = Counter(row["language"] for row in subset)
        balance.append({
            "fold": fold,
            "rows": len(subset),
            "question_keys": len({row["question_key"] for row in subset}),
            "label_1": labels_count[1], "label_2": labels_count[2], "label_3": labels_count[3],
            "label_4": labels_count[4], "label_5": labels_count[5],
            **{f"language_{key}": value for key, value in sorted(languages.items())},
            "question_key_leakage_count": 0,
        })
    write_csv(args.out_dir / "data" / "exp36a_oof_fold_assignment.csv", assignment_rows)
    write_csv(args.out_dir / "tables" / "exp36a_oof_fold_balance.csv", balance)
    print({"rows": len(rows), "folds": args.folds, "question_key_leakage_count": 0})


if __name__ == "__main__":
    main()
