"""Prepare locked train-only StratifiedGroupKFold assignments for Exp40A."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import StratifiedGroupKFold

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    PAIR_VARIANTS,
    ROOT,
    TRAIN_PATH,
    read_jsonl,
    sample_id,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp40a_pairwise_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("GroupCV preparation is blocked by Exp40A pairwise qualification NO-GO")
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError("Exp40A GroupCV requires the 2654-row paper-like train split")
    labels = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    groups = [str(row["question_key"]) for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    assignments: dict[str, int] = {}
    for fold, (_, heldout) in enumerate(splitter.split(rows, labels, groups)):
        for index in heldout:
            sid = sample_id(rows[index])
            if sid in assignments:
                raise RuntimeError(f"Duplicate fold assignment: {sid}")
            assignments[sid] = fold
    qkey_folds: dict[str, int] = {}
    for row in rows:
        qkey, fold = str(row["question_key"]), assignments[sample_id(row)]
        if qkey in qkey_folds and qkey_folds[qkey] != fold:
            raise RuntimeError(f"Question key crosses folds: {qkey}")
        qkey_folds[qkey] = fold
    assignment_rows = [
        {"sample_id": sample_id(row), "question_key": row["question_key"], "fold": assignments[sample_id(row)]}
        for row in rows
    ]
    write_csv(args.out_dir / "private/data/exp40a_groupcv_fold_assignment.csv", assignment_rows)
    balance = []
    for fold in range(5):
        subset = [row for row in rows if assignments[sample_id(row)] == fold]
        counts = Counter((int(row["label_5"]), str(row.get("language") or "unknown")) for row in subset)
        for (label, language), count in sorted(counts.items()):
            balance.append(
                {
                    "fold": fold, "label": label, "language": language, "rows": count,
                    "question_keys": len({row["question_key"] for row in subset}),
                }
            )
    write_csv(args.out_dir / "tables/exp40a_groupcv_fold_balance.csv", balance)
    leakage = [{"variant": "v0h_human_soft", "pairs": 0, "unknown_qkeys": 0, "cross_fold_pairs": 0, "leakage_pass": True}]
    for variant in PAIR_VARIANTS:
        pairs = read_jsonl(args.out_dir / f"private/data/exp40a_{variant}.jsonl")
        unknown = [row for row in pairs if str(row["question_key"]) not in qkey_folds]
        mismatched = [
            row for row in pairs
            if str(row["better"].get("question_key")) != str(row["question_key"])
            or str(row["worse"].get("question_key")) != str(row["question_key"])
        ]
        leakage.append(
            {
                "variant": variant, "pairs": len(pairs), "unknown_qkeys": len(unknown),
                "cross_fold_pairs": len(mismatched), "leakage_pass": not unknown and not mismatched,
            }
        )
    write_csv(args.out_dir / "tables/exp40a_groupcv_leakage_audit.csv", leakage)
    if not all(row["leakage_pass"] for row in leakage):
        raise RuntimeError("Exp40A GroupCV pair leakage audit failed")
    print(
        json.dumps(
            {
                "status": "PREPARED", "rows": len(rows), "question_keys": len(qkey_folds),
                "fold_rows": dict(Counter(assignments.values())), "dev_access_count": 0, "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
