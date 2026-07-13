"""Prepare locked question-key-disjoint Exp39A GroupCV folds."""

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

from thesis_exp.exp39_educfa.common import ROOT, TRAIN_PATH, VARIANTS, read_jsonl, sample_id, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp39a_data_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("GroupCV preparation is blocked by data qualification NO-GO")
    rows = read_jsonl(args.train_jsonl)
    labels = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    groups = [str(row["question_key"]) for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    assignments = {}
    for fold, (_, heldout) in enumerate(splitter.split(rows, labels, groups)):
        for index in heldout:
            sid = sample_id(rows[index])
            if sid in assignments:
                raise RuntimeError(f"Duplicate fold assignment: {sid}")
            assignments[sid] = fold
    if len(assignments) != 2654:
        raise RuntimeError("Every original train row must receive one fold")
    qkey_folds = {}
    for row in rows:
        qkey, fold = str(row["question_key"]), assignments[sample_id(row)]
        if qkey in qkey_folds and qkey_folds[qkey] != fold:
            raise RuntimeError(f"Question key crosses folds: {qkey}")
        qkey_folds[qkey] = fold
    assignment_rows = [{"sample_id": sample_id(row), "question_key": row["question_key"], "fold": assignments[sample_id(row)]} for row in rows]
    output_path = args.out_dir / "private/data/exp39a_groupcv_fold_assignment.csv"
    write_csv(output_path, assignment_rows)
    balance = []
    for fold in range(5):
        subset = [row for row in rows if assignments[sample_id(row)] == fold]
        counts = Counter((int(row["label_5"]), str(row.get("language") or "unknown")) for row in subset)
        for (label, language), count in sorted(counts.items()):
            balance.append({"fold": fold, "label": label, "language": language, "rows": count, "question_keys": len({row["question_key"] for row in subset})})
    write_csv(args.out_dir / "tables/exp39a_groupcv_fold_balance.csv", balance)
    leakage = []
    for variant in VARIANTS:
        variant_rows = read_jsonl(args.out_dir / f"private/data/exp39a_{variant}.jsonl")
        unknown = [row for row in variant_rows if str(row["question_key"]) not in qkey_folds]
        leakage.append({
            "variant": variant, "rows": len(variant_rows), "unknown_question_key_rows": len(unknown),
            "question_key_cross_fold_count": 0, "synthetic_eval_rows": sum(row.get("synthetic") and row.get("original_evaluation_row") for row in variant_rows),
            "leakage_pass": not unknown and not any(row.get("synthetic") and row.get("original_evaluation_row") for row in variant_rows),
        })
    write_csv(args.out_dir / "tables/exp39a_groupcv_leakage_audit.csv", leakage)
    if not all(row["leakage_pass"] for row in leakage):
        raise RuntimeError("Exp39A GroupCV leakage audit failed")
    print(json.dumps({
        "status": "PREPARED", "rows": len(rows), "question_keys": len(qkey_folds),
        "fold_rows": dict(Counter(assignments.values())), "dev_access_count": 0, "test_access_count": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
