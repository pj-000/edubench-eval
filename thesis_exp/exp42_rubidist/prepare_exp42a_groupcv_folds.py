"""Reuse or exactly reproduce the locked Exp41A question-key GroupCV folds."""

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

from thesis_exp.exp42_rubidist.common import (  # noqa: E402
    EXP41_ROOT,
    ROOT,
    TRAIN_PATH,
    read_fold_assignment,
    read_jsonl,
    sample_id,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--exp41-root", type=Path, default=EXP41_ROOT)
    return parser.parse_args()


def regenerate(rows: list[dict]) -> list[dict]:
    labels = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    groups = [str(row["question_key"]) for row in rows]
    assignments: dict[str, int] = {}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, heldout) in enumerate(splitter.split(rows, labels, groups)):
        for index in heldout:
            assignments[sample_id(rows[index])] = fold
    return [{"sample_id": sample_id(row), "question_key": row["question_key"], "fold": assignments[sample_id(row)]} for row in rows]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError("Exp42A requires 2654 paper-like train rows")
    exp41_hashes = json.loads((args.exp41_root / "hashes/exp41a_training_config_hashes.json").read_text(encoding="utf-8"))
    expected_hash = str(exp41_hashes["fold_assignment_sha256"])
    source = args.exp41_root / "private/data/exp41a_groupcv_fold_assignment.csv"
    if source.exists():
        source_rows = read_fold_assignment(source)
        by_id = {row["sample_id"]: row for row in source_rows}
        assignments = [{"sample_id": sample_id(row), "question_key": row["question_key"], "fold": int(by_id[sample_id(row)]["fold"])} for row in rows]
        source_mode = "copied_from_exp41_private_assignment"
    else:
        assignments = regenerate(rows)
        source_mode = "regenerated_exact_exp41_procedure"
    actual_hash = stable_hash(assignments)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Exp42A fold hash mismatch: expected={expected_hash} actual={actual_hash}")
    fold_by_id = {row["sample_id"]: int(row["fold"]) for row in assignments}
    qkey_fold: dict[str, int] = {}
    for row in rows:
        qkey, fold = str(row["question_key"]), fold_by_id[sample_id(row)]
        if qkey in qkey_fold and qkey_fold[qkey] != fold:
            raise RuntimeError(f"Question key crosses folds: {qkey}")
        qkey_fold[qkey] = fold
    private_path = args.out_dir / "private/data/exp42a_groupcv_fold_assignment.csv"
    write_csv(private_path, assignments)
    balance, leakage = [], []
    for fold in range(5):
        subset = [row for row in rows if fold_by_id[sample_id(row)] == fold]
        counts = Counter((int(row["label_5"]), str(row.get("language") or "unknown")) for row in subset)
        for (label, language), count in sorted(counts.items()):
            balance.append({"fold": fold, "label": label, "language": language, "rows": count, "fold_total_rows": len(subset), "question_keys": len({row["question_key"] for row in subset})})
        train_keys = {key for key, value in qkey_fold.items() if value != fold}
        heldout_keys = {key for key, value in qkey_fold.items() if value == fold}
        leakage.append({
            "fold": fold,
            "train_question_keys": len(train_keys),
            "heldout_question_keys": len(heldout_keys),
            "question_key_overlap": len(train_keys & heldout_keys),
            "leakage_pass": not (train_keys & heldout_keys),
            "dev_access_count": 0,
            "test_access_count": 0,
        })
    write_csv(args.out_dir / "tables/exp42a_fold_balance.csv", balance)
    write_csv(args.out_dir / "tables/exp42a_leakage_audit.csv", leakage)
    write_json(args.out_dir / "hashes/exp42a_fold_hashes.json", {
        "source_mode": source_mode,
        "exp41_expected_fold_assignment_sha256": expected_hash,
        "exp42_fold_assignment_sha256": actual_hash,
        "fold_hash_equal_exp41": True,
        "private_fold_path": str(private_path),
        "private_fold_file_sha256": sha256_file(private_path),
        "fold_rows": dict(Counter(int(row["fold"]) for row in assignments)),
        "question_keys": len(qkey_fold),
        "dev_access_count": 0,
        "test_access_count": 0,
    })
    print(json.dumps({"status": "FOLDS_LOCKED", "source_mode": source_mode, "fold_hash_equal_exp41": True, "rows": 2654, "question_keys": len(qkey_fold), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
