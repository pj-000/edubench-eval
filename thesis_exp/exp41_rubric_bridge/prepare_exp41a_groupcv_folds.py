"""Prepare locked train-only question-key-disjoint Exp41A GroupCV folds."""

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

from thesis_exp.exp41_rubric_bridge.common import ROOT, TRAIN_PATH, read_jsonl, sample_id, sha256_file, stable_hash, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp41a_compiler_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("GroupCV blocked by compiler qualification NO-GO")
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError("Exp41A requires 2654 paper-like train rows")
    labels = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    groups = [str(row["question_key"]) for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    assignments: dict[str, int] = {}
    for fold, (_, heldout) in enumerate(splitter.split(rows, labels, groups)):
        for index in heldout:
            assignments[sample_id(rows[index])] = fold
    if len(assignments) != len(rows):
        raise RuntimeError("Incomplete fold assignment")
    qkey_folds: dict[str, int] = {}
    for row in rows:
        qkey, fold = str(row["question_key"]), assignments[sample_id(row)]
        if qkey in qkey_folds and qkey_folds[qkey] != fold:
            raise RuntimeError(f"Question key crosses folds: {qkey}")
        qkey_folds[qkey] = fold
    private_rows = [{"sample_id": sample_id(row), "question_key": row["question_key"], "fold": assignments[sample_id(row)]} for row in rows]
    write_csv(args.out_dir / "private/data/exp41a_groupcv_fold_assignment.csv", private_rows)
    balance = []
    for fold in range(5):
        subset = [row for row in rows if assignments[sample_id(row)] == fold]
        counts = Counter((int(row["label_5"]), str(row.get("language") or "unknown")) for row in subset)
        for (label, language), count in sorted(counts.items()):
            balance.append({"fold": fold, "label": label, "language": language, "rows": count,
                            "fold_total_rows": len(subset), "question_keys": len({row["question_key"] for row in subset})})
    leakage = []
    for fold in range(5):
        train_qkeys = {key for key, value in qkey_folds.items() if value != fold}
        heldout_qkeys = {key for key, value in qkey_folds.items() if value == fold}
        leakage.append({"fold": fold, "train_question_keys": len(train_qkeys), "heldout_question_keys": len(heldout_qkeys),
                        "question_key_overlap": len(train_qkeys & heldout_qkeys), "leakage_pass": not (train_qkeys & heldout_qkeys),
                        "label_free_inference_preprocessing": True, "dev_access_count": 0, "test_access_count": 0})
    write_csv(args.out_dir / "tables/exp41a_groupcv_fold_balance.csv", balance)
    write_csv(args.out_dir / "tables/exp41a_groupcv_leakage_audit.csv", leakage)
    write_json(args.out_dir / "hashes/exp41a_training_config_hashes.json", {
        "fold_assignment_sha256": stable_hash(private_rows), "folds": 5, "random_state": args.seed,
        "group": "question_key", "stratify": "rounded_label_x_language", "label_free_inference_preprocessing": True,
        "compiler_protocol_sha256": sha256_file(args.out_dir / "configs/exp41a_compiler_protocol_lock.json"),
        "groupcv_config_sha256": sha256_file(args.out_dir / "configs/exp41a_groupcv_config.json"),
        "training_matrix_sha256": sha256_file(args.out_dir / "configs/exp41a_training_matrix_lock.json"),
        "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "PREPARED", "rows": len(rows), "question_keys": len(qkey_folds),
                      "fold_rows": dict(Counter(assignments.values())), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
