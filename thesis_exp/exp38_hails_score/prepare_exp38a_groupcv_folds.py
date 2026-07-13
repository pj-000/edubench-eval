"""Create locked five-fold question-key-disjoint Exp38A assignments."""

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

from thesis_exp.exp38_hails_score.common import ROOT, TRAIN_PATH, read_jsonl, sample_id, stable_hash, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.out_dir / "decision/exp38a_range_qualification_decision.json").read_text(encoding="utf-8"))
    if not decision.get("recommend_groupcv_training"):
        raise RuntimeError("GroupCV fold preparation is blocked by range qualification NO-GO")
    rows = read_jsonl(args.train_jsonl)
    groups = [str(row["question_key"]) for row in rows]
    strata = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    assignment: dict[int, int] = {}
    for fold, (_, heldout) in enumerate(splitter.split(rows, strata, groups)):
        for index in heldout:
            if int(index) in assignment:
                raise ValueError("Row assigned to more than one fold")
            assignment[int(index)] = fold
    if len(assignment) != len(rows):
        raise ValueError("Every row must receive one GroupCV fold")
    output = []
    fold_qkeys: dict[int, set[str]] = {fold: set() for fold in range(5)}
    for index, row in enumerate(rows):
        fold = assignment[index]
        fold_qkeys[fold].add(str(row["question_key"]))
        output.append({
            "sample_id": sample_id(row), "sample_id_hash": stable_hash(sample_id(row)),
            "question_key": row["question_key"], "question_key_hash": stable_hash(row["question_key"]),
            "fold": fold, "rounded_label": int(row["label_5"]), "language": row.get("language") or "unknown",
        })
    for left in range(5):
        for right in range(left + 1, 5):
            if fold_qkeys[left] & fold_qkeys[right]:
                raise RuntimeError(f"Question-key leakage between folds {left}/{right}")
    assignment_path = args.out_dir / "data/exp38a_groupcv_fold_assignment.csv"
    write_csv(assignment_path, output)
    balance = []
    for fold in range(5):
        subset = [row for row in output if row["fold"] == fold]
        counts = Counter((row["rounded_label"], row["language"]) for row in subset)
        for (label, language), count in sorted(counts.items()):
            balance.append({"fold": fold, "rounded_label": label, "language": language, "count": count, "fold_rows": len(subset), "unique_question_keys": len(fold_qkeys[fold])})
    write_csv(args.out_dir / "tables/exp38a_groupcv_fold_balance.csv", balance)
    write_json(args.out_dir / "configs/exp38a_groupcv_config.json", {
        "folds": 5, "group": "question_key", "stratify": "rounded_label_x_language", "seed": args.seed,
        "model": "Qwen3-Reranker-0.6B", "input_template": "A2_question_answer_metric",
        "epochs": 10, "learning_rate": 2e-5, "optimizer": "AdamW", "weight_decay": 0.01,
        "micro_batch": 4, "gradient_accumulation": 32, "effective_batch": 128, "max_length": 2048,
        "checkpoint_selection": "none_fixed_epoch_10", "custom_loss": False,
        "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "PREPARED", "rows": len(output), "unique_question_keys": len(set(groups)), "qkey_leakage": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
