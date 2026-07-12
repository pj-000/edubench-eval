"""Validate Exp36A OOF coverage, supervision invariants, and private artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import ROOT, VARIANTS, dynamic_target, read_csv, read_jsonl, sample_id, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--require-private", action="store_true")
    args = parser.parse_args()
    checks = []
    assignment = read_csv(args.out_dir / "data/exp36a_oof_fold_assignment.csv")
    checks.append(("fold_assignment_rows", len(assignment) == 2654, len(assignment)))
    qkey_folds = {}
    for row in assignment:
        qkey_folds.setdefault(row["question_key"], set()).add(row["fold"])
    checks.append(("question_key_leakage_zero", all(len(v) == 1 for v in qkey_folds.values()), sum(len(v)>1 for v in qkey_folds.values())))
    oof_ids = []
    for fold in range(5):
        path = args.out_dir / "private/oof_folds" / f"fold_{fold}_predictions.jsonl"
        if path.exists():
            oof_ids.extend(sample_id(row) for row in read_jsonl(path))
        elif args.require_private:
            checks.append((f"oof_fold_{fold}_exists", False, str(path)))
    if oof_ids:
        checks.append(("oof_prediction_rows", len(oof_ids) == 2654, len(oof_ids)))
        checks.append(("oof_prediction_unique", len(oof_ids) == len(set(oof_ids)), len(set(oof_ids))))
    hashes_path = args.out_dir / "hashes/exp36a_dataset_hashes.json"
    if args.require_private:
        checks.append(("dataset_hashes_exist", hashes_path.exists(), str(hashes_path)))
    if hashes_path.exists():
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        for variant in VARIANTS:
            path = args.out_dir / "private/data" / f"exp36a_{variant}_train.jsonl"
            if not path.exists():
                checks.append((f"{variant}_exists", False, str(path)))
                continue
            rows = read_jsonl(path)
            valid = len(rows) == 2654 and all(
                abs(sum(map(float, row["human_target_5"])) - 1) < 1e-6
                and abs(sum(map(float, row["teacher_target_5"])) - 1) < 1e-6
                and float(row["sample_weight"]) == 1.0
                and 0 <= float(row["teacher_lambda"]) <= 0.4
                for row in rows
            )
            checks.append((f"{variant}_valid", valid, len(rows)))
            blocked = all(float(row["teacher_lambda"]) == 0 for row in rows if int(row["rounded_human_label"]) <= 2 and int(row["teacher_score"]) >= 4)
            checks.append((f"{variant}_low_high_anchor", blocked, "lambda_zero"))
        v5_path = args.out_dir / "private/data/exp36a_v5_safer_score_train.jsonl"
        if v5_path.exists():
            v5_rows = read_jsonl(v5_path)
            changing = [row for row in v5_rows if float(row["teacher_lambda"]) > 0]
            dynamic_ok = bool(changing) and all(
                dynamic_target(row, 3, "v5_safer_score") == row["human_target_5"]
                and dynamic_target(row, 6, "v5_safer_score") != row["human_target_5"]
                for row in changing
            )
            checks.append(("dynamic_target_changes_by_epoch", dynamic_ok, len(changing)))
            zero_mask_rows = sum(int(row["failure_mask"]) == 0 for row in v5_rows)
            checks.append(("zero_failure_mask_cases_present", zero_mask_rows > 0, zero_mask_rows))
    rows_out = [{"check": name, "pass": passed, "detail": detail} for name, passed, detail in checks]
    write_csv(args.out_dir / "tables/exp36a_supervision_validation.csv", rows_out)
    passed = all(row[1] for row in checks)
    write_json(args.out_dir / "decision/exp36a_supervision_validation.json", {
        "status": "PASS" if passed else "FAIL", "checks": len(checks),
        "failed_checks": [name for name, ok, _ in checks if not ok], "test_access_count": 0,
    })
    if not passed:
        raise SystemExit("Exp36A supervision validation failed")
    print({"status": "PASS", "checks": len(checks)})


if __name__ == "__main__":
    main()
