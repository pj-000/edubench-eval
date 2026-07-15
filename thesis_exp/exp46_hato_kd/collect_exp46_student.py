"""Collect Exp46A student HATO-KD results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp46_hato_kd.common import (
    EXPECTED_FOLDS,
    ROOT,
    RUN_ROOT,
    STUDENT_VARIANTS,
    load_predictions,
    metric_row,
    run_dir,
    validate_no_eval_access,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completion = []
    donor_rates = []
    for variant in STUDENT_VARIANTS:
        for fold in EXPECTED_FOLDS:
            root = run_dir(variant, fold, args.run_root)
            summary_path = root / "run_summary.json"
            prediction_path = root / "heldout_predictions.jsonl"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            valid = prediction_path.exists() and validate_no_eval_access(summary) and summary.get("fixed_final_epoch") == 10 and int(summary.get("teacher_logit_train_coverage", 0)) == int(summary.get("train_rows", -1))
            donor_rate = float(summary.get("shuffled_teacher_logit_donor_change_rate", 0.0))
            if variant == "K3_shuffled_hato_control":
                donor_rates.append(donor_rate)
            completion.append({"variant": variant, "fold": fold, "status": summary.get("status", "MISSING"), "valid": valid, "train_rows": summary.get("train_rows"), "eval_rows": summary.get("eval_rows"), "global_step": summary.get("global_step"), "teacher_logit_train_coverage": summary.get("teacher_logit_train_coverage"), "donor_change_rate": donor_rate, "question_key_overlap": summary.get("question_key_overlap"), "dev_access_count": summary.get("dev_access_count", 0), "test_access_count": summary.get("test_access_count", 0)})
    if args.require_complete and not all(row["valid"] for row in completion):
        raise RuntimeError("Exp46 student runs are missing or invalid")
    metrics = []
    if all(row["valid"] for row in completion):
        for variant in ("K0_E4", "C1_strongest_point", *STUDENT_VARIANTS):
            metrics.append(metric_row(variant, load_predictions(variant, args.run_root)))
    write_csv(args.out_dir / "tables/exp46a_student_run_completion.csv", completion)
    write_csv(args.out_dir / "tables/exp46a_student_metrics.csv", metrics)
    write_csv(args.out_dir / "tables/exp46a_student_leakage_audit.csv", [{"scope": "student_groupcv", "runs": len(completion), "question_key_overlap": sum(int(row.get("question_key_overlap") or 0) for row in completion), "dev_access_count": sum(int(row["dev_access_count"]) for row in completion), "test_access_count": sum(int(row["test_access_count"]) for row in completion), "status": "PASS" if completion and all(row["valid"] for row in completion) else "FAIL"}])
    write_json(args.out_dir / "hashes/exp46a_student_collection.json", {"completed_runs": sum(bool(row["valid"]) for row in completion), "expected_runs": 15, "k3_mean_donor_change_rate": sum(donor_rates) / len(donor_rates) if donor_rates else 0.0, "test_access_count": 0})
    print(json.dumps({"status": "COLLECTED" if metrics else "INCOMPLETE", "completed_runs": sum(bool(row["valid"]) for row in completion), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
