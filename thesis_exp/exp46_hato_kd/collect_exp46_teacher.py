"""Collect Exp46A 4B teacher capacity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp46_hato_kd.common import (
    EXPECTED_FOLDS,
    ROOT,
    RUN_ROOT,
    TEACHER_VARIANT,
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
    for fold in EXPECTED_FOLDS:
        root = run_dir(TEACHER_VARIANT, fold, args.run_root)
        summary_path = root / "run_summary.json"
        prediction_path = root / "heldout_predictions.jsonl"
        logit_path = root / "teacher_logits_all.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        valid = prediction_path.exists() and logit_path.exists() and validate_no_eval_access(summary) and summary.get("fixed_final_epoch") == 10
        completion.append({"variant": TEACHER_VARIANT, "fold": fold, "status": summary.get("status", "MISSING"), "valid": valid, "train_rows": summary.get("train_rows"), "eval_rows": summary.get("eval_rows"), "global_step": summary.get("global_step"), "question_key_overlap": summary.get("question_key_overlap"), "dev_access_count": summary.get("dev_access_count", 0), "test_access_count": summary.get("test_access_count", 0)})
    if args.require_complete and not all(row["valid"] for row in completion):
        raise RuntimeError("Exp46 teacher runs are missing or invalid")
    metrics = []
    if all(row["valid"] for row in completion):
        metrics = [metric_row("K0_E4", load_predictions("K0_E4")), metric_row(TEACHER_VARIANT, load_predictions(TEACHER_VARIANT, args.run_root))]
    write_csv(args.out_dir / "tables/exp46a_teacher_run_completion.csv", completion)
    write_csv(args.out_dir / "tables/exp46a_teacher_capacity_metrics.csv", metrics)
    write_csv(args.out_dir / "tables/exp46a_teacher_leakage_audit.csv", [{"scope": "teacher_groupcv", "runs": len(completion), "question_key_overlap": sum(int(row.get("question_key_overlap") or 0) for row in completion), "dev_access_count": sum(int(row["dev_access_count"]) for row in completion), "test_access_count": sum(int(row["test_access_count"]) for row in completion), "status": "PASS" if completion and all(row["valid"] for row in completion) else "FAIL"}])
    write_json(args.out_dir / "hashes/exp46a_teacher_collection.json", {"completed_runs": sum(bool(row["valid"]) for row in completion), "expected_runs": 5, "test_access_count": 0})
    print(json.dumps({"status": "COLLECTED" if metrics else "INCOMPLETE", "completed_runs": sum(bool(row["valid"]) for row in completion), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
