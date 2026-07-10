"""Prepare validated Exp27L-R1 StratifiedGroupKFold assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    DEFAULT_DEV,
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    OUTER_SEEDS,
    SEED,
    canonical_rows,
    safe_split_ids,
    stratified_group_assignments,
    validate_outer,
    write_csv,
    write_json,
    write_text,
)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    rows = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    if len(rows) != 180:
        raise ValueError(f"Expected 180 Exp27J rows, got {len(rows)}")
    train_ids = {row["sample_id"] for row in rows}
    train_qkeys = {row["question_key"] for row in rows}
    dev_ids, dev_qkeys = safe_split_ids(args.dev_jsonl)
    test_ids, test_qkeys = safe_split_ids(args.test_jsonl)
    overlap = {
        "dev_sample_overlap": len(train_ids & dev_ids),
        "dev_question_overlap": len(train_qkeys & dev_qkeys),
        "test_sample_overlap": len(train_ids & test_ids),
        "test_question_overlap": len(train_qkeys & test_qkeys),
    }
    if any(overlap.values()):
        raise ValueError(f"Train-only Exp27L-R1 leakage guard failed: {overlap}")

    sensitivity: list[dict[str, object]] = []
    primary_assignment: dict[str, int] | None = None
    primary_balance: list[dict[str, object]] | None = None
    for seed in OUTER_SEEDS:
        assignment = stratified_group_assignments(rows, seed=seed, inner=False)
        balance = validate_outer(rows, assignment)
        for record in balance:
            sensitivity.append({"seed": seed, **record})
        if seed == args.seed:
            primary_assignment, primary_balance = assignment, balance
    if primary_assignment is None or primary_balance is None:
        raise ValueError(f"Primary seed {args.seed} is not in configured seed sensitivity")

    assignment_rows = [
        {
            "sample_id": row["sample_id"],
            "question_key_hash": row["question_key_hash"],
            "outer_fold": primary_assignment[row["sample_id"]],
            "view": row["view"],
            "score_stratum": row["score_stratum"],
            "severe_human_silver_conflict": row["severe_human_silver_conflict"],
            "design_weight": row["design_weight"],
        }
        for row in rows
    ]
    write_csv(args.out_dir / "data" / "exp27lr1_outer_fold_assignment_seed42.csv", assignment_rows)
    write_csv(args.out_dir / "tables" / "exp27lr1_outer_fold_balance.csv", primary_balance)
    write_csv(args.out_dir / "tables" / "exp27lr1_split_seed_sensitivity.csv", sensitivity)
    write_csv(
        args.out_dir / "tables" / "exp27lr1_review_leakage_audit.csv",
        [
            {"check": "train_calibration_rows", "count": len(rows)},
            {"check": "dev_sample_overlap", "count": overlap["dev_sample_overlap"]},
            {"check": "dev_question_overlap", "count": overlap["dev_question_overlap"]},
            {"check": "test_sample_overlap", "count": overlap["test_sample_overlap"]},
            {"check": "test_question_overlap", "count": overlap["test_question_overlap"]},
            {"check": "dev_label_used", "count": 0},
            {"check": "test_label_read", "count": 0},
            {"check": "teacher_api_calls", "count": 0},
            {"check": "gpu_training_runs", "count": 0},
        ],
    )
    decision = {
        "experiment": "exp27lr1_balanced_group_crossfit",
        "outer_seed": args.seed,
        "split_sensitivity_seeds": list(OUTER_SEEDS),
        "outer_constraints_pass": True,
        "dev_label_used": False,
        "test_label_read": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "proceed_to_full_3326_calibrated_dataset": False,
        "proceed_to_qwen3_reranker_training": False,
    }
    write_json(args.out_dir / "decision" / "exp27lr1_prepare_decision.json", decision)
    write_text(
        args.out_dir / "reports" / "exp27lr1_prepare_protocol.md",
        "# Exp27L-R1 Balanced Group Crossfit Preparation\n\n"
        "Exp27L-R1 uses sklearn StratifiedGroupKFold with question-key groups. "
        "It fails fast on any outer-fold balance violation and has no fallback to Exp27L's retired greedy allocator. "
        "All 180 rows are train-only; dev/test identifiers are used only for zero-overlap guards.\n",
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare(parse_args()))
