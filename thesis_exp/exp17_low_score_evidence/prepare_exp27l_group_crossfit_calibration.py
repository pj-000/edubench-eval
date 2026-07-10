"""Prepare the locked, question-key grouped Exp27L calibration protocol.

This stage only creates folds and protocol artifacts.  It never contacts a
teacher, trains a scorer, opens dev/test, or exposes Exp27J silver labels to
an external review packet.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27l_common import (  # noqa: E402
    DEFAULT_EXP27I,
    DEFAULT_EXP27J,
    DEFAULT_EXP27K,
    DEFAULT_OUT,
    DEFAULT_TRAIN,
    OUTER_FOLDS,
    REPRESENTATIVE_VIEW,
    RISK_ENRICHED_VIEW,
    SEED,
    canonical_rows,
    grouped_folds,
    read_csv,
    write_csv,
    write_json,
    write_text,
)


def fold_balance(rows: list[dict[str, Any]], folds: dict[str, int]) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, str, str, int]] = Counter()
    for row in rows:
        counts[(folds[row["question_key"]], row["view"], row["score_stratum"], row["severe_human_silver_conflict"])] += 1
    return [
        {
            "outer_fold": fold,
            "view": view,
            "score_stratum": stratum,
            "severe_human_silver_conflict": severe,
            "row_count": count,
        }
        for (fold, view, stratum, severe), count in sorted(counts.items())
    ]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows = canonical_rows(args.train_jsonl, args.exp27j_dir, args.exp27k_dir)
    if len(rows) != 180:
        raise ValueError(f"Expected 180 Exp27J rows, got {len(rows)}")
    counts = Counter(row["view"] for row in rows)
    if counts != {REPRESENTATIVE_VIEW: 120, RISK_ENRICHED_VIEW: 60}:
        raise ValueError(f"Unexpected Exp27J views: {counts}")
    folds = grouped_folds(rows, OUTER_FOLDS, args.seed)
    if len(folds) < OUTER_FOLDS:
        raise ValueError("Fold assignment does not contain five question-key folds")
    leakage = {
        row["check"]: int(row["count"])
        for row in read_csv(args.exp27j_dir / "tables" / "exp27j_leakage_audit.csv")
    }
    required_clean = (
        "dev_sample_overlap",
        "dev_question_overlap",
        "test_sample_overlap",
        "test_question_overlap",
    )
    if any(leakage.get(key) != 0 for key in required_clean):
        raise ValueError(f"Exp27J inherited leakage guard failed: {leakage}")

    fold_rows = [
        {
            "sample_id": row["sample_id"],
            "question_key_hash": row["question_key_hash"],
            "outer_fold": folds[row["question_key"]],
            "view": row["view"],
            "score_stratum": row["score_stratum"],
            "severe_human_silver_conflict": row["severe_human_silver_conflict"],
            "design_weight": row["design_weight"],
        }
        for row in rows
    ]
    write_csv(args.out_dir / "data" / "exp27l_question_key_fold_assignment.csv", fold_rows)
    write_csv(args.out_dir / "tables" / "exp27l_fold_balance.csv", fold_balance(rows, folds))
    write_csv(
        args.out_dir / "tables" / "exp27l_leakage_audit.csv",
        [
            {"check": "source_rows_from_train", "count": len(rows)},
            {"check": "representative_rows", "count": counts[REPRESENTATIVE_VIEW]},
            {"check": "risk_enriched_rows", "count": counts[RISK_ENRICHED_VIEW]},
            {"check": "inherited_exp27j_dev_sample_overlap", "count": leakage["dev_sample_overlap"]},
            {"check": "inherited_exp27j_dev_question_overlap", "count": leakage["dev_question_overlap"]},
            {"check": "inherited_exp27j_test_sample_overlap", "count": leakage["test_sample_overlap"]},
            {"check": "inherited_exp27j_test_question_overlap", "count": leakage["test_question_overlap"]},
            {"check": "dev_test_files_opened_by_exp27l_prepare", "count": 0},
            {"check": "teacher_api_calls", "count": 0},
            {"check": "gpu_training_runs", "count": 0},
        ],
    )
    decision = {
        "experiment": "exp27l_group_crossfit_calibration",
        "seed": args.seed,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": 3,
        "rows": len(rows),
        "question_key_groups": len(folds),
        "representative_rows": counts[REPRESENTATIVE_VIEW],
        "risk_enriched_rows": counts[RISK_ENRICHED_VIEW],
        "score_calibrator_fit_population": "representative_only",
        "risk_calibrator_fit_population": "representative_only",
        "risk_target": "severe_human_silver_conflict",
        "risk_target_definition": "abs(original_score - silver_final_score) >= 2",
        "risk_stress_used_for_fit": False,
        "dev_test_files_opened": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "proceed_to_exp27m_train": False,
        "proceed_to_reranker_training": False,
        "reason": "OOF calibration and external human review are required before any trainer dataset is constructed.",
    }
    write_json(args.out_dir / "decision" / "exp27l_prepare_decision.json", decision)
    write_text(
        args.out_dir / "reports" / "exp27l_crossfit_protocol.md",
        "\n".join(
            [
                "# Exp27L Question-Key Cross-Fitted Calibration Protocol",
                "",
                "Exp27L is a CPU-only calibration audit over the locked 180 train-only Exp27J rows.",
                "Each question key belongs to one outer fold, so no answer from the same question can appear in both fit and held-out evaluation.",
                "",
                "- Outer folds: 5",
                "- Inner folds: 3, grouped by question key",
                "- Score/risk fitting rows: representative view only (120 rows)",
                "- Risk-stress rows: held-out OOF stress evaluation only (60 rows)",
                "- Risk target: severe human-silver conflict, defined as an absolute score difference of at least 2",
                "- No teacher API, no GPU, no train/dev/test model training",
                "",
                "Exp27J silver fields are evaluation targets only. They are not allowed in the score fusion inputs, risk features, or external blind-review packets.",
                "",
            ]
        ),
    )
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27j-dir", type=Path, default=DEFAULT_EXP27J)
    parser.add_argument("--exp27k-dir", type=Path, default=DEFAULT_EXP27K)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(prepare(parse_args()))
