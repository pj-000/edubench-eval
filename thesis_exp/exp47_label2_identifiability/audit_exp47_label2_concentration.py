"""Audit label-2 concentration by question and metadata fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from thesis_exp.exp47_label2_identifiability.common import (
    ROOT,
    ensure_dirs,
    fold_assignments,
    label2_flags,
    load_canonical_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "hard_label2": rows,
        "stable_label2": [row for row in rows if row["stable_label2"]],
        "strict_label2": [row for row in rows if row["strict_label2"]],
        "ambiguous_label2": [row for row in rows if row["ambiguous_label2"]],
    }


def concentration(values: list[str]) -> tuple[int, float, float, float]:
    counts = Counter(values)
    n = max(len(values), 1)
    hhi = sum((count / n) ** 2 for count in counts.values())
    return len(counts), hhi, 1.0 / hhi if hhi else 0.0, max(counts.values(), default=0) / n


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    assignments = fold_assignments()
    label2 = []
    for row in load_canonical_rows():
        if int(row["gold_label_5"]) == 2:
            label2.append({**row, **label2_flags(row)})

    question_rows = []
    distribution_rows = []
    fold_rows = []
    fields = ("metric", "subject", "language", "scenario", "education_level", "generator_model")
    for name, rows in subsets(label2).items():
        unique, hhi, effective, max_share = concentration([str(row["question_key"]) for row in rows])
        question_rows.append(
            {
                "label2_subset": name,
                "rows": len(rows),
                "unique_question_keys": unique,
                "question_hhi": hhi,
                "effective_question_keys": effective,
                "max_question_key_rate": max_share,
            }
        )
        for field in fields:
            counts = Counter(str(row.get(field) or "unknown") for row in rows)
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                distribution_rows.append(
                    {
                        "label2_subset": name,
                        "field": field,
                        "value": value,
                        "count": count,
                        "share": count / len(rows) if rows else 0.0,
                    }
                )
        fold_counts = Counter(assignments[row["sample_id"]] for row in rows)
        for fold in range(5):
            fold_rows.append(
                {
                    "label2_subset": name,
                    "fold": fold,
                    "count": fold_counts[fold],
                    "share": fold_counts[fold] / len(rows) if rows else 0.0,
                }
            )
    write_csv(args.out_dir / "tables/exp47_label2_question_concentration.csv", question_rows)
    write_csv(args.out_dir / "tables/exp47_label2_metric_subject_distribution.csv", distribution_rows)
    write_csv(args.out_dir / "tables/exp47_label2_fold_distribution.csv", fold_rows)
    print(json.dumps({"status": "CONCENTRATION_AUDITED", "label2": len(label2), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
