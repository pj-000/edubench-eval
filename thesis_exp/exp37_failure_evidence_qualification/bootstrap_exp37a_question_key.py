"""Standalone question-key bootstrap for aligned versus shuffled signals."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    ROOT, read_csv, tie_safe_average_precision, write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--question-key-column", default="question_key")
    parser.add_argument("--target-column", default="severe")
    parser.add_argument("--aligned-column", default="aligned_verified_failure")
    parser.add_argument("--shuffled-column", default="shuffled_verified_failure")
    parser.add_argument("--out", type=Path, default=ROOT / "tables/exp37a_r1_question_key_bootstrap_ci.csv")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_csv(args.input_csv)
    required = {args.question_key_column, args.target_column, args.aligned_column, args.shuffled_column}
    if not rows or not required <= set(rows[0]):
        write_csv(args.out, [{"status": "MISSING_REQUIRED_COLUMNS", "required_columns": "|".join(sorted(required)), "test_access_count": 0}])
        return
    groups = defaultdict(list)
    for row in rows:
        groups[row[args.question_key_column]].append(row)
    keys = sorted(groups); rng = np.random.default_rng(args.seed); values = []
    for _ in range(args.resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        selected = [row for key in sampled for row in groups[str(key)]]
        target = [int(float(row[args.target_column])) for row in selected]
        aligned = [float(row[args.aligned_column]) for row in selected]
        shuffled = [float(row[args.shuffled_column]) for row in selected]
        values.append(tie_safe_average_precision(target, aligned) - tie_safe_average_precision(target, shuffled))
    write_csv(args.out, [{"metric": "aligned_minus_shuffled_AUPRC", "bootstrap_mean": float(np.mean(values)), "ci_lower_95": float(np.quantile(values, 0.025)), "ci_upper_95": float(np.quantile(values, 0.975)), "resamples": args.resamples, "cluster_unit": "question_key", "test_access_count": 0}])
    print({"status": "PASS", "rows": len(rows), "resamples": args.resamples, "test_access_count": 0})


if __name__ == "__main__":
    main()
