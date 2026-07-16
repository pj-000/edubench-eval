"""Audit human-score patterns behind the aggregate hard label 2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from thesis_exp.exp47_label2_identifiability.common import (
    ROOT,
    ensure_dirs,
    label2_flags,
    load_canonical_rows,
    quantile,
    sanitize_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    label2 = []
    for row in load_canonical_rows():
        if int(row["gold_label_5"]) == 2:
            label2.append({**row, **label2_flags(row)})
    if not label2:
        raise RuntimeError("No hard label-2 rows found")

    pattern_counts = Counter((row["human_score_pattern"], row["label2_subtype"]) for row in label2)
    pattern_rows = [
        {
            "human_score_pattern": pattern,
            "label2_subtype": subtype,
            "count": count,
            "share": count / len(label2),
        }
        for (pattern, subtype), count in sorted(pattern_counts.items())
    ]
    write_csv(args.out_dir / "tables/exp47_label2_human_pattern_distribution.csv", pattern_rows)

    groups: dict[str, list[dict]] = defaultdict(list)
    groups["all_hard_label2"] = label2
    for row in label2:
        groups[row["label2_subtype"]].append(row)
    entropy_rows = []
    for subtype, rows in groups.items():
        entropy = [float(row["human_entropy"]) for row in rows]
        ranges = [int(row["human_score_range"]) for row in rows]
        means = [float(row["expected_human_score"]) for row in rows]
        variances = [float(np.var(row["human_scores"])) for row in rows]
        entropy_rows.append(
            {
                "label2_subtype": subtype,
                "n": len(rows),
                "entropy_mean": float(np.mean(entropy)),
                "entropy_median": float(np.median(entropy)),
                "entropy_p90": quantile(entropy, 0.90),
                "score_range_mean": float(np.mean(ranges)),
                "score_range_median": float(np.median(ranges)),
                "score_range_max": max(ranges),
                "human_mean_mean": float(np.mean(means)),
                "human_variance_mean": float(np.mean(variances)),
            }
        )
    write_csv(args.out_dir / "tables/exp47_label2_entropy_summary.csv", sanitize_rows(entropy_rows))

    subtype_counts = Counter(row["label2_subtype"] for row in label2)
    strict = subtype_counts["strict_222"]
    stable_majority = subtype_counts["stable_majority_non_strict"]
    ambiguous = subtype_counts["ambiguous"]
    summary = [
        {"group": "all_hard_label2", "count": len(label2), "share_of_all_label2": 1.0},
        {"group": "strict_label2", "count": strict, "share_of_all_label2": strict / len(label2)},
        {"group": "stable_majority_non_strict", "count": stable_majority, "share_of_all_label2": stable_majority / len(label2)},
        {"group": "stable_label2_total", "count": strict + stable_majority, "share_of_all_label2": (strict + stable_majority) / len(label2)},
        {"group": "ambiguous_label2", "count": ambiguous, "share_of_all_label2": ambiguous / len(label2)},
        {"group": "no_annotator_two", "count": sum(row["annotator_two_count"] == 0 for row in label2), "share_of_all_label2": sum(row["annotator_two_count"] == 0 for row in label2) / len(label2)},
        {"group": "one_two_only", "count": sum(row["annotator_two_count"] == 1 for row in label2), "share_of_all_label2": sum(row["annotator_two_count"] == 1 for row in label2) / len(label2)},
        {"group": "score_range_ge_3", "count": sum(int(row["human_score_range"]) >= 3 for row in label2), "share_of_all_label2": sum(int(row["human_score_range"]) >= 3 for row in label2) / len(label2)},
    ]
    write_csv(args.out_dir / "tables/exp47_label2_stable_vs_ambiguous.csv", summary)
    print(json.dumps({"status": "HUMAN_PATTERNS_AUDITED", "label2": len(label2), "stable": strict + stable_majority, "ambiguous": ambiguous, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
