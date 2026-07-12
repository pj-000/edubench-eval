"""Question-key cluster bootstrap for locked Exp36A seed42 comparisons."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import ROOT, metrics, read_jsonl, write_csv


METRICS = ("MAE_argmax", "Exact_Match", "QWK", "low_to_high_rate", "label2_recall")


def bootstrap(left: list[dict[str, Any]], right: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    left_by = {row["sample_id"]: row for row in left}
    right_by = {row["sample_id"]: row for row in right}
    if set(left_by) != set(right_by):
        raise ValueError("Bootstrap predictions are not sample-aligned")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sid, row in left_by.items():
        grouped[str(row["question_key"])].append(sid)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in METRICS}
    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        ids = [sid for key in sampled for sid in grouped[str(key)]]
        lm = metrics([left_by[sid] for sid in ids])
        rm = metrics([right_by[sid] for sid in ids])
        for name in METRICS:
            values[name].append(float(lm[name]) - float(rm[name]))
    rows = []
    for name, samples in values.items():
        arr = np.asarray(samples, dtype=float)
        finite = arr[np.isfinite(arr)]
        rows.append({
            "metric": name, "left_minus_right_mean": float(np.mean(finite)),
            "ci_lower_95": float(np.quantile(finite, 0.025)), "ci_upper_95": float(np.quantile(finite, 0.975)),
            "resamples": resamples, "cluster_unit": "question_key",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    left = read_jsonl(args.out_dir / "private/dev_predictions" / args.left / "seed_42.jsonl")
    right = read_jsonl(args.out_dir / "private/dev_predictions" / args.right / "seed_42.jsonl")
    rows = bootstrap(left, right, args.resamples, args.seed)
    for row in rows:
        row.update({"left_variant": args.left, "right_variant": args.right})
    output = args.output or args.out_dir / "tables/exp36a_seed42_question_key_bootstrap_ci.csv"
    write_csv(output, rows)
    print({"rows": len(rows), "output": str(output)})


if __name__ == "__main__":
    main()

