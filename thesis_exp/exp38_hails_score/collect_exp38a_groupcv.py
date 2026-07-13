"""Collect Exp38A held-out GroupCV predictions and aggregate locked metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp38_hails_score.common import ROOT, VARIANTS, prediction_metrics, read_jsonl, write_csv

FORMAL_VARIANTS = ("v0h_human_empirical", "v2_qwen_interval_only", "v3_naive_interval", "v4_hails", "v5_shuffled_interval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(FORMAL_VARIANTS))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_variant(out_dir: Path, variant: str, allow_incomplete: bool) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        path = out_dir / f"private/groupcv_predictions/{variant}/fold_{fold}/heldout_predictions.jsonl"
        if not path.exists():
            if allow_incomplete:
                continue
            raise FileNotFoundError(f"Missing GroupCV prediction: {path}")
        rows.extend(read_jsonl(path))
    ids = [row["sample_id"] for row in rows]
    if not allow_incomplete and (len(rows) != 2654 or len(set(ids)) != 2654):
        raise ValueError(f"{variant} must have 2654 unique OOF predictions, found {len(rows)}/{len(set(ids))}")
    return rows


def main() -> None:
    args = parse_args()
    by_variant = {variant: load_variant(args.out_dir, variant, args.allow_incomplete) for variant in args.variants}
    metric_rows = [{"variant": variant, **prediction_metrics(rows)} for variant, rows in by_variant.items()]
    write_csv(args.out_dir / "tables/exp38a_groupcv_metrics.csv", metric_rows)
    recalls = [{"variant": row["variant"], **{f"label{label}_recall": row[f"label{label}_recall"] for label in range(1, 6)}} for row in metric_rows]
    write_csv(args.out_dir / "tables/exp38a_groupcv_label_recall.csv", recalls)
    stratified = []
    for variant, rows in by_variant.items():
        dimensions = {
            "gold_label": lambda row: str(row["gold_label_5"]),
            "language": lambda row: str(row.get("language") or "unknown"),
            "metric_group": lambda row: str(row.get("metric_group") or "unknown"),
            "subject": lambda row: str(row.get("subject_canonical") or "unknown"),
            "gold_region": lambda row: "low" if int(row["gold_label_5"]) <= 2 else ("high" if int(row["gold_label_5"]) >= 4 else "mid"),
        }
        for dimension, getter in dimensions.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[getter(row)].append(row)
            for value, subset in sorted(grouped.items()):
                stratified.append({"variant": variant, "dimension": dimension, "value": value, **prediction_metrics(subset)})
    write_csv(args.out_dir / "tables/exp38a_groupcv_stratified_metrics.csv", stratified)
    baseline = next((row for row in metric_rows if row["variant"] == "v0h_human_empirical"), None)
    differences = []
    if baseline and "v4_hails" in by_variant:
        hails = next(row for row in metric_rows if row["variant"] == "v4_hails")
        comparison_names = [name for name in ("v0h_human_empirical", "v2_qwen_interval_only", "v3_naive_interval", "v5_shuffled_interval") if name in by_variant]
        by_name = {row["variant"]: row for row in metric_rows}
        for name in comparison_names:
            row = by_name[name]
            differences.append({
                "left_variant": hails["variant"], "right_variant": row["variant"],
                **{f"delta_{key}": float(hails[key]) - float(row[key]) for key in ("MAE", "expected_score_MAE", "Signed_Bias", "Exact_Match", "Kendall_tau", "QWK", "Bin_Agreement", "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall")},
            })
    write_csv(args.out_dir / "tables/exp38a_groupcv_pairwise_differences.csv", differences)
    print(json.dumps({"status": "COLLECTED", "variants": list(by_variant), "rows": {key: len(value) for key, value in by_variant.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
