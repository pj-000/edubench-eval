"""Collect Exp41A OOF predictions and locked aggregate/stratified metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    FORMAL_VARIANTS, ROOT, human_distribution_metrics, prediction_metrics, read_jsonl, write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--variants", nargs="+", choices=FORMAL_VARIANTS, default=list(FORMAL_VARIANTS))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_variant(out_dir: Path, variant: str, allow_incomplete: bool) -> list[dict]:
    predictions = []
    for fold in range(5):
        path = out_dir / f"private/groupcv_predictions/{variant}/fold_{fold}/heldout_predictions.jsonl"
        if not path.exists():
            if allow_incomplete:
                continue
            raise FileNotFoundError(path)
        predictions.extend(read_jsonl(path))
    ids = [row["sample_id"] for row in predictions]
    if not allow_incomplete and (len(predictions) != 2654 or len(set(ids)) != 2654):
        raise ValueError(f"{variant} must have 2654 unique OOF predictions")
    return predictions


def entropy_band(value: float) -> str:
    if value < 1e-9:
        return "zero"
    if value <= 0.64:
        return "low"
    return "high"


def main() -> None:
    args = parse_args()
    by_variant = {variant: load_variant(args.out_dir, variant, args.allow_incomplete) for variant in args.variants}
    metrics = [{"variant": variant, **prediction_metrics(rows)} for variant, rows in by_variant.items()]
    write_csv(args.out_dir / "tables/exp41a_groupcv_metrics.csv", metrics)
    write_csv(args.out_dir / "tables/exp41a_groupcv_label_recall.csv", [
        {"variant": row["variant"], **{f"label{label}_recall": row[f"label{label}_recall"] for label in range(1, 6)}}
        for row in metrics
    ])
    write_csv(args.out_dir / "tables/exp41a_groupcv_human_distribution_metrics.csv", [
        {"variant": variant, "n": len(rows), **human_distribution_metrics(rows)} for variant, rows in by_variant.items()
    ])
    stratified = []
    for variant, rows in by_variant.items():
        strata = {
            "language": sorted({str(row.get("language") or "unknown") for row in rows}),
            "metric_group": sorted({str(row.get("metric_group") or "unknown") for row in rows}),
            "original_label": [str(label) for label in range(1, 6)],
            "human_entropy_band": ["zero", "low", "high"],
        }
        for field, values in strata.items():
            for value in values:
                if field == "original_label":
                    subset = [row for row in rows if int(row["gold_label_5"]) == int(value)]
                elif field == "human_entropy_band":
                    subset = [row for row in rows if entropy_band(float(row["human_entropy"])) == value]
                else:
                    subset = [row for row in rows if str(row.get(field) or "unknown") == value]
                if subset:
                    stratified.append({"variant": variant, "stratum_type": field, "stratum_value": value,
                                       **prediction_metrics(subset), **human_distribution_metrics(subset)})
    write_csv(args.out_dir / "tables/exp41a_groupcv_stratified_metrics.csv", stratified)
    by_name = {row["variant"]: row for row in metrics}
    differences = []
    if "v3_rubric_bridge" in by_name:
        left = by_name["v3_rubric_bridge"]
        for variant in args.variants:
            if variant == "v3_rubric_bridge":
                continue
            right = by_name[variant]
            differences.append({
                "left_variant": "v3_rubric_bridge", "right_variant": variant,
                **{f"delta_{key}": float(left[key]) - float(right[key]) for key in (
                    "MAE", "expected_score_MAE", "Signed_Bias", "abs_Signed_Bias", "Exact_Match",
                    "Kendall_tau", "QWK", "Bin_Agreement", "low_to_high_rate", "high_to_low_rate",
                    "label2_recall", "label5_recall",
                )},
            })
    write_csv(args.out_dir / "tables/exp41a_groupcv_pairwise_differences.csv", differences)
    print(json.dumps({"status": "COLLECTED", "rows": {variant: len(rows) for variant, rows in by_variant.items()},
                      "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
