"""Collect Exp40A OOF predictions and locked aggregate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    ROOT,
    VARIANTS,
    human_distribution_metrics,
    prediction_metrics,
    read_jsonl,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_variant(out_dir: Path, variant: str, allow_incomplete: bool) -> tuple[list[dict], list[dict]]:
    predictions, summaries = [], []
    for fold in range(5):
        root = out_dir / f"private/groupcv_predictions/{variant}/fold_{fold}"
        prediction_path, summary_path = root / "heldout_predictions.jsonl", root / "run_summary.json"
        if not prediction_path.exists() or not summary_path.exists():
            if allow_incomplete:
                continue
            raise FileNotFoundError(prediction_path if not prediction_path.exists() else summary_path)
        predictions.extend(read_jsonl(prediction_path))
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    ids = [row["sample_id"] for row in predictions]
    if not allow_incomplete and (len(predictions) != 2654 or len(set(ids)) != 2654 or len(summaries) != 5):
        raise ValueError(f"{variant} must have 2654 unique OOF predictions and five summaries")
    return predictions, summaries


def main() -> None:
    args = parse_args()
    loaded = {variant: load_variant(args.out_dir, variant, args.allow_incomplete) for variant in args.variants}
    by_variant = {variant: value[0] for variant, value in loaded.items()}
    metrics = [{"variant": variant, **prediction_metrics(rows)} for variant, rows in by_variant.items()]
    write_csv(args.out_dir / "tables/exp40a_groupcv_metrics.csv", metrics)
    write_csv(
        args.out_dir / "tables/exp40a_groupcv_label_recall.csv",
        [
            {"variant": row["variant"], **{f"label{label}_recall": row[f"label{label}_recall"] for label in range(1, 6)}}
            for row in metrics
        ],
    )
    write_csv(
        args.out_dir / "tables/exp40a_groupcv_human_distribution_metrics.csv",
        [{"variant": variant, "n": len(rows), **human_distribution_metrics(rows)} for variant, rows in by_variant.items()],
    )
    diagnostic_rows = []
    for variant, (_, summaries) in loaded.items():
        correct = sum(int(row.get("heldout_pair_diagnostic_correct", 0)) for row in summaries)
        total = sum(int(row.get("heldout_pair_diagnostic_total", 0)) for row in summaries)
        diagnostic_rows.append(
            {
                "variant": variant, "correct": correct, "pairs": total,
                "source_gt_counterfactual_accuracy": correct / total if total else "",
                "diagnostic_only": True,
            }
        )
    write_csv(args.out_dir / "tables/exp40a_groupcv_pair_ranking_diagnostic.csv", diagnostic_rows)
    by_name = {row["variant"]: row for row in metrics}
    differences = []
    if "v3_edupair_cf" in by_name:
        left = by_name["v3_edupair_cf"]
        for variant in args.variants:
            if variant == "v3_edupair_cf":
                continue
            right = by_name[variant]
            differences.append(
                {
                    "left_variant": "v3_edupair_cf", "right_variant": variant,
                    **{
                        f"delta_{key}": float(left[key]) - float(right[key])
                        for key in (
                            "MAE", "expected_score_MAE", "Signed_Bias", "Exact_Match", "Kendall_tau", "QWK",
                            "Bin_Agreement", "low_to_high_rate", "high_to_low_rate", "label2_recall", "label5_recall",
                        )
                    },
                }
            )
    write_csv(args.out_dir / "tables/exp40a_groupcv_pairwise_differences.csv", differences)
    print(json.dumps({"status": "COLLECTED", "rows": {key: len(value) for key, value in by_variant.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
