"""Collect locked Exp42A multiseed OOF metrics without reading dev/test."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp42_rubidist.common import (  # noqa: E402
    COMPARISONS,
    METRICS,
    ROOT,
    RUN_ROOT,
    SEEDS,
    VARIANTS,
    all_metrics,
    entropy_band,
    load_oof_predictions,
    mean_std,
    run_summary_path,
    stable_hash,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    run_audit = []
    predictions: dict[tuple[str, int], list[dict]] = {}
    for variant in args.variants:
        for seed in args.seeds:
            for fold in range(5):
                path = run_summary_path(args.run_root, variant, seed, fold)
                if not path.exists():
                    raise FileNotFoundError(path)
                summary = json.loads(path.read_text(encoding="utf-8"))
                valid = (
                    summary.get("status") == "COMPLETED"
                    and summary.get("fixed_final_epoch") == 10
                    and summary.get("question_key_overlap") == 0
                    and summary.get("nan_count") == 0
                    and summary.get("oom_count") == 0
                    and summary.get("dev_access_count") == 0
                    and summary.get("test_access_count") == 0
                )
                run_audit.append({"variant": variant, "seed": seed, "fold": fold, "valid": valid, **summary})
                if not valid:
                    raise RuntimeError(f"Invalid formal run summary: {path}")
            rows = load_oof_predictions(args.run_root, variant, seed)
            predictions[(variant, seed)] = rows
            summaries.append({"variant": variant, "seed": seed, **all_metrics(rows)})

    write_csv(args.out_dir / "tables/exp42a_selected_metrics_by_seed.csv", summaries)
    summary_by_key = {(row["variant"], int(row["seed"])): row for row in summaries}
    multiseed = []
    for variant in args.variants:
        row = {"variant": variant, "seeds": len(args.seeds), "oof_rows_per_seed": 2654}
        for metric in METRICS:
            mean, std = mean_std([float(summary_by_key[(variant, seed)][metric]) for seed in args.seeds])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        multiseed.append(row)
    write_csv(args.out_dir / "tables/exp42a_multiseed_summary.csv", multiseed)

    differences = []
    for left, right, comparison in COMPARISONS:
        for seed in args.seeds:
            left_row, right_row = summary_by_key[(left, seed)], summary_by_key[(right, seed)]
            differences.append(
                {
                    "comparison": comparison,
                    "left_variant": left,
                    "right_variant": right,
                    "seed": seed,
                    **{f"delta_{metric}": float(left_row[metric]) - float(right_row[metric]) for metric in METRICS},
                }
            )
    write_csv(args.out_dir / "tables/exp42a_pairwise_differences.csv", differences)
    write_csv(
        args.out_dir / "tables/exp42a_label_recall.csv",
        [
            {
                "variant": row["variant"],
                "seed": row["seed"],
                **{f"label{label}_recall": row[f"label{label}_recall"] for label in range(1, 6)},
                "low_to_high_count": row["low_to_high_count"],
                "low_to_high_rate": row["low_to_high_rate"],
                "high_to_low_count": row["high_to_low_count"],
                "high_to_low_rate": row["high_to_low_rate"],
                **{f"pred_count_{label}": row[f"pred_count_{label}"] for label in range(1, 6)},
            }
            for row in summaries
        ],
    )
    write_csv(
        args.out_dir / "tables/exp42a_human_distribution_metrics.csv",
        [
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "human_CE": row["human_CE"],
                "human_Brier": row["human_Brier"],
                "human_RPS": row["human_RPS"],
                "expected_score_MAE": row["expected_score_MAE"],
            }
            for row in summaries
        ],
    )

    stratified = []
    for (variant, seed), rows in predictions.items():
        strata = {
            "language": sorted({str(row.get("language") or "unknown") for row in rows}),
            "metric_group": sorted({str(row.get("metric_group") or "unknown") for row in rows}),
            "original_label": [str(label) for label in range(1, 6)],
            "human_entropy_band": ["zero", "low", "high"],
            "subject": sorted({str(row.get("subject") or "unknown") for row in rows}),
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
                    stratified.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "stratum_type": field,
                            "stratum_value": value,
                            **all_metrics(subset),
                        }
                    )
    write_csv(args.out_dir / "tables/exp42a_stratified_metrics.csv", stratified)
    write_json(
        args.out_dir / "hashes/exp42a_training_config_hashes.json",
        {
            "training_config_sha256": stable_hash(json.loads((args.out_dir / "configs/exp42a_training_config.json").read_text(encoding="utf-8"))),
            "formal_runs": len(run_audit),
            "completed_runs": sum(bool(row["valid"]) for row in run_audit),
            "variants": list(args.variants),
            "seeds": list(args.seeds),
            "folds": 5,
            "fixed_final_epoch": 10,
            "oom_count": sum(int(row.get("oom_count", 0)) for row in run_audit),
            "nan_count": sum(int(row.get("nan_count", 0)) for row in run_audit),
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "COLLECTED",
                "formal_runs": len(run_audit),
                "completed_runs": len(run_audit),
                "oof_rows": {f"{variant}/seed{seed}": len(rows) for (variant, seed), rows in predictions.items()},
                "dev_access_count": 0,
                "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
