"""Collect available Exp43 GroupCV runs without reading dev or test."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from thesis_exp.exp43_rubimor.common import ROOT, RUN_ROOT, SEEDS, VARIANTS, mean_std, prediction_metrics, read_jsonl, write_csv, write_json
from thesis_exp.exp43_rubimor.train_exp43_groupcv import run_identity

METRICS = ("MAE", "QWK", "Exact_Match", "Kendall_tau", "Signed_Bias", "abs_Signed_Bias", "Bin_Agreement", "expected_score_MAE", "human_CE", "human_Brier", "human_RPS", "low_to_high_rate", "high_to_low_rate", "label1_recall", "label2_recall", "label3_recall", "label4_recall", "label5_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-fingerprints", action="store_true")
    return parser.parse_args()


def expected_fingerprint(args: argparse.Namespace, variant: str, seed: int, fold: int) -> str:
    identity_args = SimpleNamespace(
        variant=variant, mode="groupcv", fold=fold, seed=seed,
        epochs=10, learning_rate=2e-5, weight_decay=.01, warmup_ratio=.05,
        batch_size=4, eval_batch_size=4, gradient_accumulation=32,
        max_length=2048, max_train_rows=None, max_eval_rows=None,
        max_updates=None, out_dir=args.out_dir,
    )
    return str(run_identity(identity_args)["run_fingerprint"])


def paths(run_root: Path, variant: str, seed: int, fold: int) -> tuple[Path, Path]:
    root = run_root / "groupcv" / variant / f"seed_{seed}" / f"fold_{fold}"
    return root / "run_summary.json", root / "heldout_predictions.jsonl"


def pair_accuracy(rows: list[dict], fold: int, out_dir: Path) -> tuple[int, float]:
    by_id = {row["sample_id"]: float(row["pred_score_expected"]) for row in rows}
    with (out_dir / "private/data/exp43_groupcv_fold_assignment.csv").open("r", encoding="utf-8", newline="") as handle:
        fold_by_id = {row["sample_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    pairs = read_jsonl(out_dir / "private/pairs/exp43_train_pairs.jsonl")
    heldout = [pair for pair in pairs if fold_by_id[pair["positive_id"]] == fold and fold_by_id[pair["negative_id"]] == fold]
    valid = [pair for pair in heldout if pair["positive_id"] in by_id and pair["negative_id"] in by_id]
    correct = sum(by_id[pair["positive_id"]] > by_id[pair["negative_id"]] for pair in valid)
    return len(valid), correct / len(valid) if valid else float("nan")


def main() -> None:
    args = parse_args()
    per_seed, run_completion, label_rows, pair_rows, residual_rows, strata_rows = [], [], [], [], [], []
    prediction_cache: dict[tuple[str, int], list[dict]] = {}
    for variant in args.variants:
        for seed in args.seeds:
            combined = []
            complete = True
            for fold in range(5):
                summary_path, prediction_path = paths(args.run_root, variant, seed, fold)
                exists = summary_path.exists() and prediction_path.exists()
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                expected = expected_fingerprint(args, variant, seed, fold) if args.require_fingerprints else None
                fingerprint_match = summary.get("run_fingerprint") == expected if args.require_fingerprints else True
                valid = exists and summary.get("status") == "COMPLETED" and summary.get("fixed_final_epoch") == 10 and summary.get("question_key_overlap") == 0 and summary.get("nan_count") == 0 and summary.get("oom_count") == 0 and summary.get("test_access_count") == 0 and fingerprint_match
                run_completion.append({"variant": variant, "seed": seed, "fold": fold, "exists": exists, "valid": valid, "status": summary.get("status", "MISSING"), "epochs": summary.get("fixed_final_epoch"), "fingerprint_required": args.require_fingerprints, "fingerprint_match": fingerprint_match, "run_fingerprint": summary.get("run_fingerprint", "MISSING"), "expected_fingerprint": expected or "NOT_REQUIRED", "oom_count": summary.get("oom_count", 0), "nan_count": summary.get("nan_count", 0), "test_access_count": summary.get("test_access_count", 0)})
                if not valid:
                    complete = False
                    continue
                fold_rows = read_jsonl(prediction_path)
                combined.extend(fold_rows)
                n_pairs, accuracy = pair_accuracy(fold_rows, fold, args.out_dir)
                pair_rows.append({"variant": variant, "seed": seed, "fold": fold, "heldout_pairs": n_pairs, "pair_accuracy": accuracy})
                for row in summary.get("metric_residual_norms", []):
                    residual_rows.append({"variant": variant, "seed": seed, "fold": fold, **row})
            if not complete:
                if args.require_complete:
                    raise RuntimeError(f"Missing or invalid formal runs for {variant} seed {seed}")
                continue
            if len(combined) != 2654 or len({row["sample_id"] for row in combined}) != 2654:
                raise RuntimeError(f"OOF coverage invalid for {variant} seed {seed}: {len(combined)}")
            metrics = prediction_metrics(combined)
            prediction_cache[(variant, seed)] = combined
            per_seed.append({"variant": variant, "seed": seed, **metrics})
            label_rows.append({"variant": variant, "seed": seed, **{key: value for key, value in metrics.items() if key.startswith("label") or key.startswith("low_") or key.startswith("high_") or key.startswith("pred_count")}})
            for field in ("metric", "subject", "language", "scenario", "education_level"):
                for value in sorted({str(row.get(field) or "unknown") for row in combined}):
                    subset = [row for row in combined if str(row.get(field) or "unknown") == value]
                    strata_rows.append({"variant": variant, "seed": seed, "stratum_type": field, "stratum_value": value, **prediction_metrics(subset)})
    multiseed = []
    for variant in args.variants:
        values = [row for row in per_seed if row["variant"] == variant]
        if not values:
            continue
        summary: dict[str, Any] = {"variant": variant, "seeds": len(values), "oof_rows_per_seed": 2654}
        for metric in METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = mean_std([float(row[metric]) for row in values])
        multiseed.append(summary)
    differences = []
    comparisons = (("E3", "E0"), ("E4", "E3"), ("E5", "E4"), ("E6", "E5"), ("E6", "E6N"), ("E6", "E0"), ("E6", "E3"))
    by_key = {(row["variant"], int(row["seed"])): row for row in per_seed}
    for left, right in comparisons:
        for seed in args.seeds:
            if (left, seed) not in by_key or (right, seed) not in by_key:
                continue
            differences.append({"comparison": f"{left}_vs_{right}", "left": left, "right": right, "seed": seed, **{f"delta_{metric}": float(by_key[(left, seed)][metric]) - float(by_key[(right, seed)][metric]) for metric in METRICS}})
    write_csv(args.out_dir / "tables/exp43_run_completion.csv", run_completion)
    write_csv(args.out_dir / "tables/exp43_groupcv_metrics_by_seed.csv", per_seed)
    write_csv(args.out_dir / "tables/exp43_groupcv_multiseed_summary.csv", multiseed)
    write_csv(args.out_dir / "tables/exp43_groupcv_pairwise_differences.csv", differences)
    write_csv(args.out_dir / "tables/exp43_groupcv_label_recall.csv", label_rows)
    write_csv(args.out_dir / "tables/exp43_groupcv_human_distribution_metrics.csv", [{"variant": row["variant"], "seed": row["seed"], **{key: row[key] for key in ("human_CE", "human_Brier", "human_RPS", "expected_score_MAE")}} for row in per_seed])
    write_csv(args.out_dir / "tables/exp43_groupcv_pair_metrics.csv", pair_rows)
    write_csv(args.out_dir / "tables/exp43_groupcv_stratified_metrics.csv", strata_rows)
    write_csv(args.out_dir / "tables/exp43_metric_residual_norms.csv", residual_rows)
    write_json(args.out_dir / "hashes/exp43_training_config_hashes.json", {"available_complete_variant_seeds": sorted(f"{key[0]}:{key[1]}" for key in prediction_cache), "completed_runs": sum(bool(row["valid"]) for row in run_completion), "oom_count": sum(int(row["oom_count"]) for row in run_completion), "nan_count": sum(int(row["nan_count"]) for row in run_completion), "test_access_count": 0})
    print(json.dumps({"status": "COLLECTED", "variant_seeds": len(per_seed), "completed_runs": sum(bool(row["valid"]) for row in run_completion), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
