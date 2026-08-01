#!/usr/bin/env python3
"""Export immutable HMSA paper data and development-set subgroup diagnostics.

This script never reads the frozen raw test split. Test statistics are exported
only from the completed canonical report and per-checkpoint metric files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


SEEDS = (42, 43, 44)
ARMS = ("b0", "exp51")
PRIMARY_METRICS = (
    "MAE_human_mean",
    "Exact_rounded",
    "Kendall_human_mean",
    "Bias_human_mean",
    "BinAgreement_paper_3way",
    "L2H_count",
    "H2L_count",
    "QWK_rounded",
    "ECE_rounded",
    "NLL_rounded",
    "Brier_rounded",
)
MECHANISM_METRICS = (
    "MAE_human_mean",
    "Exact_rounded",
    "Kendall_human_mean",
    "Bias_human_mean",
    "QWK_rounded",
    "L2H_count",
)
MECHANISM_ARMS = (
    ("hard_only", "Hard-only", "thesis_exp/outputs/exp49_cphce/runs/b0_hard_ce"),
    (
        "within_label_shuffled_soft",
        "Shuffled-soft",
        "thesis_exp/outputs/exp55_within_label_shuffle/runs/within_label_shuffled_soft",
    ),
    ("hmsa_true_soft", "HMSA", "thesis_exp/outputs/exp51_hmsa/runs/hmsa_lambda1"),
)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_repo = script.parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--output-dir", type=Path, default=script.parents[1] / "data")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwk(true: list[int], predicted: list[int]) -> float:
    if len(true) != len(predicted) or not true:
        raise ValueError("QWK inputs must be non-empty and aligned")
    classes = range(1, 6)
    observed = {(i, j): 0.0 for i in classes for j in classes}
    true_counts = Counter(true)
    pred_counts = Counter(predicted)
    for gold, guess in zip(true, predicted):
        observed[(gold, guess)] += 1.0
    numerator = 0.0
    denominator = 0.0
    n = float(len(true))
    for i in classes:
        for j in classes:
            weight = ((i - j) ** 2) / 16.0
            numerator += weight * observed[(i, j)]
            denominator += weight * true_counts[i] * pred_counts[j] / n
    return 1.0 - numerator / denominator if denominator else 0.0


def classify_raters(row: dict[str, Any]) -> str:
    ratings = [int(row[f"human_{index}_5"]) for index in (1, 2, 3)]
    counts = sorted(Counter(ratings).values())
    distinct = sorted(set(ratings))
    if len(distinct) == 1:
        return "unanimous"
    if len(distinct) == 2 and counts == [1, 2] and distinct[1] - distinct[0] == 1:
        return "adjacent_2_to_1"
    return "other"


def prediction_path(repo: Path, arm: str, seed: int) -> Path:
    if arm == "b0":
        root = repo / f"thesis_exp/outputs/exp49_cphce/runs/b0_hard_ce/seed_{seed}"
    else:
        root = repo / f"thesis_exp/outputs/exp51_hmsa/runs/hmsa_lambda1/seed_{seed}"
    candidates = (
        root / "predictions_dev_best.jsonl",
        root / "predictions/predictions_dev_best.jsonl",
        root / "predictions/predictions_dev.jsonl",
        root / "predictions_dev.jsonl",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No selected dev predictions found under {root}")


def group_metrics(rows: list[dict[str, Any]], predictions: dict[str, int]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot evaluate an empty subgroup")
    gold = [int(row["label_5"]) for row in rows]
    human_mean = [float(row["human_mean_5"]) for row in rows]
    guessed = [int(predictions[str(row["record_id"])]) for row in rows]
    return {
        "n": float(len(rows)),
        "MAE_human_mean": mean(abs(guess - reference) for guess, reference in zip(guessed, human_mean)),
        "Exact_rounded": mean(guess == reference for guess, reference in zip(guessed, gold)),
        "Bias_human_mean": mean(guess - reference for guess, reference in zip(guessed, human_mean)),
        "QWK_rounded": qwk(gold, guessed),
    }


def aggregate_dev_subgroups(repo: Path, output_dir: Path) -> None:
    dev_rows = load_jsonl(repo / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl")
    groups = {"all": dev_rows, "unanimous": [], "adjacent_2_to_1": [], "other": []}
    for row in dev_rows:
        groups[classify_raters(row)].append(row)
    if groups["other"]:
        raise ValueError(f"Unexpected dev disagreement patterns: {len(groups['other'])}")

    by_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        arm_predictions: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            path = prediction_path(repo, arm, seed)
            prediction_rows = load_jsonl(path)
            arm_predictions[arm] = {
                str(row["record_id"]): int(row["pred_label_5"]) for row in prediction_rows
            }
            if len(arm_predictions[arm]) != len(dev_rows):
                raise ValueError(f"Unexpected dev prediction count for {arm} seed {seed}")
        for group in ("all", "unanimous", "adjacent_2_to_1"):
            b0 = group_metrics(groups[group], arm_predictions["b0"])
            exp51 = group_metrics(groups[group], arm_predictions["exp51"])
            row: dict[str, Any] = {"seed": seed, "group": group, "n": int(b0["n"])}
            for metric in ("MAE_human_mean", "Exact_rounded", "Bias_human_mean", "QWK_rounded"):
                row[f"b0_{metric}"] = b0[metric]
                row[f"exp51_{metric}"] = exp51[metric]
                row[f"delta_{metric}"] = exp51[metric] - b0[metric]
            by_seed.append(row)

    by_seed_fields = list(by_seed[0])
    write_csv(output_dir / "dev_disagreement_by_seed.csv", by_seed, by_seed_fields)

    summary_rows: list[dict[str, Any]] = []
    for group in ("all", "unanimous", "adjacent_2_to_1"):
        selected = [row for row in by_seed if row["group"] == group]
        summary: dict[str, Any] = {"group": group, "n": selected[0]["n"]}
        for key in by_seed_fields:
            if key in {"seed", "group", "n"}:
                continue
            values = [float(row[key]) for row in selected]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_sd"] = stdev(values)
        summary_rows.append(summary)
    write_csv(output_dir / "dev_disagreement_summary.csv", summary_rows, list(summary_rows[0]))

    split_rows = []
    for split in ("train", "dev"):
        rows = load_jsonl(repo / f"thesis_exp/data/splits/paper_like_triple_seed42/{split}.jsonl")
        counts = Counter(classify_raters(row) for row in rows)
        labels = Counter(int(row["label_5"]) for row in rows)
        split_rows.append(
            {
                "split": split,
                "n": len(rows),
                "unanimous": counts["unanimous"],
                "adjacent_2_to_1": counts["adjacent_2_to_1"],
                "label_1": labels[1],
                "label_2": labels[2],
                "label_3": labels[3],
                "label_4": labels[4],
                "label_5": labels[5],
            }
        )
    # Test label totals come from the completed frozen metrics, not the raw test split.
    test_metric = load_json(
        repo / "thesis_exp/outputs/exp51_hmsa/final_test/b0/seed_42/test_metrics.json"
    )
    split_rows.append(
        {
            "split": "test",
            "n": int(test_metric["n"]),
            "unanimous": "not_reopened",
            "adjacent_2_to_1": "not_reopened",
            **{f"label_{label}": int(test_metric[f"Recall_{label}_total"]) for label in range(1, 6)},
        }
    )
    write_csv(output_dir / "dataset_summary.csv", split_rows, list(split_rows[0]))


def export_dev_mechanism_control(
    repo: Path, output_dir: Path
) -> tuple[tuple[Path, Path], dict[str, Any]]:
    comparison_path = (
        repo
        / "thesis_exp/outputs/exp55_within_label_shuffle/decision/dev_mechanism_comparison.json"
    )
    shuffle_audit_path = (
        repo / "thesis_exp/outputs/exp55_within_label_shuffle/audit/shuffle_audit.json"
    )
    comparison = load_json(comparison_path)
    shuffle_audit = load_json(shuffle_audit_path)

    expected_seeds = list(SEEDS)
    if comparison["seeds"] != expected_seeds:
        raise ValueError(f"Unexpected mechanism-control seeds: {comparison['seeds']}")
    if comparison["test_access_count"] != 0 or shuffle_audit["test_access_count"] != 0:
        raise ValueError("The development-only mechanism control reports test access")
    if shuffle_audit["checks"].get("all_label_multisets_preserved") is not True:
        raise ValueError("The shuffled-soft target multiset audit did not pass")

    shutil.copyfile(comparison_path, output_dir / "dev_mechanism_comparison.json")

    summary_rows: list[dict[str, Any]] = []
    by_seed_rows: list[dict[str, Any]] = []
    for metric in MECHANISM_METRICS:
        for source_key, paper_name, _ in MECHANISM_ARMS:
            source_metrics = comparison[source_key]
            values = source_metrics[metric]["values"]
            if len(values) != len(SEEDS):
                raise ValueError(f"Unexpected value count for {source_key}/{metric}")
            summary_rows.append(
                {
                    "method": paper_name,
                    "metric": metric,
                    "mean": source_metrics[metric]["mean"],
                    "sample_sd": source_metrics[metric]["sample_sd"],
                    **{
                        f"seed{seed}": int(value) if metric == "L2H_count" else value
                        for seed, value in zip(SEEDS, values)
                    },
                }
            )

    for _, paper_name, run_root in MECHANISM_ARMS:
        for seed in SEEDS:
            selected_path = repo / run_root / f"seed_{seed}/selected_dev_metrics.json"
            selected = load_json(selected_path)
            by_seed_rows.append(
                {
                    "seed": seed,
                    "method": paper_name,
                    "selected_epoch": int(selected["epoch"]),
                    "mae": selected["MAE_human_mean"],
                    "exact": selected["Exact_rounded"],
                    "kendall": selected["Kendall_human_mean"],
                    "bias": selected["Bias_human_mean"],
                    "qwk": selected["QWK_rounded"],
                    "l2h": int(selected["L2H_count"]),
                }
            )

    write_csv(
        output_dir / "dev_mechanism_control_summary.csv",
        summary_rows,
        ["method", "metric", "mean", "sample_sd", *[f"seed{seed}" for seed in SEEDS]],
    )
    method_order = ("Hard-only", "Shuffled-soft", "HMSA")
    by_seed_rows.sort(
        key=lambda row: (int(row["seed"]), method_order.index(str(row["method"])))
    )
    write_csv(
        output_dir / "dev_mechanism_control_by_seed.csv",
        by_seed_rows,
        ["seed", "method", "selected_epoch", "mae", "exact", "kendall", "bias", "qwk", "l2h"],
    )

    audit = {
        "status": "PASS",
        "control": "within_label_shuffled_soft",
        "evidence_role": "posthoc_dev_only",
        "shuffle_seed": shuffle_audit["shuffle_seed"],
        "mapping_shared_across_model_seeds": True,
        "mapping_sha256": shuffle_audit["mapping_sha256"],
        "training_rows": shuffle_audit["rows"],
        "effective_target_changes": shuffle_audit["effective_target_changes"],
        "effective_change_rate": shuffle_audit["effective_change_rate"],
        "preserved": [
            "training inputs and hard targets",
            "dual-head architecture and initialization rule",
            "per-hard-label soft-target multiset and entropy distribution",
            "optimizer, scheduler, auxiliary weight, checkpoint rule, and hard-head inference",
        ],
        "development_targets": "original_unshuffled_human_distributions",
        "test_access_count": 0,
        "source_sha256": {
            "dev_mechanism_comparison.json": sha256(comparison_path),
            "shuffle_audit.json": sha256(shuffle_audit_path),
        },
    }
    with (output_dir / "dev_mechanism_control_audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
        file.write("\n")

    manifest_entry = {
        "evidence_role": "posthoc_dev_only",
        "mapping_sha256": shuffle_audit["mapping_sha256"],
        "test_access_count": 0,
    }
    return (comparison_path, shuffle_audit_path), manifest_entry


def export_frozen_test(repo: Path, output_dir: Path) -> None:
    report_path = repo / "thesis_exp/outputs/exp51_hmsa/final_test/canonical_final_test_report.json"
    formal_path = repo / "thesis_exp/outputs/exp51_hmsa/decision/formal_decision.json"
    integrity_path = repo / "thesis_exp/outputs/exp51_hmsa/audit/six_checkpoint_dev_reload_integrity.json"
    for source, destination in (
        (report_path, output_dir / "canonical_final_test_report.json"),
        (formal_path, output_dir / "formal_decision.json"),
        (integrity_path, output_dir / "six_checkpoint_dev_reload_integrity.json"),
    ):
        shutil.copyfile(source, destination)

    report = load_json(report_path)
    main_rows = []
    for metric in PRIMARY_METRICS:
        main_rows.append(
            {
                "metric": metric,
                "b0_mean": report["arm_aggregates"]["b0"]["mean"][metric],
                "b0_sd": report["arm_aggregates"]["b0"]["standard_deviation"][metric],
                "exp51_mean": report["arm_aggregates"]["exp51"]["mean"][metric],
                "exp51_sd": report["arm_aggregates"]["exp51"]["standard_deviation"][metric],
                "delta_mean": report["paired_delta_exp51_minus_b0"]["mean"][metric],
                "delta_sd": report["paired_delta_exp51_minus_b0"]["standard_deviation"][metric],
            }
        )
    write_csv(output_dir / "test_main_results.csv", main_rows, list(main_rows[0]))

    seed_rows = []
    for row in report["per_seed"]:
        for metric in ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean"):
            seed_rows.append(
                {
                    "seed": row["seed"],
                    "metric": metric,
                    "b0": row["b0"][metric],
                    "exp51": row["exp51"][metric],
                    "delta": row["delta_exp51_minus_b0"][metric],
                }
            )
    write_csv(output_dir / "test_seed_primary.csv", seed_rows, list(seed_rows[0]))

    recall_rows = []
    first_metrics_path = (
        repo / "thesis_exp/outputs/exp51_hmsa/final_test/b0/seed_42/test_metrics.json"
    )
    first_metrics = load_json(first_metrics_path)
    for label in range(1, 6):
        metric = f"Recall_{label}_rounded"
        recall_rows.append(
            {
                "label": label,
                "n": int(first_metrics[f"Recall_{label}_total"]),
                "b0_mean": report["arm_aggregates"]["b0"]["mean"][metric],
                "b0_sd": report["arm_aggregates"]["b0"]["standard_deviation"][metric],
                "exp51_mean": report["arm_aggregates"]["exp51"]["mean"][metric],
                "exp51_sd": report["arm_aggregates"]["exp51"]["standard_deviation"][metric],
                "delta": report["paired_delta_exp51_minus_b0"]["mean"][metric],
            }
        )
    write_csv(output_dir / "test_recall_by_label.csv", recall_rows, list(recall_rows[0]))

    mechanism_sources, mechanism_manifest = export_dev_mechanism_control(repo, output_dir)
    manifest_sources = (
        report_path,
        formal_path,
        integrity_path,
        first_metrics_path,
        *mechanism_sources,
    )
    manifest = {
        "status": "PAPER_DATA_EXPORTED_FROM_FROZEN_RESULTS",
        "delta_convention": "Exp51_minus_B0",
        "raw_test_split_reopened": False,
        "sources": {
            str(path.relative_to(repo)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in manifest_sources
        },
        "bootstrap": report["paired_mae_bootstrap"],
        "mechanism_control": mechanism_manifest,
    }
    with (output_dir / "export_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_frozen_test(repo, output_dir)
    aggregate_dev_subgroups(repo, output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
