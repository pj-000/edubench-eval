"""Select Exp54 checkpoints and summarize the complete multi-seed dev study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np


ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
EPOCHS = (1, 2, 3)
BOOTSTRAP_SEED = 20260727
BOOTSTRAP_REPLICATES = 2000
SCORE_METRICS = (
    "Exact",
    "MAE",
    "Kendall",
    "Signed_Bias",
    "L2H_count",
    "L2H_rate",
    "Label_2_correct_count",
    "Recall_2",
    "Label_5_correct_count",
    "Recall_5",
)
RATIONALE_METRICS = (
    "nonempty_rationale_rate",
    "mean_rationale_tokens",
    "median_rationale_tokens",
    "explicit_score_leakage_rate",
    "language_consistency_rate",
)
EXECUTION_METRICS = (
    "forced_completion_count",
    "forced_completion_rate",
    "max_token_hit_count",
    "max_token_hit_rate",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)


def select_checkpoint(
    epoch_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return min(
        epoch_metrics,
        key=lambda value: (
            -float(value["score"]["Exact"]),
            float(value["score"]["MAE"]),
            int(value["epoch"]),
        ),
    )


def prediction_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if any(not row["parse_success"] for row in rows):
        raise ValueError("selected prediction file has parse failures")
    gold = np.asarray([int(row["label_5"]) for row in rows])
    pred = np.asarray([int(row["prediction"]["score"]) for row in rows])
    record_ids = [str(row["record_id"]) for row in rows]
    return gold, pred, record_ids


def sampled_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    sampled_gold = gold[indices]
    sampled_pred = pred[indices]
    low = sampled_gold <= 2
    label2 = sampled_gold == 2
    label5 = sampled_gold == 5
    return {
        "Exact": float(np.mean(sampled_pred == sampled_gold)),
        "MAE": float(np.mean(np.abs(sampled_pred - sampled_gold))),
        "L2H_rate": float(np.mean(sampled_pred[low] >= 4)),
        "Recall_2": float(np.mean(sampled_pred[label2] == 2)),
        "Recall_5": float(np.mean(sampled_pred[label5] == 5)),
    }


def paired_hierarchical_bootstrap(
    selected_predictions: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    canonical_ids: list[str] | None = None
    for arm in (left_arm, right_arm):
        for seed in SEEDS:
            gold, pred, record_ids = prediction_arrays(
                selected_predictions[(arm, seed)]
            )
            if canonical_ids is None:
                canonical_ids = record_ids
            if record_ids != canonical_ids:
                raise ValueError("selected dev record order differs")
            arrays[(arm, seed)] = (gold, pred)
    assert canonical_ids is not None
    randomizer = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = randomizer.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_rows = randomizer.integers(
            0,
            len(canonical_ids),
            size=len(canonical_ids),
        )
        left_values: dict[str, list[float]] = defaultdict(list)
        right_values: dict[str, list[float]] = defaultdict(list)
        for seed in sampled_seeds:
            left_gold, left_pred = arrays[(left_arm, int(seed))]
            right_gold, right_pred = arrays[(right_arm, int(seed))]
            if not np.array_equal(left_gold, right_gold):
                raise ValueError("paired dev gold vectors differ")
            for name, value in sampled_metrics(
                left_gold,
                left_pred,
                sampled_rows,
            ).items():
                left_values[name].append(value)
            for name, value in sampled_metrics(
                right_gold,
                right_pred,
                sampled_rows,
            ).items():
                right_values[name].append(value)
        for name in left_values:
            deltas[name].append(
                float(np.mean(left_values[name]))
                - float(np.mean(right_values[name]))
            )
    return {
        "comparison": f"{left_arm}_minus_{right_arm}",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "deltas": {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
            for name, values in sorted(deltas.items())
        },
    }


def collect(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    all_epoch_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_predictions: dict[
        tuple[str, int], list[dict[str, Any]]
    ] = {}
    for arm in ARMS:
        for seed in SEEDS:
            candidates = []
            for epoch in EPOCHS:
                run_dir = (
                    args.dev_root
                    / arm.lower()
                    / f"seed{seed}"
                    / f"epoch{epoch}"
                )
                metrics = read_json(run_dir / "metrics.json")
                protocol = read_json(run_dir / "protocol.json")
                predictions = read_jsonl(
                    run_dir / "predictions.jsonl"
                )
                if (
                    metrics.get("arm") != arm
                    or metrics.get("seed") != seed
                    or metrics.get("epoch") != epoch
                    or metrics.get("test_accessed") is not False
                    or protocol.get("dev_accessed") is not True
                    or protocol.get("test_accessed") is not False
                ):
                    raise ValueError(f"{run_dir}: dev result metadata differs")
                if (
                    len(predictions) != int(metrics["execution"]["rows"])
                    or any(
                        row.get("parse_success") is not True
                        for row in predictions
                    )
                ):
                    raise ValueError(
                        f"{run_dir}: prediction completeness differs"
                    )
                forced_completion_count = sum(
                    bool(row.get("forced_completion", False))
                    for row in predictions
                )
                max_token_hit_count = sum(
                    int(
                        row.get(
                            "backend_generated_token_count",
                            row["generated_token_count"],
                        )
                    )
                    >= 256
                    for row in predictions
                )
                execution_audit = {
                    "forced_completion_count": (
                        forced_completion_count
                    ),
                    "forced_completion_rate": (
                        forced_completion_count / len(predictions)
                    ),
                    "max_token_hit_count": max_token_hit_count,
                    "max_token_hit_rate": (
                        max_token_hit_count / len(predictions)
                    ),
                }
                metrics["_execution_audit"] = execution_audit
                candidates.append(metrics)
                all_epoch_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        **{
                            name: metrics["score"][name]
                            for name in SCORE_METRICS
                        },
                        **{
                            name: metrics["rationale"][name]
                            for name in RATIONALE_METRICS
                        },
                        **execution_audit,
                    }
                )
            selected = select_checkpoint(candidates)
            epoch = int(selected["epoch"])
            selected_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_epoch": epoch,
                    "selection_rule": (
                        "max Exact, lower MAE, earlier epoch"
                    ),
                    **{
                        name: selected["score"][name]
                        for name in SCORE_METRICS
                    },
                    **{
                        name: selected["rationale"][name]
                        for name in RATIONALE_METRICS
                    },
                    **selected["_execution_audit"],
                }
            )
            selected_predictions[(arm, seed)] = read_jsonl(
                args.dev_root
                / arm.lower()
                / f"seed{seed}"
                / f"epoch{epoch}"
                / "predictions.jsonl"
            )

    aggregates = []
    for arm in ARMS:
        arm_rows = [row for row in selected_rows if row["arm"] == arm]
        aggregate: dict[str, Any] = {
            "arm": arm,
            "seeds": list(SEEDS),
        }
        for name in (
            *SCORE_METRICS,
            *RATIONALE_METRICS,
            *EXECUTION_METRICS,
        ):
            values = [float(row[name]) for row in arm_rows]
            aggregate[f"{name}_mean"] = mean(values)
            aggregate[f"{name}_sample_std"] = stdev(values)
        aggregates.append(aggregate)

    bootstraps = [
        paired_hierarchical_bootstrap(
            selected_predictions,
            left_arm="R3",
            right_arm=right,
        )
        for right in ("S0", "R1", "R2")
    ]
    write_csv(args.output_dir / "all_epoch_metrics.csv", all_epoch_rows)
    write_csv(args.output_dir / "selected_checkpoints.csv", selected_rows)
    write_csv(args.output_dir / "multiseed_summary.csv", aggregates)
    write_json(args.output_dir / "paired_bootstrap.json", bootstraps)
    write_json(
        args.output_dir / "final_results.json",
        {
            "status": "EXP54_DEV_STUDY_COMPLETE",
            "checkpoint_selection_rule": (
                "maximum dev Exact; lower dev MAE; earlier epoch"
            ),
            "selected_runs": selected_rows,
            "multiseed_summary": aggregates,
            "paired_bootstrap": bootstraps,
            "model_based_rationale_preference_status": (
                "NOT_RUN_EVALUATOR_MODELS_NOT_CONFIGURED"
            ),
            "inference_completion_note": (
                "Score is emitted before rationale. At the 256-token "
                "boundary, an unfinished schema-valid rationale prefix is "
                "deterministically truncated and JSON-closed without "
                "changing score. Arm-specific completion rates must be "
                "considered when interpreting rationale diagnostics."
            ),
            "dev_accessed": True,
            "test_accessed": False,
        },
    )

    lines = [
        "# Exp54 RAR-SFT dev results",
        "",
        "Checkpoint selection: maximum dev Exact, then lower MAE, "
        "then earlier epoch.",
        "",
        "| Arm | Exact | MAE | Kendall | L2H | Recall-2 | Recall-5 "
        "| Forced close |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['arm']} "
            f"| {row['Exact_mean']:.4f}±{row['Exact_sample_std']:.4f} "
            f"| {row['MAE_mean']:.4f}±{row['MAE_sample_std']:.4f} "
            f"| {row['Kendall_mean']:.4f}±{row['Kendall_sample_std']:.4f} "
            f"| {row['L2H_count_mean']:.2f} "
            f"| {row['Recall_2_mean']:.4f} "
            f"| {row['Recall_5_mean']:.4f} "
            f"| {row['forced_completion_rate_mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The report includes deterministic rationale diagnostics. "
            "Model-based blind rationale preference is not run because "
            "the two evaluator model identities and credentials are not "
            "configured.",
            "",
            "Forced close only truncates the tail of a rationale that "
            "reaches the fixed 256-token boundary; the score has already "
            "been emitted and is unchanged. Unequal arm-specific forced-"
            "close rates limit direct rationale-quality interpretation.",
            "",
            "Dev accessed: yes. Test accessed: no.",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
