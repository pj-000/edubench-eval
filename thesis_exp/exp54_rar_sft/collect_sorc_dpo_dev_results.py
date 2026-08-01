"""Collect formal SORC-DPO dev metrics and preregistered pairwise contrasts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np

from thesis_exp.exp54_rar_sft import REPO_ROOT


ARMS = {
    "P1_FIELD_DPO": (42, 43, 44),
    "P2_SORC_SCORE": (42, 43, 44),
    "P3_JOINT_SORC": (42, 43, 44),
    "P1_SYN_SEED42": (42,),
}
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
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_REPLICATES = 2000
DEFAULT_DEV_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_dev_runs_vllm"
)
DEFAULT_SFT_DEV_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/dev_runs_vllm"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_dev_summary_vllm"
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
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def _execution_audit(
    predictions: list[dict[str, Any]],
) -> dict[str, float | int]:
    forced = sum(bool(row.get("forced_completion", False)) for row in predictions)
    max_hits = sum(
        int(
            row.get(
                "backend_generated_token_count",
                row["generated_token_count"],
            )
        )
        >= 256
        for row in predictions
    )
    return {
        "forced_completion_count": forced,
        "forced_completion_rate": forced / len(predictions),
        "max_token_hit_count": max_hits,
        "max_token_hit_rate": max_hits / len(predictions),
    }


def _load_run(
    run_dir: Path,
    *,
    arm: str,
    seed: int,
    preference: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = read_json(run_dir / "metrics.json")
    protocol = read_json(run_dir / "protocol.json")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    expected_status = (
        "EXP54_SORC_DPO_DEV_METRICS_VLLM_COMPLETE"
        if preference
        else "EXP54_DEV_METRICS_VLLM_COMPLETE"
    )
    if (
        metrics.get("status") != expected_status
        or metrics.get("arm") != arm
        or int(metrics.get("seed")) != seed
        or metrics.get("test_accessed") is not False
        or protocol.get("dev_accessed") is not True
        or protocol.get("test_accessed") is not False
        or len(predictions) != 664
        or any(row.get("parse_success") is not True for row in predictions)
    ):
        raise ValueError(f"{run_dir}: dev result contract differs")
    metrics["_execution_audit"] = _execution_audit(predictions)
    return metrics, predictions


def _arrays(
    predictions: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return (
        np.asarray([int(row["label_5"]) for row in predictions]),
        np.asarray([int(row["prediction"]["score"]) for row in predictions]),
        [str(row["record_id"]) for row in predictions],
    )


def _sampled_metrics(
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


def paired_bootstrap(
    predictions: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    left: str,
    right: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    canonical_ids: list[str] | None = None
    for arm in (left, right):
        for seed in seeds:
            gold, pred, record_ids = _arrays(predictions[(arm, seed)])
            if canonical_ids is None:
                canonical_ids = record_ids
            if record_ids != canonical_ids:
                raise ValueError("paired dev row order differs")
            arrays[(arm, seed)] = (gold, pred)
    assert canonical_ids is not None
    randomizer = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = randomizer.choice(seeds, size=len(seeds), replace=True)
        sampled_rows = randomizer.integers(
            0,
            len(canonical_ids),
            size=len(canonical_ids),
        )
        left_values: dict[str, list[float]] = defaultdict(list)
        right_values: dict[str, list[float]] = defaultdict(list)
        for seed_value in sampled_seeds:
            seed = int(seed_value)
            left_gold, left_pred = arrays[(left, seed)]
            right_gold, right_pred = arrays[(right, seed)]
            if not np.array_equal(left_gold, right_gold):
                raise ValueError("paired dev gold vectors differ")
            for name, value in _sampled_metrics(
                left_gold,
                left_pred,
                sampled_rows,
            ).items():
                left_values[name].append(value)
            for name, value in _sampled_metrics(
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
        "comparison": f"{left}_minus_{right}",
        "seeds": list(seeds),
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
    selected_arms = (
        {
            arm: seeds
            for arm, seeds in ARMS.items()
            if arm != "P1_SYN_SEED42"
        }
        if args.three_arm_only
        else ARMS
    )
    per_seed_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for arm, seeds in selected_arms.items():
        for seed in seeds:
            metrics, rows = _load_run(
                args.dev_root / arm.lower() / f"seed_{seed}",
                arm=arm,
                seed=seed,
                preference=True,
            )
            execution = metrics["_execution_audit"]
            per_seed_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    **{
                        name: metrics["score"][name]
                        for name in SCORE_METRICS
                    },
                    **{
                        name: metrics["rationale"][name]
                        for name in RATIONALE_METRICS
                    },
                    **execution,
                }
            )
            predictions[(arm, seed)] = rows
    for seed in (42, 43, 44):
        _metrics, rows = _load_run(
            args.sft_dev_root / "r3" / f"seed{seed}" / "epoch3",
            arm="R3",
            seed=seed,
            preference=False,
        )
        predictions[("R3", seed)] = rows

    aggregates = []
    for arm, seeds in selected_arms.items():
        arm_rows = [row for row in per_seed_rows if row["arm"] == arm]
        aggregate: dict[str, Any] = {"arm": arm, "seeds": list(seeds)}
        for name in (
            *SCORE_METRICS,
            *RATIONALE_METRICS,
            *EXECUTION_METRICS,
        ):
            values = [float(row[name]) for row in arm_rows]
            aggregate[f"{name}_mean"] = mean(values)
            aggregate[f"{name}_sample_std"] = (
                stdev(values) if len(values) > 1 else None
            )
        aggregates.append(aggregate)
    bootstraps = [
        paired_bootstrap(
            predictions,
            left="P1_FIELD_DPO",
            right="R3",
            seeds=(42, 43, 44),
        ),
        paired_bootstrap(
            predictions,
            left="P2_SORC_SCORE",
            right="P1_FIELD_DPO",
            seeds=(42, 43, 44),
        ),
        paired_bootstrap(
            predictions,
            left="P3_JOINT_SORC",
            right="P2_SORC_SCORE",
            seeds=(42, 43, 44),
        ),
        paired_bootstrap(
            predictions,
            left="P3_JOINT_SORC",
            right="R3",
            seeds=(42, 43, 44),
        ),
    ]
    if not args.three_arm_only:
        bootstraps.append(
            paired_bootstrap(
                predictions,
                left="P1_FIELD_DPO",
                right="P1_SYN_SEED42",
                seeds=(42,),
            )
        )
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(args.output_dir / "multiseed_summary.csv", aggregates)
    write_json(args.output_dir / "paired_bootstrap.json", bootstraps)
    final = {
        "status": "EXP54_SORC_DPO_DEV_STUDY_COMPLETE",
        "preference_runs": per_seed_rows,
        "multiseed_summary": aggregates,
        "paired_bootstrap": bootstraps,
        "comparisons": [
            "P1_FIELD_DPO vs frozen R3-SFT epoch3",
            "P2_SORC_SCORE vs P1_FIELD_DPO",
            "P3_JOINT_SORC vs P2_SORC_SCORE",
            "P3_JOINT_SORC vs frozen R3-SFT epoch3",
        ]
        + (
            []
            if args.three_arm_only
            else ["P1_FIELD_DPO seed42 vs P1_SYN_SEED42"]
        ),
        "inference_protocol": "RAR_SFT_VLLM_COMPACT_JSON_V1",
        "dev_accessed": True,
        "test_accessed": False,
    }
    write_json(args.output_dir / "final_results.json", final)

    lines = [
        "# Exp54 SORC-DPO dev results",
        "",
        "| Arm | Exact | MAE | Kendall | L2H | Recall-2 | Recall-5 | Forced close |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        standard = row["Exact_sample_std"]
        exact = (
            f"{row['Exact_mean']:.4f}±{standard:.4f}"
            if standard is not None
            else f"{row['Exact_mean']:.4f} (seed42)"
        )
        lines.append(
            f"| {row['arm']} "
            f"| {exact} "
            f"| {row['MAE_mean']:.4f} "
            f"| {row['Kendall_mean']:.4f} "
            f"| {row['L2H_count_mean']:.2f} "
            f"| {row['Recall_2_mean']:.4f} "
            f"| {row['Recall_5_mean']:.4f} "
            f"| {row['forced_completion_rate_mean']:.2%} |"
        )
    lines.extend(["", "Dev accessed: yes. Test accessed: no."])
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument(
        "--sft-dev-root",
        type=Path,
        default=DEFAULT_SFT_DEV_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--three-arm-only",
        action="store_true",
        help="Collect P1/P2/P3 without the seed-42 P1-SYN diagnostic arm.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
