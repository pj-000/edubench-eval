"""Locked seed-42 MeanAux gate against frozen Hard-only and HMSA dev runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp56_meanaux import (
    OUTPUT_ROOT,
    baseline_run_dir,
    hmsa_run_dir,
    run_output_dir,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gate_checks(
    baseline: dict[str, Any],
    meanaux: dict[str, Any],
) -> dict[str, bool]:
    baseline_correct = round(
        float(baseline["Exact_rounded"]) * int(baseline["n"])
    )
    meanaux_correct = round(
        float(meanaux["Exact_rounded"]) * int(meanaux["n"])
    )
    bias_improvement = abs(float(baseline["Bias_human_mean"])) - abs(
        float(meanaux["Bias_human_mean"])
    )
    return {
        "mae_improvement_at_least_0p005": (
            float(baseline["MAE_human_mean"])
            - float(meanaux["MAE_human_mean"])
            >= 0.005
        ),
        "exact_loses_at_most_two_rows": (
            meanaux_correct >= baseline_correct - 2
        ),
        "kendall_noninferior": (
            float(meanaux["Kendall_human_mean"])
            >= float(baseline["Kendall_human_mean"]) - 0.003
        ),
        "label4_loses_at_most_three_rows": (
            int(meanaux["Recall_4_correct"])
            >= int(baseline["Recall_4_correct"]) - 3
        ),
        "label5_loses_at_most_three_rows": (
            int(meanaux["Recall_5_correct"])
            >= int(baseline["Recall_5_correct"]) - 3
        ),
        "l2h_does_not_increase": (
            int(meanaux["L2H_count"]) <= int(baseline["L2H_count"])
        ),
        "bias_or_l2h_improves": (
            int(meanaux["L2H_count"]) <= int(baseline["L2H_count"]) - 1
            or bias_improvement >= 0.005
        ),
        "finite": all(
            math.isfinite(float(metrics[key]))
            for metrics in (baseline, meanaux)
            for key in (
                "MAE_human_mean",
                "Exact_rounded",
                "Kendall_human_mean",
                "Bias_human_mean",
            )
        ),
    }


def keyed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in read_jsonl(path)}


def paired_deltas(
    left_path: Path,
    right_path: Path,
) -> np.ndarray:
    left = keyed_predictions(left_path)
    right = keyed_predictions(right_path)
    if set(left) != set(right):
        raise ValueError("Paired prediction IDs differ")
    return np.asarray(
        [
            abs(
                float(right[key]["pred_label_5"])
                - float(left[key]["human_mean_5"])
            )
            - abs(
                float(left[key]["pred_label_5"])
                - float(left[key]["human_mean_5"])
            )
            for key in sorted(left)
        ],
        dtype=float,
    )


def bootstrap(
    deltas: np.ndarray,
    *,
    resamples: int = 10_000,
    rng_seed: int = 56,
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    values = np.empty(resamples)
    for index in range(resamples):
        values[index] = float(
            deltas[rng.integers(0, len(deltas), len(deltas))].mean()
        )
    return {
        "mean_delta_mae": float(deltas.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "resamples": resamples,
        "rng_seed": rng_seed,
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (
            "MAE_human_mean",
            "Exact_rounded",
            "Kendall_human_mean",
            "Bias_human_mean",
            "QWK_rounded",
            "L2H_count",
            "Recall_1_rounded",
            "Recall_2_rounded",
            "Recall_3_rounded",
            "Recall_4_rounded",
            "Recall_5_rounded",
        )
    }


def seed42_gate() -> dict[str, Any]:
    seed = 42
    baseline_summary = read_json(baseline_run_dir(seed) / "run_summary.json")
    hmsa_summary = read_json(hmsa_run_dir(seed) / "run_summary.json")
    meanaux_summary = read_json(run_output_dir(seed) / "run_summary.json")
    for key in (
        "train_text_hash",
        "dev_text_hash",
        "model_name_or_path",
        "scheduler",
    ):
        if baseline_summary[key] != meanaux_summary[key]:
            raise ValueError(f"Hard-only/MeanAux parity failure: {key}")
        if hmsa_summary[key] != meanaux_summary[key]:
            raise ValueError(f"HMSA/MeanAux parity failure: {key}")
    if (
        int(meanaux_summary["test_access_count"]) != 0
        or meanaux_summary["inference"] != "hard_head_raw_logit_argmax"
    ):
        raise ValueError("MeanAux inference/test contract failure")
    baseline = baseline_summary["selected_metrics"]
    hmsa = hmsa_summary["selected_metrics"]
    meanaux = meanaux_summary["selected_metrics"]
    checks = gate_checks(baseline, meanaux)
    baseline_predictions = (
        baseline_run_dir(seed) / "predictions" / "predictions_dev.jsonl"
    )
    hmsa_predictions = (
        hmsa_run_dir(seed) / "predictions" / "predictions_dev.jsonl"
    )
    meanaux_predictions = (
        run_output_dir(seed) / "predictions" / "predictions_dev.jsonl"
    )
    meanaux_minus_baseline = paired_deltas(
        baseline_predictions,
        meanaux_predictions,
    )
    meanaux_minus_hmsa = paired_deltas(
        hmsa_predictions,
        meanaux_predictions,
    )
    passed = all(checks.values())
    return {
        "status": "EXP56_SEED42_PASS" if passed else "EXP56_SEED42_NO_GO",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "hard_only": compact_metrics(baseline),
            "hmsa": compact_metrics(hmsa),
            "meanaux": compact_metrics(meanaux),
        },
        "paired_bootstrap": {
            "meanaux_minus_hard_only": bootstrap(meanaux_minus_baseline),
            "meanaux_minus_hmsa": bootstrap(meanaux_minus_hmsa),
        },
        "interpretation_contract": (
            "This post-hoc dev control compares target geometry/loss, not "
            "information content; no test prediction is authorized."
        ),
        "test_access_count": 0,
    }


def main() -> None:
    decision = seed42_gate()
    decision_dir = OUTPUT_ROOT / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "seed42_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for name, metrics in decision["metrics"].items():
        rows.append({"arm": name, **metrics})
    write_csv(OUTPUT_ROOT / "tables" / "seed42_comparison.csv", rows)
    write_text(
        OUTPUT_ROOT / "seed42_decision.md",
        "# Exp56 MeanAux seed42 decision\n\n"
        f"Status: **{decision['status']}**\n\n"
        + "\n".join(
            f"- {key}: {'PASS' if value else 'FAIL'}"
            for key, value in decision["checks"].items()
        )
        + "\n",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
