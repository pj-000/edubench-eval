"""Paired seed42/formal decision logic for Exp49."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp49_cphce import FORMAL_SEEDS, OUTPUT_ROOT, run_output_dir
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(variant: str, seed: int) -> dict[str, Any]:
    path = run_output_dir(variant, seed) / "selected_dev_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def _summary(variant: str, seed: int) -> dict[str, Any]:
    path = run_output_dir(variant, seed) / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def _prediction_map(variant: str, seed: int) -> dict[str, dict[str, Any]]:
    path = run_output_dir(variant, seed) / "predictions" / "predictions_dev.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read_jsonl(path)
    return {str(row.get("record_id") or row.get("id")): row for row in rows}


def paired_delta_row(seed: int) -> list[float]:
    baseline = _prediction_map("b0_hard_ce", seed)
    treatment = _prediction_map("m1_human_soft", seed)
    if set(baseline) != set(treatment):
        raise ValueError(f"Prediction ID mismatch for seed {seed}")
    output: list[float] = []
    for key in sorted(baseline):
        b = baseline[key]
        m = treatment[key]
        human_mean = float(b["human_mean_5"])
        output.append(abs(float(m["pred_label_5"]) - human_mean) - abs(float(b["pred_label_5"]) - human_mean))
    return output


def bootstrap_ci(seeds: list[int], resamples: int = 10_000, rng_seed: int = 42) -> dict[str, float]:
    deltas = np.asarray([paired_delta_row(seed) for seed in seeds], dtype=float)
    if deltas.ndim != 2 or deltas.shape[1] == 0:
        raise ValueError("No paired rows for bootstrap")
    rng = np.random.default_rng(rng_seed)
    values = np.empty(resamples, dtype=float)
    n = deltas.shape[1]
    for index in range(resamples):
        sample = rng.integers(0, n, size=n)
        values[index] = float(deltas[:, sample].mean())
    return {
        "mean_delta_mae": float(deltas.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "resamples": resamples,
    }


def comparison_row(seed: int) -> dict[str, Any]:
    b = _metrics("b0_hard_ce", seed)
    m = _metrics("m1_human_soft", seed)
    b_summary = _summary("b0_hard_ce", seed)
    m_summary = _summary("m1_human_soft", seed)
    parity = {
        "train_text_hash": b_summary["train_text_hash"] == m_summary["train_text_hash"],
        "dev_text_hash": b_summary["dev_text_hash"] == m_summary["dev_text_hash"],
        "model_name_or_path": b_summary["model_name_or_path"] == m_summary["model_name_or_path"],
        "scheduler": b_summary["scheduler"] == m_summary["scheduler"],
        "runtime_head": b_summary["runtime_head"] == m_summary["runtime_head"],
    }
    if not all(parity.values()):
        raise ValueError(f"B0/M1 contract parity failed for seed {seed}: {parity}")
    row: dict[str, Any] = {"seed": seed}
    keys = (
        "MAE_human_mean",
        "Exact_rounded",
        "Kendall_human_mean",
        "Bias_human_mean",
        "L2H_count",
        "L2H_rounded",
        "Recall_4_rounded",
        "Recall_5_rounded",
        "BinAgreement_paper_3way",
    )
    for key in keys:
        row[f"b0_{key}"] = b[key]
        row[f"m1_{key}"] = m[key]
        row[f"delta_{key}"] = float(m[key]) - float(b[key])
    row["abs_bias_improvement"] = abs(float(b["Bias_human_mean"])) - abs(float(m["Bias_human_mean"]))
    row["mae_improvement"] = float(b["MAE_human_mean"]) - float(m["MAE_human_mean"])
    return row


def seed42_gate(resamples: int = 10_000) -> dict[str, Any]:
    row = comparison_row(42)
    ci = bootstrap_ci([42], resamples=resamples)
    checks = {
        "mae_improvement_at_least_0p005": row["mae_improvement"] >= 0.005,
        "exact_noninferior_two_rows": row["delta_Exact_rounded"] >= -(2.0 / 664.0),
        "kendall_noninferior": row["delta_Kendall_human_mean"] >= -0.003,
        "finite": all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))),
    }
    diagnostics = {
        "label4_delta": row["delta_Recall_4_rounded"],
        "label5_delta": row["delta_Recall_5_rounded"],
        "l2h_count_delta": row["delta_L2H_count"],
        "abs_bias_improvement": row["abs_bias_improvement"],
        "note": "Low-tail and per-label dev counts are reported but are not seed42 vetoes because dev has only 20 low rows.",
    }
    passed = all(checks.values())
    return {
        "status": "SEED42_PASS" if passed else "SEED42_NO_GO",
        "passed": passed,
        "checks": checks,
        "diagnostics": diagnostics,
        "comparison": row,
        "paired_bootstrap": ci,
    }


def formal_gate(resamples: int = 10_000) -> dict[str, Any]:
    rows = [comparison_row(seed) for seed in FORMAL_SEEDS]
    mean = {key: float(np.mean([float(row[key]) for row in rows])) for key in rows[0] if key != "seed"}
    ci = bootstrap_ci(list(FORMAL_SEEDS), resamples=resamples)
    checks = {
        "mean_mae_improvement_at_least_0p005": mean["mae_improvement"] >= 0.005,
        "mae_improves_two_of_three": sum(float(row["mae_improvement"]) > 0 for row in rows) >= 2,
        "mean_exact_noninferior": mean["delta_Exact_rounded"] >= -0.003,
        "exact_noninferior_two_of_three": sum(float(row["delta_Exact_rounded"]) >= -0.003 for row in rows) >= 2,
        "mean_kendall_noninferior": mean["delta_Kendall_human_mean"] >= -0.003,
        "bias_or_l2h_improves": mean["abs_bias_improvement"] >= 0.010 or mean["delta_L2H_count"] < 0,
        "paired_mae_ci_upper_below_zero": ci["ci_high"] < 0,
    }
    diagnostics = {
        "mean_label4_delta": mean["delta_Recall_4_rounded"],
        "mean_label5_delta": mean["delta_Recall_5_rounded"],
        "mean_l2h_count_delta": mean["delta_L2H_count"],
        "note": "Per-label and low-tail values remain mandatory reports, not standalone vetoes on the original imbalanced split.",
    }
    passed = all(checks.values())
    return {
        "status": "FORMAL_PASS" if passed else "FORMAL_NO_GO",
        "passed": passed,
        "checks": checks,
        "diagnostics": diagnostics,
        "per_seed": rows,
        "mean": mean,
        "paired_bootstrap": ci,
    }


def write_decision(mode: str, decision: dict[str, Any]) -> Path:
    decision_dir = OUTPUT_ROOT / "decision"
    table_dir = OUTPUT_ROOT / "tables"
    decision_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    path = decision_dir / f"{mode}_decision.json"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = decision.get("per_seed") or [decision["comparison"]]
    write_csv(table_dir / f"{mode}_comparison.csv", rows)
    write_text(
        OUTPUT_ROOT / f"{mode}_decision.md",
        "\n".join(
            [
                f"# Exp49 {mode} decision",
                "",
                f"Status: **{decision['status']}**",
                "",
                *[f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in decision["checks"].items()],
            ]
        )
        + "\n",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("seed42", "formal"), required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    decision = seed42_gate(args.bootstrap_resamples) if args.mode == "seed42" else formal_gate(args.bootstrap_resamples)
    path = write_decision(args.mode, decision)
    print(json.dumps({"decision_path": str(path), **decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
