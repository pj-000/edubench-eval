"""Locked seed42 gate comparing CAHS-0.5 with the frozen Exp49 B0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp50_cahs import OUTPUT_ROOT, baseline_run_dir, run_output_dir
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_checks(b: dict[str, Any], c: dict[str, Any]) -> dict[str, bool]:
    b_correct = round(float(b["Exact_rounded"]) * int(b["n"]))
    c_correct = round(float(c["Exact_rounded"]) * int(c["n"]))
    bias_improvement = abs(float(b["Bias_human_mean"])) - abs(float(c["Bias_human_mean"]))
    return {
        "mae_improvement_at_least_0p005": float(b["MAE_human_mean"]) - float(c["MAE_human_mean"]) >= 0.005,
        "exact_loses_at_most_two_rows": c_correct >= b_correct - 2,
        "kendall_noninferior": float(c["Kendall_human_mean"]) >= float(b["Kendall_human_mean"]) - 0.003,
        "label4_loses_at_most_three_rows": int(c["Recall_4_correct"]) >= int(b["Recall_4_correct"]) - 3,
        "label5_loses_at_most_three_rows": int(c["Recall_5_correct"]) >= int(b["Recall_5_correct"]) - 3,
        "l2h_does_not_increase": int(c["L2H_count"]) <= int(b["L2H_count"]),
        "bias_or_l2h_improves": int(c["L2H_count"]) <= int(b["L2H_count"]) - 1 or bias_improvement >= 0.005,
        "finite": all(math.isfinite(float(metrics[key])) for metrics in (b, c) for key in ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean", "Bias_human_mean")),
    }


def paired_deltas(seed: int) -> tuple[np.ndarray, int, int]:
    b_rows = {str(row["record_id"]): row for row in read_jsonl(baseline_run_dir(seed) / "predictions" / "predictions_dev.jsonl")}
    c_rows = {str(row["record_id"]): row for row in read_jsonl(run_output_dir(seed) / "predictions" / "predictions_dev.jsonl")}
    if set(b_rows) != set(c_rows):
        raise ValueError("B0/CAHS prediction IDs differ")
    deltas: list[float] = []
    b_only = c_only = 0
    for ident in sorted(b_rows):
        b, c = b_rows[ident], c_rows[ident]
        mean = float(b["human_mean_5"])
        deltas.append(abs(float(c["pred_label_5"]) - mean) - abs(float(b["pred_label_5"]) - mean))
        b_correct = int(b["pred_label_5"]) == int(b["label_5"])
        c_correct = int(c["pred_label_5"]) == int(c["label_5"])
        b_only += int(b_correct and not c_correct)
        c_only += int(c_correct and not b_correct)
    return np.asarray(deltas), b_only, c_only


def bootstrap(deltas: np.ndarray, resamples: int = 10_000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    values = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(deltas), len(deltas))
        values[index] = float(deltas[sample].mean())
    return {"mean_delta_mae": float(deltas.mean()), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "resamples": resamples}


def mcnemar_exact(b_only: int, c_only: int) -> dict[str, Any]:
    discordant = b_only + c_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(0, min(b_only, c_only) + 1)) / (2**discordant)
        p = min(1.0, 2.0 * tail)
    return {"b0_correct_cahs_wrong": b_only, "b0_wrong_cahs_correct": c_only, "discordant": discordant, "two_sided_exact_p": p}


def seed42_gate() -> dict[str, Any]:
    b_summary = read_json(baseline_run_dir(42) / "run_summary.json")
    c_summary = read_json(run_output_dir(42) / "run_summary.json")
    for key in ("train_text_hash", "dev_text_hash", "model_name_or_path", "scheduler"):
        if b_summary[key] != c_summary[key]:
            raise ValueError(f"B0/CAHS parity failure: {key}")
    b, c = b_summary["selected_metrics"], c_summary["selected_metrics"]
    checks = gate_checks(b, c)
    deltas, b_only, c_only = paired_deltas(42)
    comparison = {
        "seed": 42,
        "b0_selected_epoch": b_summary["selected_epoch"],
        "cahs_selected_epoch": c_summary["selected_epoch"],
        "b0_exact_correct": round(float(b["Exact_rounded"]) * int(b["n"])),
        "cahs_exact_correct": round(float(c["Exact_rounded"]) * int(c["n"])),
        "mae_improvement": float(b["MAE_human_mean"]) - float(c["MAE_human_mean"]),
        "delta_exact": float(c["Exact_rounded"]) - float(b["Exact_rounded"]),
        "delta_kendall": float(c["Kendall_human_mean"]) - float(b["Kendall_human_mean"]),
        "b0_label4_correct": int(b["Recall_4_correct"]),
        "cahs_label4_correct": int(c["Recall_4_correct"]),
        "b0_label5_correct": int(b["Recall_5_correct"]),
        "cahs_label5_correct": int(c["Recall_5_correct"]),
        "b0_l2h_count": int(b["L2H_count"]),
        "cahs_l2h_count": int(c["L2H_count"]),
        "b0_bias": float(b["Bias_human_mean"]),
        "cahs_bias": float(c["Bias_human_mean"]),
    }
    passed = all(checks.values())
    return {"status": "SEED42_PASS" if passed else "SEED42_NO_GO", "passed": passed, "checks": checks, "comparison": comparison, "paired_bootstrap": bootstrap(deltas), "mcnemar": mcnemar_exact(b_only, c_only)}


def main() -> None:
    decision = seed42_gate()
    decision_dir = OUTPUT_ROOT / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    path = decision_dir / "seed42_decision.json"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUTPUT_ROOT / "tables" / "seed42_comparison.csv", [decision["comparison"]])
    write_text(OUTPUT_ROOT / "seed42_decision.md", "# Exp50 seed42 decision\n\nStatus: **" + decision["status"] + "**\n\n" + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in decision["checks"].items()) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
