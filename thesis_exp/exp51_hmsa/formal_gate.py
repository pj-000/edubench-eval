"""Locked three-seed formal dev Gate for Exp51 versus paired B0 runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp51_hmsa import FORMAL_SEEDS, OUTPUT_ROOT, baseline_run_dir, run_output_dir
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


FORMAL_LOCK_PATH = Path(__file__).resolve().parents[1] / "configs" / "exp51_hmsa" / "formal_lock.json"
FORMAL_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "configs" / "exp51_hmsa" / "formal_protocol.json"
SOURCE_LOCK_PATH = Path(__file__).resolve().parents[1] / "configs" / "exp51_hmsa" / "source_lock.json"


def verify_formal_lock(path: Path = FORMAL_LOCK_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Exp51 formal protocol requires {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    for relative, expected in lock["files"].items():
        actual = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Exp51 formal-lock mismatch: {relative}: {actual} != {expected}")
    lock["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return lock


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def exact_correct(metrics: dict[str, Any]) -> int:
    return sum(int(metrics[f"Recall_{label}_correct"]) for label in range(1, 6))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summaries(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return read_json(baseline_run_dir(seed) / "run_summary.json"), read_json(run_output_dir(seed) / "run_summary.json")


def comparison_row(seed: int, protocol: dict[str, Any]) -> dict[str, Any]:
    b_summary, e_summary = summaries(seed)
    source_lock = read_json(SOURCE_LOCK_PATH)
    expected_source_manifest = hashlib.sha256(SOURCE_LOCK_PATH.read_bytes()).hexdigest()
    parity = {key: b_summary[key] == e_summary[key] for key in ("train_text_hash", "dev_text_hash", "model_name_or_path", "scheduler")}
    if not all(parity.values()):
        raise ValueError(f"B0/Exp51 parity failure for seed {seed}: {parity}")
    for key in ("train_text_hash", "dev_text_hash", "model_name_or_path", "scheduler"):
        if b_summary[key] != protocol[key]:
            raise ValueError(f"Formal protocol mismatch for seed {seed}: {key}")
    if int(b_summary["test_access_count"]) != 0 or int(e_summary["test_access_count"]) != 0:
        raise ValueError(f"Unexpected test access for seed {seed}")
    if b_summary["status"] != "COMPLETED" or e_summary["status"] != "COMPLETED" or int(b_summary["seed"]) != seed or int(e_summary["seed"]) != seed:
        raise ValueError(f"Incomplete or mismatched run summary for seed {seed}")
    if bool(b_summary["nan_or_inf"]) or bool(e_summary["nan_or_inf"]):
        raise ValueError(f"Non-finite run for seed {seed}")
    if e_summary["locked_source_commit"] != source_lock["source_commit"] or e_summary["source_lock_manifest_sha256"] != expected_source_manifest:
        raise ValueError(f"Exp51 source-lock provenance failure for seed {seed}")
    if e_summary["inference"] != "hard_head_raw_logit_argmax" or float(e_summary["aux_weight"]) != 1.0:
        raise ValueError(f"Exp51 inference/loss contract failure for seed {seed}")
    head_contract = e_summary["initial_head_contract"]
    if head_contract["hard_head_hash"] != head_contract["soft_head_hash"] or not head_contract["storage_independent"]:
        raise ValueError(f"Exp51 head initialization contract failure for seed {seed}")
    b, e = b_summary["selected_metrics"], e_summary["selected_metrics"]
    if int(b["n"]) != 664 or int(e["n"]) != 664:
        raise ValueError(f"Unexpected dev size for seed {seed}")
    row: dict[str, Any] = {
        "seed": seed,
        "b0_selected_epoch": b_summary["selected_epoch"],
        "exp51_selected_epoch": e_summary["selected_epoch"],
        "b0_exact_correct": exact_correct(b),
        "exp51_exact_correct": exact_correct(e),
        "mae_improvement": float(b["MAE_human_mean"]) - float(e["MAE_human_mean"]),
        "delta_exact": float(e["Exact_rounded"]) - float(b["Exact_rounded"]),
        "delta_kendall": float(e["Kendall_human_mean"]) - float(b["Kendall_human_mean"]),
        "delta_label4_recall": float(e["Recall_4_rounded"]) - float(b["Recall_4_rounded"]),
        "delta_label5_recall": float(e["Recall_5_rounded"]) - float(b["Recall_5_rounded"]),
        "delta_l2h_count": int(e["L2H_count"]) - int(b["L2H_count"]),
        "abs_bias_improvement": abs(float(b["Bias_human_mean"])) - abs(float(e["Bias_human_mean"])),
        "b0_mae": float(b["MAE_human_mean"]),
        "exp51_mae": float(e["MAE_human_mean"]),
        "b0_exact": float(b["Exact_rounded"]),
        "exp51_exact": float(e["Exact_rounded"]),
        "b0_kendall": float(b["Kendall_human_mean"]),
        "exp51_kendall": float(e["Kendall_human_mean"]),
        "b0_label4_correct": int(b["Recall_4_correct"]),
        "exp51_label4_correct": int(e["Recall_4_correct"]),
        "b0_label5_correct": int(b["Recall_5_correct"]),
        "exp51_label5_correct": int(e["Recall_5_correct"]),
        "b0_l2h_count": int(b["L2H_count"]),
        "exp51_l2h_count": int(e["L2H_count"]),
        "b0_bias": float(b["Bias_human_mean"]),
        "exp51_bias": float(e["Bias_human_mean"]),
        "b0_hard_ce": float(b["eval_hard_ce_loss"]),
        "exp51_hard_ce": float(e["eval_hard_ce_loss"]),
        "exp51_soft_aux_ce": float(e["soft_aux_ce"]),
    }
    return row


def paired_delta_rows(seed: int) -> np.ndarray:
    b = {str(row["record_id"]): row for row in read_jsonl(baseline_run_dir(seed) / "predictions" / "predictions_dev.jsonl")}
    e = {str(row["record_id"]): row for row in read_jsonl(run_output_dir(seed) / "predictions" / "predictions_dev.jsonl")}
    if set(b) != set(e):
        raise ValueError(f"Prediction ID mismatch for seed {seed}")
    return np.asarray(
        [abs(float(e[key]["pred_label_5"]) - float(b[key]["human_mean_5"])) - abs(float(b[key]["pred_label_5"]) - float(b[key]["human_mean_5"])) for key in sorted(b)],
        dtype=float,
    )


def bootstrap_ci(seeds: tuple[int, ...], resamples: int, rng_seed: int) -> dict[str, float]:
    deltas = np.stack([paired_delta_rows(seed) for seed in seeds], axis=0)
    per_row_seed_mean = deltas.mean(axis=0)
    rng = np.random.default_rng(rng_seed)
    values = np.empty(resamples)
    for index in range(resamples):
        values[index] = float(per_row_seed_mean[rng.integers(0, len(per_row_seed_mean), len(per_row_seed_mean))].mean())
    return {
        "mean_delta_mae": float(per_row_seed_mean.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "resamples": resamples,
    }


def checks_from_rows(rows: list[dict[str, Any]], ci_high: float, protocol: dict[str, Any] | None = None) -> tuple[dict[str, float], dict[str, float], dict[str, bool]]:
    protocol = read_json(FORMAL_PROTOCOL_PATH) if protocol is None else protocol
    numeric_keys = [key for key, value in rows[0].items() if key != "seed" and isinstance(value, (int, float))]
    means = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
    standard_deviations = {key: float(np.std([float(row[key]) for row in rows], ddof=1)) for key in numeric_keys}
    checks = {
        "mean_mae_improvement_at_least_0p005": means["mae_improvement"] >= float(protocol["mean_mae_improvement_min"]),
        "mae_improves_two_of_three": sum(float(row["mae_improvement"]) > 0 for row in rows) >= int(protocol["mae_improves_min_seeds"]),
        "mean_exact_delta_noninferior": means["delta_exact"] >= float(protocol["mean_exact_delta_min"]),
        "exact_noninferior_two_of_three": sum(int(row["exp51_exact_correct"]) >= int(row["b0_exact_correct"]) - int(protocol["exact_max_correct_drop_per_seed"]) for row in rows) >= int(protocol["exact_noninferior_min_seeds"]),
        "mean_kendall_delta_noninferior": means["delta_kendall"] >= float(protocol["mean_kendall_delta_min"]),
        "mean_label4_recall_delta_noninferior": means["delta_label4_recall"] >= float(protocol["mean_label4_recall_delta_min"]),
        "mean_label5_recall_delta_noninferior": means["delta_label5_recall"] >= float(protocol["mean_label5_recall_delta_min"]),
        "mean_l2h_does_not_increase": means["delta_l2h_count"] <= float(protocol["mean_l2h_delta_max"]),
        "mean_bias_or_l2h_improves": means["abs_bias_improvement"] >= float(protocol["mean_abs_bias_improvement_min"]) or means["delta_l2h_count"] <= -float(protocol["mean_l2h_improvement_alternative"]),
        "paired_mae_ci_upper_below_zero": ci_high < float(protocol["paired_mae_ci_high_strict_max"]),
        "finite": all(math.isfinite(float(value)) for row in rows for value in row.values() if isinstance(value, (int, float))),
    }
    return means, standard_deviations, checks


def formal_gate(resamples: int | None = None) -> dict[str, Any]:
    formal_lock = verify_formal_lock()
    protocol = read_json(FORMAL_PROTOCOL_PATH)
    protocol_seeds = tuple(int(seed) for seed in protocol["seeds"])
    if protocol_seeds != FORMAL_SEEDS:
        raise ValueError(f"Formal seed mismatch: {protocol_seeds} != {FORMAL_SEEDS}")
    rows = [comparison_row(seed, protocol) for seed in protocol_seeds]
    actual_resamples = int(protocol["bootstrap_resamples"]) if resamples is None else resamples
    ci = bootstrap_ci(protocol_seeds, actual_resamples, int(protocol["bootstrap_rng_seed"]))
    means, standard_deviations, checks = checks_from_rows(rows, ci["ci_high"], protocol)
    passed = all(checks.values())
    return {
        "status": "EXP51_FORMAL_PASS" if passed else "EXP51_FORMAL_NO_GO",
        "passed": passed,
        "checks": checks,
        "per_seed": rows,
        "mean": means,
        "standard_deviation": standard_deviations,
        "paired_bootstrap": ci,
        "formal_protocol": protocol,
        "formal_protocol_commit": formal_lock["protocol_commit"],
        "formal_lock_manifest_sha256": formal_lock["manifest_sha256"],
        "test_access_count": 0,
    }


def main() -> None:
    decision = formal_gate()
    decision_dir = OUTPUT_ROOT / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    path = decision_dir / "formal_decision.json"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUTPUT_ROOT / "tables" / "formal_comparison.csv", decision["per_seed"])
    write_text(OUTPUT_ROOT / "formal_decision.md", "# Exp51 formal decision\n\nStatus: **" + decision["status"] + "**\n\n" + "\n".join(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in decision["checks"].items()) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
