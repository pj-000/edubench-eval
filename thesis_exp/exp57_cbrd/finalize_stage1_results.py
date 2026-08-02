"""Post-result integrity audit for the completed Exp57 Stage 1 campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp57_cbrd.metrics import add_boundary_diagnostics, compute_metrics


SCIENTIFIC = (
    "dual_hard",
    "consensus_only",
    "routed_hmsa",
    "residual_only",
    "sign_flipped",
    "shuffled_residual",
)
SEEDS = (42, 43, 44)
PRIMARY_KEYS = ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    run_root = OUTPUT_ROOT / "runs"
    expected = [(variant, seed) for variant in SCIENTIFIC for seed in SEEDS]
    expected.append(("detached_soft", 42))
    rows: list[dict[str, Any]] = []
    all_checks: list[bool] = []
    train_hashes: set[str] = set()
    dev_hashes: set[str] = set()
    source_locks: set[str] = set()
    for variant, seed in expected:
        directory = run_root / variant / f"seed_{seed}"
        summary = read_json(directory / "run_summary.json")
        selected = read_json(directory / "selected_dev_metrics.json")
        history = read_json(directory / "dev_metrics_history.json")
        predictions = read_jsonl(directory / "predictions" / "predictions_dev.jsonl")
        recomputed = add_boundary_diagnostics(compute_metrics(predictions), predictions)
        metric_differences = {
            key: abs(float(selected[key]) - float(recomputed[key])) for key in PRIMARY_KEYS
        }
        best_exact = max(float(item["Exact_rounded"]) for item in history)
        expected_epoch = min(
            int(item["epoch"])
            for item in history
            if float(item["Exact_rounded"]) == best_exact
        )
        checks = {
            "completed": summary.get("status") == "COMPLETED",
            "variant_seed_match": summary.get("variant") == variant and int(summary.get("seed")) == seed,
            "test_access_zero": int(summary.get("test_access_count", -1)) == 0,
            "dev_prediction_count_664": len(predictions) == 664,
            "dev_record_ids_unique": len({str(item["record_id"]) for item in predictions}) == 664,
            "primary_metrics_recompute": max(metric_differences.values()) <= 1e-12,
            "checkpoint_rule_recomputed": int(summary["selected_epoch"]) == expected_epoch,
            "ten_epoch_history": len(history) == 10,
            "checkpoint_exists": Path(summary["checkpoint_path"]).joinpath("state_dict.pt").is_file(),
        }
        all_checks.extend(checks.values())
        train_hashes.add(str(summary["train_text_hash"]))
        dev_hashes.add(str(summary["dev_text_hash"]))
        source_locks.add(str(summary["frozen_contract_files"]["source_lock_sha256"]))
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "selected_epoch": summary["selected_epoch"],
                "MAE_human_mean": selected["MAE_human_mean"],
                "Exact_rounded": selected["Exact_rounded"],
                "Kendall_human_mean": selected["Kendall_human_mean"],
                "checks": checks,
                "metric_recompute_abs_differences": metric_differences,
                "source_lock_sha256": summary["frozen_contract_files"]["source_lock_sha256"],
            }
        )
    pilot = read_json(OUTPUT_ROOT / "decision" / "stage1_pilot_decision.json")
    development = read_json(OUTPUT_ROOT / "decision" / "stage1_development_decision.json")
    test_artifacts = [
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.is_file() and ("predictions_test" in path.name or "logits_test" in path.name)
    ]
    canceled_a6000_summaries = list((OUTPUT_ROOT / "runs_a6000").glob("*/seed_*/run_summary.json"))
    global_checks = {
        "nineteen_expected_runs": len(rows) == 19,
        "all_per_run_checks": all(all_checks),
        "single_train_hash": len(train_hashes) == 1,
        "single_dev_hash": len(dev_hashes) == 1,
        "no_test_prediction_artifacts": not test_artifacts,
        "no_completed_a6000_runs": not canceled_a6000_summaries,
        "pilot_gate_recorded": pilot.get("status") == "PILOT_NO_GO",
        "development_gate_recorded": development.get("status") == "GO_EXTEND_PRIMARY_TO_SEEDS_45_46",
    }
    report = {
        "status": (
            "INTEGRITY_PASS_WITH_PILOT_PARITY_QUALIFICATION"
            if all(global_checks.values())
            else "INTEGRITY_FAIL"
        ),
        "global_checks": global_checks,
        "runs": rows,
        "train_text_hashes": sorted(train_hashes),
        "dev_text_hashes": sorted(dev_hashes),
        "source_lock_hashes": sorted(source_locks),
        "pilot_status": pilot["status"],
        "development_status": development["status"],
        "development_gates": development["gates"],
        "test_artifacts": test_artifacts,
        "test_access_count": 0,
        "interpretation": "All result artifacts are internally complete and recomputable. The development mechanism comparisons are diagnostically strong, but the preregistered historical full-run parity gate failed and must remain an explicit qualification.",
    }
    write_json(OUTPUT_ROOT / "audit" / "stage1_final_integrity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == "INTEGRITY_FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
