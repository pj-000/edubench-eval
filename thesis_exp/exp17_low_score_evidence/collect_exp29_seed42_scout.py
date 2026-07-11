"""Collect the Exp29 seed42 dev-only dual-target scout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_EXP28 = Path("thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/runs")
DEFAULT_OUT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp29_dual_target_ce_seed42")
VARIANTS = ("c1_audited_dual_target", "c2_selected_exposure_control", "c3_random_dual_target_control")
METRICS = ("MAE_label", "Exact Match", "Kendall tau", "Bin Agreement", "low_to_high_rate", "Acc@1", "Acc@2", "Acc@3", "Acc@4", "Acc@5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp28-runs", type=Path, default=DEFAULT_EXP28)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    baseline_path = args.exp28_runs / "b0_original_human" / "seed_42" / "metrics.json"
    baseline = json.loads(baseline_path.read_text())[0]
    rows = [{"variant": "b0_original_human", **baseline}]
    for variant in VARIANTS:
        path = args.out_dir / "runs" / variant / "seed_42" / "metrics.json"
        rows.append({"variant": variant, **json.loads(path.read_text())[0]})
    table = args.out_dir / "tables" / "exp29_seed42_dev_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    by = {row["variant"]: row for row in rows}; base = by["b0_original_human"]; main = by[VARIANTS[0]]
    checks = {
        "mae_guard": main["MAE_label"] <= base["MAE_label"] + 0.01,
        "exact_guard": main["Exact Match"] >= base["Exact Match"] - 0.01,
        "kendall_guard": main["Kendall tau"] >= base["Kendall tau"] - 0.01,
        "bin_guard": main["Bin Agreement"] >= base["Bin Agreement"] - 0.005,
        "low_to_high_improves": main["low_to_high_rate"] < base["low_to_high_rate"],
        "label5_guard": main["Acc@5"] >= base["Acc@5"] - 0.03,
        "beats_exposure_control_mae": main["MAE_label"] < by[VARIANTS[1]]["MAE_label"],
        "beats_random_control_mae": main["MAE_label"] < by[VARIANTS[2]]["MAE_label"],
    }
    decision = {
        "status": "READY_FOR_SEEDS_43_44" if all(checks.values()) else "SEED42_SCOUT_NOT_SUPPORTED",
        "checks": checks,
        "test_read": False,
        "next_action": "run seeds 43/44" if all(checks.values()) else "stop dual-target augmentation and keep test closed",
    }
    path = args.out_dir / "decision" / "exp29_seed42_scout_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(decision, indent=2) + "\n")
    report = args.out_dir / "reports" / "exp29_seed42_scout_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"# Exp29 Seed42 Scout\n\n- decision: **{decision['status']}**\n- test read: no\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
