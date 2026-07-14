"""Collect headline dev runs and apply the locked dev gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp43_rubimor.common import ROOT, RUN_ROOT, SEEDS, mean_std, read_jsonl, sha256_file, write_csv, write_json

VARIANTS = ("E0", "E3", "E5", "E6", "E6N")
METRICS = ("MAE", "QWK", "Exact_Match", "Kendall_tau", "Signed_Bias", "abs_Signed_Bias", "Bin_Agreement", "low_to_high_rate", "high_to_low_rate", "label1_recall", "label2_recall", "label3_recall", "label4_recall", "label5_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); rows = []; checkpoints = []
    for variant in VARIANTS:
        for seed in SEEDS:
            root = args.run_root / "headline" / variant / f"seed_{seed}"
            summary_path, checkpoint = root / "run_summary.json", root / "best_checkpoint.pt"
            if not summary_path.exists() or not checkpoint.exists():
                raise FileNotFoundError(f"Incomplete headline run: {root}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "COMPLETED" or summary.get("test_access_count") != 0:
                raise RuntimeError(f"Invalid headline summary: {summary_path}")
            rows.append({"variant": variant, "seed": seed, "selected_epoch": summary["selected_epoch"], **summary["metrics"]})
            checkpoints.append({"variant": variant, "seed": seed, "selected_epoch": summary["selected_epoch"], "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)})
    summary_rows = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        out = {"variant": variant, "seeds": len(subset)}
        for metric in METRICS:
            out[f"{metric}_mean"], out[f"{metric}_std"] = mean_std([float(row[metric]) for row in subset])
        summary_rows.append(out)
    by_variant = {row["variant"]: row for row in summary_rows}
    e0, e3, e6, e6n = (by_variant[name] for name in ("E0", "E3", "E6", "E6N"))
    primary = float(e0["MAE_mean"])-float(e6["MAE_mean"]) >= .015 or float(e6["QWK_mean"])-float(e0["QWK_mean"]) >= .020 or float(e6["Kendall_tau_mean"])-float(e0["Kendall_tau_mean"]) >= .015
    favorable_seeds = sum(any((float(next(r for r in rows if r["variant"]=="E6" and r["seed"]==seed)[metric]) - float(next(r for r in rows if r["variant"]=="E0" and r["seed"]==seed)[metric])) * (-1 if metric=="MAE" else 1) > 0 for metric in ("MAE","QWK","Kendall_tau")) for seed in SEEDS) >= 2
    protection = float(e6["Exact_Match_mean"]) >= float(e0["Exact_Match_mean"])-.005 and float(e6["Bin_Agreement_mean"]) >= float(e0["Bin_Agreement_mean"])-.005 and float(e6["abs_Signed_Bias_mean"]) <= float(e0["abs_Signed_Bias_mean"])+.01 and float(e6["label5_recall_mean"]) >= float(e0["label5_recall_mean"])-.02 and float(e6["high_to_low_rate_mean"]) <= float(e0["high_to_low_rate_mean"])+.01
    incremental = float(e3["MAE_mean"])-float(e6["MAE_mean"]) >= .005 or float(e6["QWK_mean"])-float(e3["QWK_mean"]) >= .010 or float(e6["Kendall_tau_mean"])-float(e3["Kendall_tau_mean"]) >= .0075
    noise = float(e6n["MAE_mean"])-float(e6["MAE_mean"]) >= .003 or float(e6["QWK_mean"])-float(e6n["QWK_mean"]) >= .005
    groupcv = json.loads((args.out_dir / "decision/exp43_groupcv_decision.json").read_text(encoding="utf-8"))
    direction = groupcv["status"] in {"RUBIMOR_FULL_GROUPCV_GO", "RUBIMOR_OVERALL_GROUPCV_GO"}
    status = "HEADLINE_DEV_GO" if all((primary, favorable_seeds, protection, incremental, noise, direction)) else "HEADLINE_DEV_STOP"
    write_csv(args.out_dir / "tables/exp43_headline_dev_metrics.csv", rows)
    write_csv(args.out_dir / "tables/exp43_headline_dev_multiseed_summary.csv", summary_rows)
    write_csv(args.out_dir / "tables/exp43_headline_selected_checkpoints.csv", checkpoints)
    write_json(args.out_dir / "hashes/exp43_checkpoint_hashes.json", {"checkpoints": checkpoints})
    decision = {"status": status, "checks": {"primary_gain": primary, "two_of_three_seeds": favorable_seeds, "protection": protection, "incremental_over_E3": incremental, "beats_E6N": noise, "groupcv_direction_reproduced": direction}, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp43_headline_dev_decision.json", decision)
    report = ["# Exp43 Headline Dev Report", "", f"- Decision: **{status}**", "", *[f"- {key}: {value}" for key,value in decision["checks"].items()], "", "Dev selected the highest Exact Match, then lower MAE, then earlier epoch. No test data were accessed."]
    (args.out_dir / "reports/exp43_headline_dev_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()

