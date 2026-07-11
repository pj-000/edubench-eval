"""Collect the locked Exp28G B0/B2 three-seed final test results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test"
)
VARIANTS = ("b0_original_human", "b2_selective_dual_teacher")
SEEDS = (42, 43, 44)
METRICS = (
    "MAE_label", "Signed Bias label", "Exact Match", "Quadratic Weighted Kappa", "Kendall tau",
    "Bin Agreement", "low_to_high_rate", "high_to_mid_or_low_rate",
    "Acc@1", "Acc@2", "Acc@3", "Acc@4", "Acc@5",
)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = args.out_dir / "runs" / variant / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            test_rows = [row for row in payload if row.get("split") == "test"]
            if len(test_rows) != 1:
                raise ValueError(f"Expected one test metric row: {path}")
            rows.append({"variant": variant, "seed": seed, **test_rows[0]})
    summary_rows = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        summary: dict[str, Any] = {"variant": variant, "n_seeds": 3}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
        summary_rows.append(summary)
    baseline, main = summary_rows
    deltas = {metric: float(main[f"{metric}_mean"]) - float(baseline[f"{metric}_mean"]) for metric in METRICS}
    write_csv(args.out_dir / "tables" / "exp28g_test_seed_metrics.csv", rows, list(rows[0]))
    write_csv(args.out_dir / "tables" / "exp28g_test_multiseed_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(
        args.out_dir / "tables" / "exp28g_test_main_vs_baseline.csv",
        [{"metric": metric, "baseline": baseline[f"{metric}_mean"], "main": main[f"{metric}_mean"], "delta": delta} for metric, delta in deltas.items()],
        ["metric", "baseline", "main", "delta"],
    )
    decision = {
        "status": "FINAL_TEST_COMPLETED",
        "baseline": "b0_original_human",
        "main": "b2_selective_dual_teacher",
        "test_rows": 2218,
        "seeds": list(SEEDS),
        "test_used_for_selection": False,
        "deltas": deltas,
        "paper_claim": "pending paired test bootstrap and interpretation",
    }
    path = args.out_dir / "decision" / "exp28g_final_test_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Exp28G One-Shot Final Test",
            "",
            "- status: **FINAL TEST COMPLETED**",
            "- test rows: 2218",
            "- variants: B0 original labels and B2 selective dual teacher",
            "- seeds: 42, 43, 44",
            "- test used for selection: no",
            "",
            "Final claims remain pending paired seed/triple bootstrap. No method or checkpoint may be changed after this result.",
        ]
    )
    report_path = args.out_dir / "reports" / "exp28g_final_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
