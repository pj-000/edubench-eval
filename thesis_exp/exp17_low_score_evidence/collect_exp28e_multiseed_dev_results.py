"""Collect Exp28E three-seed dev-only ordinary-CE results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


DEFAULT_RUN_ROOT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/runs"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev"
)
VARIANTS = (
    "b0_original_human",
    "b1_primary_teacher_all",
    "b2_selective_dual_teacher",
    "b3_filter_unresolved",
    "b4_random_transition_control",
)
SEEDS = (42, 43, 44)
METRICS = (
    "MAE_label",
    "Signed Bias label",
    "Exact Match",
    "Quadratic Weighted Kappa",
    "Kendall tau",
    "low_to_high_rate",
    "high_to_mid_or_low_rate",
    "Acc@1",
    "Acc@2",
    "Acc@5",
)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def collect(args: argparse.Namespace) -> dict[str, Any]:
    selected_rows = []
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = args.run_root / variant / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("split") != "dev":
                raise ValueError(f"Expected exactly one dev-only metric row: {path}")
            selected_rows.append({"variant": variant, "seed": seed, **payload[0]})

    if missing:
        decision = {
            "status": "WAITING_FOR_DEV_TRAINING",
            "completed_runs": len(selected_rows),
            "expected_runs": len(VARIANTS) * len(SEEDS),
            "missing_runs": missing,
            "test_read": False,
        }
        write_json(args.out_dir / "decision" / "exp28e_multiseed_dev_decision.json", decision)
        return decision

    summary_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[row["variant"]].append(row)
    for variant in VARIANTS:
        summary: dict[str, Any] = {"variant": variant, "seeds": "42|43|44", "n_seeds": 3}
        for metric in METRICS:
            values = [float(row[metric]) for row in grouped[variant] if finite(row.get(metric))]
            summary[f"{metric}_mean"] = mean(values) if values else ""
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
        summary_rows.append(summary)

    baseline = next(row for row in summary_rows if row["variant"] == "b0_original_human")
    comparison_rows = []
    for row in summary_rows:
        if row["variant"] == "b0_original_human":
            continue
        comparison = {"variant": row["variant"], "baseline": "b0_original_human"}
        for metric in METRICS:
            left = row.get(f"{metric}_mean")
            right = baseline.get(f"{metric}_mean")
            comparison[f"delta_{metric}"] = float(left) - float(right) if finite(left) and finite(right) else ""
        comparison_rows.append(comparison)

    write_csv(
        args.out_dir / "tables" / "exp28e_selected_dev_metrics.csv",
        selected_rows,
        ["variant", "seed", *[key for key in selected_rows[0] if key not in {"variant", "seed"}]],
    )
    summary_fields = list(summary_rows[0])
    write_csv(args.out_dir / "tables" / "exp28e_multiseed_dev_summary.csv", summary_rows, summary_fields)
    write_csv(
        args.out_dir / "tables" / "exp28e_dev_deltas_vs_original.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )

    # Dev-only recommendation: the main method must beat the random-transition
    # control and preserve aggregate/high-score quality. Formal significance is
    # added by the later question/triple-cluster bootstrap step.
    selective = next(row for row in summary_rows if row["variant"] == "b2_selective_dual_teacher")
    random_control = next(row for row in summary_rows if row["variant"] == "b4_random_transition_control")
    checks = {
        "mae_not_worse_than_baseline": float(selective["MAE_label_mean"]) <= float(baseline["MAE_label_mean"]) + 0.01,
        "exact_not_worse_than_baseline": float(selective["Exact Match_mean"]) >= float(baseline["Exact Match_mean"]) - 0.01,
        "kendall_not_worse_than_baseline": float(selective["Kendall tau_mean"]) >= float(baseline["Kendall tau_mean"]) - 0.01,
        "low_to_high_improves": float(selective["low_to_high_rate_mean"]) < float(baseline["low_to_high_rate_mean"]),
        "label5_guard": float(selective["Acc@5_mean"]) >= float(baseline["Acc@5_mean"]) - 0.03,
        "beats_random_control_on_mae": float(selective["MAE_label_mean"]) < float(random_control["MAE_label_mean"]),
    }
    passed = all(checks.values())
    decision = {
        "status": "READY_FOR_BOOTSTRAP_AND_FINAL_DEV_LOCK" if passed else "DEV_SUCCESS_CRITERIA_NOT_MET",
        "checks": checks,
        "recommended_variant": "b2_selective_dual_teacher" if passed else None,
        "test_read": False,
        "test_open_authorized": False,
        "next_action": (
            "Run paired clustered bootstrap, freeze the final data variant/checkpoint, then request final test authorization."
            if passed
            else "Analyze data/protocol failure on train and dev; do not open test."
        ),
    }
    write_json(args.out_dir / "decision" / "exp28e_multiseed_dev_decision.json", decision)
    report = f"""# Exp28E Three-Seed Dev-Only Results

- runs: {len(selected_rows)}/{len(VARIANTS) * len(SEEDS)}
- model: Qwen3-Reranker-0.6B
- input: question + answer + evaluation dimension
- loss: ordinary cross-entropy
- epochs: 10
- effective batch size: 128
- checkpoint selection: validation accuracy
- test read: no
- decision: **{decision['status']}**

The selective method is compared with the original-label baseline, full primary-teacher labels,
uncertainty filtering, and a matched random-transition control. Test remains closed until the
paired clustered bootstrap and final dev lock pass.
"""
    report_path = args.out_dir / "reports" / "exp28e_multiseed_dev_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), ensure_ascii=False, sort_keys=True))
