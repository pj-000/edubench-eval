"""Diagnose Exp28 selective hard-relabeling trade-offs from dev-only summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/"
    "tables/exp28e_training_variant_summary.csv"
)
DEFAULT_DEV_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/"
    "tables/exp28e_multiseed_dev_summary.csv"
)
DEFAULT_DEV_DECISION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/"
    "decision/exp28e_multiseed_dev_decision.json"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28i_teacher_relabeling_diagnosis_seed42"
)
BASELINE = "b0_original_human"
MAIN = "b2_selective_dual_teacher"
RANDOM = "b4_random_transition_control"
METRICS = (
    "MAE_label",
    "Signed Bias label",
    "Exact Match",
    "Quadratic Weighted Kappa",
    "Kendall tau",
    "Bin Agreement",
    "low_to_high_rate",
    "high_to_mid_or_low_rate",
    "Acc@1",
    "Acc@2",
    "Acc@3",
    "Acc@4",
    "Acc@5",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.data_summary, args.dev_summary, args.dev_decision):
        if not path.exists():
            raise FileNotFoundError(path)
    data = {row["variant"]: row for row in read_csv(args.data_summary)}
    dev = {row["variant"]: row for row in read_csv(args.dev_summary)}
    decision = json.loads(args.dev_decision.read_text(encoding="utf-8"))
    required = {BASELINE, MAIN, RANDOM}
    if not required <= set(data) or not required <= set(dev):
        raise ValueError("B0, B2, and B4 summaries are required")

    metric_rows = []
    for metric in METRICS:
        baseline = float(dev[BASELINE][f"{metric}_mean"])
        main = float(dev[MAIN][f"{metric}_mean"])
        random = float(dev[RANDOM][f"{metric}_mean"])
        metric_rows.append(
            {
                "metric": metric,
                "b0_original": baseline,
                "b2_selective": main,
                "b4_random": random,
                "b2_minus_b0": main - baseline,
                "b2_minus_b4": main - random,
            }
        )
    write_csv(
        args.out_dir / "tables" / "exp28i_metric_tradeoff.csv",
        metric_rows,
        ["metric", "b0_original", "b2_selective", "b4_random", "b2_minus_b0", "b2_minus_b4"],
    )

    label_rows = []
    for label in range(1, 6):
        baseline_count = int(data[BASELINE][f"label_{label}"])
        main_count = int(data[MAIN][f"label_{label}"])
        random_count = int(data[RANDOM][f"label_{label}"])
        baseline_recall = float(dev[BASELINE][f"Acc@{label}_mean"])
        main_recall = float(dev[MAIN][f"Acc@{label}_mean"])
        random_recall = float(dev[RANDOM][f"Acc@{label}_mean"])
        label_rows.append(
            {
                "label": label,
                "b0_train_count": baseline_count,
                "b2_train_count": main_count,
                "b4_train_count": random_count,
                "b2_minus_b0_train_count": main_count - baseline_count,
                "b2_dev_recall_minus_b0": main_recall - baseline_recall,
                "b2_dev_recall_minus_b4": main_recall - random_recall,
            }
        )
    write_csv(
        args.out_dir / "tables" / "exp28i_label_shift_diagnosis.csv",
        label_rows,
        [
            "label", "b0_train_count", "b2_train_count", "b4_train_count",
            "b2_minus_b0_train_count", "b2_dev_recall_minus_b0", "b2_dev_recall_minus_b4",
        ],
    )

    lookup = {row["metric"]: row for row in metric_rows}
    overall_worse = (
        lookup["MAE_label"]["b2_minus_b0"] > 0
        or lookup["Exact Match"]["b2_minus_b0"] < 0
        or lookup["Kendall tau"]["b2_minus_b0"] < 0
    )
    risk_better = lookup["low_to_high_rate"]["b2_minus_b0"] < 0
    targeting_better_than_random = lookup["low_to_high_rate"]["b2_minus_b4"] < 0
    status = (
        "SELECTIVE_HARD_RELABELING_SUPPORTED"
        if not overall_worse and risk_better
        else "RISK_ACCURACY_TRADEOFF"
        if risk_better
        else "SELECTIVE_HARD_RELABELING_NOT_SUPPORTED"
    )
    result = {
        "status": status,
        "dev_campaign_status": decision.get("status"),
        "overall_metrics_worse_than_b0": overall_worse,
        "low_to_high_better_than_b0": risk_better,
        "low_to_high_better_than_random_control": targeting_better_than_random,
        "test_read": False,
        "recommend_open_test": decision.get("status") == "READY_FOR_BOOTSTRAP_AND_FINAL_DEV_LOCK",
        "interpretation": (
            "Selective targeting carries risk information, but hard score replacement shifts the class boundary "
            "and is not supported as the final training-label method."
            if status == "RISK_ACCURACY_TRADEOFF"
            else "Follow the locked dev decision and confidence intervals."
        ),
    }
    write_json(args.out_dir / "decision" / "exp28i_teacher_relabeling_decision.json", result)
    report = f"""# Exp28I Teacher Relabeling Diagnosis

- decision: **{status}**
- dev campaign: **{decision.get('status')}**
- B2 overall worse than B0: {overall_worse}
- B2 low-to-high better than B0: {risk_better}
- B2 low-to-high better than matched random relabeling: {targeting_better_than_random}
- test read: no

This diagnosis separates two questions: whether the selected rows contain useful risk information,
and whether replacing their hard labels is a valid training intervention. A lower low-to-high rate
with worse MAE, Exact Match, or Kendall tau supports the former but rejects the latter. Teacher
scores remain model-generated silver supervision and are not treated as corrected human gold.
"""
    report_path = args.out_dir / "reports" / "exp28i_teacher_relabeling_diagnosis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-summary", type=Path, default=DEFAULT_DATA_SUMMARY)
    parser.add_argument("--dev-summary", type=Path, default=DEFAULT_DEV_SUMMARY)
    parser.add_argument("--dev-decision", type=Path, default=DEFAULT_DEV_DECISION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), ensure_ascii=False, sort_keys=True))
