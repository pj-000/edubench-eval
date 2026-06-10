"""Collect QD-S2 results and compare against QD-B0/QD-B1/QD-S1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_training import (
    EXP06_TRAINING_RUNS_DIR,
    EXP06_TRAINING_TABLES_DIR,
    QD_S2_RUN_DIR,
    QD_S2_RUN_ID,
    ensure_exp06_training_dirs,
)
from thesis_exp.src.edujudge.exp06_training.collect_qds1_results import (
    SUMMARY_FIELDS,
    _as_float,
    _fmt,
    _metric_value,
    _run_split_extras,
    _test_row,
    load_baseline_summary,
    load_qds1_summary,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def load_qds2_summary(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "tables" / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing QD-S2 metrics summary: {relpath(path)}")
    rows = _read_csv_if_exists(path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    out = []
    for row in rows:
        split = row.get("split", "")
        extras = _run_split_extras(run_dir, split)
        out.append(
            {
                "run_id": QD_S2_RUN_ID,
                "setting": "human_plus_384_synthetic L1 weighted ordinal",
                "split": split,
                "n": row.get("n", ""),
                "Accuracy": _metric_value(row, "Accuracy", "Exact Match"),
                "MAE_label": row.get("MAE_label", ""),
                "MAE_expected": row.get("MAE_expected", ""),
                "Macro-F1": row.get("Macro-F1", ""),
                "Quadratic Weighted Kappa": row.get("Quadratic Weighted Kappa", ""),
                "Kendall tau": row.get("Kendall tau", ""),
                "Spearman rho": row.get("Spearman rho", ""),
                "severe_error_rate": row.get("severe_error_rate", ""),
                "low_to_high_rate": _metric_value(row, "low_to_high_rate") or extras["low_to_high_rate"],
                "Acc@5": extras["Acc@5"],
                "high_to_mid_or_low_rate": extras["high_to_mid_or_low_rate"],
                "best_epoch": str(metadata.get("best_epoch") or row.get("epoch", "")),
                "best_global_step": str(metadata.get("best_global_step") or row.get("global_step", "")),
            }
        )
    return out


def delta_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    qds2 = _test_row(rows, "QD-S2")
    out = []
    for baseline in ["QD-B0", "QD-B1", "QD-S1"]:
        base = _test_row(rows, baseline)
        if not base or not qds2:
            continue
        for metric, direction in [
            ("MAE_label", "lower_better"),
            ("Accuracy", "higher_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("low_to_high_rate", "lower_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
        ]:
            q_value = _as_float(qds2.get(metric))
            b_value = _as_float(base.get(metric))
            if q_value is None or b_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-S2_minus_{baseline}",
                    "metric": metric,
                    "baseline_value": b_value,
                    "qds2_value": q_value,
                    "delta": q_value - b_value,
                    "direction": direction,
                }
            )
    return out


def class_weight_lines(run_dir: Path) -> list[str]:
    path = run_dir / "tables" / "class_weights.csv"
    if not path.exists():
        return ["Class weights: NA"]
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lines = ["Class weights computed from QD-S2 train only:"]
    for row in rows:
        lines.append(
            f"- label {row.get('label_5')}: count={row.get('train_count')}, "
            f"weight={_fmt(row.get('clipped_weight'), 6)}"
        )
    return lines


def qds2_gate(rows: list[dict[str, str]], deltas: list[dict[str, Any]]) -> str:
    qds2 = _test_row(rows, "QD-S2")
    qds1 = _test_row(rows, "QD-S1")
    b1 = _test_row(rows, "QD-B1")
    if not qds2 or not qds1 or not b1:
        return "REVIEW_REQUIRED"
    low_vs_s1 = next((row for row in deltas if row["comparison"] == "QD-S2_minus_QD-S1" and row["metric"] == "low_to_high_rate"), {})
    low_vs_b1 = next((row for row in deltas if row["comparison"] == "QD-S2_minus_QD-B1" and row["metric"] == "low_to_high_rate"), {})
    acc5_vs_b1 = next((row for row in deltas if row["comparison"] == "QD-S2_minus_QD-B1" and row["metric"] == "Acc@5"), {})
    improves_s1 = _as_float(low_vs_s1.get("delta")) is not None and float(low_vs_s1["delta"]) < 0
    beats_b1 = _as_float(low_vs_b1.get("delta")) is not None and float(low_vs_b1["delta"]) <= 0
    acc5_ok = _as_float(acc5_vs_b1.get("delta")) is not None and float(acc5_vs_b1["delta"]) > -0.03
    return "YES" if improves_s1 and beats_b1 and acc5_ok else "REVIEW_REQUIRED"


def write_reports(rows: list[dict[str, str]], deltas: list[dict[str, Any]], run_dir: Path) -> None:
    b0 = _test_row(rows, "QD-B0")
    b1 = _test_row(rows, "QD-B1")
    qds1 = _test_row(rows, "QD-S1")
    qds2 = _test_row(rows, "QD-S2")
    can_qds3 = qds2_gate(rows, deltas)

    report_lines = [
        "# Exp6 QD-S2 Human + Synthetic L1 Weighted Ordinal",
        "",
        "This report compares QD-S2 against QD-B0, QD-B1, and QD-S1.",
        "Synthetic rows are pseudo-label augmentation rows and are not human labels.",
        "",
        "## Test Metrics",
        "",
        "| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1, qds2]:
        if row:
            report_lines.append(
                f"| {row['run_id']} | {_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
                f"{_fmt(row.get('Accuracy'))} | {_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} |"
            )
    report_lines.extend(["", "## Class Weights", "", *class_weight_lines(run_dir), "", "## Gate", ""])
    report_lines.extend(
        [
            f"- Can QD-S3 start? **{can_qds3}**",
            "- Model weights/checkpoints are written under `thesis_exp/artifacts/` and must not be committed.",
        ]
    )
    report_text = "\n".join(report_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "report_qds2.md", report_text)
    write_text(run_dir / "report.md", report_text)

    review_lines = [
        "# Exp6 QD-S2 Review Package",
        "",
        f"Can QD-S3 start? **{can_qds3}**",
        "",
        "QD-S3 should start only after reviewing whether QD-S2 improves low-score behavior without unacceptable high-score damage.",
        "",
        "Remaining blockers:",
        "",
    ]
    if can_qds3 == "YES":
        review_lines.append("- None for QD-S3 planning.")
    else:
        review_lines.append("- Review QD-S2 against QD-B1 and QD-S1 before starting QD-S3.")
    review_text = "\n".join(review_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "review_package_qds2.md", review_text)
    write_text(run_dir / "review_package.md", review_text)

    notion_lines = [
        "# Exp6 QD-S2 Synthetic + L1 Summary",
        "",
        "- Setting: question-disjoint `question_seed42`.",
        "- Train: 3326 human + 384 final synthetic low-score pseudo-label rows.",
        "- Dev/test: human-only.",
        "- Loss: L1 weighted ordinal; class weights computed from QD-S2 train only.",
        "",
        "## Test Comparison",
        "",
        "| Run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1, qds2]:
        if row:
            notion_lines.append(
                f"| {row['run_id']} | {_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
                f"{_fmt(row.get('Accuracy'))} | {_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} |"
            )
    notion_lines.extend(["", "## Class Weights", "", *class_weight_lines(run_dir), "", f"- Can QD-S3 start: **{can_qds3}**"])
    notion_text = "\n".join(notion_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "notion_exp06_qds2_summary.md", notion_text)
    write_text(run_dir / "notion_exp06_qds2_summary.md", notion_text)


def collect(run_dir: Path = QD_S2_RUN_DIR) -> None:
    ensure_exp06_training_dirs()
    baseline_rows = load_baseline_summary()
    qds1_rows = load_qds1_summary()
    qds2_rows = load_qds2_summary(run_dir)
    rows = baseline_rows + qds1_rows + qds2_rows
    deltas = delta_rows(rows)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_training_results_summary.csv", rows, SUMMARY_FIELDS)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_qds2_comparison.csv", deltas)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_low_score_comparison.csv", [row for row in deltas if row["metric"] == "low_to_high_rate"])
    write_csv(
        EXP06_TRAINING_TABLES_DIR / "exp06_high_score_comparison.csv",
        [row for row in deltas if row["metric"] in {"Acc@5", "high_to_mid_or_low_rate"}],
    )
    write_reports(rows, deltas, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp6 QD-S2 training results.")
    parser.add_argument("--run_dir", type=Path, default=QD_S2_RUN_DIR)
    args = parser.parse_args()
    collect(args.run_dir)
    print(f"Wrote Exp6 QD-S2 summaries: {relpath(EXP06_TRAINING_RUNS_DIR)}")


if __name__ == "__main__":
    main()
