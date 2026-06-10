"""Collect QD-S3 results and compare against QD-B0/QD-B1/QD-S1/QD-S2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_training import (
    EXP06_TRAINING_RUNS_DIR,
    EXP06_TRAINING_TABLES_DIR,
    QD_S1_RUN_DIR,
    QD_S2_RUN_DIR,
    QD_S3_RUN_DIR,
    QD_S3_RUN_ID,
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
from thesis_exp.src.edujudge.exp06_training.collect_qds2_results import load_qds2_summary
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def load_qds3_summary(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "tables" / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing QD-S3 metrics summary: {relpath(path)}")
    rows = _read_csv_if_exists(path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    out = []
    for row in rows:
        split = row.get("split", "")
        extras = _run_split_extras(run_dir, split)
        out.append(
            {
                "run_id": QD_S3_RUN_ID,
                "setting": "synthetic pretrain then human-only finetune ordinary ordinal",
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
                "best_epoch": str(metadata.get("stage2_best_epoch") or metadata.get("best_epoch") or row.get("epoch", "")),
                "best_global_step": str(
                    metadata.get("stage2_best_global_step") or metadata.get("best_global_step") or row.get("global_step", "")
                ),
            }
        )
    return out


def delta_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    qds3 = _test_row(rows, "QD-S3")
    out = []
    for baseline in ["QD-B0", "QD-B1", "QD-S1", "QD-S2"]:
        base = _test_row(rows, baseline)
        if not base or not qds3:
            continue
        for metric, direction in [
            ("MAE_label", "lower_better"),
            ("Accuracy", "higher_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("low_to_high_rate", "lower_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
        ]:
            q_value = _as_float(qds3.get(metric))
            b_value = _as_float(base.get(metric))
            if q_value is None or b_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-S3_minus_{baseline}",
                    "metric": metric,
                    "baseline_value": b_value,
                    "qds3_value": q_value,
                    "delta": q_value - b_value,
                    "direction": direction,
                }
            )
    return out


def qds3_gate(rows: list[dict[str, str]], deltas: list[dict[str, Any]]) -> str:
    qds3 = _test_row(rows, "QD-S3")
    b0 = _test_row(rows, "QD-B0")
    b1 = _test_row(rows, "QD-B1")
    if not qds3 or not b0 or not b1:
        return "REVIEW_REQUIRED"
    low_vs_b0 = next((row for row in deltas if row["comparison"] == "QD-S3_minus_QD-B0" and row["metric"] == "low_to_high_rate"), {})
    mae_vs_b0 = next((row for row in deltas if row["comparison"] == "QD-S3_minus_QD-B0" and row["metric"] == "MAE_label"), {})
    acc5_vs_b1 = next((row for row in deltas if row["comparison"] == "QD-S3_minus_QD-B1" and row["metric"] == "Acc@5"), {})
    low_improves_b0 = _as_float(low_vs_b0.get("delta")) is not None and float(low_vs_b0["delta"]) < 0
    mae_not_bad = _as_float(mae_vs_b0.get("delta")) is not None and float(mae_vs_b0["delta"]) <= 0.03
    acc5_ok = _as_float(acc5_vs_b1.get("delta")) is not None and float(acc5_vs_b1["delta"]) > -0.03
    return "YES" if low_improves_b0 and mae_not_bad and acc5_ok else "REVIEW_REQUIRED"


def write_reports(rows: list[dict[str, str]], deltas: list[dict[str, Any]], run_dir: Path) -> None:
    b0 = _test_row(rows, "QD-B0")
    b1 = _test_row(rows, "QD-B1")
    qds1 = _test_row(rows, "QD-S1")
    qds2 = _test_row(rows, "QD-S2")
    qds3 = _test_row(rows, "QD-S3")
    matrix_complete = "YES" if qds3 else "NO"
    training_dataset_ready = qds3_gate(rows, deltas)

    report_lines = [
        "# Exp6 QD-S3 Synthetic Pretrain then Human Fine-tune",
        "",
        "This report compares QD-S3 against QD-B0, QD-B1, QD-S1, and QD-S2.",
        "Stage 1 uses synthetic low-score pseudo labels only; stage 2 returns to human-only question_seed42 training.",
        "Synthetic rows are never treated as human labels.",
        "",
        "## Test Metrics",
        "",
        "| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1, qds2, qds3]:
        if row:
            report_lines.append(
                f"| {row['run_id']} | {_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
                f"{_fmt(row.get('Accuracy'))} | {_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} |"
            )
    report_lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Exp6 training matrix complete? **{matrix_complete}**",
            f"- Can Exp6 training conclusion draft start? **{training_dataset_ready}**",
            "- Model weights/checkpoints are written under `thesis_exp/artifacts/` and must not be committed.",
        ]
    )
    report_text = "\n".join(report_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "report_qds3.md", report_text)
    write_text(run_dir / "report.md", report_text)

    review_lines = [
        "# Exp6 QD-S3 Review Package",
        "",
        f"Exp6 training matrix complete? **{matrix_complete}**",
        f"Can Exp6 training conclusion draft start? **{training_dataset_ready}**",
        "",
        "Main review questions:",
        "",
        "1. Does synthetic pretraining reduce low-to-high errors relative to QD-B0?",
        "2. Does stage2 human fine-tuning avoid QD-S1/QD-S2 degradation?",
        "3. Does Acc@5 remain acceptable compared with QD-B1?",
        "",
        "Remaining blockers:",
        "",
    ]
    if training_dataset_ready == "YES":
        review_lines.append("- None for drafting the Exp6 training conclusion, pending human review.")
    else:
        review_lines.append("- Review QD-S3 tradeoffs before claiming synthetic pretraining is beneficial.")
    review_text = "\n".join(review_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "review_package_qds3.md", review_text)
    write_text(run_dir / "review_package.md", review_text)

    notion_lines = [
        "# Exp6 QD-S3 Synthetic Pretrain -> Human Fine-tune Summary",
        "",
        "- Setting: question-disjoint `question_seed42`.",
        "- Stage 1: 384 final synthetic low-score pseudo-label rows.",
        "- Stage 2: 3326 human-only train rows.",
        "- Dev/test: human-only.",
        "- Loss: ordinary ordinal BCEWithLogitsLoss, no class weights, no asymmetric penalty.",
        "",
        "## Test Comparison",
        "",
        "| Run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1, qds2, qds3]:
        if row:
            notion_lines.append(
                f"| {row['run_id']} | {_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
                f"{_fmt(row.get('Accuracy'))} | {_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} |"
            )
    notion_lines.extend(
        [
            "",
            f"- Exp6 training matrix complete: **{matrix_complete}**",
            f"- Can conclusion draft start: **{training_dataset_ready}**",
        ]
    )
    notion_text = "\n".join(notion_lines)
    write_text(EXP06_TRAINING_RUNS_DIR / "notion_exp06_qds3_summary.md", notion_text)
    write_text(run_dir / "notion_exp06_qds3_summary.md", notion_text)


def collect(run_dir: Path = QD_S3_RUN_DIR) -> None:
    ensure_exp06_training_dirs()
    baseline_rows = load_baseline_summary()
    qds1_rows = load_qds1_summary(QD_S1_RUN_DIR)
    qds2_rows = load_qds2_summary(QD_S2_RUN_DIR)
    qds3_rows = load_qds3_summary(run_dir)
    rows = baseline_rows + qds1_rows + qds2_rows + qds3_rows
    deltas = delta_rows(rows)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_training_results_summary.csv", rows, SUMMARY_FIELDS)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_qds3_comparison.csv", deltas)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_low_score_comparison.csv", [row for row in deltas if row["metric"] == "low_to_high_rate"])
    write_csv(
        EXP06_TRAINING_TABLES_DIR / "exp06_high_score_comparison.csv",
        [row for row in deltas if row["metric"] in {"Acc@5", "high_to_mid_or_low_rate"}],
    )
    write_reports(rows, deltas, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp6 QD-S3 training results.")
    parser.add_argument("--run_dir", type=Path, default=QD_S3_RUN_DIR)
    args = parser.parse_args()
    collect(args.run_dir)
    print(f"Wrote Exp6 QD-S3 summaries: {relpath(EXP06_TRAINING_RUNS_DIR)}")


if __name__ == "__main__":
    main()
