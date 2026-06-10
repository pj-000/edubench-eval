"""Collect QD-S1 results and compare against question-disjoint baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_training import (
    EXP06_TRAINING_RUNS_DIR,
    EXP06_TRAINING_TABLES_DIR,
    QD_BASELINE_TABLES_DIR,
    QD_BASELINE_RUNS_DIR,
    QD_S1_RUN_DIR,
    QD_S1_RUN_ID,
    ensure_exp06_training_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


SUMMARY_FIELDS = [
    "run_id",
    "setting",
    "split",
    "n",
    "Accuracy",
    "MAE_label",
    "MAE_expected",
    "Macro-F1",
    "Quadratic Weighted Kappa",
    "Kendall tau",
    "Spearman rho",
    "severe_error_rate",
    "low_to_high_rate",
    "Acc@5",
    "high_to_mid_or_low_rate",
    "best_epoch",
    "best_global_step",
]


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _as_float(value)
    if number is None:
        return str(value) if value not in (None, "") else "NA"
    return f"{number:.{digits}f}"


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _metric_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _split_table_row(path: Path, split: str) -> dict[str, str]:
    for row in _read_csv_if_exists(path):
        if row.get("split") == split:
            return row
    return {}


def _run_split_extras(run_dir: Path, split: str) -> dict[str, str]:
    low = _split_table_row(run_dir / "tables" / "low_score_metrics.csv", split)
    high = _split_table_row(run_dir / "tables" / "high_score_metrics.csv", split)
    return {
        "low_to_high_rate": _metric_value(low, "low_to_high_rate"),
        "Acc@5": _metric_value(high, "Acc@5"),
        "high_to_mid_or_low_rate": _metric_value(high, "high_to_mid_or_low_rate"),
    }


def load_qds1_summary(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "tables" / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing QD-S1 metrics summary: {relpath(path)}")
    rows = _read_csv_if_exists(path)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    out = []
    for row in rows:
        split = row.get("split", "")
        extras = _run_split_extras(run_dir, split)
        out.append(
            {
                "run_id": QD_S1_RUN_ID,
                "setting": "human_plus_384_synthetic ordinary ordinal",
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


def load_baseline_summary() -> list[dict[str, str]]:
    path = QD_BASELINE_TABLES_DIR / "qd_baseline_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing question-disjoint baseline table: {relpath(path)}. "
            "Provide QD-B0/QD-B1 baseline outputs before comparing QD-S1."
        )
    out = []
    for row in _read_csv_if_exists(path):
        run_id = row.get("run_id", "")
        split = row.get("split", "")
        extras = _run_split_extras(QD_BASELINE_RUNS_DIR / run_id, split)
        setting = "human-only ordinary ordinal" if "QD-B0" in run_id else "human-only L1 weighted ordinal"
        out.append(
            {
                "run_id": run_id,
                "setting": setting,
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
                "best_epoch": row.get("epoch", ""),
                "best_global_step": row.get("global_step", ""),
            }
        )
    return out


def _test_row(rows: list[dict[str, str]], run_prefix: str) -> dict[str, str]:
    for row in rows:
        if row.get("split") == "test" and row.get("run_id", "").startswith(run_prefix):
            return row
    return {}


def delta_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    qds1 = _test_row(rows, "QD-S1")
    out = []
    for baseline in ["QD-B0", "QD-B1"]:
        base = _test_row(rows, baseline)
        if not base or not qds1:
            continue
        for metric, direction in [
            ("MAE_label", "lower_better"),
            ("Accuracy", "higher_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("low_to_high_rate", "lower_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
        ]:
            q_value = _as_float(qds1.get(metric))
            b_value = _as_float(base.get(metric))
            if q_value is None or b_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-S1_minus_{baseline}",
                    "metric": metric,
                    "baseline_value": b_value,
                    "qds1_value": q_value,
                    "delta": q_value - b_value,
                    "direction": direction,
                }
            )
    return out


def write_reports(rows: list[dict[str, str]], deltas: list[dict[str, Any]]) -> None:
    qds1 = _test_row(rows, "QD-S1")
    b0 = _test_row(rows, "QD-B0")
    b1 = _test_row(rows, "QD-B1")
    low_delta_b0 = next((row for row in deltas if row["comparison"] == "QD-S1_minus_QD-B0" and row["metric"] == "low_to_high_rate"), {})
    acc5_delta_b0 = next((row for row in deltas if row["comparison"] == "QD-S1_minus_QD-B0" and row["metric"] == "Acc@5"), {})
    can_qds2 = "PENDING_REVIEW"
    if qds1 and b0:
        low_ok = _as_float(low_delta_b0.get("delta")) is not None and float(low_delta_b0["delta"]) < 0
        acc5_delta = _as_float(acc5_delta_b0.get("delta"))
        acc5_ok = acc5_delta is None or acc5_delta > -0.03
        can_qds2 = "YES" if low_ok and acc5_ok else "REVIEW_REQUIRED"

    report_lines = [
        "# Exp6 QD-S1 Human + Synthetic Ordinary Ordinal",
        "",
        "This report compares QD-S1 against the existing question-disjoint QD-B0/QD-B1 baselines.",
        "Synthetic rows are pseudo-label augmentation rows and are not human labels.",
        "",
        "## Test Metrics",
        "",
        "| run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1]:
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
            f"- Can QD-S2 start? **{can_qds2}**",
            "- Model weights/checkpoints are written under `thesis_exp/artifacts/` and must not be committed.",
        ]
    )
    write_text(EXP06_TRAINING_RUNS_DIR / "report.md", "\n".join(report_lines))

    review_lines = [
        "# Exp6 QD-S1 Review Package",
        "",
        f"Can QD-S2 start? **{can_qds2}**",
        "",
        "QD-S2 should start only after reviewing QD-S1's low-score improvement and high-score tradeoff.",
        "",
        "Remaining blockers:",
        "",
    ]
    if can_qds2 == "YES":
        review_lines.append("- None for QD-S2 planning.")
    else:
        review_lines.append("- Review QD-S1 test metrics against QD-B0/QD-B1 before starting QD-S2.")
    write_text(EXP06_TRAINING_RUNS_DIR / "review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp6 QD-S1 Synthetic Augmentation Summary",
        "",
        "- Setting: question-disjoint `question_seed42`.",
        "- Train: 3326 human + 384 final synthetic low-score pseudo-label rows.",
        "- Dev/test: human-only.",
        "- Loss: ordinary ordinal BCEWithLogitsLoss, no class weights, no asymmetric loss.",
        "",
        "## Test Comparison",
        "",
        "| Run | MAE_label | QWK | Accuracy | low_to_high | Acc@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qds1]:
        if row:
            notion_lines.append(
                f"| {row['run_id']} | {_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
                f"{_fmt(row.get('Accuracy'))} | {_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} |"
            )
    notion_lines.extend(["", f"- Can QD-S2 start: **{can_qds2}**"])
    write_text(EXP06_TRAINING_RUNS_DIR / "notion_exp06_qds1_summary.md", "\n".join(notion_lines))


def collect(run_dir: Path = QD_S1_RUN_DIR) -> None:
    ensure_exp06_training_dirs()
    baseline_rows = load_baseline_summary()
    qds1_rows = load_qds1_summary(run_dir)
    rows = baseline_rows + qds1_rows
    deltas = delta_rows(rows)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_training_results_summary.csv", rows, SUMMARY_FIELDS)
    write_csv(EXP06_TRAINING_TABLES_DIR / "exp06_low_score_comparison.csv", [row for row in deltas if row["metric"] == "low_to_high_rate"])
    write_csv(
        EXP06_TRAINING_TABLES_DIR / "exp06_high_score_comparison.csv",
        [row for row in deltas if row["metric"] in {"Acc@5", "high_to_mid_or_low_rate"}],
    )
    write_reports(rows, deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp6 QD-S1 training results.")
    parser.add_argument("--run_dir", type=Path, default=QD_S1_RUN_DIR)
    args = parser.parse_args()
    collect(args.run_dir)
    print(f"Wrote Exp6 QD-S1 summaries: {relpath(EXP06_TRAINING_RUNS_DIR)}")


if __name__ == "__main__":
    main()
