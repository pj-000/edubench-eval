"""Collect Exp9 QD-PR1 outputs and compare against question-disjoint baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    EXP09_OUTPUT_DIR,
    EXP09_REPORTS_DIR,
    EXP09_RUN_ID,
    EXP09_TABLES_DIR,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
    QD_ER1_RUN_DIR,
    QD_ER1_RUN_ID,
    QD_R1_RUN_DIR,
    QD_R1_RUN_ID,
    ensure_exp09_dirs,
    exp09_run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_text


SUMMARY_FIELDS = [
    "run_id",
    "setting",
    "status",
    "split",
    "n",
    "Accuracy",
    "MAE_label",
    "MAE_expected",
    "RMSE_label",
    "Signed Bias label",
    "Signed Bias expected",
    "Macro-F1",
    "Weighted-F1",
    "Quadratic Weighted Kappa",
    "Kendall tau",
    "Spearman rho",
    "severe_error_rate",
    "low_to_high_rate",
    "Acc@5",
    "high_to_mid_or_low_rate",
    "monotonic_violation_rate",
    "best_epoch",
    "best_global_step",
]

DELTA_FIELDS = [
    "comparison",
    "metric",
    "baseline_run_id",
    "baseline_value",
    "qdpr1_value",
    "delta",
    "direction",
    "status",
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
        return "NA" if value in (None, "") else str(value)
    return f"{number:.{digits}f}"


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_status(run_dir: Path, default: str) -> str:
    metadata = _metadata(run_dir)
    return str(metadata.get("status") or default)


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _run_rows(run_id: str, setting: str, run_dir: Path, default_status: str = "completed") -> list[dict[str, Any]]:
    metrics_rows = _read_csv_if_exists(run_dir / "tables" / "metrics_summary.csv")
    mono_rows = {row.get("split", ""): row for row in _read_csv_if_exists(run_dir / "tables" / "monotonicity_metrics.csv")}
    metadata = _metadata(run_dir)
    status = _run_status(run_dir, default_status)
    out: list[dict[str, Any]] = []
    for row in metrics_rows:
        split = row.get("split", "")
        mono = mono_rows.get(split, {})
        out.append(
            {
                "run_id": run_id,
                "setting": setting,
                "status": status,
                "split": split,
                "n": row.get("n", ""),
                "Accuracy": _metric(row, "Accuracy", "Exact Match"),
                "MAE_label": row.get("MAE_label", ""),
                "MAE_expected": row.get("MAE_expected", ""),
                "RMSE_label": row.get("RMSE_label", ""),
                "Signed Bias label": row.get("Signed Bias label", ""),
                "Signed Bias expected": row.get("Signed Bias expected", ""),
                "Macro-F1": row.get("Macro-F1", ""),
                "Weighted-F1": row.get("Weighted-F1", ""),
                "Quadratic Weighted Kappa": row.get("Quadratic Weighted Kappa", ""),
                "Kendall tau": row.get("Kendall tau", ""),
                "Spearman rho": row.get("Spearman rho", ""),
                "severe_error_rate": row.get("severe_error_rate", ""),
                "low_to_high_rate": row.get("low_to_high_rate", ""),
                "Acc@5": row.get("Acc@5", ""),
                "high_to_mid_or_low_rate": row.get("high_to_mid_or_low_rate", ""),
                "monotonic_violation_rate": row.get("monotonic_violation_rate", mono.get("monotonic_violation_rate", "")),
                "best_epoch": metadata.get("best_epoch") or row.get("epoch", ""),
                "best_global_step": metadata.get("best_global_step") or row.get("global_step", ""),
            }
        )
    return out


def _pending_qdpr1_row() -> dict[str, Any]:
    return {
        "run_id": EXP09_RUN_ID,
        "setting": "human-only risk-aware pairwise independent ordinal",
        "status": "pending_formal_training",
        "split": "test",
        "n": "",
        "Accuracy": "",
        "MAE_label": "",
        "MAE_expected": "",
        "RMSE_label": "",
        "Signed Bias label": "",
        "Signed Bias expected": "",
        "Macro-F1": "",
        "Weighted-F1": "",
        "Quadratic Weighted Kappa": "",
        "Kendall tau": "",
        "Spearman rho": "",
        "severe_error_rate": "",
        "low_to_high_rate": "",
        "Acc@5": "",
        "high_to_mid_or_low_rate": "",
        "monotonic_violation_rate": "",
        "best_epoch": "",
        "best_global_step": "",
    }


def _test_row(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return row
    return {}


def load_all_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(
        _run_rows(
            QD_B0_RUN_ID,
            "human-only ordinary independent ordinal",
            QD_BASELINE_RUNS_DIR / QD_B0_RUN_ID,
        )
    )
    rows.extend(
        _run_rows(
            QD_B1_RUN_ID,
            "human-only L1 weighted independent ordinal",
            QD_BASELINE_RUNS_DIR / QD_B1_RUN_ID,
        )
    )
    rows.extend(_run_rows(QD_R1_RUN_ID, "human-only CORAL rank-consistent ordinal", QD_R1_RUN_DIR))
    rows.extend(_run_rows(QD_ER1_RUN_ID, "human-only EduRisk rank-consistent ordinal", QD_ER1_RUN_DIR))
    qdpr1_rows = _run_rows(EXP09_RUN_ID, "human-only risk-aware pairwise independent ordinal", run_dir)
    rows.extend(qdpr1_rows or [_pending_qdpr1_row()])
    return rows


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qdpr1 = _test_row(rows, EXP09_RUN_ID)
    out: list[dict[str, Any]] = []
    if not qdpr1 or qdpr1.get("status") == "pending_formal_training":
        return out
    for baseline in [QD_B0_RUN_ID, QD_B1_RUN_ID, QD_R1_RUN_ID, QD_ER1_RUN_ID]:
        base = _test_row(rows, baseline)
        if not base:
            continue
        for metric, direction in [
            ("low_to_high_rate", "lower_better"),
            ("MAE_label", "lower_better"),
            ("MAE_expected", "lower_better"),
            ("Accuracy", "higher_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("Kendall tau", "higher_better"),
            ("Spearman rho", "higher_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
            ("monotonic_violation_rate", "lower_better"),
        ]:
            new_value = _as_float(qdpr1.get(metric))
            base_value = _as_float(base.get(metric))
            if new_value is None or base_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-PR1_minus_{baseline.split('_')[0]}",
                    "metric": metric,
                    "baseline_run_id": baseline,
                    "baseline_value": base_value,
                    "qdpr1_value": new_value,
                    "delta": new_value - base_value,
                    "direction": direction,
                }
            )
    return out


def pair_comparability_report_lines() -> list[str]:
    path = EXP09_TABLES_DIR / "pair_comparability_distribution.csv"
    rows = _read_csv_if_exists(path)
    if not rows:
        return [
            "",
            "## Pair Comparability Audit",
            "",
            "Pair comparability audit is pending; rerun Exp9 setup sanity to regenerate it.",
        ]
    lines = [
        "",
        "## Pair Comparability Audit",
        "",
        "Pair construction follows priority `same_question > same_metric_language > same_metric`; "
        "an `any_valid` fallback is used only if needed to fill the configured pair budget.",
        "Actual comparability rates are reported below. Formal training results should be interpreted in light of these rates.",
        "",
        "| split | pair_type | pairs | same_question | same_metric | same_language | same_metric_language |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('split')} | {row.get('pair_type')} | {row.get('pair_count')} | "
            f"{_fmt(row.get('same_question_rate'))} | {_fmt(row.get('same_metric_rate'))} | "
            f"{_fmt(row.get('same_language_rate'))} | {_fmt(row.get('same_metric_language_rate'))} |"
        )
    return lines


def write_reports(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    test_rows = [row for row in rows if row.get("split") == "test"]
    qdpr1 = _test_row(rows, EXP09_RUN_ID)
    status = qdpr1.get("status", "pending_formal_training") if qdpr1 else "pending_formal_training"
    lines = [
        "# Exp9 QD-PR1 Pairwise Ordinal Review Package",
        "",
        f"Formal training status: `{status}`",
        "Training executed by setup stage: `no`.",
        "API called: `no`.",
        "Synthetic generated: `no`.",
        "",
        "## Method",
        "",
        "QD-PR1 adds risk-aware ordinal preference pairs on top of QD-B1-style weighted ordinal BCE.",
        "Pairs are built from QD-S0 human-only train rows; dev pairs are diagnostic only.",
        "",
        "## Formal Run Defaults",
        "",
        "The formal run keeps `epochs=10`, `train_pairs=20000`, `dev_pairs=5000`, and "
        "`effective_batch_size=128` unchanged. For 24GB RTX 3090 execution, the default "
        "micro-batch is `per_device_train_batch_size=1` with "
        "`gradient_accumulation_steps=128`, `per_device_eval_batch_size=2`, "
        "`max_length=2048`, and `gradient_checkpointing=true`.",
        *pair_comparability_report_lines(),
        "",
        "## Test Comparison",
        "",
        "| run | status | MAE_label | QWK | Accuracy | low_to_high | Acc@5 | monotonic_violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        lines.append(
            f"| {row.get('run_id')} | {row.get('status')} | {_fmt(row.get('MAE_label'))} | "
            f"{_fmt(row.get('Quadratic Weighted Kappa'))} | {_fmt(row.get('Accuracy'))} | "
            f"{_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} | "
            f"{_fmt(row.get('monotonic_violation_rate'))} |"
        )
    if deltas:
        lines.extend(["", "## Deltas", "", "| comparison | metric | delta | direction |", "| --- | --- | ---: | --- |"])
        for row in deltas:
            lines.append(f"| {row['comparison']} | {row['metric']} | {_fmt(row['delta'])} | {row['direction']} |")
    else:
        lines.extend(["", "QD-PR1 formal metrics are pending; this package records setup readiness only."])
    report = "\n".join(lines)
    write_text(EXP09_REPORTS_DIR / "review_package.md", report)
    write_text(EXP09_REPORTS_DIR / "notion_exp09_pairwise_summary.md", report)
    write_text(EXP09_OUTPUT_DIR / "review_package.md", report)
    write_text(EXP09_OUTPUT_DIR / "report.md", report)


def collect(run_dir: Path | None = None) -> list[dict[str, Any]]:
    ensure_exp09_dirs()
    run_dir = run_dir or exp09_run_dir(False)
    rows = load_all_rows(run_dir)
    deltas = delta_rows(rows)
    delta_table_rows = deltas or [
        {
            "comparison": "pending_formal_training",
            "metric": "",
            "baseline_run_id": "",
            "baseline_value": "",
            "qdpr1_value": "",
            "delta": "",
            "direction": "",
            "status": "QD-PR1 formal metrics are not available yet.",
        }
    ]
    low_rows = [row for row in deltas if row.get("metric") == "low_to_high_rate"] or [
        {
            "comparison": "pending_formal_training",
            "metric": "low_to_high_rate",
            "baseline_run_id": "",
            "baseline_value": "",
            "qdpr1_value": "",
            "delta": "",
            "direction": "lower_better",
            "status": "QD-PR1 formal metrics are not available yet.",
        }
    ]
    high_rows = [row for row in deltas if row.get("metric") in {"Acc@5", "high_to_mid_or_low_rate"}] or [
        {
            "comparison": "pending_formal_training",
            "metric": "Acc@5/high_to_mid_or_low_rate",
            "baseline_run_id": "",
            "baseline_value": "",
            "qdpr1_value": "",
            "delta": "",
            "direction": "mixed",
            "status": "QD-PR1 formal metrics are not available yet.",
        }
    ]
    write_csv(EXP09_TABLES_DIR / "exp09_main_comparison.csv", rows, fieldnames=SUMMARY_FIELDS)
    write_csv(EXP09_TABLES_DIR / "exp09_delta_vs_baselines.csv", delta_table_rows, fieldnames=DELTA_FIELDS)
    write_csv(EXP09_TABLES_DIR / "exp09_low_score_comparison.csv", low_rows, fieldnames=DELTA_FIELDS)
    write_csv(EXP09_TABLES_DIR / "exp09_high_score_comparison.csv", high_rows, fieldnames=DELTA_FIELDS)
    write_reports(rows, deltas)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp9 QD-PR1 results.")
    parser.add_argument("--run_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.run_dir)
    print(f"Exp9 results collected: {len(rows)} rows")


if __name__ == "__main__":
    main()
