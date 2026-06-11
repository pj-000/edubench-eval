"""Collect Exp8 QD-ER1 outputs and compare with QD baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp08_edurisk import (
    EXP08_OUTPUT_DIR,
    EXP08_REPORTS_DIR,
    EXP08_RUN_ID,
    EXP08_TABLES_DIR,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_R1_RUN_DIR,
    QD_R1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
    ensure_exp08_dirs,
    exp08_run_dir,
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
    "expected_edurisk",
    "best_epoch",
    "best_global_step",
]

DELTA_FIELDS = [
    "comparison",
    "metric",
    "direction",
    "qder1_value",
    "baseline_value",
    "delta",
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


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _run_rows(run_id: str, setting: str, run_dir: Path, default_status: str = "completed") -> list[dict[str, Any]]:
    metrics_rows = _read_csv_if_exists(run_dir / "tables" / "metrics_summary.csv")
    metadata = _metadata(run_dir)
    status = str(metadata.get("status") or default_status)
    rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        rows.append(
            {
                "run_id": run_id,
                "setting": setting,
                "status": status,
                "split": row.get("split", ""),
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
                "monotonic_violation_rate": row.get("monotonic_violation_rate", ""),
                "expected_edurisk": row.get("expected_edurisk", ""),
                "best_epoch": metadata.get("best_epoch") or row.get("epoch", ""),
                "best_global_step": metadata.get("best_global_step") or row.get("global_step", ""),
            }
        )
    return rows


def _pending_qder1_row() -> dict[str, Any]:
    return {
        "run_id": EXP08_RUN_ID,
        "setting": "human-only EduRisk rank-consistent ordinal",
        "status": "implementation_ready_training_not_run",
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
        "expected_edurisk": "",
        "best_epoch": "",
        "best_global_step": "",
    }


def load_rows() -> list[dict[str, Any]]:
    rows = []
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
    rows.extend(
        _run_rows(
            QD_R1_RUN_ID,
            "human-only CORAL rank-consistent ordinal",
            QD_R1_RUN_DIR,
        )
    )
    exp8_rows = _run_rows(EXP08_RUN_ID, "human-only EduRisk rank-consistent ordinal", exp08_run_dir(False))
    rows.extend(exp8_rows or [_pending_qder1_row()])
    return rows


def _test_row(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return row
    return {}


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    er1 = _test_row(rows, EXP08_RUN_ID)
    if not er1 or er1.get("status") == "implementation_ready_training_not_run":
        return []
    out: list[dict[str, Any]] = []
    for baseline in [QD_B0_RUN_ID, QD_B1_RUN_ID, QD_R1_RUN_ID]:
        base = _test_row(rows, baseline)
        if not base:
            continue
        for metric, direction in [
            ("low_to_high_rate", "lower_better"),
            ("MAE_label", "lower_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
            ("Signed Bias label", "closer_to_zero_better"),
            ("monotonic_violation_rate", "lower_better"),
        ]:
            er_value = _as_float(er1.get(metric))
            base_value = _as_float(base.get(metric))
            if er_value is None or base_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-ER1_minus_{baseline}",
                    "metric": metric,
                    "direction": direction,
                    "qder1_value": er_value,
                    "baseline_value": base_value,
                    "delta": er_value - base_value,
                }
            )
    return out


def write_report(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    er1 = _test_row(rows, EXP08_RUN_ID)
    b1 = _test_row(rows, QD_B1_RUN_ID)
    status = er1.get("status", "implementation_ready_training_not_run")
    lines = [
        "# Exp8 EduRisk Ordinal Loss Report",
        "",
        f"Status: `{status}`",
        "",
        "Training executed by this package: `no` until the formal train script is run.",
        "API calls: `no`.",
        "Synthetic data: `no`.",
        "",
        "## Method",
        "",
        "QD-ER1 uses a CORAL-style rank-consistent head and converts cumulative probabilities into a legal 5-class distribution.",
        "The training objective combines soft ordinal cross entropy, normalized low-score risk, effective-number class balancing, and cumulative BCE.",
        "",
        "## Current Comparison",
        "",
        "| run | status | low_to_high | MAE | QWK | Acc@5 | signed bias | monotonic violation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id in [QD_B0_RUN_ID, QD_B1_RUN_ID, QD_R1_RUN_ID, EXP08_RUN_ID]:
        row = _test_row(rows, run_id)
        lines.append(
            "| "
            f"{run_id} | {row.get('status', 'missing')} | {_fmt(row.get('low_to_high_rate'))} | "
            f"{_fmt(row.get('MAE_label'))} | {_fmt(row.get('Quadratic Weighted Kappa'))} | "
            f"{_fmt(row.get('Acc@5'))} | {_fmt(row.get('Signed Bias label'))} | "
            f"{_fmt(row.get('monotonic_violation_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- Primary low-score target: QD-ER1 `low_to_high_rate` should be lower than QD-B1 (`{_fmt(b1.get('low_to_high_rate'))}`).",
            "- MAE should be no worse than QD-B1 + 0.02.",
            "- QWK should be no worse than QD-B1 - 0.03.",
            "- Acc@5 should be no worse than QD-B1 - 0.05, not QD-R1.",
            "- high_to_mid_or_low should be no worse than QD-B1 + 0.03.",
            "- Signed bias should be closer to zero than QD-R1.",
            "- monotonic_violation_rate must remain 0.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if status == "implementation_ready_training_not_run":
        lines.append("Run server smoke first, then formal QD-ER1 only if the smoke sanity checks pass.")
    elif deltas:
        lines.append("Use the delta table to decide whether QD-ER1 qualifies as the thesis training-method candidate.")
    else:
        lines.append("Collect formal QD-ER1 metrics before making a thesis-method decision.")
    text = "\n".join(lines)
    write_text(EXP08_OUTPUT_DIR / "report.md", text)
    write_text(EXP08_REPORTS_DIR / "report.md", text)
    write_text(EXP08_OUTPUT_DIR / "review_package.md", text)
    write_text(EXP08_REPORTS_DIR / "review_package.md", text)
    write_text(EXP08_OUTPUT_DIR / "notion_exp08_edurisk_summary.md", text)
    write_text(EXP08_REPORTS_DIR / "notion_exp08_edurisk_summary.md", text)


def collect() -> dict[str, Any]:
    ensure_exp08_dirs()
    rows = load_rows()
    deltas = delta_rows(rows)
    delta_output = deltas or [
        {
            "comparison": "pending_formal_training",
            "metric": "not_available",
            "direction": "not_applicable",
            "qder1_value": "",
            "baseline_value": "",
            "delta": "",
        }
    ]
    write_csv(EXP08_TABLES_DIR / "exp08_comparison_metrics.csv", rows, fieldnames=SUMMARY_FIELDS)
    write_csv(EXP08_TABLES_DIR / "exp08_delta_vs_baselines.csv", delta_output, fieldnames=DELTA_FIELDS)
    write_report(rows, deltas)
    return {"rows": len(rows), "deltas": len(deltas)}


def main() -> None:
    result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
