"""Collect Exp7 QD-R1 outputs and compare with QD-B0/QD-B1 baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_OUTPUT_DIR,
    EXP07_REPORTS_DIR,
    EXP07_RUN_ID,
    EXP07_TABLES_DIR,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
    ensure_exp07_dirs,
    exp07_run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


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
    "mean_violation_magnitude",
    "best_epoch",
    "best_global_step",
]


MONOTONICITY_FIELDS = [
    "run_id",
    "setting",
    "status",
    "split",
    "n",
    "monotonic_violation_rate",
    "mean_violation_magnitude",
    "p1_ge_p2_rate",
    "p2_ge_p3_rate",
    "p3_ge_p4_rate",
    "mean_prob_gt_1",
    "mean_prob_gt_2",
    "mean_prob_gt_3",
    "mean_prob_gt_4",
    "prob_gt_1_positive_rate",
    "prob_gt_2_positive_rate",
    "prob_gt_3_positive_rate",
    "prob_gt_4_positive_rate",
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


def _split_row(path: Path, split: str) -> dict[str, str]:
    for row in _read_csv_if_exists(path):
        if row.get("split") == split:
            return row
    return {}


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_status(run_dir: Path, default: str) -> str:
    metadata = _metadata(run_dir)
    if metadata.get("status"):
        return str(metadata["status"])
    return default


def _run_rows(run_id: str, setting: str, run_dir: Path, default_status: str = "completed") -> list[dict[str, Any]]:
    metrics_rows = _read_csv_if_exists(run_dir / "tables" / "metrics_summary.csv")
    mono_rows = {row.get("split", ""): row for row in _read_csv_if_exists(run_dir / "tables" / "monotonicity_metrics.csv")}
    metadata = _metadata(run_dir)
    status = _run_status(run_dir, default_status)
    out = []
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
                "mean_violation_magnitude": mono.get("mean_violation_magnitude", ""),
                "best_epoch": metadata.get("best_epoch") or row.get("epoch", ""),
                "best_global_step": metadata.get("best_global_step") or row.get("global_step", ""),
            }
        )
    return out


def _pending_qdr1_row() -> dict[str, Any]:
    return {
        "run_id": EXP07_RUN_ID,
        "setting": "human-only CORAL rank-consistent ordinal",
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
        "mean_violation_magnitude": "",
        "best_epoch": "",
        "best_global_step": "",
    }


def load_baseline_rows() -> list[dict[str, Any]]:
    return _run_rows(
        QD_B0_RUN_ID,
        "human-only ordinary independent ordinal",
        QD_BASELINE_RUNS_DIR / QD_B0_RUN_ID,
    ) + _run_rows(
        QD_B1_RUN_ID,
        "human-only L1 weighted independent ordinal",
        QD_BASELINE_RUNS_DIR / QD_B1_RUN_ID,
    )


def load_qdr1_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = _run_rows(EXP07_RUN_ID, "human-only CORAL rank-consistent ordinal", run_dir)
    return rows or [_pending_qdr1_row()]


def _test_row(rows: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return row
    return {}


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qdr1 = _test_row(rows, EXP07_RUN_ID)
    out: list[dict[str, Any]] = []
    for baseline in [QD_B0_RUN_ID, QD_B1_RUN_ID]:
        base = _test_row(rows, baseline)
        if not base or not qdr1 or qdr1.get("status") == "pending_formal_training":
            continue
        for metric, direction in [
            ("monotonic_violation_rate", "lower_better"),
            ("mean_violation_magnitude", "lower_better"),
            ("low_to_high_rate", "lower_better"),
            ("MAE_label", "lower_better"),
            ("MAE_expected", "lower_better"),
            ("Accuracy", "higher_better"),
            ("Quadratic Weighted Kappa", "higher_better"),
            ("Kendall tau", "higher_better"),
            ("Spearman rho", "higher_better"),
            ("Acc@5", "higher_better"),
            ("high_to_mid_or_low_rate", "lower_better"),
        ]:
            q_value = _as_float(qdr1.get(metric))
            b_value = _as_float(base.get(metric))
            if q_value is None or b_value is None:
                continue
            out.append(
                {
                    "comparison": f"QD-R1_minus_{baseline.split('_')[0]}",
                    "metric": metric,
                    "baseline_run_id": baseline,
                    "baseline_value": b_value,
                    "qdr1_value": q_value,
                    "delta": q_value - b_value,
                    "direction": direction,
                }
            )
    return out


def monotonicity_rows(rows: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        split = str(row.get("split", ""))
        source_dir = run_dir if run_id == EXP07_RUN_ID else QD_BASELINE_RUNS_DIR / run_id
        mono = _split_row(source_dir / "tables" / "monotonicity_metrics.csv", split)
        out.append(
            {
                "run_id": run_id,
                "setting": row.get("setting", ""),
                "status": row.get("status", ""),
                "split": split,
                "n": row.get("n", ""),
                "monotonic_violation_rate": row.get("monotonic_violation_rate", ""),
                "mean_violation_magnitude": row.get("mean_violation_magnitude", mono.get("mean_violation_magnitude", "")),
                "p1_ge_p2_rate": mono.get("p1_ge_p2_rate", ""),
                "p2_ge_p3_rate": mono.get("p2_ge_p3_rate", ""),
                "p3_ge_p4_rate": mono.get("p3_ge_p4_rate", ""),
                "mean_prob_gt_1": mono.get("mean_prob_gt_1", ""),
                "mean_prob_gt_2": mono.get("mean_prob_gt_2", ""),
                "mean_prob_gt_3": mono.get("mean_prob_gt_3", ""),
                "mean_prob_gt_4": mono.get("mean_prob_gt_4", ""),
                "prob_gt_1_positive_rate": mono.get("prob_gt_1_positive_rate", ""),
                "prob_gt_2_positive_rate": mono.get("prob_gt_2_positive_rate", ""),
                "prob_gt_3_positive_rate": mono.get("prob_gt_3_positive_rate", ""),
                "prob_gt_4_positive_rate": mono.get("prob_gt_4_positive_rate", ""),
            }
        )
    return out


def exp7b_gate(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> str:
    qdr1 = _test_row(rows, EXP07_RUN_ID)
    if not qdr1 or qdr1.get("status") == "pending_formal_training":
        return "PENDING_FORMAL_TRAINING"
    mono = _as_float(qdr1.get("monotonic_violation_rate"))
    if mono is None or mono > 1e-6:
        return "IMPLEMENTATION_REVIEW_REQUIRED"
    low_delta = next(
        (row for row in deltas if row["comparison"] == "QD-R1_minus_QD-B0" and row["metric"] == "low_to_high_rate"),
        {},
    )
    mae_delta = next(
        (row for row in deltas if row["comparison"] == "QD-R1_minus_QD-B0" and row["metric"] == "MAE_label"),
        {},
    )
    low_ok = _as_float(low_delta.get("delta")) is not None and float(low_delta["delta"]) < 0
    mae_ok = _as_float(mae_delta.get("delta")) is None or float(mae_delta["delta"]) <= 0.03
    return "YES" if low_ok and mae_ok else "NOT_RECOMMENDED_YET"


def _answer_hurts_acc5(deltas: list[dict[str, Any]]) -> str:
    row = next((item for item in deltas if item["comparison"] == "QD-R1_minus_QD-B0" and item["metric"] == "Acc@5"), {})
    if not row:
        return "PENDING_FORMAL_TRAINING"
    delta = float(row["delta"])
    if delta >= 0:
        return "NO; Acc@5 increases, but overestimation bias also increases"
    return "YES"


def write_reports(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    b0 = _test_row(rows, QD_B0_RUN_ID)
    b1 = _test_row(rows, QD_B1_RUN_ID)
    qdr1 = _test_row(rows, EXP07_RUN_ID)
    gate = exp7b_gate(rows, deltas)
    warning = ""
    mono = _as_float(qdr1.get("monotonic_violation_rate")) if qdr1 else None
    if mono is not None and mono > 1e-6:
        warning = "Implementation warning: CORAL monotonic violation rate is non-zero."
    report_lines = [
        "# Exp7-A CORAL-style Rank-consistent Ordinal Scorer",
        "",
        "Exp7-A fixes the input and data setting, then changes only the ordinal scorer structure.",
        "Exp3 selected A4 as the best input template, so QD-R1 uses the existing A4 text field.",
        "Exp4 showed ordinal scoring is a reasonable objective for the five-point score.",
        "The previous independent ordinal head can produce rank inconsistency such as P(score>3) > P(score>2).",
        "Exp5 loss ablations did not stably solve low-to-high errors, and Exp6 synthetic augmentation",
        "did not stably exceed human-only baselines. Exp7-A therefore changes the scorer head instead",
        "of generating data or adding another loss penalty.",
        "",
        "This is scoring model fine-tuning, not generative SFT.",
        "",
        "## Test Metrics",
        "",
        "| run | status | MAE_label | QWK | Accuracy | low_to_high | monotonic_violation | Acc@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qdr1]:
        if row:
            report_lines.append(
                f"| {row['run_id']} | {row.get('status', '')} | {_fmt(row.get('MAE_label'))} | "
                f"{_fmt(row.get('Quadratic Weighted Kappa'))} | {_fmt(row.get('Accuracy'))} | "
                f"{_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('monotonic_violation_rate'))} | "
                f"{_fmt(row.get('Acc@5'))} |"
            )
    report_lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"- Does QD-R1 reduce monotonic_violation_rate? {_answer_delta(deltas, 'monotonic_violation_rate')}",
            f"- Does QD-R1 reduce low_to_high_rate? {_answer_delta(deltas, 'low_to_high_rate')}",
            f"- Does QD-R1 improve MAE_label/QWK/Kendall? {_answer_quality(deltas)}",
            f"- Does QD-R1 hurt Acc@5? {_answer_hurts_acc5(deltas)}",
            "- Interpretation: rank consistency solved monotonic violation but did not solve low-score overestimation.",
            f"- Should Exp7-B start? **{gate}**",
            "- Next recommended step: calibration feasibility/export planning, not Exp7-B.",
        ]
    )
    if warning:
        report_lines.extend(["", f"**{warning}**"])
    report_lines.extend(["", "Model checkpoints live under `thesis_exp/artifacts/` and must not be committed."])
    report_text = "\n".join(report_lines)
    write_text(EXP07_OUTPUT_DIR / "report.md", report_text)
    write_text(EXP07_REPORTS_DIR / "report.md", report_text)

    review_lines = [
        "# Exp7-A Review Package",
        "",
        f"Can Exp7-B start? **{gate}**",
        "",
        "Scope:",
        "",
        "- Implemented only QD-R1 CORAL rank-consistent ordinal scorer.",
        "- No CORN, class-balanced loss, focal loss, calibration, API calls, or synthetic generation.",
        f"- Formal training status: {qdr1.get('status', 'pending') if qdr1 else 'pending'}.",
        "- Rank consistency is solved, but low-score overestimation is not solved.",
        "- Next recommended step: calibration feasibility/export planning, not Exp7-B.",
        "",
        "Review focus:",
        "",
        "- Confirm QD-S0 train/dev/test are human-only and A4 text is used directly.",
        "- Confirm CORAL logits satisfy z1 >= z2 >= z3 >= z4 and probabilities are monotonic.",
        "- Use `reports/qdr1_diagnosis.md` before deciding whether any Exp7-B variant is justified.",
    ]
    write_text(EXP07_OUTPUT_DIR / "review_package.md", "\n".join(review_lines))
    write_text(EXP07_REPORTS_DIR / "review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp7-A QD-R1 CORAL Summary",
        "",
        "- Dataset: `QD-S0_human_only`, question_seed42.",
        "- Input: fixed A4 text field.",
        "- Head: shared latent quality score plus ordered thresholds.",
        "- Loss: unweighted ordinal BCEWithLogits.",
        "- Synthetic data: no.",
        "- Class weights, focal loss, calibration: no.",
        f"- Formal status: `{qdr1.get('status', 'pending') if qdr1 else 'pending'}`.",
        f"- Exp7-B gate: **{gate}**.",
        "- Rank consistency solved monotonic violation, but low-score overestimation remains.",
        "- Next recommended step: calibration feasibility/export planning, not Exp7-B.",
    ]
    write_text(EXP07_OUTPUT_DIR / "notion_exp07_r1_summary.md", "\n".join(notion_lines))


def _answer_delta(deltas: list[dict[str, Any]], metric: str, higher_better: bool = False) -> str:
    row = next((item for item in deltas if item["comparison"] == "QD-R1_minus_QD-B0" and item["metric"] == metric), {})
    if not row:
        return "PENDING_FORMAL_TRAINING"
    delta = float(row["delta"])
    improves = delta > 0 if higher_better else delta < 0
    return "YES" if improves else "NO"


def _answer_quality(deltas: list[dict[str, Any]]) -> str:
    wanted = {
        "MAE_label": "lower",
        "Quadratic Weighted Kappa": "higher",
        "Kendall tau": "higher",
    }
    found = []
    for metric, direction in wanted.items():
        row = next((item for item in deltas if item["comparison"] == "QD-R1_minus_QD-B0" and item["metric"] == metric), {})
        if not row:
            continue
        delta = float(row["delta"])
        found.append(delta < 0 if direction == "lower" else delta > 0)
    if not found:
        return "PENDING_FORMAL_TRAINING"
    return "YES" if all(found) else "NO"


def collect(run_dir: Path = exp07_run_dir()) -> None:
    ensure_exp07_dirs()
    baseline_rows = load_baseline_rows()
    qdr1_rows = load_qdr1_rows(run_dir)
    rows = baseline_rows + qdr1_rows
    deltas = delta_rows(rows)
    write_csv(EXP07_TABLES_DIR / "exp07_r1_comparison.csv", rows, SUMMARY_FIELDS)
    write_csv(EXP07_TABLES_DIR / "exp07_monotonicity_comparison.csv", monotonicity_rows(rows, run_dir), MONOTONICITY_FIELDS)
    write_csv(
        EXP07_TABLES_DIR / "exp07_low_score_comparison.csv",
        [row for row in deltas if row["metric"] == "low_to_high_rate"],
    )
    write_csv(
        EXP07_TABLES_DIR / "exp07_high_score_comparison.csv",
        [row for row in deltas if row["metric"] in {"Acc@5", "high_to_mid_or_low_rate"}],
    )
    write_reports(rows, deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp7 QD-R1 results.")
    parser.add_argument("--run_dir", type=Path, default=exp07_run_dir())
    args = parser.parse_args()
    collect(args.run_dir)
    print(f"Wrote Exp7 summaries: {relpath(EXP07_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
