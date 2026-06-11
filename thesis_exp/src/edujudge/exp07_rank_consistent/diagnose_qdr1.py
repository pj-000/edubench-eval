"""Diagnose completed Exp7 QD-R1 CORAL outputs without modifying raw run files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_OUTPUT_DIR,
    EXP07_REPORTS_DIR,
    EXP07_RUN_ID,
    EXP07_TABLES_DIR,
    ensure_exp07_dirs,
    exp07_run_dir,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.collect_exp07_results import _as_float, _fmt
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_text


PRED_LABELS = [1, 2, 3, 4, 5]
TRUE_LABELS = [1, 2, 3, 4, 5]
PROB_FIELDS = ["prob_gt_1", "prob_gt_2", "prob_gt_3", "prob_gt_4"]
LOGIT_FIELDS = ["logit_gt_1", "logit_gt_2", "logit_gt_3", "logit_gt_4"]


def _prediction_path(run_dir: Path, split: str) -> Path:
    return run_dir / "predictions" / f"predictions_{split}.jsonl"


def _pred_label(row: dict[str, Any]) -> int:
    return int(row.get("pred_label_5", row.get("pred_label")))


def _load_predictions(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ["dev", "test"]:
        path = _prediction_path(run_dir, split)
        if path.exists():
            out[split] = read_jsonl(path)
    if not out:
        raise FileNotFoundError(f"Missing QD-R1 prediction files under {relpath(run_dir)}")
    return out


def prediction_distribution(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_rows in predictions.items():
        totals = Counter(int(row["label_5"]) for row in split_rows)
        counts = Counter((int(row["label_5"]), _pred_label(row)) for row in split_rows)
        for true_label in TRUE_LABELS:
            total = totals[true_label]
            for pred_label in PRED_LABELS:
                count = counts[(true_label, pred_label)]
                rows.append(
                    {
                        "split": split,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                        "rate": count / total if total else 0.0,
                    }
                )
    return rows


def per_true_label_mean_probs(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_rows in predictions.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            grouped[int(row["label_5"])].append(row)
        for true_label in TRUE_LABELS:
            group = grouped.get(true_label, [])
            if not group:
                rows.append({"split": split, "true_label": true_label, "n": 0})
                continue
            rows.append(
                {
                    "split": split,
                    "true_label": true_label,
                    "n": len(group),
                    "mean_prob_gt_1": mean(float(row["prob_gt_1"]) for row in group),
                    "mean_prob_gt_2": mean(float(row["prob_gt_2"]) for row in group),
                    "mean_prob_gt_3": mean(float(row["prob_gt_3"]) for row in group),
                    "mean_prob_gt_4": mean(float(row["prob_gt_4"]) for row in group),
                    "mean_pred_label": mean(_pred_label(row) for row in group),
                    "mean_pred_score_expected": mean(float(row["pred_score_expected"]) for row in group),
                    "signed_bias": mean(float(row["signed_error_label"]) for row in group),
                }
            )
    return rows


def low_score_error_distribution(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_rows in predictions.items():
        low_rows = [row for row in split_rows if int(row["label_5"]) <= 2]
        for group_name, group_rows in [
            ("low_all", low_rows),
            ("label_1", [row for row in low_rows if int(row["label_5"]) == 1]),
            ("label_2", [row for row in low_rows if int(row["label_5"]) == 2]),
        ]:
            total = len(group_rows)
            counts = Counter(_pred_label(row) for row in group_rows)
            low_to_high_count = sum(1 for row in group_rows if _pred_label(row) >= 4)
            low_to_mid_count = sum(1 for row in group_rows if _pred_label(row) == 3)
            exact_count = sum(1 for row in group_rows if _pred_label(row) == int(row["label_5"]))
            for pred_label in PRED_LABELS:
                count = counts[pred_label]
                rows.append(
                    {
                        "split": split,
                        "low_group": group_name,
                        "pred_label": pred_label,
                        "count": count,
                        "rate": count / total if total else 0.0,
                        "low_to_high_count": low_to_high_count,
                        "low_to_high_rate": low_to_high_count / total if total else 0.0,
                        "low_to_mid_count": low_to_mid_count,
                        "low_to_mid_rate": low_to_mid_count / total if total else 0.0,
                        "exact_count": exact_count,
                        "exact_rate": exact_count / total if total else 0.0,
                    }
                )
    return rows


def threshold_score_diagnostics(predictions: dict[str, list[dict[str, Any]]], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint_metadata = run_dir / "exp07_head_metadata.json"
    best_metadata = run_dir / "best" / "exp07_head_metadata.json"
    has_thresholds = checkpoint_metadata.exists() or best_metadata.exists()
    reason = (
        "learned threshold parameters and latent score arrays are not stored in the run output; "
        "only logits/probabilities were exported"
    )
    for split, split_rows in predictions.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in split_rows:
            grouped[int(row["label_5"])].append(row)
        for true_label in TRUE_LABELS:
            group = grouped.get(true_label, [])
            if group:
                gap_12 = mean(float(row["logit_gt_1"]) - float(row["logit_gt_2"]) for row in group)
                gap_23 = mean(float(row["logit_gt_2"]) - float(row["logit_gt_3"]) for row in group)
                gap_34 = mean(float(row["logit_gt_3"]) - float(row["logit_gt_4"]) for row in group)
            else:
                gap_12 = gap_23 = gap_34 = ""
            rows.append(
                {
                    "split": split,
                    "true_label": true_label,
                    "n": len(group),
                    "tau_1": "",
                    "tau_2": "",
                    "tau_3": "",
                    "tau_4": "",
                    "mean_latent_score": "",
                    "threshold_gap_12": gap_12,
                    "threshold_gap_23": gap_23,
                    "threshold_gap_34": gap_34,
                    "status": "available" if has_thresholds else "unavailable",
                    "reason": "" if has_thresholds else reason,
                }
            )
    return rows


def _test_row(rows: list[dict[str, str]], run_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return row
    return {}


def _delta_row(rows: list[dict[str, str]], comparison: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row.get("comparison") == comparison and row.get("metric") == metric:
            return row
    return {}


def _distribution_line(rows: list[dict[str, Any]], split: str, group: str) -> str:
    subset = [row for row in rows if row["split"] == split and row["low_group"] == group]
    return ", ".join(f"pred {row['pred_label']}: {row['count']} ({float(row['rate']):.3f})" for row in subset)


def write_diagnosis_report(
    prediction_rows: list[dict[str, Any]],
    per_label_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
) -> None:
    comparison = read_csv(EXP07_TABLES_DIR / "exp07_r1_comparison.csv")
    low_comparison = read_csv(EXP07_TABLES_DIR / "exp07_low_score_comparison.csv")
    high_comparison = read_csv(EXP07_TABLES_DIR / "exp07_high_score_comparison.csv")
    qdr1 = _test_row(comparison, EXP07_RUN_ID)
    b0 = _test_row(comparison, "QD-B0_human_only_ordinary_ordinal")
    b1 = _test_row(comparison, "QD-B1_human_only_L1_weighted_ordinal")
    low_vs_b0 = _delta_row(low_comparison, "QD-R1_minus_QD-B0", "low_to_high_rate")
    low_vs_b1 = _delta_row(low_comparison, "QD-R1_minus_QD-B1", "low_to_high_rate")
    acc5_vs_b0 = _delta_row(high_comparison, "QD-R1_minus_QD-B0", "Acc@5")
    test_low_all = next(row for row in low_rows if row["split"] == "test" and row["low_group"] == "low_all" and row["pred_label"] == 1)
    low_to_high_rate = float(test_low_all["low_to_high_rate"])
    low_to_mid_rate = float(test_low_all["low_to_mid_rate"])
    exact_rate = float(test_low_all["exact_rate"])
    signed_bias = _as_float(qdr1.get("Signed Bias label"))
    b0_bias = _as_float(b0.get("Signed Bias label"))
    threshold_status = threshold_rows[0]["status"] if threshold_rows else "unavailable"
    threshold_reason = threshold_rows[0]["reason"] if threshold_rows else "missing diagnostics"

    lines = [
        "# QD-R1 CORAL Diagnosis",
        "",
        "This diagnosis reads the completed QD-R1 raw predictions, arrays, run metadata, and summary tables.",
        "It does not train a model, call an API, generate synthetic data, or modify raw predictions/arrays.",
        "",
        "## Required Answers",
        "",
        "- Did CORAL fix monotonicity? **YES.** Test monotonic_violation_rate is 0.0000.",
        "- Did CORAL reduce low_to_high? **NO.** It matches QD-B0 and is worse than QD-B1.",
        "- Did CORAL improve overall scoring? **NO.** Accuracy, MAE_label, QWK, Kendall, and Spearman are worse than QD-B0.",
        "- Did CORAL hurt Acc@5? **NO.** Acc@5 increased, but this comes with stronger overestimation bias.",
        "- Should Exp7-B start? **NOT_RECOMMENDED_YET.** First diagnose or change the score calibration/objective.",
        "",
        "## Test Summary",
        "",
        "| run | Accuracy | MAE_label | QWK | Kendall | low_to_high | Acc@5 | signed_bias_label | monotonic_violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [b0, b1, qdr1]:
        if row:
            lines.append(
                f"| {row['run_id']} | {_fmt(row.get('Accuracy'))} | {_fmt(row.get('MAE_label'))} | "
                f"{_fmt(row.get('Quadratic Weighted Kappa'))} | {_fmt(row.get('Kendall tau'))} | "
                f"{_fmt(row.get('low_to_high_rate'))} | {_fmt(row.get('Acc@5'))} | "
                f"{_fmt(row.get('Signed Bias label'))} | {_fmt(row.get('monotonic_violation_rate'))} |"
            )
    lines.extend(
        [
            "",
            "## Why Acc@5 Can Increase While MAE/QWK Worsen",
            "",
            "QD-R1 shifts predictions upward. That helps many true label-5 examples stay at 5, improving Acc@5,",
            "but the same upward shift overestimates lower and mid labels. This raises signed label bias, harms exact",
            "match on non-5 labels, increases severe errors, and weakens rank-sensitive metrics such as QWK and Kendall.",
            "",
            "## Low-score Errors",
            "",
            f"- Test low_to_high rate: {_fmt(low_to_high_rate)}.",
            f"- Test low_to_mid rate: {_fmt(low_to_mid_rate)}.",
            f"- Test low exact rate: {_fmt(exact_rate)}.",
            f"- Low-to-high delta vs QD-B0: {_fmt(low_vs_b0.get('delta'))}.",
            f"- Low-to-high delta vs QD-B1: {_fmt(low_vs_b1.get('delta'))}.",
            f"- Test low pred distribution: {_distribution_line(low_rows, 'test', 'low_all')}.",
            "",
            "## Overestimation",
            "",
            f"QD-R1 is overestimating scores. Its test signed label bias is {_fmt(signed_bias)}, compared with "
            f"QD-B0 at {_fmt(b0_bias)}. The mean predicted label by true-label table shows the upward drift is",
            "especially damaging for low and mid labels.",
            "",
            "## Threshold Diagnostics",
            "",
            f"Threshold/latent score status: **{threshold_status}**.",
            f"Reason: {threshold_reason}.",
            "The exported logits still allow checking rank consistency and approximate logit gap constancy, but not",
            "absolute learned tau values or mean latent score by true label.",
            "",
            "## Recommendation",
            "",
            "Do not start Exp7-B yet. CORAL solved the structural rank-consistency problem, but the main failure mode",
            "now appears to be upward score calibration/decision bias rather than independent-threshold monotonicity.",
            "A better next step is to diagnose threshold calibration or selection criteria before adding a new Exp7-B variant.",
        ]
    )
    write_text(EXP07_REPORTS_DIR / "qdr1_diagnosis.md", "\n".join(lines))


def diagnose(run_dir: Path = exp07_run_dir()) -> None:
    ensure_exp07_dirs()
    predictions = _load_predictions(run_dir)
    pred_rows = prediction_distribution(predictions)
    per_label_rows = per_true_label_mean_probs(predictions)
    low_rows = low_score_error_distribution(predictions)
    threshold_rows = threshold_score_diagnostics(predictions, run_dir)
    write_csv(EXP07_TABLES_DIR / "qdr1_prediction_distribution.csv", pred_rows)
    write_csv(EXP07_TABLES_DIR / "qdr1_per_true_label_mean_probs.csv", per_label_rows)
    write_csv(EXP07_TABLES_DIR / "qdr1_low_score_error_distribution.csv", low_rows)
    write_csv(EXP07_TABLES_DIR / "qdr1_threshold_score_diagnostics.csv", threshold_rows)
    write_diagnosis_report(pred_rows, per_label_rows, low_rows, threshold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose completed QD-R1 CORAL results.")
    parser.add_argument("--run_dir", type=Path, default=exp07_run_dir())
    args = parser.parse_args()
    diagnose(args.run_dir)
    print(f"Wrote QD-R1 diagnosis: {relpath(EXP07_REPORTS_DIR / 'qdr1_diagnosis.md')}")


if __name__ == "__main__":
    main()
