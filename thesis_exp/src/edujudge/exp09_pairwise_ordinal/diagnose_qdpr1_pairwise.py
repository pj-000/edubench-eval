"""Diagnose the Exp9 QD-PR1 formal pairwise result without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    EXP09_OUTPUT_DIR,
    EXP09_REPORTS_DIR,
    EXP09_RUN_ID,
    EXP09_TABLES_DIR,
    QD_B1_RUN_ID,
    ensure_exp09_dirs,
    exp09_run_dir,
)
from thesis_exp.src.edujudge.utils.io import read_csv, write_csv, write_text


LABELS = [1, 2, 3, 4, 5]
PROB_FIELDS = ["prob_gt_1", "prob_gt_2", "prob_gt_3", "prob_gt_4"]
PREDICTION_DISTRIBUTION_FIELDS = ["split", "pred_label", "count", "rate"]
PER_TRUE_LABEL_FIELDS = ["split", "true_label", "pred_label", "count", "rate"]
LOW_SCORE_FIELDS = [
    "split",
    "true_label",
    "n",
    "exact_count",
    "exact_rate",
    "low_to_mid_count",
    "low_to_mid_rate",
    "low_to_high_count",
    "low_to_high_rate",
    "mean_pred_label",
    "mean_pred_score_expected",
    "mean_signed_error",
    "pred_1_count",
    "pred_2_count",
    "pred_3_count",
    "pred_4_count",
    "pred_5_count",
]
MONOTONIC_FIELDS = [
    "split",
    "true_label",
    "n",
    "monotonic_violation_count",
    "monotonic_violation_rate",
    "mean_prob_gt_1",
    "mean_prob_gt_2",
    "mean_prob_gt_3",
    "mean_prob_gt_4",
    "mean_pred_label",
    "mean_pred_score_expected",
]
PAIRWISE_GAP_FIELDS = [
    "split",
    "group_field",
    "group_value",
    "pair_count",
    "pair_accuracy",
    "margin_satisfied_rate",
    "mean_score_gap",
    "median_score_gap",
    "mean_pair_margin",
    "min_score_gap",
    "max_score_gap",
]
LOSS_DIAGNOSTIC_FIELDS = [
    "scope",
    "row_count",
    "epoch_start",
    "epoch_end",
    "global_step_start",
    "global_step_end",
    "mean_L_total",
    "mean_L_point",
    "mean_L_pair",
    "mean_weighted_L_pair",
    "mean_score_gap",
    "mean_low_high_pair_loss",
    "mean_adjacent_pair_loss",
    "mean_point_base_loss",
    "mean_point_sample_weight",
    "min_point_sample_weight",
    "max_point_sample_weight",
]


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    number = _float(value)
    if number is None:
        return default
    return int(round(number))


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return "NA" if value in (None, "") else str(value)
    return f"{number:.{digits}f}"


def _mean(values: Iterable[Any]) -> float:
    numbers = [_float(value) for value in values]
    clean = [value for value in numbers if value is not None]
    return statistics.fmean(clean) if clean else 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _prediction_path(run_dir: Path, split: str) -> Path:
    nested = run_dir / "predictions" / f"predictions_{split}.jsonl"
    if nested.exists():
        return nested
    flat = run_dir / f"predictions_{split}.jsonl"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"Missing predictions for split={split}: {nested}")


def load_predictions(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: _read_jsonl(_prediction_path(run_dir, split)) for split in ["dev", "test"]}


def true_label(row: dict[str, Any]) -> int:
    value = _int(row.get("label_5", row.get("true_label")))
    if value is None:
        raise ValueError(f"Prediction row missing label_5: {row.keys()}")
    return value


def pred_label(row: dict[str, Any]) -> int:
    value = _int(row.get("pred_label_5", row.get("pred_label")))
    if value is None:
        raise ValueError(f"Prediction row missing pred_label_5: {row.keys()}")
    return value


def pred_score_expected(row: dict[str, Any]) -> float:
    return _float(row.get("pred_score_expected"), float(pred_label(row))) or float(pred_label(row))


def monotonic_violation(row: dict[str, Any]) -> bool:
    if "monotonic_violation" in row:
        return bool(row["monotonic_violation"])
    probs = [_float(row.get(field), 0.0) or 0.0 for field in PROB_FIELDS]
    return any(probs[idx + 1] > probs[idx] + 1e-7 for idx in range(len(probs) - 1))


def prediction_distribution(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in predictions.items():
        counts = Counter(pred_label(row) for row in rows)
        total = len(rows)
        for label in LABELS:
            out.append(
                {
                    "split": split,
                    "pred_label": label,
                    "count": counts[label],
                    "rate": _rate(counts[label], total),
                }
            )
    return out


def per_true_label_distribution(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in predictions.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[true_label(row)].append(row)
        for label in LABELS:
            label_rows = grouped.get(label, [])
            counts = Counter(pred_label(row) for row in label_rows)
            total = len(label_rows)
            for pred in LABELS:
                out.append(
                    {
                        "split": split,
                        "true_label": label,
                        "pred_label": pred,
                        "count": counts[pred],
                        "rate": _rate(counts[pred], total),
                    }
                )
    return out


def low_score_error_analysis(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in predictions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            label = true_label(row)
            if label <= 2:
                grouped[str(label)].append(row)
                grouped["all_low"].append(row)
        for label_key in ["1", "2", "all_low"]:
            label_rows = grouped.get(label_key, [])
            n = len(label_rows)
            pred_counts = Counter(pred_label(row) for row in label_rows)
            exact = sum(1 for row in label_rows if pred_label(row) == true_label(row))
            low_to_mid = pred_counts[3]
            low_to_high = sum(pred_counts[label] for label in [4, 5])
            out.append(
                {
                    "split": split,
                    "true_label": label_key,
                    "n": n,
                    "exact_count": exact,
                    "exact_rate": _rate(exact, n),
                    "low_to_mid_count": low_to_mid,
                    "low_to_mid_rate": _rate(low_to_mid, n),
                    "low_to_high_count": low_to_high,
                    "low_to_high_rate": _rate(low_to_high, n),
                    "mean_pred_label": _mean(pred_label(row) for row in label_rows),
                    "mean_pred_score_expected": _mean(pred_score_expected(row) for row in label_rows),
                    "mean_signed_error": _mean(pred_label(row) - true_label(row) for row in label_rows),
                    **{f"pred_{label}_count": pred_counts[label] for label in LABELS},
                }
            )
    return out


def monotonic_violation_by_label(predictions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in predictions.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[true_label(row)].append(row)
        for label in LABELS:
            label_rows = grouped.get(label, [])
            n = len(label_rows)
            violations = sum(1 for row in label_rows if monotonic_violation(row))
            out.append(
                {
                    "split": split,
                    "true_label": label,
                    "n": n,
                    "monotonic_violation_count": violations,
                    "monotonic_violation_rate": _rate(violations, n),
                    "mean_prob_gt_1": _mean(row.get("prob_gt_1") for row in label_rows),
                    "mean_prob_gt_2": _mean(row.get("prob_gt_2") for row in label_rows),
                    "mean_prob_gt_3": _mean(row.get("prob_gt_3") for row in label_rows),
                    "mean_prob_gt_4": _mean(row.get("prob_gt_4") for row in label_rows),
                    "mean_pred_label": _mean(pred_label(row) for row in label_rows),
                    "mean_pred_score_expected": _mean(pred_score_expected(row) for row in label_rows),
                }
            )
    return out


def pairwise_gap_by_pair_type(run_dir: Path) -> list[dict[str, Any]]:
    accuracy_rows = read_csv(run_dir / "tables" / "pairwise_dev_accuracy.csv")
    gap_rows = read_csv(run_dir / "tables" / "pairwise_score_gap_metrics.csv")
    gap_by_key = {
        (row.get("split"), row.get("group_field"), row.get("group_value")): row
        for row in gap_rows
    }
    out: list[dict[str, Any]] = []
    for row in accuracy_rows:
        key = (row.get("split"), row.get("group_field"), row.get("group_value"))
        gap = gap_by_key.get(key, {})
        out.append(
            {
                "split": row.get("split", ""),
                "group_field": row.get("group_field", ""),
                "group_value": row.get("group_value", ""),
                "pair_count": row.get("pair_count", ""),
                "pair_accuracy": row.get("pair_accuracy", ""),
                "margin_satisfied_rate": row.get("margin_satisfied_rate", ""),
                "mean_score_gap": gap.get("mean_score_gap", ""),
                "median_score_gap": gap.get("median_score_gap", ""),
                "mean_pair_margin": gap.get("mean_pair_margin", ""),
                "min_score_gap": gap.get("min_score_gap", ""),
                "max_score_gap": gap.get("max_score_gap", ""),
            }
        )
    return out


def _loss_summary(scope: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    epochs = [_int(row.get("epoch"), 0) or 0 for row in rows]
    global_steps = [_int(row.get("global_step"), 0) or 0 for row in rows]
    return {
        "scope": scope,
        "row_count": len(rows),
        "epoch_start": min(epochs) if epochs else "",
        "epoch_end": max(epochs) if epochs else "",
        "global_step_start": min(global_steps) if global_steps else "",
        "global_step_end": max(global_steps) if global_steps else "",
        "mean_L_total": _mean(row.get("L_total") for row in rows),
        "mean_L_point": _mean(row.get("L_point") for row in rows),
        "mean_L_pair": _mean(row.get("L_pair") for row in rows),
        "mean_weighted_L_pair": _mean(row.get("weighted_L_pair") for row in rows),
        "mean_score_gap": _mean(row.get("mean_score_gap") for row in rows),
        "mean_low_high_pair_loss": _mean(row.get("low_high_pair_loss") for row in rows),
        "mean_adjacent_pair_loss": _mean(row.get("adjacent_pair_loss") for row in rows),
        "mean_point_base_loss": _mean(row.get("mean_point_base_loss") for row in rows),
        "mean_point_sample_weight": _mean(row.get("mean_point_sample_weight") for row in rows),
        "min_point_sample_weight": min((_float(row.get("min_point_sample_weight"), 0.0) or 0.0 for row in rows), default=0.0),
        "max_point_sample_weight": max((_float(row.get("max_point_sample_weight"), 0.0) or 0.0 for row in rows), default=0.0),
    }


def loss_component_diagnostics(run_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(run_dir / "logs" / "loss_debug_history.csv")
    if not rows:
        return []
    out = [
        _loss_summary("first_100_steps", rows[:100]),
        _loss_summary("last_100_steps", rows[-100:]),
        _loss_summary("all_steps", rows),
    ]
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        epoch = _int(row.get("epoch"))
        if epoch is not None:
            by_epoch[epoch].append(row)
    for epoch in sorted(by_epoch):
        out.append(_loss_summary(f"epoch_{epoch}", by_epoch[epoch]))
    return out


def _find_test_row(rows: list[dict[str, str]], run_id: str) -> dict[str, str]:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return row
    return {}


def _find_delta(rows: list[dict[str, str]], baseline_id: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row.get("baseline_run_id") == baseline_id and row.get("metric") == metric:
            return row
    return {}


def _lookup(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return {}


def write_diagnosis_report(
    predictions: dict[str, list[dict[str, Any]]],
    low_rows: list[dict[str, Any]],
    mono_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
) -> None:
    main_rows = read_csv(EXP09_TABLES_DIR / "exp09_main_comparison.csv")
    delta_rows = read_csv(EXP09_TABLES_DIR / "exp09_delta_vs_baselines.csv")
    qdpr1 = _find_test_row(main_rows, EXP09_RUN_ID)
    qdb1 = _find_test_row(main_rows, QD_B1_RUN_ID)
    lth_delta = _find_delta(delta_rows, QD_B1_RUN_ID, "low_to_high_rate")
    mae_delta = _find_delta(delta_rows, QD_B1_RUN_ID, "MAE_label")
    qwk_delta = _find_delta(delta_rows, QD_B1_RUN_ID, "Quadratic Weighted Kappa")
    acc5_delta = _find_delta(delta_rows, QD_B1_RUN_ID, "Acc@5")
    mono_delta = _find_delta(delta_rows, QD_B1_RUN_ID, "monotonic_violation_rate")
    low_all_test = _lookup(low_rows, split="test", true_label="all_low")
    low_label_2 = _lookup(low_rows, split="test", true_label="2")
    mono_label_2 = _lookup(mono_rows, split="test", true_label=2)
    pair_overall = _lookup(pair_rows, split="dev", group_field="overall", group_value="overall")
    pair_low_high = _lookup(pair_rows, split="dev", group_field="pair_type", group_value="low_high")
    first_loss = _lookup(loss_rows, scope="first_100_steps")
    last_loss = _lookup(loss_rows, scope="last_100_steps")
    pred_counts = Counter(pred_label(row) for row in predictions["test"])
    pred_total = len(predictions["test"])
    pred_summary = ", ".join(
        f"{label}: {pred_counts[label]} ({_fmt(_rate(pred_counts[label], pred_total))})" for label in LABELS
    )
    lines = [
        "# QD-PR1 Pairwise Formal Result Diagnosis",
        "",
        "## Scope",
        "",
        "This diagnosis reads existing QD-PR1 raw run outputs, summary tables, predictions, pairwise diagnostics, "
        "and loss/debug history. It does not train a model, call an API, generate synthetic data, modify raw "
        "predictions/arrays/logs, submit checkpoint files, or start QD-PR2.",
        "",
        "## Required Answers",
        "",
        f"- Did QD-PR1 reduce low-to-high? **NO.** Test low_to_high is `{_fmt(qdpr1.get('low_to_high_rate'))}`, "
        f"worse than QD-B1 `{_fmt(qdb1.get('low_to_high_rate'))}`; delta vs QD-B1 is `{_fmt(lth_delta.get('delta'))}`.",
        f"- Did QD-PR1 beat QD-B1? **NO.** MAE delta `{_fmt(mae_delta.get('delta'))}`, QWK delta "
        f"`{_fmt(qwk_delta.get('delta'))}`, and Acc@5 delta `{_fmt(acc5_delta.get('delta'))}` are all unfavorable.",
        f"- Did pairwise training damage ordinal monotonicity? **YES.** Test monotonic violation is "
        f"`{_fmt(qdpr1.get('monotonic_violation_rate'))}`; delta vs QD-B1 is `{_fmt(mono_delta.get('delta'))}`.",
        f"- Did pairwise training improve pairwise score gaps on dev? **YES, partially.** Dev pair accuracy is "
        f"`{_fmt(pair_overall.get('pair_accuracy'))}`, but only `{_fmt(pair_overall.get('margin_satisfied_rate'))}` "
        "of pairs satisfy the configured margin.",
        f"- Are low-high pairs better separated after training? **Partially.** Dev low-high pair accuracy is "
        f"`{_fmt(pair_low_high.get('pair_accuracy'))}`, mean score gap `{_fmt(pair_low_high.get('mean_score_gap'))}`, "
        f"but margin satisfaction is only `{_fmt(pair_low_high.get('margin_satisfied_rate'))}`.",
        "- Is pointwise calibration worse? **YES.** The model improves pair separation while worsening test MAE, QWK, "
        "Acc@5, and monotonicity relative to QD-B1.",
        f"- Does label=2 remain problematic? **YES.** On test label=2, low_to_high is "
        f"`{_fmt(low_label_2.get('low_to_high_rate'))}` and monotonic violation is "
        f"`{_fmt(mono_label_2.get('monotonic_violation_rate'))}`.",
        "- Should QD-PR2 start? **Recommended only as a controlled anchored fine-tuning experiment, not as an "
        "unanchored from-scratch rerun.**",
        "",
        "## Main Test Metrics",
        "",
        "| model | low_to_high | MAE_label | QWK | Acc@5 | monotonic_violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| QD-B1 raw | {_fmt(qdb1.get('low_to_high_rate'))} | {_fmt(qdb1.get('MAE_label'))} | "
        f"{_fmt(qdb1.get('Quadratic Weighted Kappa'))} | {_fmt(qdb1.get('Acc@5'))} | "
        f"{_fmt(qdb1.get('monotonic_violation_rate'))} |",
        f"| QD-PR1 formal | {_fmt(qdpr1.get('low_to_high_rate'))} | {_fmt(qdpr1.get('MAE_label'))} | "
        f"{_fmt(qdpr1.get('Quadratic Weighted Kappa'))} | {_fmt(qdpr1.get('Acc@5'))} | "
        f"{_fmt(qdpr1.get('monotonic_violation_rate'))} |",
        "",
        "## Prediction Distribution",
        "",
        f"Test pred_label distribution is `{pred_summary}`. The model still concentrates predictions in high labels, "
        "while label=1/2 cases are frequently promoted into 4/5.",
        "",
        "## Low-score Failure Pattern",
        "",
        f"Across test true labels 1-2, low_to_high is `{_fmt(low_all_test.get('low_to_high_rate'))}` "
        f"({low_all_test.get('low_to_high_count')}/{low_all_test.get('n')}). Label=2 is the key failure: "
        f"low_to_high `{_fmt(low_label_2.get('low_to_high_rate'))}`, mean predicted label "
        f"`{_fmt(low_label_2.get('mean_pred_label'))}`, signed error `{_fmt(low_label_2.get('mean_signed_error'))}`.",
        "",
        "## Pairwise vs Pointwise Diagnostics",
        "",
        f"Loss history supports a split between pairwise separation and pointwise calibration. In the first 100 steps, "
        f"mean score gap is `{_fmt(first_loss.get('mean_score_gap'))}`, L_point `{_fmt(first_loss.get('mean_L_point'))}`, "
        f"and L_pair `{_fmt(first_loss.get('mean_L_pair'))}`. In the last 100 steps, mean score gap rises to "
        f"`{_fmt(last_loss.get('mean_score_gap'))}`, L_point falls to `{_fmt(last_loss.get('mean_L_point'))}`, "
        f"and L_pair remains `{_fmt(last_loss.get('mean_L_pair'))}`. This is consistent with pairwise ordering "
        "being optimized without preserving calibrated cumulative probabilities.",
        "",
        "## Interpretation",
        "",
        "QD-PR1 is a negative formal result. The pairwise objective learns useful ranking/gap signals on dev pairs, "
        "but the independent ordinal head is not anchored strongly enough to preserve valid cumulative behavior. "
        "The result should not be presented as an effective method. It should be presented as evidence that "
        "pairwise supervision must be constrained by a stronger pointwise/monotonic anchor.",
        "",
        "## QD-PR2 Recommendation",
        "",
        "QD-PR2 is recommended only with controlled changes:",
        "",
        "- Initialize from the QD-B1 checkpoint instead of training from scratch.",
        "- Sweep `lambda_pair` only in `{0.05, 0.1}`.",
        "- Add monotonic regularization or use a rank-consistent head so pairwise gaps cannot break cumulative order.",
        "- Use high-comparability pairs only, prioritizing `same_question` and then `same_metric_language`.",
        "- Fine-tune for 2-3 epochs rather than running a full from-scratch 10-epoch training job.",
        "",
        "Do not start QD-PR2 from this script; this report only proposes the controlled follow-up.",
    ]
    report = "\n".join(lines)
    write_text(EXP09_REPORTS_DIR / "qdpr1_pairwise_diagnosis.md", report)
    notion_lines = [
        "# Exp9 QD-PR1 Pairwise Result Summary",
        "",
        f"Formal status: `completed`.",
        "",
        f"QD-PR1 did not reduce low_to_high: `{_fmt(qdpr1.get('low_to_high_rate'))}` vs QD-B1 "
        f"`{_fmt(qdb1.get('low_to_high_rate'))}`.",
        "",
        f"QD-PR1 did not beat QD-B1: MAE `{_fmt(qdpr1.get('MAE_label'))}` vs "
        f"`{_fmt(qdb1.get('MAE_label'))}`, QWK `{_fmt(qdpr1.get('Quadratic Weighted Kappa'))}` vs "
        f"`{_fmt(qdb1.get('Quadratic Weighted Kappa'))}`, Acc@5 `{_fmt(qdpr1.get('Acc@5'))}` vs "
        f"`{_fmt(qdb1.get('Acc@5'))}`.",
        "",
        f"Main failure: pairwise training damaged ordinal monotonicity (`{_fmt(qdpr1.get('monotonic_violation_rate'))}`) "
        "and therefore worsened pointwise calibration, despite useful dev pairwise ranking signal.",
        "",
        "Interpretation: negative QD-PR1 result; pairwise direction remains promising only as anchored fine-tuning.",
        "",
        "QD-PR2 recommendation: initialize from QD-B1, use `lambda_pair` in `{0.05, 0.1}`, add monotonic "
        "regularization, use high-comparability pairs only, and fine-tune for 2-3 epochs.",
    ]
    write_text(EXP09_REPORTS_DIR / "notion_exp09_pairwise_result_summary.md", "\n".join(notion_lines))


def diagnose(run_dir: Path | None = None) -> None:
    ensure_exp09_dirs()
    run_dir = run_dir or exp09_run_dir(False)
    predictions = load_predictions(run_dir)
    distribution_rows = prediction_distribution(predictions)
    per_true_rows = per_true_label_distribution(predictions)
    low_rows = low_score_error_analysis(predictions)
    mono_rows = monotonic_violation_by_label(predictions)
    pair_rows = pairwise_gap_by_pair_type(run_dir)
    loss_rows = loss_component_diagnostics(run_dir)
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_prediction_distribution.csv",
        distribution_rows,
        fieldnames=PREDICTION_DISTRIBUTION_FIELDS,
    )
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_per_true_label_prediction_distribution.csv",
        per_true_rows,
        fieldnames=PER_TRUE_LABEL_FIELDS,
    )
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_low_score_error_analysis.csv",
        low_rows,
        fieldnames=LOW_SCORE_FIELDS,
    )
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_monotonic_violation_by_label.csv",
        mono_rows,
        fieldnames=MONOTONIC_FIELDS,
    )
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_pairwise_gap_by_pair_type.csv",
        pair_rows,
        fieldnames=PAIRWISE_GAP_FIELDS,
    )
    write_csv(
        EXP09_TABLES_DIR / "qdpr1_loss_component_diagnostics.csv",
        loss_rows,
        fieldnames=LOSS_DIAGNOSTIC_FIELDS,
    )
    write_diagnosis_report(predictions, low_rows, mono_rows, pair_rows, loss_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Exp9 QD-PR1 pairwise formal results.")
    parser.add_argument("--run_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnose(args.run_dir)
    print("Exp9 QD-PR1 diagnosis generated")


if __name__ == "__main__":
    main()
