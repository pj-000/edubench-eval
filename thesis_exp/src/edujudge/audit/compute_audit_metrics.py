"""Compute Exp1 evaluator-vs-human audit metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from thesis_exp.src.edujudge.audit import (
    EVALUATORS,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    ensure_exp01_dirs,
    relpath,
)


def _nan() -> float:
    return float("nan")


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else _nan()


def _safe_rate(mask: pd.Series | np.ndarray) -> float:
    series = pd.Series(mask).dropna()
    return float(series.mean()) if len(series) else _nan()


def _macro_weighted_f1(y_true: list[int], y_pred: list[int], labels: list[int]) -> tuple[float, float]:
    f1s: list[float] = []
    weights: list[int] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
        weights.append(sum(1 for t in y_true if t == label))
    macro = float(np.mean(f1s)) if f1s else _nan()
    weighted = float(np.average(f1s, weights=weights)) if sum(weights) else _nan()
    return macro, weighted


def _qwk(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    if not y_true:
        return _nan()
    n = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    observed = np.zeros((n, n), dtype=float)
    for true, pred in zip(y_true, y_pred):
        observed[label_to_idx[true], label_to_idx[pred]] += 1
    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / observed.sum()
    weights = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return _nan()
    return 1 - numerator / denominator


def _kendall_tau_b(x_values: list[float], y_values: list[float]) -> float:
    n = len(x_values)
    if n < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return _nan()
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        xi = x_values[i]
        yi = y_values[i]
        for j in range(i + 1, n):
            dx = (x_values[j] > xi) - (x_values[j] < xi)
            dy = (y_values[j] > yi) - (y_values[j] < yi)
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return _nan()
    return (concordant - discordant) / denom


def _rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return _nan()
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return _nan()
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(set(x_values)) <= 1 or len(set(y_values)) <= 1:
        return _nan()
    return _pearson(_rank_average(x_values), _rank_average(y_values))


def _valid_eval_df(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    pred_col = f"pred_{suffix}"
    label_col = f"pred_label_{suffix}"
    valid_col = f"valid_{suffix}"
    valid = df[df[valid_col].astype(bool)].copy()
    valid[pred_col] = pd.to_numeric(valid[pred_col], errors="coerce")
    valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce").astype("Int64")
    valid["label_5"] = pd.to_numeric(valid["label_5"], errors="coerce").astype("Int64")
    valid["human_mean_5"] = pd.to_numeric(valid["human_mean_5"], errors="coerce")
    return valid.dropna(subset=[pred_col, label_col, "label_5", "human_mean_5"])


def overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [1, 2, 3, 4, 5]
    n_total = len(df)
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        pred_col = f"pred_{suffix}"
        label_col = f"pred_label_{suffix}"
        valid_col = f"valid_{suffix}"
        aligned = df[f"raw_pred_{suffix}"].notna()
        valid = _valid_eval_df(df, suffix)
        n_valid = len(valid)
        y_true = valid["label_5"].astype(int).tolist()
        y_pred = valid[label_col].astype(int).tolist()
        diffs = valid[pred_col] - valid["human_mean_5"]
        abs_diffs = diffs.abs()
        macro_f1, weighted_f1 = _macro_weighted_f1(y_true, y_pred, labels)
        rows.append(
            {
                "evaluator": spec.name,
                "n_total": n_total,
                "n_valid": n_valid,
                "invalid_rate": float((aligned.sum() - n_valid) / n_total) if n_total else _nan(),
                "coverage": float(n_valid / n_total) if n_total else _nan(),
                "MAE": _safe_mean(abs_diffs),
                "RMSE": float(np.sqrt(np.mean(np.square(diffs)))) if n_valid else _nan(),
                "Signed Bias": _safe_mean(diffs),
                "Exact Match": _safe_rate(valid[label_col].astype(int) == valid["label_5"].astype(int)),
                "Within-1 Accuracy": _safe_rate((valid[label_col].astype(int) - valid["label_5"].astype(int)).abs() <= 1),
                "Macro-F1": macro_f1,
                "Weighted-F1": weighted_f1,
                "Quadratic Weighted Kappa": _qwk(y_true, y_pred, labels),
                "Kendall tau": _kendall_tau_b(valid["human_mean_5"].tolist(), valid[pred_col].tolist()),
                "Spearman rho": _spearman(valid["human_mean_5"].tolist(), valid[pred_col].tolist()),
                "Bin Agreement": _nan(),
                "notes": "Bin Agreement not reproduced because the PDF binning definition is unavailable.",
            }
        )
    return pd.DataFrame(rows)


def per_bin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        pred_col = f"pred_{suffix}"
        label_col = f"pred_label_{suffix}"
        valid_col = f"valid_{suffix}"
        for label in [1, 2, 3, 4, 5]:
            bin_df = df[df["label_5"].astype(int) == label].copy()
            valid = bin_df[bin_df[valid_col].astype(bool)].copy()
            if len(valid):
                valid[pred_col] = pd.to_numeric(valid[pred_col], errors="coerce")
                valid[label_col] = pd.to_numeric(valid[label_col], errors="coerce").astype("Int64")
                valid["human_mean_5"] = pd.to_numeric(valid["human_mean_5"], errors="coerce")
            pred_labels = valid[label_col].astype(float) if len(valid) else pd.Series(dtype=float)
            rows.append(
                {
                    "evaluator": spec.name,
                    "label_5": label,
                    "n": len(bin_df),
                    "n_valid": len(valid),
                    "accuracy": _safe_rate(pred_labels == label),
                    "mean_human": _safe_mean(valid["human_mean_5"]) if len(valid) else _nan(),
                    "mean_pred": _safe_mean(valid[pred_col]) if len(valid) else _nan(),
                    "signed_bias": _safe_mean(valid[pred_col] - valid["human_mean_5"]) if len(valid) else _nan(),
                    "MAE": _safe_mean((valid[pred_col] - valid["human_mean_5"]).abs()) if len(valid) else _nan(),
                    "overestimation_rate": _safe_rate(pred_labels > label),
                    "severe_overestimation_rate": _safe_rate((pred_labels - label) >= 2),
                    "underestimation_rate": _safe_rate(pred_labels < label),
                    "severe_underestimation_rate": _safe_rate((label - pred_labels) >= 2),
                    "pred_1_rate": _safe_rate(pred_labels == 1),
                    "pred_2_rate": _safe_rate(pred_labels == 2),
                    "pred_3_rate": _safe_rate(pred_labels == 3),
                    "pred_4_rate": _safe_rate(pred_labels == 4),
                    "pred_5_rate": _safe_rate(pred_labels == 5),
                }
            )
    return pd.DataFrame(rows)


def low_score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    low_df = df[df["label_5"].astype(int).isin([1, 2])].copy()
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        pred_col = f"pred_{suffix}"
        label_col = f"pred_label_{suffix}"
        valid = _valid_eval_df(low_df, suffix)
        pred_labels = valid[label_col].astype(int) if len(valid) else pd.Series(dtype=int)
        rows.append(
            {
                "evaluator": spec.name,
                "n_low": len(low_df),
                "n_valid_low": len(valid),
                "low_exact_match": _safe_rate(pred_labels == valid["label_5"].astype(int)) if len(valid) else _nan(),
                "low_recall": _safe_rate(pred_labels <= 2),
                "low_MAE": _safe_mean((valid[pred_col] - valid["human_mean_5"]).abs()) if len(valid) else _nan(),
                "low_signed_bias": _safe_mean(valid[pred_col] - valid["human_mean_5"]) if len(valid) else _nan(),
                "low_overestimation_rate": _safe_rate(pred_labels > valid["label_5"].astype(int)) if len(valid) else _nan(),
                "low_severe_overestimation_rate": _safe_rate((pred_labels - valid["label_5"].astype(int)) >= 2)
                if len(valid)
                else _nan(),
                "low_to_high_rate": _safe_rate(pred_labels >= 4),
                "mean_pred_low": _safe_mean(valid[pred_col]) if len(valid) else _nan(),
                "mean_human_low": _safe_mean(valid["human_mean_5"]) if len(valid) else _nan(),
            }
        )
    return pd.DataFrame(rows)


def high_score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    high_df = df[df["label_5"].astype(int) == 5].copy()
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        pred_col = f"pred_{suffix}"
        label_col = f"pred_label_{suffix}"
        valid = _valid_eval_df(high_df, suffix)
        pred_labels = valid[label_col].astype(int) if len(valid) else pd.Series(dtype=int)
        rows.append(
            {
                "evaluator": spec.name,
                "n_high": len(high_df),
                "n_valid_high": len(valid),
                "acc_at_5": _safe_rate(pred_labels == 5),
                "high_MAE": _safe_mean((valid[pred_col] - valid["human_mean_5"]).abs()) if len(valid) else _nan(),
                "high_signed_bias": _safe_mean(valid[pred_col] - valid["human_mean_5"]) if len(valid) else _nan(),
                "high_to_mid_or_low_rate": _safe_rate(pred_labels <= 3),
                "mean_pred_high": _safe_mean(valid[pred_col]) if len(valid) else _nan(),
                "mean_human_high": _safe_mean(valid["human_mean_5"]) if len(valid) else _nan(),
            }
        )
    return pd.DataFrame(rows)


def stratified_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        pred_col = f"pred_{suffix}"
        label_col = f"pred_label_{suffix}"
        for group_value, group in df.groupby(group_col, dropna=False):
            valid = _valid_eval_df(group, suffix)
            if len(valid):
                pred_labels = valid[label_col].astype(int)
                labels = valid["label_5"].astype(int)
                diffs = valid[pred_col] - valid["human_mean_5"]
                kendall = _kendall_tau_b(valid["human_mean_5"].tolist(), valid[pred_col].tolist())
            else:
                pred_labels = pd.Series(dtype=int)
                labels = pd.Series(dtype=int)
                diffs = pd.Series(dtype=float)
                kendall = _nan()
            rows.append(
                {
                    "evaluator": spec.name,
                    group_col: group_value,
                    "n": len(group),
                    "n_valid": len(valid),
                    "MAE": _safe_mean(diffs.abs()) if len(valid) else _nan(),
                    "Signed Bias": _safe_mean(diffs) if len(valid) else _nan(),
                    "Exact Match": _safe_rate(pred_labels == labels) if len(valid) else _nan(),
                    "Kendall tau": kendall,
                }
            )
    return pd.DataFrame(rows)


PDF_OVERALL_REFERENCE = {
    "EduBenchEvaluator": {"MAE": 0.430, "Signed Bias": 0.246, "Exact Match": 0.725, "Kendall tau": 0.508, "Bin Agreement": 0.897},
    "DeepSeek-V3": {"MAE": 0.576, "Signed Bias": 0.458, "Exact Match": 0.602, "Kendall tau": 0.326, "Bin Agreement": 0.867},
    "DeepSeek-R1": {"MAE": 0.589, "Signed Bias": 0.335, "Exact Match": 0.585, "Kendall tau": 0.319, "Bin Agreement": 0.854},
    "QwQ-plus": {"MAE": 0.593, "Signed Bias": 0.402, "Exact Match": 0.604, "Kendall tau": 0.301, "Bin Agreement": 0.860},
    "GPT-4o": {"MAE": 0.598, "Signed Bias": 0.475, "Exact Match": 0.575, "Kendall tau": 0.278, "Bin Agreement": 0.868},
}

PDF_PER_BIN_REFERENCE = {
    "EduBenchEvaluator": {1: 48.1, 2: 23.4, 3: 21.1, 4: 66.1, 5: 87.7},
    "QwQ-plus": {1: 0.0, 2: 15.2, 3: 16.3, 4: 28.2, 5: 91.0},
    "DeepSeek-V3": {1: 7.7, 2: 0.0, 3: 4.1, 4: 31.4, 5: 91.4},
    "DeepSeek-R1": {1: 7.7, 2: 10.6, 3: 18.0, 4: 30.6, 5: 86.0},
    "GPT-4o": {1: 0.0, 2: 0.0, 3: 5.9, 4: 25.1, 5: 90.0},
}


def pdf_reference_comparison(overall: pd.DataFrame, per_bin: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluator, metrics in PDF_OVERALL_REFERENCE.items():
        current = overall[overall["evaluator"] == evaluator]
        for metric_name, reference in metrics.items():
            current_value = _nan()
            if not current.empty and metric_name in current.columns:
                current_value = float(current.iloc[0][metric_name])
            if metric_name == "Bin Agreement":
                status = "not_comparable"
            elif math.isnan(current_value):
                status = "missing_current"
            else:
                threshold = 0.12 if metric_name in {"MAE", "Signed Bias"} else 0.15
                status = "matched_trend" if abs(current_value - reference) <= threshold else "differs"
            rows.append(
                {
                    "evaluator": evaluator,
                    "metric_name": metric_name,
                    "pdf_reference": reference,
                    "current_value": current_value,
                    "delta": current_value - reference if not math.isnan(current_value) else _nan(),
                    "status": status,
                }
            )
    for evaluator, bins in PDF_PER_BIN_REFERENCE.items():
        for label, reference in bins.items():
            current = per_bin[(per_bin["evaluator"] == evaluator) & (per_bin["label_5"] == label)]
            current_value = _nan()
            if not current.empty:
                current_value = float(current.iloc[0]["accuracy"]) * 100
            status = "missing_current" if math.isnan(current_value) else (
                "matched_trend" if abs(current_value - reference) <= 15 else "differs"
            )
            rows.append(
                {
                    "evaluator": evaluator,
                    "metric_name": f"Acc@{label}",
                    "pdf_reference": reference,
                    "current_value": current_value,
                    "delta": current_value - reference if not math.isnan(current_value) else _nan(),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def run_metrics() -> dict[str, pd.DataFrame]:
    ensure_exp01_dirs()
    df = pd.read_json(EXP01_OUTPUT_DIR / "predictions_aligned.jsonl", lines=True)
    for col in ["label_5", "human_mean_5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    outputs = {
        "evaluator_metrics": overall_metrics(df),
        "per_bin_metrics": per_bin_metrics(df),
        "low_score_metrics": low_score_metrics(df),
        "high_score_metrics": high_score_metrics(df),
    }
    stratified = {
        "metric_level_metrics": "metric_canonical",
        "scenario_level_metrics": "scenario_canonical",
        "subject_level_metrics": "subject_canonical",
        "language_level_metrics": "language",
        "education_level_metrics": "education_level_canonical",
        "generator_model_level_metrics": "generator_model",
    }
    for name, col in stratified.items():
        outputs[name] = stratified_metrics(df, col)
    outputs["pdf_reference_comparison"] = pdf_reference_comparison(
        outputs["evaluator_metrics"], outputs["per_bin_metrics"]
    )

    for name, table in outputs.items():
        table.to_csv(EXP01_TABLES_DIR / f"{name}.csv", index=False)
    return outputs


def main() -> None:
    outputs = run_metrics()
    metrics = outputs["evaluator_metrics"]
    print("Evaluator metrics:")
    for _, row in metrics.iterrows():
        print(
            f"{row['evaluator']}: n_valid={int(row['n_valid'])}, "
            f"MAE={row['MAE']:.3f}, Exact={row['Exact Match']:.3f}, Kendall={row['Kendall tau']:.3f}"
        )
    print(f"Outputs: {relpath(EXP01_TABLES_DIR / 'evaluator_metrics.csv')}")


if __name__ == "__main__":
    main()

