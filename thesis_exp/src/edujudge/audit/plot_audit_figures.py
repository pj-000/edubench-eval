"""Generate Exp1 audit figures with matplotlib."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_exp.src.edujudge.audit import (
    EVALUATORS,
    EXP01_FIGURES_DIR,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    ensure_exp01_dirs,
    relpath,
)
from thesis_exp.src.edujudge.data.normalize_fields import METRIC_SPECS, SCENARIO_SPECS


def savefig(name: str) -> None:
    EXP01_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        plt.savefig(EXP01_FIGURES_DIR / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close()


def short_labels(labels: list[str], width: int = 16) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) for label in labels]


def plot_grouped_metrics(metrics: pd.DataFrame) -> None:
    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, metrics["MAE"], width, label="MAE")
    plt.bar(x + width / 2, metrics["Signed Bias"], width, label="Signed Bias")
    plt.xticks(x, metrics["evaluator"], rotation=25, ha="right")
    plt.ylabel("Score difference")
    plt.title("Evaluator MAE and Signed Bias")
    plt.legend()
    savefig("fig01_evaluator_overall_mae_bias")

    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, metrics["Exact Match"], width, label="Exact Match")
    plt.bar(x + width / 2, metrics["Kendall tau"], width, label="Kendall tau")
    plt.xticks(x, metrics["evaluator"], rotation=25, ha="right")
    plt.ylabel("Agreement")
    plt.ylim(0, 1)
    plt.title("Evaluator Exact Match and Kendall Tau")
    plt.legend()
    savefig("fig01_evaluator_exact_kendall")


def plot_per_bin(per_bin: pd.DataFrame) -> None:
    pivot = per_bin.pivot(index="label_5", columns="evaluator", values="accuracy")
    plt.figure(figsize=(8.5, 5))
    for evaluator in pivot.columns:
        plt.plot(pivot.index, pivot[evaluator], marker="o", label=evaluator)
    plt.xlabel("Human label")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Per-bin Accuracy by True Human Label")
    plt.legend()
    savefig("fig01_per_bin_accuracy")

    bias = per_bin.pivot(index="label_5", columns="evaluator", values="signed_bias")
    plt.figure(figsize=(8.5, 5))
    for evaluator in bias.columns:
        plt.plot(bias.index, bias[evaluator], marker="o", label=evaluator)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Human label")
    plt.ylabel("Signed bias")
    plt.title("Bias by True Human Score")
    plt.legend()
    savefig("fig01_bias_by_true_score")

    mean_pred = per_bin.pivot(index="label_5", columns="evaluator", values="mean_pred")
    plt.figure(figsize=(8.5, 5))
    for evaluator in mean_pred.columns:
        plt.plot(mean_pred.index, mean_pred[evaluator], marker="o", label=evaluator)
    plt.plot([1, 5], [1, 5], linestyle="--", label="Ideal")
    plt.xlabel("Human label")
    plt.ylabel("Mean predicted score")
    plt.ylim(1, 5.2)
    plt.title("Calibration Curve")
    plt.legend()
    savefig("fig01_calibration_curve")

    plt.figure(figsize=(8.5, 5))
    for evaluator in mean_pred.columns:
        plt.plot(mean_pred.index, mean_pred[evaluator], marker="o", label=evaluator)
    plt.xlabel("Human label")
    plt.ylabel("Mean predicted score")
    plt.ylim(1, 5.2)
    plt.title("Mean Prediction by True Label")
    plt.legend()
    savefig("fig01_mean_pred_by_true_label")


def plot_low_score(low: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(low["evaluator"], low["low_to_high_rate"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.title("Low-score Items Predicted as High (label<=2 -> pred>=4)")
    savefig("fig01_low_score_overestimation_rate")


def plot_confusions(aligned: pd.DataFrame) -> None:
    labels = [1, 2, 3, 4, 5]
    for spec in EVALUATORS:
        suffix = spec.field_suffix
        valid = aligned[aligned[f"valid_{suffix}"].astype(bool)].copy()
        if valid.empty:
            continue
        matrix = np.zeros((5, 5), dtype=int)
        for _, row in valid.iterrows():
            true_label = int(row["label_5"])
            pred_label = int(row[f"pred_label_{suffix}"])
            matrix[true_label - 1, pred_label - 1] += 1
        plt.figure(figsize=(6.2, 5.4))
        plt.imshow(matrix)
        plt.colorbar(label="Count")
        plt.xticks(np.arange(5), labels)
        plt.yticks(np.arange(5), labels)
        plt.xlabel("Predicted label")
        plt.ylabel("Human label")
        plt.title(f"Confusion Matrix: {spec.name}")
        for i in range(5):
            for j in range(5):
                plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
        savefig(f"fig01_confusion_matrix_{suffix}")


def heatmap(table: pd.DataFrame, group_col: str, name: str, title: str, preferred_columns: list[str] | None = None) -> None:
    pivot = table.pivot(index="evaluator", columns=group_col, values="MAE")
    if preferred_columns:
        columns = [col for col in preferred_columns if col in pivot.columns]
        columns.extend([col for col in pivot.columns if col not in columns])
        pivot = pivot[columns]
    if group_col == "subject_canonical" and len(pivot.columns) > 25:
        counts = table.groupby(group_col)["n_valid"].sum().sort_values(ascending=False)
        pivot = pivot[[col for col in counts.index[:25] if col in pivot.columns]]
    values = pivot.to_numpy(dtype=float)
    fig_width = max(9, len(pivot.columns) * 0.55)
    plt.figure(figsize=(fig_width, 4.8))
    plt.imshow(values, aspect="auto")
    plt.colorbar(label="MAE")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), short_labels(list(pivot.columns), width=12), rotation=45, ha="right")
    plt.title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not math.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    savefig(name)


def plot_label_distribution(aligned: pd.DataFrame) -> None:
    counts = aligned["label_5"].astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    low_count = int(counts.loc[1] + counts.loc[2])
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Human label")
    plt.ylabel("Count")
    plt.title(f"Test Label Distribution (low-score count={low_count})")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    savefig("fig01_test_label_distribution")


def run_plots() -> list[Path]:
    ensure_exp01_dirs()
    metrics = pd.read_csv(EXP01_TABLES_DIR / "evaluator_metrics.csv")
    per_bin = pd.read_csv(EXP01_TABLES_DIR / "per_bin_metrics.csv")
    low = pd.read_csv(EXP01_TABLES_DIR / "low_score_metrics.csv")
    aligned = pd.read_json(EXP01_OUTPUT_DIR / "predictions_aligned.jsonl", lines=True)

    plot_grouped_metrics(metrics)
    plot_per_bin(per_bin)
    plot_low_score(low)
    plot_confusions(aligned)
    heatmap(
        pd.read_csv(EXP01_TABLES_DIR / "metric_level_metrics.csv"),
        "metric_canonical",
        "fig01_metric_mae_heatmap",
        "Metric-level MAE Heatmap",
        [spec["canonical_metric"] for spec in METRIC_SPECS],
    )
    heatmap(
        pd.read_csv(EXP01_TABLES_DIR / "scenario_level_metrics.csv"),
        "scenario_canonical",
        "fig01_scenario_mae_heatmap",
        "Scenario-level MAE Heatmap",
        [spec["canonical_scenario"] for spec in SCENARIO_SPECS],
    )
    heatmap(
        pd.read_csv(EXP01_TABLES_DIR / "subject_level_metrics.csv"),
        "subject_canonical",
        "fig01_subject_mae_heatmap",
        "Subject-level MAE Heatmap (local enriched metadata)",
    )
    plot_label_distribution(aligned)
    return sorted(EXP01_FIGURES_DIR.glob("fig01_*.*"))


def main() -> None:
    figures = run_plots()
    png_count = len([path for path in figures if path.suffix == ".png"])
    pdf_count = len([path for path in figures if path.suffix == ".pdf"])
    print(f"Generated figures: {png_count} png / {pdf_count} pdf")
    print(f"Outputs: {relpath(EXP01_FIGURES_DIR)}")


if __name__ == "__main__":
    main()

