"""Plot Exp3 input-ablation figures from available runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from thesis_exp.src.edujudge.exp03 import EXP03_FIGURES_DIR, EXP03_PREDICTIONS_DIR, EXP03_TABLES_DIR, TEMPLATE_NAMES, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.compute_input_ablation_metrics import compute_input_ablation_metrics
from thesis_exp.src.edujudge.exp03.collect_exp03_results import float_or_none
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_text


def save_figure(fig: plt.Figure, stem: str) -> None:
    EXP03_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(EXP03_FIGURES_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def template_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("ablation_id") or row.get("template_name")) for row in rows]


def pending_figure(stem: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(fig, stem)


def numeric(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        parsed = float_or_none(value)
        out.append(float("nan") if parsed is None else parsed)
    return out


def figure_overall(summary: list[dict[str, Any]]) -> None:
    labels = template_labels(summary)
    metrics = [
        ("test_accuracy", "Accuracy"),
        ("test_MAE_label", "MAE label"),
        ("test_kendall_tau", "Kendall tau"),
        ("test_signed_bias", "Signed bias"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        ax.bar(labels, numeric([row.get(key) for row in summary]))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Exp3 Input Ablation Overall Metrics")
    save_figure(fig, "fig03_input_ablation_overall_metrics")


def figure_low_score(summary: list[dict[str, Any]]) -> None:
    labels = template_labels(summary)
    metrics = [
        ("test_acc_at_1", "Acc@1"),
        ("test_acc_at_2", "Acc@2"),
        ("test_low_to_high_rate", "low_to_high_rate"),
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for idx, (key, label) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, numeric([row.get(key) for row in summary]), width=width, label=label)
    ax.set_xticks(x, labels, rotation=30)
    ax.set_ylim(0, 1)
    ax.set_title("Exp3 Low-Score Metrics")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "fig03_input_ablation_low_score")


def figure_per_bin() -> None:
    path = EXP03_TABLES_DIR / "input_ablation_per_bin.csv"
    if not path.exists():
        pending_figure("fig03_per_bin_accuracy_by_template", "Exp3 Per-Bin Accuracy", "No per-bin metrics are available yet.")
        return
    rows = [row for row in read_csv(path) if row.get("split") == "test"]
    if not rows:
        pending_figure("fig03_per_bin_accuracy_by_template", "Exp3 Per-Bin Accuracy", "No completed test runs are available yet.")
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["template_name"]].append(row)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for template_name in TEMPLATE_NAMES:
        subset = sorted(grouped.get(template_name, []), key=lambda row: int(float(row.get("label_5") or 0)))
        if not subset:
            continue
        ax.plot(
            [int(float(row["label_5"])) for row in subset],
            numeric([row.get("accuracy") for row in subset]),
            marker="o",
            label=template_name.split("_", 1)[0],
        )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 1)
    ax.set_xlabel("True label_5")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp3 Per-Bin Accuracy by Template")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    save_figure(fig, "fig03_per_bin_accuracy_by_template")


def figure_low_to_high(summary: list[dict[str, Any]]) -> None:
    labels = template_labels(summary)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.bar(labels, numeric([row.get("test_low_to_high_rate") for row in summary]))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Exp3 Low-to-High Rate by Template")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "fig03_low_to_high_rate_by_template")


def figure_metric_delta_heatmap() -> None:
    path = EXP03_TABLES_DIR / "input_ablation_metric_delta_vs_a2.csv"
    rows = read_csv(path) if path.exists() else []
    rows = [
        row
        for row in rows
        if row.get("template_name") == "A3_question_answer_metric_rubric"
        and float_or_none(row.get("delta_MAE_label_vs_a2")) is not None
    ]
    if not rows:
        pending_figure(
            "fig03_metric_delta_heatmap_rubric_vs_baseline",
            "A3 - A2 Metric-Level MAE Delta",
            "A3 metric-level results are pending. Run formal A3 training, then rerun postprocess.",
        )
        return
    rows = sorted(rows, key=lambda row: row["metric_canonical"])
    values = np.array([[float(row["delta_MAE_label_vs_a2"])] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, max(4.2, len(rows) * 0.34)))
    im = ax.imshow(values, aspect="auto")
    ax.set_yticks(np.arange(len(rows)), [row["metric_canonical"] for row in rows])
    ax.set_xticks([0], ["A3 - A2 MAE"])
    ax.set_title("Metric-Level MAE Delta: Rubric vs Baseline")
    for idx, row in enumerate(rows):
        ax.text(0, idx, f"{float(row['delta_MAE_label_vs_a2']):+.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "fig03_metric_delta_heatmap_rubric_vs_baseline")


def figure_calibration(summary: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plotted = False
    for row in summary:
        template_name = row["template_name"]
        pred_path = EXP03_PREDICTIONS_DIR / f"{template_name}_predictions_test.jsonl"
        if not pred_path.exists():
            continue
        preds = read_jsonl(pred_path)
        grouped: dict[int, list[float]] = defaultdict(list)
        for pred in preds:
            grouped[int(pred["label_5"])].append(float(pred.get("pred_score_expected", pred.get("pred_score_5"))))
        labels = sorted(grouped)
        if not labels:
            continue
        ax.plot(labels, [float(np.mean(grouped[label])) for label in labels], marker="o", label=row.get("ablation_id"))
        plotted = True
    ax.plot([1, 5], [1, 5], linestyle="--", label="ideal")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("True label_5")
    ax.set_ylabel("Mean predicted expected score")
    ax.set_title("Exp3 Calibration Curve by Template")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    if not plotted:
        ax.text(0.5, 0.5, "No prediction files are available yet.", transform=ax.transAxes, ha="center", va="center")
    save_figure(fig, "fig03_calibration_curve_by_template")


def figure_token_length() -> None:
    path = EXP03_TABLES_DIR / "input_ablation_token_length.csv"
    rows = read_csv(path) if path.exists() else []
    rows = [row for row in rows if row.get("split") == "test"]
    if not rows:
        pending_figure("fig03_token_length_by_template", "Exp3 Token Length by Template", "Token length stats are pending.")
        return
    labels = [str(row.get("template_name")).split("_", 1)[0] for row in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(labels, numeric([row.get("mean_token_length") for row in rows]))
    ax.set_ylabel("Mean token length")
    ax.set_title("Exp3 Token Length by Template")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "fig03_token_length_by_template")


def figure_truncation(summary: list[dict[str, Any]]) -> None:
    labels = template_labels(summary)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.bar(labels, numeric([row.get("truncation_rate") for row in summary]))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Truncation rate")
    ax.set_title("Exp3 Truncation Rate by Template")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "fig03_truncation_rate_by_template")


def plot_input_ablation_figures() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    summary = compute_input_ablation_metrics()
    figure_overall(summary)
    figure_low_score(summary)
    figure_per_bin()
    figure_low_to_high(summary)
    figure_metric_delta_heatmap()
    figure_calibration(summary)
    figure_token_length()
    figure_truncation(summary)
    write_text(
        EXP03_FIGURES_DIR / "figure_manifest.md",
        "\n".join(
            [
                "# Exp3 Figure Manifest",
                "",
                "- fig03_input_ablation_overall_metrics.png/pdf",
                "- fig03_input_ablation_low_score.png/pdf",
                "- fig03_per_bin_accuracy_by_template.png/pdf",
                "- fig03_low_to_high_rate_by_template.png/pdf",
                "- fig03_metric_delta_heatmap_rubric_vs_baseline.png/pdf",
                "- fig03_calibration_curve_by_template.png/pdf",
                "- fig03_token_length_by_template.png/pdf",
                "- fig03_truncation_rate_by_template.png/pdf",
            ]
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Exp3 input-ablation figures.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    plot_input_ablation_figures()
    print(f"Exp3 figures written to {relpath(EXP03_FIGURES_DIR)}")


if __name__ == "__main__":
    main()
