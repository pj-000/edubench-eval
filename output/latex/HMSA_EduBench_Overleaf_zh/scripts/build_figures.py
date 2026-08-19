#!/usr/bin/env python3
"""Generate publication-style vector figures for the HMSA manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

COLORS = {
    # Restrained, color-blind-friendly academic palette. Results use one
    # accent colour for HMSA and neutral grey for the hard-only baseline.
    "ink": "#262626",
    "muted": "#666666",
    "grid": "#E7E7E7",
    "pair": "#C4C4C4",
    "baseline": "#7A7A7A",
    "method": "#3C5488",
    "hard": "#3C5488",
    "soft": "#A66A4B",
    "backbone": "#F1F3F5",
    "hard_fill": "#EDF1F7",
    "soft_fill": "#F7F1ED",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10,
    linestyle: str = "-",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
    )
    return box


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    linestyle: str | tuple = "-",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=color,
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def architecture_figure() -> None:
    """Compact two-branch architecture figure with an explicit deployment path."""
    # Draw close to the final full-width print size.  Oversized canvases make
    # otherwise reasonable font sizes shrink below 7 pt when LaTeX scales the
    # PDF down to the text width.
    fig, axis = plt.subplots(figsize=(8.0, 3.25))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Quiet background bands establish hierarchy without enclosing every item.
    axis.add_patch(Rectangle((0.565, 0.54), 0.41, 0.29, facecolor="#F5F7FA", edgecolor="none", zorder=0))
    axis.add_patch(Rectangle((0.565, 0.18), 0.41, 0.27, facecolor="#F3F8F7", edgecolor="none", zorder=0))
    axis.text(0.578, 0.795, "PRIMARY  ·  TRAINING + SCORING", ha="left", va="center", fontsize=8.8, fontweight="bold", color=COLORS["hard"], fontfamily="DejaVu Sans")
    axis.text(0.578, 0.415, "AUXILIARY  ·  TRAINING ONLY", ha="left", va="center", fontsize=8.8, fontweight="bold", color="#347873", fontfamily="DejaVu Sans")

    # Input is represented as compact fields instead of a large generic box.
    axis.text(0.045, 0.755, "MODEL INPUT", ha="left", va="center", fontsize=8.8, fontweight="bold", color=COLORS["muted"], fontfamily="DejaVu Sans")
    input_labels = ("Question", "Answer", "Criterion")
    for index, label in enumerate(input_labels):
        y = 0.61 - index * 0.14
        box = FancyBboxPatch((0.045, y), 0.15, 0.095, boxstyle="round,pad=0.008,rounding_size=0.018", linewidth=0.8, edgecolor="#C8CDD3", facecolor="#F7F8FA")
        axis.add_patch(box)
        axis.text(0.12, y + 0.0475, label, ha="center", va="center", fontsize=9.4, color=COLORS["ink"], fontfamily="DejaVu Sans")

    # A shallow stack suggests the shared transformer without a bulky container.
    for offset, shade in ((0.000, "#E1E6EC"), (0.012, "#E9EDF2"), (0.024, "#F1F3F6")):
        layer = FancyBboxPatch((0.275 + offset, 0.355 + offset), 0.205, 0.285, boxstyle="round,pad=0.012,rounding_size=0.025", linewidth=0.9, edgecolor="#AEB6BF", facecolor=shade)
        axis.add_patch(layer)
    axis.text(0.401, 0.535, "Shared backbone", ha="center", va="center", fontsize=10.7, fontweight="bold", color=COLORS["ink"], fontfamily="DejaVu Sans")
    axis.text(0.401, 0.465, "Qwen3-Reranker-0.6B", ha="center", va="center", fontsize=9.1, color=COLORS["muted"], fontfamily="DejaVu Sans")

    input_centers = (0.6575, 0.5175, 0.3775)
    for center in input_centers:
        axis.plot([0.195, 0.215], [center, center], color=COLORS["muted"], linewidth=0.8)
    axis.plot([0.215, 0.215], [input_centers[-1], input_centers[0]], color=COLORS["muted"], linewidth=0.8)
    arrow(axis, (0.215, 0.5175), (0.275, 0.5175), COLORS["ink"])
    arrow(axis, (0.504, 0.51), (0.585, 0.677), COLORS["hard"])
    arrow(axis, (0.504, 0.49), (0.585, 0.317), "#347873", (0, (3, 2)))

    # Heads are compact modules; the coloured result box makes deployment explicit.
    hard_head = FancyBboxPatch((0.585, 0.625), 0.165, 0.105, boxstyle="round,pad=0.010,rounding_size=0.018", linewidth=1.0, edgecolor=COLORS["hard"], facecolor="#E9EEF6")
    soft_head = FancyBboxPatch((0.585, 0.265), 0.165, 0.105, boxstyle="round,pad=0.010,rounding_size=0.018", linewidth=1.0, edgecolor="#347873", facecolor="#E7F1F0")
    axis.add_patch(hard_head)
    axis.add_patch(soft_head)
    axis.text(0.6675, 0.677, "Hard-label head\n5-class logits $z^{(h)}$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"], fontfamily="DejaVu Sans", linespacing=1.25)
    axis.text(0.6675, 0.317, "Soft-label head\n5-class logits $z^{(s)}$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLORS["ink"], fontfamily="DejaVu Sans", linespacing=1.25)

    score_box = FancyBboxPatch((0.835, 0.60), 0.115, 0.155, boxstyle="round,pad=0.010,rounding_size=0.020", linewidth=0, facecolor=COLORS["hard"])
    loss_box = FancyBboxPatch((0.82, 0.265), 0.13, 0.105, boxstyle="round,pad=0.010,rounding_size=0.020", linewidth=1.0, edgecolor="#347873", facecolor="white", linestyle="--")
    axis.add_patch(score_box)
    axis.add_patch(loss_box)
    axis.text(0.8925, 0.696, "FINAL SCORE", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#DDE6F4", fontfamily="DejaVu Sans")
    axis.text(0.8925, 0.642, "$\\hat{y}\\in\\{1,\\ldots,5\\}$", ha="center", va="center", fontsize=11.0, color="white")
    axis.text(0.885, 0.317, "$\\mathcal{L}_{s}=\\mathrm{CE}(z^{(s)},d)$", ha="center", va="center", fontsize=9.2, color="#2C6965")
    axis.text(0.885, 0.215, "discarded at inference", ha="center", va="center", fontsize=8.2, color=COLORS["muted"], fontfamily="DejaVu Sans")

    arrow(axis, (0.75, 0.677), (0.835, 0.677), COLORS["hard"])
    arrow(axis, (0.75, 0.317), (0.82, 0.317), "#347873", (0, (3, 2)))
    axis.text(0.792, 0.643, "$\\mathrm{argmax}$", ha="center", va="top", fontsize=8.2, color=COLORS["hard"])
    axis.text(0.6675, 0.565, "$\\mathcal{L}_{h}=\\mathrm{CE}(z^{(h)},y)$", ha="center", va="center", fontsize=9.2, color=COLORS["hard"])

    axis.text(0.565, 0.110, "JOINT TRAINING OBJECTIVE", ha="left", va="center", fontsize=8.8, fontweight="bold", color=COLORS["muted"], fontfamily="DejaVu Sans")
    axis.text(0.565, 0.052, "$\\mathcal{L}=\\mathcal{L}_{h}+\\lambda\\mathcal{L}_{s}\\quad(\\lambda=1)$", ha="left", va="center", fontsize=10.8, color=COLORS["ink"])
    save_figure(fig, "fig1_hmsa_architecture")


def seed_pair_figure() -> None:
    rows = read_csv(DATA / "test_seed_primary.csv")
    metric_info = [
        ("MAE_human_mean", "(a) MAE $\\downarrow$", "{:.3f}"),
        ("Exact_rounded", "(b) Exact match $\\uparrow$", "{:.3f}"),
        ("Kendall_human_mean", "(c) Kendall $\\tau_b$ $\\uparrow$", "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65))
    for axis, (metric, title, formatter) in zip(axes, metric_info):
        selected = sorted((row for row in rows if row["metric"] == metric), key=lambda row: int(row["seed"]))
        for row in selected:
            b0 = float(row["b0"])
            hmsa = float(row["exp51"])
            axis.plot([0, 1], [b0, hmsa], color=COLORS["pair"], linewidth=0.9, zorder=1)
            axis.scatter(0, b0, facecolor="white", edgecolor=COLORS["baseline"], linewidth=1.0, marker="o", s=34, zorder=3)
            axis.scatter(1, hmsa, facecolor="white", edgecolor=COLORS["method"], linewidth=1.0, marker="D", s=32, zorder=3)
        b0_mean = np.mean([float(row["b0"]) for row in selected])
        hmsa_mean = np.mean([float(row["exp51"]) for row in selected])
        delta = hmsa_mean - b0_mean
        axis.plot([0, 1], [b0_mean, hmsa_mean], color=COLORS["ink"], linewidth=1.5, zorder=2)
        axis.scatter(0, b0_mean, color=COLORS["baseline"], marker="o", edgecolor="white", linewidth=0.7, s=58, zorder=4)
        axis.scatter(1, hmsa_mean, color=COLORS["method"], marker="D", edgecolor="white", linewidth=0.7, s=55, zorder=4)
        axis.set_xticks([0, 1], ["Hard-only", "HMSA"])
        axis.set_title(title, fontsize=9.6, pad=7)
        axis.tick_params(axis="both", labelsize=8.5)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(COLORS["muted"])
        axis.spines[["left", "bottom"]].set_linewidth(0.75)
        sign = "+" if delta >= 0 else ""
        axis.text(0.5, 0.035, f"Mean $\\Delta$ = {sign}{formatter.format(delta)}", transform=axis.transAxes, ha="center", va="bottom", fontsize=8.0, color=COLORS["muted"])
    fig.tight_layout(w_pad=1.15)
    save_figure(fig, "fig2_seed_paired_primary")


def recall_figure() -> None:
    rows = read_csv(DATA / "test_recall_by_label.csv")
    labels = [int(row["label"]) for row in rows]
    denominators = [int(row["n"]) for row in rows]
    b0 = np.asarray([float(row["b0_mean"]) for row in rows])
    hmsa = np.asarray([float(row["exp51_mean"]) for row in rows])
    b0_sd = np.asarray([float(row["b0_sd"]) for row in rows])
    exp_sd = np.asarray([float(row["exp51_sd"]) for row in rows])
    delta = hmsa - b0

    x = np.arange(len(labels))
    offset = 0.11
    fig, axis = plt.subplots(figsize=(7.2, 3.7))
    for index in range(len(labels)):
        axis.plot(
            [x[index] - offset, x[index] + offset],
            [b0[index], hmsa[index]],
            color=COLORS["pair"],
            linewidth=1.1,
            zorder=1,
        )
    axis.errorbar(
        x - offset,
        b0,
        yerr=b0_sd,
        fmt="o",
        markersize=6.4,
        capsize=3.2,
        color=COLORS["baseline"],
        ecolor=COLORS["baseline"],
        elinewidth=1.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    axis.errorbar(
        x + offset,
        hmsa,
        yerr=exp_sd,
        fmt="D",
        markersize=6.1,
        capsize=3.2,
        color=COLORS["method"],
        ecolor=COLORS["method"],
        elinewidth=1.0,
        markerfacecolor=COLORS["method"],
        markeredgewidth=0.9,
        zorder=3,
    )
    for index, value in enumerate(delta):
        annotation_y = min(max(b0[index] + b0_sd[index], hmsa[index] + exp_sd[index]) + 0.045, 0.955)
        axis.text(
            x[index],
            annotation_y,
            f"{value:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=COLORS["ink"],
        )
    tick_labels = [
        f"Score {label}\n($n={denominator:,}$)"
        for label, denominator in zip(labels, denominators)
    ]
    axis.set_xticks(x, tick_labels)
    axis.set_ylabel(r"Recall $\uparrow$")
    axis.set_ylim(0, 1.0)
    axis.tick_params(axis="x", labelsize=8.8, pad=7)
    axis.tick_params(axis="y", labelsize=8.8)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["baseline"], markersize=6, label="Hard-only"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["method"], markeredgecolor=COLORS["method"], markersize=5.6, label="HMSA"),
    ]
    axis.legend(handles=legend, frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.4)
    axis.text(
        0.99,
        0.025,
        r"$\Delta$ recall = HMSA $-$ Hard-only",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.1,
        color=COLORS["muted"],
    )
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(COLORS["muted"])
    axis.spines[["left", "bottom"]].set_linewidth(0.75)
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, "fig3_test_recall_by_label")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["muted"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    architecture_figure()
    seed_pair_figure()
    recall_figure()
    print(f"Generated figures in {FIGURES}")


if __name__ == "__main__":
    main()
