#!/usr/bin/env python3
"""Plot the locked Exp28-31 dev-only dual-teacher campaign summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/"
    "exp32_dual_teacher_campaign_conclusion_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")


def plot_exp28(rows: list[dict[str, str]], root: Path) -> None:
    selected = [row for row in rows if row["experiment"] == "Exp28"]
    labels = ["Original", "Qwen all", "Dual teacher", "Filter", "Random control"]
    colors = ["#2f6b4f", "#b55d4c", "#c28a2c", "#7b6d8d", "#4f78a8"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
    metrics = [
        ("MAE", "MAE (lower is better)"),
        ("Exact_Match", "Exact match (higher is better)"),
        ("low_to_high", "Low-to-high rate (lower is better)"),
    ]
    for axis, (field, title) in zip(axes, metrics):
        values = [float(row[field]) for row in selected]
        axis.bar(range(len(values)), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Exp28 three-seed dev comparison")
    fig.tight_layout()
    save_figure(fig, root, "exp32_exp28_three_seed_comparison")
    plt.close(fig)


def plot_followups(rows: list[dict[str, str]], root: Path) -> None:
    selected = [row for row in rows if row["experiment"] in {"Exp29", "Exp30", "Exp31"}]
    labels = [
        "E29 audited", "E29 exposure", "E29 random",
        "E30 confirmed", "E30 random", "E31 disputed", "E31 random",
    ]
    colors = ["#c28a2c", "#7b6d8d", "#4f78a8", "#c28a2c", "#4f78a8", "#c28a2c", "#4f78a8"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.8))
    for axis, field, title in (
        (axes[0], "MAE", "Seed-42 MAE (lower is better)"),
        (axes[1], "low_to_high", "Seed-42 low-to-high rate (lower is better)"),
    ):
        values = [float(row[field]) for row in selected]
        axis.bar(range(len(values)), values, color=colors)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Exp29-31 teacher-targeted follow-up scouts and controls")
    fig.tight_layout()
    save_figure(fig, root, "exp32_followup_scout_comparison")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root / "tables" / "exp32_campaign_summary.csv")
    plot_exp28(rows, args.root)
    plot_followups(rows, args.root)
    print({"rows": len(rows), "figure_dir": str(args.root / "figures")})


if __name__ == "__main__":
    main()
