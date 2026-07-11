"""Plot lightweight Exp28 dev/final-test paper figures from aggregate CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_DEV_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/"
    "tables/exp28e_multiseed_dev_summary.csv"
)
DEFAULT_TEST_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test/"
    "tables/exp28g_test_multiseed_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test/figures"
)
VARIANT_LABELS = {
    "b0_original_human": "B0 Original",
    "b1_primary_teacher_all": "B1 Qwen all",
    "b2_selective_dual_teacher": "B2 Selective dual",
    "b3_filter_unresolved": "B3 Filter unresolved",
    "b4_random_transition_control": "B4 Random control",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, Any], metric: str) -> float:
    return float(row[f"{metric}_mean"])


def configure() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def plot_overall(plt: Any, rows: list[dict[str, Any]], out_dir: Path, prefix: str) -> None:
    metrics = [
        ("MAE_label", "MAE (lower is better)"),
        ("Exact Match", "Exact Match"),
        ("Kendall tau", "Kendall's tau"),
        ("Bin Agreement", "Bin Agreement"),
    ]
    labels = [VARIANT_LABELS.get(row["variant"], row["variant"]) for row in rows]
    colors = ["#5B6770", "#8A9A5B", "#167D8D", "#A86F3D", "#9B6A88"][: len(rows)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.8))
    for axis, (metric, title) in zip(axes, metrics):
        values = [value(row, metric) for row in rows]
        axis.bar(range(len(rows)), values, color=colors, width=0.72)
        axis.set_title(title)
        axis.set_xticks(range(len(rows)), labels, rotation=34, ha="right")
        axis.grid(axis="y", alpha=0.22)
        for index, observed in enumerate(values):
            axis.text(index, observed, f"{observed:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"Exp28 {prefix}: evaluator-human agreement", y=1.03)
    fig.tight_layout()
    save(fig, out_dir / f"exp28_{prefix.lower().replace(' ', '_')}_overall_metrics.png")
    plt.close(fig)


def plot_per_label(plt: Any, rows: list[dict[str, Any]], out_dir: Path, prefix: str) -> None:
    labels = [VARIANT_LABELS.get(row["variant"], row["variant"]) for row in rows]
    scores = [1, 2, 3, 4, 5]
    width = 0.8 / max(1, len(rows))
    colors = ["#5B6770", "#8A9A5B", "#167D8D", "#A86F3D", "#9B6A88"][: len(rows)]
    fig, axis = plt.subplots(figsize=(7.8, 3.6))
    for index, (row, label, color) in enumerate(zip(rows, labels, colors)):
        offset = (index - (len(rows) - 1) / 2) * width
        axis.bar(
            [score + offset for score in scores],
            [value(row, f"Acc@{score}") for score in scores],
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(scores, [f"Score {score}" for score in scores])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Per-label accuracy")
    axis.set_title(f"Exp28 {prefix}: accuracy across the full score scale")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, out_dir / f"exp28_{prefix.lower().replace(' ', '_')}_per_label_accuracy.png")
    plt.close(fig)


def plot_risk(plt: Any, rows: list[dict[str, Any]], out_dir: Path, prefix: str) -> None:
    metrics = [
        ("low_to_high_rate", "Low-to-high"),
        ("high_to_mid_or_low_rate", "Score-5 downgraded to <=3"),
        ("Acc@5", "Score-5 recall"),
    ]
    labels = [VARIANT_LABELS.get(row["variant"], row["variant"]) for row in rows]
    colors = ["#5B6770", "#8A9A5B", "#167D8D", "#A86F3D", "#9B6A88"][: len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for axis, (metric, title) in zip(axes, metrics):
        observed = [value(row, metric) for row in rows]
        axis.bar(range(len(rows)), observed, color=colors, width=0.72)
        axis.set_title(title)
        axis.set_ylim(0.0, max(0.05, min(1.0, max(observed) * 1.22)))
        axis.set_xticks(range(len(rows)), labels, rotation=34, ha="right")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(f"Exp28 {prefix}: tail risk and high-score protection", y=1.03)
    fig.tight_layout()
    save(fig, out_dir / f"exp28_{prefix.lower().replace(' ', '_')}_risk_protection.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-summary", type=Path, default=DEFAULT_DEV_SUMMARY)
    parser.add_argument("--test-summary", type=Path, default=DEFAULT_TEST_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    plt = configure()
    dev_rows = read_csv(args.dev_summary)
    plot_overall(plt, dev_rows, args.out_dir, "Dev")
    plot_per_label(plt, dev_rows, args.out_dir, "Dev")
    plot_risk(plt, dev_rows, args.out_dir, "Dev")
    if args.include_test:
        test_rows = read_csv(args.test_summary)
        plot_overall(plt, test_rows, args.out_dir, "Final test")
        plot_per_label(plt, test_rows, args.out_dir, "Final test")
        plot_risk(plt, test_rows, args.out_dir, "Final test")


if __name__ == "__main__":
    main()
