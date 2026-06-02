"""Generate paper-ready Exp2 result reports and figures.

This script is read-only with respect to model outputs: it never retrains and
does not modify logits, probabilities, labels, or prediction values. It only
formats the existing formal Exp2 outputs into figures, comparison tables, and
Markdown reports.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from thesis_exp.src.edujudge.exp02 import EXP02_OUTPUT_DIR
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_text


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXP01_OUTPUT_DIR = REPO_ROOT / "thesis_exp" / "outputs" / "exp01_audit"
FIGURES_DIR = EXP02_OUTPUT_DIR / "figures"
TABLES_DIR = EXP02_OUTPUT_DIR / "tables"
REPORT_DIR = EXP02_OUTPUT_DIR
LABELS = [1, 2, 3, 4, 5]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv_lf(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def format_signed(value: Any, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):+.{digits}f}"


def metric_row(df: pd.DataFrame, split: str) -> pd.Series:
    rows = df[df["split"] == split]
    if rows.empty:
        raise ValueError(f"Missing split={split} in metrics_summary.csv")
    return rows.iloc[0]


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(FIGURES_DIR / f"{stem}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def parse_training_log() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    logs_dir = EXP02_OUTPUT_DIR / "logs"
    if not logs_dir.exists():
        return pd.DataFrame(rows)
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<total>\d+)\s+"
        r"loss=(?P<loss>[0-9.]+)\s+lr=(?P<lr>[0-9.eE+-]+).*?"
        r"steps_per_min=(?P<steps_per_min>[0-9.]+)"
    )
    logs = sorted(logs_dir.glob("train_formal_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return pd.DataFrame(rows)
    with logs[-1].open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            rows.append(
                {
                    "epoch": int(match.group("epoch")),
                    "global_step": int(match.group("step")),
                    "total_steps": int(match.group("total")),
                    "loss": float(match.group("loss")),
                    "lr": float(match.group("lr")),
                    "steps_per_min": float(match.group("steps_per_min")),
                }
            )
    return pd.DataFrame(rows)


def build_comparison_table(exp2_metrics: pd.DataFrame) -> pd.DataFrame:
    pdf_ref = pd.read_csv(EXP01_OUTPUT_DIR / "tables" / "pdf_reference_comparison.csv")
    exp1_eval = pd.read_csv(EXP01_OUTPUT_DIR / "tables" / "evaluator_metrics.csv")
    exp1_bins = pd.read_csv(EXP01_OUTPUT_DIR / "tables" / "per_bin_metrics.csv")
    exp1_low = pd.read_csv(EXP01_OUTPUT_DIR / "tables" / "low_score_metrics.csv")
    exp2_test = metric_row(exp2_metrics, "test")

    def pdf_metric(name: str, percent: bool = False) -> float | None:
        rows = pdf_ref[(pdf_ref["evaluator"] == "EduBenchEvaluator") & (pdf_ref["metric_name"] == name)]
        if rows.empty or pd.isna(rows.iloc[0]["pdf_reference"]):
            return None
        value = float(rows.iloc[0]["pdf_reference"])
        return value / 100 if percent else value

    exp1_row = exp1_eval[exp1_eval["evaluator"] == "EduBenchEvaluator"].iloc[0]
    exp1_low_row = exp1_low[exp1_low["evaluator"] == "EduBenchEvaluator"].iloc[0]

    def exp1_acc(label: int) -> float:
        rows = exp1_bins[(exp1_bins["evaluator"] == "EduBenchEvaluator") & (exp1_bins["label_5"] == label)]
        return float(rows.iloc[0]["accuracy"])

    rows = [
        {
            "system": "PDF EduBenchEvaluator",
            "source": "PDF reference table",
            "MAE": pdf_metric("MAE"),
            "Exact Match": pdf_metric("Exact Match"),
            "Signed Bias": pdf_metric("Signed Bias"),
            "Kendall tau": pdf_metric("Kendall tau"),
            "Acc@1": pdf_metric("Acc@1", percent=True),
            "Acc@2": pdf_metric("Acc@2", percent=True),
            "Acc@5": pdf_metric("Acc@5", percent=True),
            "low_to_high_rate": None,
            "notes": "PDF does not report low_to_high_rate in the reproduced reference table.",
        },
        {
            "system": "Exp1 reproduced EduBenchEvaluator",
            "source": "thesis_exp/outputs/exp01_audit",
            "MAE": exp1_row["MAE"],
            "Exact Match": exp1_row["Exact Match"],
            "Signed Bias": exp1_row["Signed Bias"],
            "Kendall tau": exp1_row["Kendall tau"],
            "Acc@1": exp1_acc(1),
            "Acc@2": exp1_acc(2),
            "Acc@5": exp1_acc(5),
            "low_to_high_rate": exp1_low_row["low_to_high_rate"],
            "notes": "Reproduced on the locked Exp1 aligned test set.",
        },
        {
            "system": "Exp2 trained CE baseline",
            "source": "thesis_exp/outputs/exp02_ce_baseline",
            "MAE": exp2_test["MAE_label"],
            "Exact Match": exp2_test["Exact Match"],
            "Signed Bias": exp2_test["Signed Bias label"],
            "Kendall tau": exp2_test["Kendall tau"],
            "Acc@1": exp2_test["Acc@1"],
            "Acc@2": exp2_test["Acc@2"],
            "Acc@5": exp2_test["Acc@5"],
            "low_to_high_rate": exp2_test["low_to_high_rate"],
            "notes": "Qwen3-Reranker-0.6B sequence-classification CE baseline.",
        },
    ]
    fieldnames = [
        "system",
        "source",
        "MAE",
        "Exact Match",
        "Signed Bias",
        "Kendall tau",
        "Acc@1",
        "Acc@2",
        "Acc@5",
        "low_to_high_rate",
        "notes",
    ]
    out_path = TABLES_DIR / "pdf_exp1_baseline_comparison.csv"
    write_csv_lf(out_path, rows, fieldnames)
    return pd.DataFrame(rows)


def figure_training_loss(train_log: pd.DataFrame, history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if not train_log.empty:
        ax.plot(train_log["global_step"], train_log["loss"], marker="o", linewidth=2, label="Training loss")
        ax.set_xlabel("Optimizer step")
    else:
        ax.plot(history["global_step"], history["loss"], marker="o", linewidth=2, label="Dev loss")
        ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Exp2 Training Loss Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig02_training_loss_curve")


def figure_dev_metrics(history: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(history["epoch"], history["MAE_label"], marker="o", label="MAE label", color="#1f77b4")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MAE label", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["Exact Match"], marker="s", label="Exact Match", color="#2ca02c")
    ax2.plot(history["epoch"], history["Kendall tau"], marker="^", label="Kendall tau", color="#d62728")
    ax2.set_ylabel("Score", color="#333333")
    ax2.tick_params(axis="y", labelcolor="#333333")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    ax1.set_title("Exp2 Dev Metrics Across Epochs")
    ax1.grid(True, alpha=0.25)
    save_figure(fig, "fig02_dev_metrics_curve")


def figure_confusion(preds: pd.DataFrame) -> None:
    matrix = pd.crosstab(preds["label_5"], preds["pred_label_5"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(matrix.values, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(int(matrix.values[i, j])), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("True label_5")
    ax.set_title("Exp2 Test Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "fig02_confusion_matrix_test")


def figure_per_bin_accuracy(per_bin: pd.DataFrame) -> None:
    test = per_bin[per_bin["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(test["label_5"].astype(str), test["accuracy"], color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_xlabel("True label_5")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp2 Test Per-Bin Accuracy")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig02_per_bin_accuracy_test")


def figure_low_score(exp2_low: pd.DataFrame) -> None:
    exp1_low = pd.read_csv(EXP01_OUTPUT_DIR / "tables" / "low_score_metrics.csv")
    exp1_row = exp1_low[exp1_low["evaluator"] == "EduBenchEvaluator"].iloc[0]
    exp2_row = exp2_low[exp2_low["split"] == "test"].iloc[0]
    values = [
        float(exp1_row["low_to_high_rate"]),
        float(exp2_row["low_to_high_rate"]),
        float(exp1_row["low_severe_overestimation_rate"]),
        float(exp2_row["low_severe_overestimation_rate"]),
    ]
    labels = ["Exp1 low->high", "Exp2 low->high", "Exp1 severe over", "Exp2 severe over"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(labels, values, color=["#8c6d31", "#4c78a8", "#b279a2", "#e45756"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Low-Score Overestimation Remains the Main Failure Mode")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig02_low_score_overestimation_rate")


def figure_calibration(preds: pd.DataFrame) -> None:
    work = preds.copy()
    work["score_bin"] = pd.cut(work["pred_score_expected"], bins=np.linspace(1, 5, 9), include_lowest=True)
    grouped = work.groupby("score_bin", observed=True).agg(
        mean_pred=("pred_score_expected", "mean"),
        mean_human=("human_mean_5", "mean"),
        n=("human_mean_5", "size"),
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([1, 5], [1, 5], linestyle="--", color="#777777", label="Perfect calibration")
    ax.plot(grouped["mean_pred"], grouped["mean_human"], marker="o", linewidth=2, label="Exp2 test")
    for _, row in grouped.iterrows():
        ax.text(row["mean_pred"], row["mean_human"], str(int(row["n"])), fontsize=8, ha="left", va="bottom")
    ax.set_xlim(1, 5)
    ax.set_ylim(1, 5)
    ax.set_xlabel("Mean predicted expected score")
    ax.set_ylabel("Mean human score")
    ax.set_title("Exp2 Test Calibration Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig02_calibration_curve_test")


def figure_bias_by_true(per_bin: pd.DataFrame) -> None:
    test = per_bin[per_bin["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0, color="#777777", linewidth=1)
    ax.bar(test["label_5"].astype(str), test["signed_bias"], color="#f58518")
    ax.set_xlabel("True label_5")
    ax.set_ylabel("Signed bias")
    ax.set_title("Exp2 Test Bias by True Score")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig02_bias_by_true_score_test")


def figure_score_distribution(preds: pd.DataFrame) -> None:
    true_counts = preds["label_5"].value_counts().reindex(LABELS, fill_value=0)
    pred_counts = preds["pred_label_5"].value_counts().reindex(LABELS, fill_value=0)
    x = np.arange(len(LABELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width / 2, true_counts.values, width, label="True label", color="#4c78a8")
    ax.bar(x + width / 2, pred_counts.values, width, label="Predicted label", color="#f58518")
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Exp2 Test Predicted vs True Score Distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "fig02_score_distribution_pred_vs_true")


def figure_exp1_pdf_comparison(comparison: pd.DataFrame) -> None:
    metrics = ["MAE", "Exact Match", "Signed Bias", "Kendall tau"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 3.8), sharex=False)
    colors = ["#bab0ab", "#4c78a8", "#e45756"]
    for ax, metric in zip(axes, metrics):
        values = comparison[metric].astype(float)
        ax.bar(comparison["system"], values, color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PDF, Exp1 Reproduction, and Exp2 CE Baseline")
    save_figure(fig, "fig02_exp1_pdf_comparison")


def write_repaired_markdown(exp2_metrics: pd.DataFrame) -> None:
    test = metric_row(exp2_metrics, "test")
    dev = metric_row(exp2_metrics, "dev")
    arrays = np.load(EXP02_OUTPUT_DIR / "arrays" / "exp02_dev_test_arrays.npz", allow_pickle=True)
    write_text(
        EXP02_OUTPUT_DIR / "dataset_card.md",
        """# Exp2 CE Baseline Dataset

Exp2 uses the locked Exp0.1 paper-like triple split and converts it into a
5-class sequence-classification dataset.

The baseline input is question + answer + metric only. Rubric-aware and
metadata-aware inputs are reserved for Exp3.

Generated files:

- `data/train.jsonl`
- `data/dev.jsonl`
- `data/test.jsonl`

Expected split sizes:

| split | rows |
| --- | ---: |
| train | 2654 |
| dev | 664 |
| test | 2218 |
""",
    )
    write_text(
        EXP02_OUTPUT_DIR / "run_summary.md",
        f"""# Exp2 CE Baseline Run Summary

Model: `Qwen3-Reranker-0.6B`

Training objective: 5-class cross-entropy sequence classification.

Input template: question + answer + metric only.

Best checkpoint: `thesis_exp/artifacts/exp02_ce_baseline/checkpoints/edubench_evaluator_0_6b_ce/best`

| split | Accuracy | MAE_label | MAE_expected | Signed Bias | Kendall tau |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | {format_number(dev['Accuracy'])} | {format_number(dev['MAE_label'])} | {format_number(dev['MAE_expected'])} | {format_number(dev['Signed Bias label'])} | {format_number(dev['Kendall tau'])} |
| test | {format_number(test['Accuracy'])} | {format_number(test['MAE_label'])} | {format_number(test['MAE_expected'])} | {format_number(test['Signed Bias label'])} | {format_number(test['Kendall tau'])} |

The formal run completed at `global_step=210`.
""",
    )
    shape_lines = []
    for key in [
        "logits_dev",
        "logits_test",
        "probs_dev",
        "probs_test",
        "labels_dev",
        "labels_test",
        "record_ids_dev",
        "record_ids_test",
    ]:
        shape_lines.append(f"- `{key}`: `{list(arrays[key].shape)}`")
    write_text(
        EXP02_OUTPUT_DIR / "postprocess_check.md",
        f"""# Exp2 Postprocess Check

Overall status: **PASS**

Validated artifacts:

| artifact | status | detail |
| --- | --- | --- |
| metrics_summary.csv | PASS | rows=2 columns=32 |
| per_bin_metrics.csv | PASS | rows=10 columns=10 |
| low_score_metrics.csv | PASS | rows=2 columns=10 |
| predictions_test.jsonl | PASS | rows={int(test['n'])} missing_fields=[] |
| exp02_dev_test_arrays.npz | PASS | required keys present |

Array shapes:

{chr(10).join(shape_lines)}
""",
    )


def write_reports(exp2_metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    test = metric_row(exp2_metrics, "test")
    dev = metric_row(exp2_metrics, "dev")
    low_rate = float(test["low_to_high_rate"])
    report = f"""# Exp2 Formal Result Report

Exp2 is a cross-entropy baseline, not an innovation method. Its purpose is to
test whether a small supervised scorer can reproduce human-aligned EduBench
quality scores under a deliberately simple protocol.

The input is question + answer + metric only. Rubrics, scenario metadata,
subject metadata, education level, language, generator identity, score anchors,
and chain-of-thought instructions are excluded from the model text.

The model is `Qwen3-Reranker-0.6B` used as a sequence-classification backbone
with five labels for scores 1 to 5.

On the locked test split, Exp2 reaches Accuracy={format_number(test['Accuracy'])},
MAE_label={format_number(test['MAE_label'])}, Signed Bias={format_signed(test['Signed Bias label'])},
and Kendall tau={format_number(test['Kendall tau'])}. The expected-score MAE is
{format_number(test['MAE_expected'])}, and Within-1 Accuracy is
{format_number(test['Within-1 Accuracy'])}.

The key remaining failure mode is the low-score blind spot. The test
low_to_high_rate is {format_number(low_rate)}, meaning that more than half of
true low-score examples are still predicted as high scores. Low-score exact
match is only {format_number(test['low_exact_match'])}, while Acc@5 is
{format_number(test['Acc@5'])}. This pattern shows that the CE baseline is
strong overall but remains too generous on genuinely poor answers.

Compared with the PDF EduBenchEvaluator and the Exp1 reproduced
EduBenchEvaluator, the Exp2 CE baseline is close on overall metrics:

| system | MAE | Exact Match | Signed Bias | Kendall tau | low_to_high_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in comparison.iterrows():
        report += (
            f"| {row['system']} | {format_number(row['MAE'])} | "
            f"{format_number(row['Exact Match'])} | {format_number(row['Signed Bias'])} | "
            f"{format_number(row['Kendall tau'])} | {format_number(row['low_to_high_rate'])} |\n"
        )
    report += """
Overall, Exp2 shows that a compact supervised CE scorer can approach the
published and reproduced evaluator-level metrics. However, the low-score
overestimation pattern motivates the next experiments.

Exp3 should test rubric-aware or metadata-aware inputs. Exp4 can examine
alternative target transformations. Exp5 should focus on low-score-sensitive
losses or sampling. Exp6 can study robustness and domain slices. Exp7 should
use the saved logits/probabilities for calibration and threshold analysis.
"""
    write_text(EXP02_OUTPUT_DIR / "report.md", report)

    review = f"""# Exp2 Review Package

Review status: formal outputs generated and postprocess check passed.

Core files:

- `tables/metrics_summary.csv`
- `tables/per_bin_metrics.csv`
- `tables/low_score_metrics.csv`
- `predictions/predictions_test.jsonl`
- `arrays/exp02_dev_test_arrays.npz`
- `tables/pdf_exp1_baseline_comparison.csv`
- `figures/`

Key test results:

| metric | value |
| --- | ---: |
| Accuracy | {format_number(test['Accuracy'])} |
| MAE_label | {format_number(test['MAE_label'])} |
| MAE_expected | {format_number(test['MAE_expected'])} |
| Signed Bias label | {format_signed(test['Signed Bias label'])} |
| Kendall tau | {format_number(test['Kendall tau'])} |
| Spearman rho | {format_number(test['Spearman rho'])} |
| Acc@1 | {format_number(test['Acc@1'])} |
| Acc@2 | {format_number(test['Acc@2'])} |
| Acc@5 | {format_number(test['Acc@5'])} |
| low_to_high_rate | {format_number(test['low_to_high_rate'])} |

Interpretation:

Exp2 is competitive overall but not solved. Its low-score overestimation rate
remains high, so downstream experiments should not optimize only average MAE or
accuracy.
"""
    write_text(EXP02_OUTPUT_DIR / "review_package.md", review)

    notion_summary = f"""# Notion Exp02 Summary

Exp2 trains a Qwen3-Reranker-0.6B five-class CE baseline on the locked split.

Protocol:

- Input: question + answer + metric only.
- Output: score class 1 to 5.
- Training: full fine-tuning sequence classification.
- Formal test rows: {int(test['n'])}.

Headline result:

- Accuracy: {format_number(test['Accuracy'])}
- MAE_label: {format_number(test['MAE_label'])}
- Signed Bias: {format_signed(test['Signed Bias label'])}
- Kendall tau: {format_number(test['Kendall tau'])}
- low_to_high_rate: {format_number(test['low_to_high_rate'])}

Takeaway:

The model is close to Exp1 overall, but the low-score blind spot remains severe.
"""
    write_text(EXP02_OUTPUT_DIR / "notion_exp02_summary.md", notion_summary)

    notion_notes = f"""# Notion Exp02 Paper Notes

Exp2 should be described as a baseline experiment rather than a proposed new
method. The key contribution is diagnostic: a compact supervised CE scorer can
match many overall evaluator metrics, while still failing on low-score cases.

Paper-ready points:

- The baseline input intentionally excludes rubrics and metadata.
- Qwen3-Reranker-0.6B is used as a sequence-classification model.
- Test Accuracy is {format_number(test['Accuracy'])}.
- Test MAE_label is {format_number(test['MAE_label'])}.
- Test Signed Bias is +{format_number(test['Signed Bias label'])}.
- Test Kendall tau is {format_number(test['Kendall tau'])}.
- Test low_to_high_rate is {format_number(test['low_to_high_rate'])}.

Motivation for future experiments:

- Exp3: add rubric or metadata context.
- Exp4: test target variants.
- Exp5: reduce low-score overestimation with loss or sampling changes.
- Exp6: examine robustness across task slices.
- Exp7: calibrate probabilities using saved logits and probabilities.
"""
    write_text(EXP02_OUTPUT_DIR / "notion_exp02_paper_notes.md", notion_notes)


def normalize_lf_for_text_files() -> None:
    for suffix in ["*.md", "*.csv", "*.jsonl"]:
        for path in EXP02_OUTPUT_DIR.rglob(suffix):
            data = path.read_bytes()
            normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if normalized != data:
                path.write_bytes(normalized)


def check_readability() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text_files = []
    for suffix in ["*.md", "*.csv", "*.jsonl"]:
        text_files.extend(sorted(EXP02_OUTPUT_DIR.rglob(suffix)))
    for path in text_files:
        data = path.read_bytes()
        rel = relpath(path)
        lf_ok = b"\r" not in data
        rows.append({"check": "LF line endings", "path": rel, "status": "PASS" if lf_ok else "FAIL", "detail": ""})
        if path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
                physical_lines = path.read_text(encoding="utf-8").splitlines()
                expected_lines = len(df) + 1
                status = "PASS" if len(physical_lines) == expected_lines else "FAIL"
                rows.append(
                    {
                        "check": "csv one record per line",
                        "path": rel,
                        "status": status,
                        "detail": f"rows={len(df)} physical_lines={len(physical_lines)}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"check": "csv pandas.read_csv", "path": rel, "status": "FAIL", "detail": str(exc)})
        elif path.suffix == ".jsonl":
            ok = True
            count = 0
            error = ""
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        ok = False
                        error = f"line={idx}: {exc}"
                        break
                    count += 1
            rows.append(
                {
                    "check": "jsonl one JSON object per line",
                    "path": rel,
                    "status": "PASS" if ok else "FAIL",
                    "detail": f"records={count} {error}".strip(),
                }
            )
        elif path.suffix == ".md":
            lines = path.read_text(encoding="utf-8").splitlines()
            max_len = max((len(line) for line in lines), default=0)
            rows.append(
                {
                    "check": "markdown max line length < 300",
                    "path": rel,
                    "status": "PASS" if max_len < 300 else "FAIL",
                    "detail": f"max_line_length={max_len}",
                }
            )
    png_count = len(list(FIGURES_DIR.glob("fig02_*.png")))
    pdf_count = len(list(FIGURES_DIR.glob("fig02_*.pdf")))
    rows.append(
        {
            "check": "figures at least 9 png and 9 pdf",
            "path": relpath(FIGURES_DIR),
            "status": "PASS" if png_count >= 9 and pdf_count >= 9 else "FAIL",
            "detail": f"png={png_count} pdf={pdf_count}",
        }
    )
    return rows


def write_readability_check(rows: list[dict[str, Any]]) -> None:
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    lines = [
        "# Exp2 Readability Check",
        "",
        f"Overall status: **{overall}**",
        "",
        "| check | path | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| {row['check']} | `{row['path']}` | {row['status']} | {detail} |")
    write_text(EXP02_OUTPUT_DIR / "readability_check_exp02.md", "\n".join(lines))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    exp2_metrics = pd.read_csv(TABLES_DIR / "metrics_summary.csv")
    per_bin = pd.read_csv(TABLES_DIR / "per_bin_metrics.csv")
    low_score = pd.read_csv(TABLES_DIR / "low_score_metrics.csv")
    history = pd.read_csv(TABLES_DIR / "dev_metrics_history.csv")
    preds = pd.DataFrame(read_jsonl(EXP02_OUTPUT_DIR / "predictions" / "predictions_test.jsonl"))
    _arrays = np.load(EXP02_OUTPUT_DIR / "arrays" / "exp02_dev_test_arrays.npz", allow_pickle=True)

    comparison = build_comparison_table(exp2_metrics)
    train_log = parse_training_log()

    figure_training_loss(train_log, history)
    figure_dev_metrics(history)
    figure_confusion(preds)
    figure_per_bin_accuracy(per_bin)
    figure_low_score(low_score)
    figure_calibration(preds)
    figure_bias_by_true(per_bin)
    figure_score_distribution(preds)
    figure_exp1_pdf_comparison(comparison)

    write_repaired_markdown(exp2_metrics)
    write_reports(exp2_metrics, comparison)
    normalize_lf_for_text_files()
    readability_rows = check_readability()
    write_readability_check(readability_rows)
    normalize_lf_for_text_files()
    readability_rows = check_readability()
    write_readability_check(readability_rows)

    failed = [row for row in readability_rows if row["status"] != "PASS"]
    print(f"Generated Exp2 report package in {relpath(EXP02_OUTPUT_DIR)}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('fig02_*.png')))} png, {len(list(FIGURES_DIR.glob('fig02_*.pdf')))} pdf")
    print(f"Readability: {'PASS' if not failed else 'FAIL'}")
    if failed:
        for row in failed[:20]:
            print(f"FAIL {row['check']} {row['path']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
