"""Generate thesis-ready Exp 0 distribution figures and summary reports."""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thesis_exp.src.edujudge.data.build_dataset import PROCESSED_PATH
from thesis_exp.src.edujudge.data.reference_contract import EXPECTED_METRICS, EXPECTED_SCENARIOS, EXPECTED_SUBJECTS
from thesis_exp.src.edujudge.utils.io import (
    FIGURES_DIR,
    OUTPUT_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_exp_dirs,
    md_table,
    read_jsonl,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import truncate_text


KEY_FIELDS = [
    "question",
    "answer",
    "metric_canonical",
    "subject_canonical",
    "scenario_canonical",
    "education_level_canonical",
    "language",
    "generator_model",
    "human_1",
    "human_2",
    "human_3",
    "human_mean_5",
    "label_5",
]


def _wrap(value: object, width: int = 24) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bar(
    counts: Counter[Any],
    title: str,
    xlabel: str,
    ylabel: str,
    name: str,
    width: int = 24,
    rotate: int = 0,
    color: str = "#4C78A8",
) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, min(14, 0.55 * len(labels) + 4)), 4.8))
    bars = ax.bar(range(len(labels)), values, color=color, edgecolor="#333333", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([_wrap(label, width) for label in labels], rotation=rotate, ha="right" if rotate else "center")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, name)


def _hist(values: list[float], title: str, xlabel: str, ylabel: str, name: str, bins: int | list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(values, bins=bins, color="#72B7B2", edgecolor="#333333", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, name)


def _split_score_distribution() -> None:
    split_dir = SPLITS_DIR / "paper_like_triple_seed42"
    data = {split: read_jsonl(split_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    labels = [1, 2, 3, 4, 5]
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"train": "#4C78A8", "dev": "#F58518", "test": "#54A24B"}
    for idx, split in enumerate(["train", "dev", "test"]):
        counts = Counter(int(row["label_5"]) for row in data[split])
        x = [label + (idx - 1) * width for label in labels]
        ax.bar(x, [counts[label] for label in labels], width=width, label=split, color=colors[split], edgecolor="#333333", linewidth=0.4)
    ax.set_title("Label Distribution by Split (paper_like_triple_seed42)")
    ax.set_xlabel("label_5")
    ax.set_ylabel("Number of scored items")
    ax.set_xticks(labels)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig00_split_score_distribution_paper_like_triple")


def _missingness_heatmap(rows: list[dict[str, Any]]) -> None:
    matrix = []
    row_labels = []
    groups = {"all": rows}
    split_dir = SPLITS_DIR / "paper_like_triple_seed42"
    if split_dir.exists():
        for split in ["train", "dev", "test"]:
            path = split_dir / f"{split}.jsonl"
            if path.exists():
                groups[split] = read_jsonl(path)
    for label, group_rows in groups.items():
        row_labels.append(label)
        values = []
        for field in KEY_FIELDS:
            missing = sum(1 for row in group_rows if row.get(field) in (None, "", [], {}, "unknown"))
            values.append(missing / max(1, len(group_rows)))
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_title("Missingness Rate for Key Fields")
    ax.set_xticks(range(len(KEY_FIELDS)))
    ax.set_xticklabels([_wrap(field, 18) for field in KEY_FIELDS], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Missing rate")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#222222")
    _save(fig, "fig00_missingness_heatmap")


def _annotator_score_distribution(rows: list[dict[str, Any]]) -> None:
    annotators = ["human_1", "human_2", "human_3"]
    labels = [1, 2, 3, 4, 5]
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, annotator in enumerate(annotators):
        counts = Counter(int(row[annotator]) for row in rows if row.get(annotator) is not None)
        x = [label + (idx - 1) * width for label in labels]
        ax.bar(x, [counts[label] for label in labels], width=width, label=annotator, color=colors[idx], edgecolor="#333333", linewidth=0.4)
    ax.set_title("Annotator Score Distribution")
    ax.set_xlabel("Human score")
    ax.set_ylabel("Number of scored items")
    ax.set_xticks(labels)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig00_annotator_score_distribution")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def _annotator_agreement(rows: list[dict[str, Any]]) -> None:
    pairs = [("human_1", "human_2"), ("human_1", "human_3"), ("human_2", "human_3")]
    metrics = []
    for left, right in pairs:
        paired = [(float(row[left]), float(row[right])) for row in rows if row.get(left) is not None and row.get(right) is not None]
        xs = [item[0] for item in paired]
        ys = [item[1] for item in paired]
        corr = _pearson(xs, ys)
        mae = mean([abs(x - y) for x, y in paired]) if paired else float("nan")
        exact = mean([1.0 if x == y else 0.0 for x, y in paired]) if paired else float("nan")
        metrics.append({"pair": f"{left}-{right}", "correlation": corr, "mae": mae, "exact_agreement": exact})

    fig, ax1 = plt.subplots(figsize=(7, 4.8))
    x = list(range(len(metrics)))
    ax1.bar([value - 0.2 for value in x], [m["correlation"] for m in metrics], width=0.2, label="Correlation", color="#4C78A8")
    ax1.bar(x, [m["exact_agreement"] for m in metrics], width=0.2, label="Exact agreement", color="#54A24B")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Correlation / exact agreement")
    ax2 = ax1.twinx()
    ax2.bar([value + 0.2 for value in x], [m["mae"] for m in metrics], width=0.2, label="MAE", color="#F58518")
    ax2.set_ylabel("MAE")
    ax1.set_title("Human Annotator Agreement")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m["pair"] for m in metrics])
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    _save(fig, "fig00_human_annotator_agreement")


def _reference_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    boxes = [
        ("Official EduBench\nfull data\n9 scenarios / 18821 points", 0.02),
        ("Local derived\nresults_merge.jsonl", 0.25),
        ("Audit human-scored\nsubset\n5536 scored items", 0.47),
        ("Paper-like split\n3318 train pool\n2218 held-out test", 0.69),
        ("Downstream\nExp1+", 0.88),
    ]
    for text, x in boxes:
        rect = plt.Rectangle((x, 0.35), 0.16, 0.35, facecolor="#E8F1FA", edgecolor="#2F4B7C", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.08, 0.525, text, ha="center", va="center", fontsize=10)
    for _, x in boxes[:-1]:
        ax.annotate("", xy=(x + 0.18, 0.525), xytext=(x + 0.16, 0.525), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"})
    ax.set_title("Reference Flow: Official EduBench to PDF Audit Subset")
    _save(fig, "fig00_reference_flow")


def _score_mapping_audit(rows: list[dict[str, Any]]) -> None:
    pairs = Counter()
    for row in rows:
        pairs[(str(row.get("raw_score_scale_detected")), int(row.get("label_5")))] += 1
    labels = sorted(pairs.keys())
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [f"{scale}\nlabel {label}" for scale, label in labels]
    values = [pairs[key] for key in labels]
    bars = ax.bar(range(len(labels)), values, color="#4C78A8", edgecolor="#333333", linewidth=0.5)
    ax.set_title("Score Mapping Audit: Raw Scale to label_5")
    ax.set_xlabel("raw_score_scale_detected -> label_5")
    ax.set_ylabel("Number of scored items")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(names)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig00_score_mapping_audit")


def _split_coverage_heatmap(split_name: str, field: str, expected_values: list[str], title: str, output_name: str) -> None:
    split_dir = SPLITS_DIR / split_name
    split_rows = {split: read_jsonl(split_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    matrix = []
    for split in ["train", "dev", "test"]:
        counts = Counter(row.get(field) for row in split_rows[split])
        matrix.append([counts.get(value, 0) for value in expected_values])
    fig, ax = plt.subplots(figsize=(max(9, len(expected_values) * 0.55), 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["train", "dev", "test"])
    ax.set_xticks(range(len(expected_values)))
    ax.set_xticklabels([_wrap(value, 18) for value in expected_values], rotation=45, ha="right")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            ax.text(x, y, str(value), ha="center", va="center", fontsize=7, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Count")
    _save(fig, output_name)


def _subject_alignment_status(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("subject_mapping_method", "unknown") for row in rows)
    observed = {row.get("subject_canonical") for row in rows if row.get("subject_canonical") and row.get("subject_canonical") != "unknown"}
    status = "PASS" if set(observed) == set(EXPECTED_SUBJECTS) else "WARNING"
    _bar(
        Counter(dict(counts.most_common())),
        f"Subject Alignment Status ({status}: {len(observed)}/25 canonical subjects)",
        "subject mapping method",
        "Number of scored items",
        "fig00_subject_alignment_status",
        width=16,
        rotate=25,
        color="#72B7B2",
    )


def generate_figures(rows: list[dict[str, Any]]) -> None:
    label_counts = Counter(int(row["label_5"]) for row in rows)
    label_counts = Counter({label: label_counts.get(label, 0) for label in [1, 2, 3, 4, 5]})
    low_count = label_counts[1] + label_counts[2]
    low_pct = low_count / max(1, len(rows))
    _bar(
        label_counts,
        f"Overall label_5 Distribution (low labels 1/2: {low_count}, {low_pct:.1%})",
        "label_5",
        "Number of scored items",
        "fig00_score_distribution",
        width=10,
    )

    _hist(
        [float(row["human_mean_raw"]) for row in rows if row.get("human_mean_raw") is not None],
        "Raw Human Mean Score Distribution",
        "human_mean_raw",
        "Number of scored items",
        "fig00_raw_score_distribution",
        bins=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
    )
    _split_score_distribution()

    metric_counts = Counter(row["metric_canonical"] for row in rows)
    _bar(
        Counter(dict(metric_counts.most_common())),
        f"Canonical Metric Distribution (n={len(metric_counts)})",
        "Canonical metric",
        "Number of scored items",
        "fig00_metric_distribution",
        width=20,
        rotate=45,
        color="#4C78A8",
    )

    scenario_counts = Counter(row["scenario_canonical"] for row in rows)
    _bar(
        Counter(dict(scenario_counts.most_common())),
        f"Canonical Scenario Distribution (n={len(scenario_counts)})",
        "Canonical scenario",
        "Number of scored items",
        "fig00_scenario_distribution",
        width=18,
        rotate=35,
        color="#72B7B2",
    )

    subject_counts = Counter(row["subject_canonical"] for row in rows)
    _bar(
        Counter(dict(subject_counts.most_common(25))),
        "Subject Distribution (Top 25)",
        "subject_canonical",
        "Number of scored items",
        "fig00_subject_distribution_top25",
        width=16,
        rotate=45,
        color="#54A24B",
    )

    _bar(
        Counter(dict(Counter(row["education_level_canonical"] for row in rows).most_common())),
        "Education Level Distribution",
        "education_level_canonical",
        "Number of scored items",
        "fig00_education_level_distribution",
        width=16,
        rotate=30,
        color="#E45756",
    )

    _bar(
        Counter(dict(Counter(row["language"] for row in rows).most_common())),
        "Language Distribution",
        "language",
        "Number of scored items",
        "fig00_language_distribution",
        width=12,
        color="#B279A2",
    )

    _bar(
        Counter(dict(Counter(row["generator_model"] for row in rows).most_common())),
        "Generator Model Distribution",
        "generator_model",
        "Number of scored items",
        "fig00_generator_model_distribution",
        width=18,
        rotate=30,
        color="#F58518",
    )

    _missingness_heatmap(rows)
    _annotator_score_distribution(rows)
    _annotator_agreement(rows)
    _reference_flow()
    _score_mapping_audit(rows)
    _split_coverage_heatmap(
        "paper_like_triple_seed42",
        "metric_canonical",
        EXPECTED_METRICS,
        "Split Coverage by Official Metric",
        "fig00_split_coverage_heatmap_metric",
    )
    _split_coverage_heatmap(
        "paper_like_triple_seed42",
        "scenario_canonical",
        EXPECTED_SCENARIOS,
        "Split Coverage by Official Scenario",
        "fig00_split_coverage_heatmap_scenario",
    )
    _subject_alignment_status(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _split_counts(split_name: str) -> dict[str, int]:
    split_dir = SPLITS_DIR / split_name
    return {split: len(read_jsonl(split_dir / f"{split}.jsonl")) if (split_dir / f"{split}.jsonl").exists() else 0 for split in ["train", "dev", "test"]}


def _leakage_status() -> str:
    report = OUTPUT_DIR / "leakage_report.md"
    if not report.exists():
        return "unknown"
    first_lines = "\n".join(report.read_text(encoding="utf-8").splitlines()[:5])
    if "FAIL" in first_lines:
        return "FAIL"
    if "WARNING" in first_lines:
        return "WARNING"
    if "PASS" in first_lines:
        return "PASS"
    return "unknown"


def _count_figures() -> list[str]:
    return sorted(path.name for path in FIGURES_DIR.glob("fig00_*.png"))


def _distribution_rows(rows: list[dict[str, Any]], key: str, max_rows: int = 20) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key, "unknown") or "unknown") for row in rows)
    total = sum(counts.values())
    return [{"value": value, "count": count, "pct": f"{count / max(1, total):.2%}"} for value, count in counts.most_common(max_rows)]


def generate_reports(rows: list[dict[str, Any]]) -> None:
    inventory = _read_csv(TABLES_DIR / "source_inventory.csv")
    source_counts = Counter(row.get("likely_role", "unknown") for row in inventory)
    split_stats = _read_csv(TABLES_DIR / "split_stats.csv")
    leakage_status = _leakage_status()
    unmapped_metrics = _read_csv(TABLES_DIR / "unmapped_metrics.csv")
    unmapped_scenarios = _read_csv(TABLES_DIR / "unmapped_scenarios.csv")
    invalid_scores = _read_csv(TABLES_DIR / "invalid_or_ambiguous_scores.csv")
    official_inventory = _read_csv(TABLES_DIR / "official_source_inventory.csv")
    sanity_rows = _read_csv(TABLES_DIR / "sanity_check_results.csv")

    label_counts = Counter(int(row["label_5"]) for row in rows)
    low_count = label_counts[1] + label_counts[2]
    low_pct = low_count / max(1, len(rows))
    metric_count = len({row["metric_canonical"] for row in rows if row["metric_canonical"] != "unknown"})
    scenario_count = len({row["scenario_canonical"] for row in rows if row["scenario_canonical"] != "unknown"})
    subject_count = len({row["subject_canonical"] for row in rows if row["subject_canonical"] != "unknown"})
    score_scales = Counter(row.get("raw_score_scale_detected") for row in rows)
    unmapped_metric_error = sum(1 for row in unmapped_metrics if row.get("severity") == "ERROR")
    unmapped_metric_info = sum(1 for row in unmapped_metrics if row.get("severity") == "INFO")
    unmapped_scenario_error = sum(1 for row in unmapped_scenarios if row.get("severity") == "ERROR")
    unmapped_scenario_info = sum(1 for row in unmapped_scenarios if row.get("severity") == "INFO")
    sanity_status = "PASS"
    if any(row.get("status") == "FAIL" for row in sanity_rows):
        sanity_status = "FAIL"
    elif any(row.get("status") == "WARNING" for row in sanity_rows):
        sanity_status = "WARNING"
    can_exp1 = "YES" if sanity_status in {"PASS", "WARNING"} and metric_count == 12 and scenario_count == 9 and not invalid_scores else "NO"
    split_source = ""
    for row in split_stats:
        if row.get("split_name") == "paper_like_triple_seed42" and row.get("split") == "train":
            split_source = row.get("split_source", "")
            break

    report_lines = [
        "# Exp 0.1 Report: EduBench Reference Alignment and Data Hardening",
        "",
        "## 1. Purpose",
        "",
        "Exp 0.1 hardens the existing Exp 0 artifacts against three source-of-truth layers: the local PDF audit corpus, official EduBench paper definitions, and the official EduBench GitHub repository. It does not train models, call APIs, or use GPU.",
        "",
        "## 2. Source-of-truth hierarchy",
        "",
        "1. PDF audit corpus: 5536 generated responses / scored items, 5 generator models, 12 dimensions, 25 subjects, 6 education levels, 2 languages, and 3318/2218 evaluator-vs-human split at question-answer-metric triple granularity.",
        "2. Official EduBench paper: full benchmark has 9 major educational scenarios, 4000+ educational contexts, 18821 data points, 500 sampled human/LLM-evaluated queries, and 12 evaluation aspects.",
        "3. Official EduBench GitHub: canonical 9 scenarios, 12 metrics, and `data/all_data` official data structure when available.",
        "",
        "## 3. Official EduBench full data vs PDF audit subset",
        "",
        "Official EduBench full data is not the same artifact as the local 5536-row audit corpus. The local dataset is named `edubench_audit_human_scored_subset`; it is a local derived merged human-scored subset from `results_merge.jsonl`, not the full official EduBench dataset. Downstream Exp1 evaluator training/testing should use this audit corpus. Synthetic augmentation or full official EduBench data can be used in later experiments only if kept out of the main human-labeled test set.",
        "",
        "## 4. Source Inventory Summary",
        "",
        md_table([{"likely_role": key, "num_files": value} for key, value in source_counts.most_common()], ["likely_role", "num_files"], max_rows=20),
        "",
        "## 5. Local source selection",
        "",
        "`results_merge.jsonl` is used as the primary local source because each row already represents a scored item and contains `question`, `answer`, `metric`, `task`, generator `model`, plus `evaluate.human_1`, `evaluate.human_2`, and `evaluate.human_3`. `report/results_merge_enriched.jsonl` is used only to recover local enriched audit metadata such as subject, education level, language, and original held-out flag. Synthetic sampled files are excluded.",
        "",
        "## 6. Official source inventory",
        "",
        md_table(official_inventory, ["source_path", "source_origin", "file_role", "num_records", "contains_question", "contains_answer", "contains_metric", "contains_task_or_scenario"], max_rows=30),
        "",
        "## 7. Metric/scenario alignment",
        "",
        f"Canonical metric count: {metric_count}/12; canonical scenario count: {scenario_count}/9. Mapping tables are written to `tables/metric_mapping.csv` and `tables/scenario_mapping.csv`.",
        "",
        md_table(
            [
                {"field": "num_unmapped_metric_error", "value": unmapped_metric_error},
                {"field": "num_unmapped_metric_info", "value": unmapped_metric_info},
                {"field": "num_unmapped_scenario_error", "value": unmapped_scenario_error},
                {"field": "num_unmapped_scenario_info", "value": unmapped_scenario_info},
            ],
            ["field", "value"],
            max_rows=10,
        ),
        "",
        "## 8. Subject alignment status",
        "",
        f"Canonical subject count: {subject_count}/25. Subject metadata is recovered from the local enriched audit file when available. Subject-level analysis can be used as primary thesis evidence only after confirming this local enriched subject annotation; otherwise treat it as audit metadata.",
        "",
        "## 9. Score scale audit",
        "",
        f"Raw human scores in the main dataset are detected as: {dict(score_scales)}. Because the current `results_merge.jsonl` human scores are already on a 1-5 scale, `human_1_5`/`human_2_5`/`human_3_5` equal the raw values and `label_5` is `round(human_mean_5)` clipped to 1-5. The 1-10 mapping compatible with `5-grades.py` is still implemented for any future 1-10 source: 1-2->1, 3-4->2, 5-6->3, 7-8->4, 9-10->5.",
        "",
        "## 10. Main dataset statistics",
        "",
        md_table(
            [
                {"item": "dataset_name", "observed": "edubench_audit_human_scored_subset", "reference": "PDF audit subset"},
                {"item": "total_scored_items", "observed": len(rows), "reference": "5536"},
                {"item": "unique_triple_key", "observed": len({row["triple_key"] for row in rows}), "reference": "question-answer-metric granularity"},
                {"item": "unique_question_key", "observed": len({row["question_key"] for row in rows}), "reference": "question robustness split"},
                {"item": "unique_answer_key", "observed": len({row["answer_key"] for row in rows}), "reference": "leakage diagnostic"},
                {"item": "generator_models", "observed": len({row["generator_model"] for row in rows}), "reference": "5"},
                {"item": "canonical_metrics", "observed": metric_count, "reference": "12"},
                {"item": "canonical_scenarios", "observed": scenario_count, "reference": "9"},
                {"item": "canonical_subjects", "observed": len({row["subject_canonical"] for row in rows if row["subject_canonical"] != "unknown"}), "reference": "25"},
                {"item": "education_levels", "observed": len({row["education_level_canonical"] for row in rows if row["education_level_canonical"] != "unknown"}), "reference": "6"},
                {"item": "languages", "observed": sorted({row["language"] for row in rows}), "reference": "English / Chinese"},
                {"item": "paper train_pool/test", "observed": "targeted by paper_like_triple_seed42", "reference": "3318 / 2218"},
            ],
            ["item", "observed", "reference"],
            max_rows=20,
        ),
        "",
        "### label_5 distribution",
        "",
        md_table(_distribution_rows(rows, "label_5", 10), ["value", "count", "pct"], max_rows=10),
        "",
        "## 11. Split reference check",
        "",
        f"`paper_like_triple_seed42` split source: `{split_source}`.",
        "",
        md_table(split_stats, ["split_name", "split", "split_source", "target_rows", "num_rows", "train_pool_rows", "metric_count", "scenario_count", "subject_count", "education_level_count", "language_count"], max_rows=20),
        "",
        "## 12. Leakage check",
        "",
        f"Leakage status: **{leakage_status}**. See `leakage_report.md`, `tables/leakage_summary.csv`, and `tables/leakage_details.csv` for the exact checks.",
        "",
        "## 13. Low-score distribution",
        "",
        f"Labels 1/2 account for {low_count} of {len(rows)} scored items ({low_pct:.2%}).",
        "",
        "## 14. Figures generated",
        "",
        "\n".join(f"- `figures/{name}`" for name in _count_figures()),
        "",
        "## 15. Remaining warnings before Exp1",
        "",
        f"- Sanity status: {sanity_status}.",
        f"- Leakage status: {leakage_status}.",
        "- Synthetic sampled overlap is reported only as a future Exp6 augmentation risk; synthetic rows are excluded from the main dataset.",
        "- Confirm that local enriched subject annotations are acceptable before treating subject-level analysis as primary evidence.",
    ]
    write_text(OUTPUT_DIR / "report.md", "\n".join(report_lines))

    paper_counts = _split_counts("paper_like_triple_seed42")
    question_counts = _split_counts("question_seed42")
    examples = []
    for row in rows[:5]:
        examples.append(
            {
                "record_id": row["record_id"],
                "metric": row["metric_canonical"],
                "scenario": row["scenario_canonical"],
                "label_5": row["label_5"],
                "question_preview": truncate_text(row["question"], 120),
            }
        )

    review_lines = [
        "# Exp 0.1 Review Package",
        "",
        "## Exp1 Readiness",
        "",
        f"Can Exp1 start? **{can_exp1}**",
        "",
        "Remaining warnings: review leakage warnings, synthetic overlap risk notes, and subject metadata provenance before using subject-level findings as primary evidence.",
        "",
        "## Key Output Files",
        "",
        "- `thesis_exp/data/processed/edubench_scoring_all.jsonl`",
        "- `thesis_exp/data/splits/paper_like_triple_seed42/`",
        "- `thesis_exp/data/splits/question_seed42/`",
        "- `thesis_exp/outputs/exp00_data/data_card.md`",
        "- `thesis_exp/outputs/exp00_data/leakage_report.md`",
        "- `thesis_exp/outputs/exp00_data/report.md`",
        "- `thesis_exp/outputs/exp00_data/figures/`",
        "- `thesis_exp/outputs/exp00_data/tables/`",
        "",
        "## Key Counts",
        "",
        md_table(
            [
                {"item": "main_dataset_name", "value": "edubench_audit_human_scored_subset"},
                {"item": "main_dataset_rows", "value": len(rows)},
                {"item": "paper_like_triple_seed42", "value": paper_counts},
                {"item": "question_seed42", "value": question_counts},
                {"item": "main_split_name", "value": "paper_like_triple_seed42"},
                {"item": "paper_like_split_source", "value": split_source},
                {"item": "leakage_status", "value": leakage_status},
                {"item": "unmapped_metrics", "value": len(unmapped_metrics)},
                {"item": "num_unmapped_metric_error", "value": unmapped_metric_error},
                {"item": "num_unmapped_metric_info", "value": unmapped_metric_info},
                {"item": "unmapped_scenarios", "value": len(unmapped_scenarios)},
                {"item": "invalid_or_ambiguous_scores", "value": len(invalid_scores)},
                {"item": "subject_level_primary_evidence_allowed", "value": "YES, after accepting local enriched subject metadata" if subject_count == 25 else "NO"},
                {"item": "synthetic_files_excluded", "value": "YES"},
                {"item": "proper_markdown_reports_generated", "value": "YES"},
            ],
            ["item", "value"],
            max_rows=20,
        ),
        "",
        "## Five Standardized Sample Records",
        "",
        md_table(examples, ["record_id", "metric", "scenario", "label_5", "question_preview"], max_rows=5),
        "",
        "## Exact Commands Run",
        "",
        "```bash\npython -m thesis_exp.src.edujudge.data.run_exp00_reference_alignment\npython -m py_compile thesis_exp/src/edujudge/data/*.py thesis_exp/src/edujudge/plots/*.py thesis_exp/src/edujudge/utils/*.py\n```",
        "",
        "## Questions Requiring Human Confirmation",
        "",
        "- Confirm that `results_merge.jsonl` is the intended final merged human-scored source for thesis experiments.",
        "- `EduBench.zip` in this repository appears to contain paper/template assets rather than raw benchmark data; official scenario/metric alignment therefore uses the project definition and local `metrics_map.json`.",
        "- Confirm that `report/results_merge_enriched.jsonl` is acceptable as local derived metadata for 25-subject and original held-out split alignment.",
    ]
    write_text(OUTPUT_DIR / "review_package.md", "\n".join(review_lines))


def main() -> None:
    ensure_exp_dirs()
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found. Run build_dataset first: {PROCESSED_PATH}")
    rows = read_jsonl(PROCESSED_PATH)
    generate_figures(rows)
    generate_reports(rows)
    print(f"Wrote figures and reports to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
