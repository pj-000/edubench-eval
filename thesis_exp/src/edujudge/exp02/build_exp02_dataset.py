"""Build fixed-split classification data for the Exp2 CE baseline.

This step does not train a model. It converts the locked Exp0.1
``paper_like_triple_seed42`` split into JSONL records consumed by
``train_ce_baseline.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp02 import (
    DEV_SPLIT_PATH,
    EXPECTED_SPLIT_ROWS,
    EXP02_DATA_DIR,
    EXP02_OUTPUT_DIR,
    EXP02_TABLES_DIR,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    ensure_exp02_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_jsonl, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


SPLIT_PATHS = {
    "train": TRAIN_SPLIT_PATH,
    "dev": DEV_SPLIT_PATH,
    "test": TEST_SPLIT_PATH,
}


def _rubric_text(row: dict[str, Any]) -> str:
    rubric = row.get("rubric")
    if isinstance(rubric, list):
        return "\n".join(f"- {stringify(item)}" for item in rubric)
    return stringify(rubric)


def make_prompt(row: dict[str, Any]) -> str:
    """Create the CE baseline input text without human or judge labels."""
    metadata = [
        f"Metric: {row.get('metric_canonical', 'unknown')}",
        f"Metric group: {row.get('metric_group', 'unknown')}",
        f"Scenario: {row.get('scenario_canonical', 'unknown')}",
        f"Subject: {row.get('subject_canonical', 'unknown')}",
        f"Education level: {row.get('education_level_canonical', 'unknown')}",
        f"Language: {row.get('language', 'unknown')}",
        f"Generator model: {row.get('generator_model', 'unknown')}",
    ]
    return "\n\n".join(
        [
            "You are EduBenchEvaluator, an educational response quality judge.",
            "Task: assign exactly one score from 1 to 5 for the assistant response under the specified metric.",
            "Score meaning: 1 is worst quality and 5 is best quality. Use the rubric below as the authority.",
            "[Metadata]\n" + "\n".join(metadata),
            "[Question]\n" + stringify(row.get("question")),
            "[Assistant Response]\n" + stringify(row.get("answer")),
            "[Rubric]\n" + _rubric_text(row),
            "Predict the final score label only.",
        ]
    )


def convert_row(row: dict[str, Any], split: str, row_index: int) -> dict[str, Any]:
    label_5 = int(row["label_5"])
    if label_5 < 1 or label_5 > 5:
        raise ValueError(f"label_5 outside 1-5 for {row.get('record_id')}: {label_5}")
    return {
        "id": row.get("record_id") or f"{split}_{row_index}",
        "split": split,
        "text": make_prompt(row),
        "label": label_5 - 1,
        "label_5": label_5,
        "human_mean_5": float(row["human_mean_5"]),
        "record_id": row.get("record_id"),
        "triple_key": row.get("triple_key"),
        "question_key": row.get("question_key"),
        "answer_key": row.get("answer_key"),
        "metric_canonical": row.get("metric_canonical"),
        "metric_abbr": row.get("metric_abbr"),
        "metric_group": row.get("metric_group"),
        "scenario_canonical": row.get("scenario_canonical"),
        "subject_canonical": row.get("subject_canonical"),
        "education_level_canonical": row.get("education_level_canonical"),
        "language": row.get("language"),
        "generator_model": row.get("generator_model"),
    }


def build_split(split: str, path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    converted = [convert_row(row, split, idx) for idx, row in enumerate(rows)]
    return converted


def write_dataset_card(stats_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp2 CE Baseline Dataset",
        "",
        "This dataset is derived from the locked Exp0.1 paper-like triple split. It is intended for",
        "training a 5-class cross-entropy baseline for EduBenchEvaluator 0.6B.",
        "",
        "No Exp0 data or split file is modified. Human labels are used only as supervised targets;",
        "existing automatic judge predictions are not used as training targets.",
        "",
        "## Source Splits",
        "",
        "| split | source | rows | expected_rows | status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in stats_rows:
        lines.append(
            f"| {row['split']} | `{row['source_path']}` | {row['rows']} | {row['expected_rows']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            "| split | label_5 | count | pct_within_split |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in label_rows:
        lines.append(f"| {row['split']} | {row['label_5']} | {row['count']} | {row['pct_within_split']:.6f} |")
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- `{relpath(EXP02_DATA_DIR / 'train.jsonl')}`",
            f"- `{relpath(EXP02_DATA_DIR / 'dev.jsonl')}`",
            f"- `{relpath(EXP02_DATA_DIR / 'test.jsonl')}`",
        ]
    )
    write_text(EXP02_OUTPUT_DIR / "dataset_card.md", "\n".join(lines))


def build_exp02_dataset() -> dict[str, list[dict[str, Any]]]:
    ensure_exp02_dirs()
    built: dict[str, list[dict[str, Any]]] = {}
    stats_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    seen_ids_by_split: dict[str, set[str]] = {}
    for split, path in SPLIT_PATHS.items():
        converted = build_split(split, path)
        built[split] = converted
        write_jsonl(EXP02_DATA_DIR / f"{split}.jsonl", converted)
        expected = EXPECTED_SPLIT_ROWS[split]
        status = "PASS" if len(converted) == expected else "WARN"
        stats_rows.append(
            {
                "split": split,
                "source_path": relpath(path),
                "output_path": relpath(EXP02_DATA_DIR / f"{split}.jsonl"),
                "rows": len(converted),
                "expected_rows": expected,
                "status": status,
            }
        )
        counter = Counter(row["label_5"] for row in converted)
        for label in [1, 2, 3, 4, 5]:
            count = counter[label]
            label_rows.append(
                {
                    "split": split,
                    "label_5": label,
                    "count": count,
                    "pct_within_split": count / len(converted) if converted else 0,
                }
            )
        seen_ids_by_split[split] = {stringify(row["id"]) for row in converted}

    overlap_rows = []
    split_names = list(SPLIT_PATHS)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = seen_ids_by_split[left] & seen_ids_by_split[right]
            overlap_rows.append({"left_split": left, "right_split": right, "record_id_overlap": len(overlap)})

    write_csv(EXP02_TABLES_DIR / "dataset_stats.csv", stats_rows)
    write_csv(EXP02_TABLES_DIR / "label_distribution.csv", label_rows)
    write_csv(EXP02_TABLES_DIR / "split_record_id_overlap.csv", overlap_rows)
    write_dataset_card(stats_rows, label_rows)
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp2 CE baseline JSONL files.")
    parser.add_argument("--print-summary", action="store_true", help="Print split and label counts after building.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build_exp02_dataset()
    if args.print_summary:
        for split, rows in built.items():
            counts = Counter(row["label_5"] for row in rows)
            print(f"{split}: {len(rows)} rows, labels={dict(sorted(counts.items()))}")
    print(f"Exp2 CE data written to {relpath(EXP02_DATA_DIR)}")


if __name__ == "__main__":
    main()

