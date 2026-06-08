"""Build A4 input dataset on the question-disjoint split for Exp6 baselines."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import clean_answer_text, clean_question_text
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import count_tokens, load_tokenizer, validate_source_row
from thesis_exp.src.edujudge.exp03.rubric_repair import AUTO_MODE, apply_rubric_mode, load_rubric_mapping, resolve_rubric_mode
from thesis_exp.src.edujudge.exp03.templates import make_prompt
from thesis_exp.src.edujudge.exp06_baselines import (
    A4_QD_DATASET_DIR,
    EXPECTED_QD_SPLIT_ROWS,
    EXP06_QD_TABLES_DIR,
    QUESTION_SPLIT_DIR,
    ensure_exp06_qd_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_jsonl, write_text


TEMPLATE_NAME = "A4_question_answer_metric_rubric_metadata"
REQUIRED_FIELDS = {
    "id",
    "record_id",
    "triple_key",
    "text",
    "label",
    "label_5",
    "human_mean_5",
    "metric_canonical",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
}


def convert_row(
    row: dict[str, Any],
    split: str,
    row_index: int,
    tokenizer: Any | None,
    max_length: int,
    rubric_mode: str,
    rubric_mapping: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    validate_source_row(row, split, row_index)
    row_for_prompt = apply_rubric_mode(row, rubric_mode, rubric_mapping)
    label_5 = int(row["label_5"])
    text = make_prompt(row_for_prompt, TEMPLATE_NAME)
    num_tokens, token_count_method = count_tokens(text, tokenizer)
    return {
        "id": row.get("record_id") or f"{split}_{row_index}",
        "record_id": row.get("record_id"),
        "triple_key": row.get("triple_key"),
        "question": clean_question_text(row.get("question")),
        "answer": clean_answer_text(row.get("answer")),
        "text": text,
        "template_name": TEMPLATE_NAME,
        "label": label_5 - 1,
        "label_5": label_5,
        "human_mean_5": float(row["human_mean_5"]),
        "metric_canonical": row.get("metric_canonical"),
        "metric_abbr": row.get("metric_abbr"),
        "metric_group": row.get("metric_group"),
        "rubric_text": row_for_prompt["rubric_text"],
        "scenario_canonical": row.get("scenario_canonical"),
        "subject_canonical": row.get("subject_canonical"),
        "education_level_canonical": row.get("education_level_canonical"),
        "language": row.get("language"),
        "generator_model": "",
        "num_chars": len(text),
        "num_tokens": num_tokens,
        "token_count_method": token_count_method,
        "truncated": bool(num_tokens > max_length),
        "rubric_mode": row_for_prompt.get("rubric_mode"),
        "rubric_source_file": row_for_prompt.get("rubric_source_file"),
        "rubric_source_field": row_for_prompt.get("rubric_source_field"),
        "rubric_requires_human_confirmation": row_for_prompt.get("rubric_requires_human_confirmation"),
        "rubric_repair_notes": row_for_prompt.get("rubric_repair_notes"),
        "generation_split_name": "question_seed42",
    }


def validate_built_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    missing_fields: set[str] = set()
    bad_labels = 0
    for row in rows:
        missing_fields.update(REQUIRED_FIELDS - set(row))
        try:
            label = int(row["label_5"])
            if label < 1 or label > 5:
                bad_labels += 1
        except Exception:
            bad_labels += 1
    status = "PASS" if len(rows) == EXPECTED_QD_SPLIT_ROWS[split] and not missing_fields and bad_labels == 0 else "FAIL"
    return {
        "split": split,
        "rows": len(rows),
        "expected_rows": EXPECTED_QD_SPLIT_ROWS[split],
        "missing_fields": ",".join(sorted(missing_fields)),
        "bad_labels": bad_labels,
        "status": status,
        "path": relpath(A4_QD_DATASET_DIR / f"{split}.jsonl"),
    }


def build_dataset(model_name_or_path: str | None = None, max_length: int = 2048, rubric_mode: str = AUTO_MODE) -> list[dict[str, Any]]:
    ensure_exp06_qd_dirs()
    resolved_rubric_mode = resolve_rubric_mode(rubric_mode)
    rubric_mapping = load_rubric_mapping() if resolved_rubric_mode == "auto" else {}
    tokenizer = load_tokenizer(model_name_or_path, local_files_only=True) if model_name_or_path else None
    sanity_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        source_rows = read_jsonl(QUESTION_SPLIT_DIR / f"{split}.jsonl")
        converted = [
            convert_row(row, split, idx, tokenizer, max_length, resolved_rubric_mode, rubric_mapping)
            for idx, row in enumerate(source_rows)
        ]
        write_jsonl(A4_QD_DATASET_DIR / f"{split}.jsonl", converted)
        sanity_rows.append(validate_built_rows(converted, split))
        counts = Counter(int(row["label_5"]) for row in converted)
        for label in sorted(counts):
            label_rows.append({"split": split, "label_5": label, "count": counts[label]})
    write_csv(EXP06_QD_TABLES_DIR / "qd_a4_dataset_sanity.csv", sanity_rows)
    write_csv(EXP06_QD_TABLES_DIR / "qd_a4_label_distribution.csv", label_rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in sanity_rows) else "FAIL"
    lines = [
        "# Exp6 Question-disjoint A4 Dataset",
        "",
        f"Overall status: **{overall}**",
        "",
        "This dataset uses `question_seed42` and the Exp3 A4 input template. It is intended for",
        "question-disjoint Exp6 baselines QD-B0 and QD-B1.",
        "",
        "| split | rows | expected | status | path |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in sanity_rows:
        lines.append(f"| {row['split']} | {row['rows']} | {row['expected_rows']} | {row['status']} | `{row['path']}` |")
    write_text(A4_QD_DATASET_DIR / "dataset_card.md", "\n".join(lines))
    if overall != "PASS":
        raise RuntimeError(f"Exp6 QD A4 dataset sanity failed: {relpath(EXP06_QD_TABLES_DIR / 'qd_a4_dataset_sanity.csv')}")
    return sanity_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--rubric_mode", default=AUTO_MODE)
    args = parser.parse_args()
    rows = build_dataset(args.model_name_or_path, args.max_length, args.rubric_mode)
    print(f"Wrote Exp6 QD A4 dataset: {[row['status'] for row in rows]}")
    print(f"Output: {relpath(A4_QD_DATASET_DIR)}")


if __name__ == "__main__":
    main()
