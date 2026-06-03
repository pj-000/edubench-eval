"""Build Exp3 A0-A4 input-ablation datasets from the locked Exp0.1 split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import clean_answer_text, clean_question_text
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import make_prompt as make_exp02_prompt
from thesis_exp.src.edujudge.exp03 import (
    EXPECTED_SPLIT_ROWS,
    EXP03_DATASETS_DIR,
    EXP03_TABLES_DIR,
    EXP03_TEMPLATES_DIR,
    SPLIT_PATHS,
    TEMPLATE_NAMES,
    ensure_exp03_dirs,
    template_dataset_dir,
)
from thesis_exp.src.edujudge.exp03.rubric_sources import audit_rubric_sources
from thesis_exp.src.edujudge.exp03.templates import (
    estimate_token_length,
    get_template_spec,
    make_prompt,
    require_rubric_text,
    template_manifest_rows,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_json, write_jsonl, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


REQUIRED_SOURCE_FIELDS = [
    "record_id",
    "triple_key",
    "question",
    "answer",
    "metric_canonical",
    "metric_abbr",
    "metric_group",
    "rubric",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
    "generator_model",
    "human_mean_5",
    "label_5",
]


def load_tokenizer(model_name_or_path: str | None, local_files_only: bool) -> Any | None:
    if not model_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only, trust_remote_code=True)
    except Exception as exc:
        print(f"[WARN] tokenizer unavailable; falling back to approximate token lengths: {type(exc).__name__}: {exc}")
        return None


def count_tokens(text: str, tokenizer: Any | None) -> tuple[int, str]:
    if tokenizer is None:
        return estimate_token_length(text), "approx_char_div4"
    return len(tokenizer.encode(text, add_special_tokens=True)), "tokenizer"


def validate_source_row(row: dict[str, Any], split: str, idx: int) -> None:
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{split} row {idx} missing required fields: {missing}")
    label_5 = int(row["label_5"])
    if label_5 < 1 or label_5 > 5:
        raise ValueError(f"{split} row {idx} label_5 outside 1..5: {label_5}")


def convert_row(
    row: dict[str, Any],
    split: str,
    row_index: int,
    template_name: str,
    tokenizer: Any | None,
    max_length: int,
) -> dict[str, Any]:
    validate_source_row(row, split, row_index)
    label_5 = int(row["label_5"])
    text = make_prompt(row, template_name)
    num_tokens, token_count_method = count_tokens(text, tokenizer)
    rubric_text = require_rubric_text(row)
    return {
        "id": row.get("record_id") or f"{split}_{row_index}",
        "record_id": row.get("record_id"),
        "triple_key": row.get("triple_key"),
        "question": clean_question_text(row.get("question")),
        "answer": clean_answer_text(row.get("answer")),
        "text": text,
        "template_name": template_name,
        "label": label_5 - 1,
        "label_5": label_5,
        "human_mean_5": float(row["human_mean_5"]),
        "metric_canonical": row.get("metric_canonical"),
        "metric_abbr": row.get("metric_abbr"),
        "metric_group": row.get("metric_group"),
        "rubric_text": rubric_text,
        "scenario_canonical": row.get("scenario_canonical"),
        "subject_canonical": row.get("subject_canonical"),
        "education_level_canonical": row.get("education_level_canonical"),
        "language": row.get("language"),
        "generator_model": row.get("generator_model"),
        "num_chars": len(text),
        "num_tokens": num_tokens,
        "token_count_method": token_count_method,
        "truncated": bool(num_tokens > max_length),
    }


def build_template_split(
    split: str,
    rows: list[dict[str, Any]],
    template_name: str,
    tokenizer: Any | None,
    max_length: int,
) -> list[dict[str, Any]]:
    return [convert_row(row, split, idx, template_name, tokenizer, max_length) for idx, row in enumerate(rows)]


def summarize_lengths(rows: list[dict[str, Any]], split: str, template_name: str) -> dict[str, Any]:
    chars = np.array([int(row["num_chars"]) for row in rows], dtype=float)
    tokens = np.array([int(row["num_tokens"]) for row in rows], dtype=float)
    trunc = np.array([bool(row["truncated"]) for row in rows], dtype=bool)
    return {
        "template_name": template_name,
        "split": split,
        "n": len(rows),
        "mean_chars": float(chars.mean()) if len(chars) else 0.0,
        "p50_chars": float(np.percentile(chars, 50)) if len(chars) else 0.0,
        "p95_chars": float(np.percentile(chars, 95)) if len(chars) else 0.0,
        "max_chars": int(chars.max()) if len(chars) else 0,
        "mean_token_length": float(tokens.mean()) if len(tokens) else 0.0,
        "p50_token_length": float(np.percentile(tokens, 50)) if len(tokens) else 0.0,
        "p95_token_length": float(np.percentile(tokens, 95)) if len(tokens) else 0.0,
        "max_token_length": int(tokens.max()) if len(tokens) else 0,
        "truncation_rate": float(trunc.mean()) if len(trunc) else 0.0,
        "token_count_method": rows[0].get("token_count_method") if rows else "",
    }


def write_template_examples(built: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    examples: list[dict[str, Any]] = []
    for template_name in TEMPLATE_NAMES:
        rows = built[template_name]["train"][:10]
        for row in rows:
            examples.append(
                {
                    "record_id": row["record_id"],
                    "template_name": template_name,
                    "text_preview": row["text"][:2000],
                    "label_5": row["label_5"],
                    "char_length": row["num_chars"],
                    "token_length": row["num_tokens"],
                    "truncated": row["truncated"],
                }
            )
    write_json(EXP03_TEMPLATES_DIR / "rendered_template_examples.json", examples)


def write_dataset_card(stats_rows: list[dict[str, Any]], length_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp3 Input Ablation Datasets",
        "",
        "Derived from the locked Exp0.1 paper-like triple split. Exp3 changes only input templates;",
        "the model family, 5-class CE objective, labels, train/dev/test split, and checkpoint",
        "selection policy remain aligned with Exp2.",
        "",
        "A2 is the Exp2-compatible question + answer + metric baseline and should normally reuse",
        "Exp2 formal outputs instead of retraining.",
        "",
        "## Dataset Stats",
        "",
        "| template | split | rows | expected | status |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in stats_rows:
        lines.append(
            f"| {row['template_name']} | {row['split']} | {row['rows']} | {row['expected_rows']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Token Length Summary",
            "",
            "| template | split | mean_token_length | p95_token_length | truncation_rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in length_rows:
        lines.append(
            f"| {row['template_name']} | {row['split']} | {row['mean_token_length']:.2f} | "
            f"{row['p95_token_length']:.2f} | {row['truncation_rate']:.6f} |"
        )
    write_text(EXP03_DATASETS_DIR / "dataset_card.md", "\n".join(lines))


def write_template_manifest() -> None:
    rows = template_manifest_rows()
    write_csv(EXP03_TEMPLATES_DIR / "template_manifest.csv", rows)
    write_json(EXP03_TEMPLATES_DIR / "template_manifest.json", rows)


def equivalence_rows(raw_by_split: dict[str, list[dict[str, Any]]], built: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    for split, raw_rows in raw_by_split.items():
        a2_rows = built["A2_question_answer_metric"][split]
        mismatches = []
        for idx, (raw, converted) in enumerate(zip(raw_rows, a2_rows)):
            expected = make_exp02_prompt(raw)
            if converted["text"] != expected:
                mismatches.append(
                    {
                        "row_index": idx,
                        "record_id": raw.get("record_id"),
                        "exp03_preview": converted["text"][:200],
                        "exp02_preview": expected[:200],
                    }
                )
        rows.append(
            {
                "split": split,
                "checked_rows": len(raw_rows),
                "mismatch_rows": len(mismatches),
                "status": "PASS" if not mismatches else "FAIL",
                "first_mismatch_record_id": mismatches[0]["record_id"] if mismatches else "",
            }
        )
    return rows


def build_exp03_datasets(
    template_names: list[str] | None = None,
    model_name_or_path: str | None = None,
    local_files_only: bool = True,
    max_length: int = 2048,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    ensure_exp03_dirs()
    template_names = template_names or TEMPLATE_NAMES
    for template_name in template_names:
        get_template_spec(template_name)

    audit_rubric_sources()
    tokenizer = load_tokenizer(model_name_or_path, local_files_only=local_files_only)
    raw_by_split = {split: read_jsonl(path) for split, path in SPLIT_PATHS.items()}

    built: dict[str, dict[str, list[dict[str, Any]]]] = {}
    stats_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    template_length_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for template_name in template_names:
        built[template_name] = {}
        out_dir = template_dataset_dir(template_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, raw_rows in raw_by_split.items():
            converted = build_template_split(split, raw_rows, template_name, tokenizer, max_length)
            built[template_name][split] = converted
            write_jsonl(out_dir / f"{split}.jsonl", converted)
            expected_rows = EXPECTED_SPLIT_ROWS[split]
            stats_rows.append(
                {
                    "template_name": template_name,
                    "ablation_id": get_template_spec(template_name).ablation_id,
                    "split": split,
                    "source_path": relpath(SPLIT_PATHS[split]),
                    "output_path": relpath(out_dir / f"{split}.jsonl"),
                    "rows": len(converted),
                    "expected_rows": expected_rows,
                    "status": "PASS" if len(converted) == expected_rows else "FAIL",
                }
            )
            length_summary = summarize_lengths(converted, split, template_name)
            length_rows.append(length_summary)
            for row in converted:
                template_length_rows.append(
                    {
                        "template_name": template_name,
                        "split": split,
                        "record_id": row["record_id"],
                        "num_chars": row["num_chars"],
                        "num_tokens": row["num_tokens"],
                        "token_count_method": row["token_count_method"],
                        "truncated": row["truncated"],
                        "label_5": row["label_5"],
                    }
                )
            counts = Counter(row["label_5"] for row in converted)
            for label in [1, 2, 3, 4, 5]:
                label_rows.append(
                    {
                        "template_name": template_name,
                        "split": split,
                        "label_5": label,
                        "count": counts[label],
                        "pct_within_split": counts[label] / len(converted) if converted else 0,
                    }
                )

    write_csv(EXP03_TABLES_DIR / "dataset_stats_by_template.csv", stats_rows)
    write_csv(EXP03_TABLES_DIR / "template_length_stats.csv", length_rows)
    write_csv(EXP03_TEMPLATES_DIR / "template_lengths.csv", template_length_rows)
    write_csv(EXP03_TABLES_DIR / "label_distribution_by_template.csv", label_rows)
    if "A2_question_answer_metric" in built:
        write_csv(EXP03_TABLES_DIR / "a2_exp2_template_equivalence.csv", equivalence_rows(raw_by_split, built))
    write_template_examples(built)
    write_template_manifest()
    write_dataset_card(stats_rows, length_rows)
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp3 input-ablation datasets.")
    parser.add_argument("--templates", nargs="+", default=TEMPLATE_NAMES, choices=TEMPLATE_NAMES)
    parser.add_argument("--model_name_or_path", default=None, help="Optional tokenizer source for token lengths.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not download tokenizer files.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build_exp03_datasets(
        template_names=args.templates,
        model_name_or_path=args.model_name_or_path,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
    )
    if args.print_summary:
        for template_name, split_map in built.items():
            for split, rows in split_map.items():
                print(f"{template_name}/{split}: {len(rows)} rows")
    print(f"Exp3 datasets written to {relpath(EXP03_DATASETS_DIR)}")


if __name__ == "__main__":
    main()
