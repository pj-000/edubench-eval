"""Prepare Exp19 first-round SFT dev prediction data for LLaMA-Factory.

This script creates a dev-only ShareGPT dataset used by LLaMA-Factory
``do_predict``. The assistant label is included only as the held-out reference
target consumed by the prediction/evaluation loop; it is not part of the user
prompt and is not used for training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    OPENAI_TAGS,
    clean,
    clamp_score,
    language,
    metric_name,
    question_group,
    sample_id,
    score_only_target,
    sft_dataset_entry,
    sft_example,
    subject,
)
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_csv, write_json, write_text  # noqa: E402


DEFAULT_DEV_PATH = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_DATASET_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42")
EVAL_DATASET_NAME = "edubench_exp19_dev_score_eval"


def load_dataset_info(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"dataset_info must be an object: {path}")
        return data
    return {}


def stable_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = sample_id(row)
    return sid if sid else f"dev_row_{idx:05d}"


def gold_label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5") or row.get("label") or row.get("gold_label"))


def eval_record(row: dict[str, Any]) -> dict[str, Any]:
    return sft_example(row, score_only_target(row))


def reference_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    return {
        "eval_index": idx,
        "split": split,
        "sample_id": stable_sample_id(row, idx),
        "record_id": clean(row.get("record_id")),
        "question_key": clean(row.get("question_key")),
        "question_group_id": question_group(row),
        "metric": metric_name(row),
        "metric_id": clean(row.get("metric_id")),
        "language": language(row),
        "subject": subject(row),
        "gold_label": gold_label(row),
        "human_1": clean(row.get("human_1_5") or row.get("human_1")),
        "human_2": clean(row.get("human_2_5") or row.get("human_2")),
        "human_3": clean(row.get("human_3_5") or row.get("human_3")),
    }


def write_dataset_info(dataset_dir: Path, eval_file: Path) -> None:
    entry = sft_dataset_entry(str(eval_file.relative_to(dataset_dir)))
    if entry.get("tags") != OPENAI_TAGS:
        raise ValueError("Unexpected ShareGPT tag mapping for eval dataset.")
    for name in ("dataset_info.json", "dataset_info_snippet.json"):
        path = dataset_dir / name
        data = load_dataset_info(path)
        data[EVAL_DATASET_NAME] = entry
        write_json(path, data)


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.dev_jsonl)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    data_dir = args.dataset_dir / "data"
    tables_dir = args.dataset_dir / "tables"
    reports_dir = args.dataset_dir / "reports"
    eval_file = data_dir / f"{EVAL_DATASET_NAME}.json"
    reference_file = tables_dir / "exp19_dev_reference.csv"

    write_json(eval_file, [eval_record(row) for row in rows])
    write_csv(reference_file, [reference_row(row, idx, args.split_name) for idx, row in enumerate(rows)])
    write_dataset_info(args.dataset_dir, eval_file)

    report = [
        "# Exp19 First-Round SFT Dev Prediction Dataset",
        "",
        f"- split: `{args.split_name}`",
        f"- source jsonl: `{args.dev_jsonl}`",
        f"- examples: {len(rows)}",
        f"- LLaMA-Factory dataset name: `{EVAL_DATASET_NAME}`",
        f"- eval data file: `{eval_file}`",
        f"- reference table: `{reference_file}`",
        "",
        "The user prompt contains only question, answer, metric, rubric, and metadata.",
        "The assistant label is present only as the held-out reference target for prediction bookkeeping.",
        "This step does not read test and does not train a model.",
    ]
    write_text(reports_dir / "exp19_sft_dev_eval_dataset_report.md", "\n".join(report))
    return {
        "dataset_name": EVAL_DATASET_NAME,
        "rows": len(rows),
        "eval_file": str(eval_file),
        "reference_file": str(reference_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp19 LLaMA-Factory dev prediction dataset.")
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-name", default="dev")
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all dev examples.")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
