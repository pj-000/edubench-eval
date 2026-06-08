"""Leakage check plan and optional checker for generated Exp6 synthetic rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ensure_generation_dirs
from thesis_exp.src.edujudge.exp06_generation.common import output_path, qa_key, question_key, split_key_sets, write_table
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import write_text
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


FIELDS = [
    "synthetic_id",
    "source_question_key_in_dev",
    "source_question_key_in_test",
    "source_triple_key_in_dev",
    "source_triple_key_in_test",
    "synthetic_question_key_in_dev",
    "synthetic_question_key_in_test",
    "synthetic_qa_key_in_dev",
    "synthetic_qa_key_in_test",
    "duplicate_synthetic_answer",
    "duplicate_with_human_test_answer",
    "leakage_status",
    "notes",
]


def leakage_plan_text() -> str:
    return """# Exp6 Generated Synthetic Leakage Check Plan

Generated rows must pass these checks before any train-only augmentation:

- `source_question_key` must not appear in dev/test.
- `source_triple_key` must not appear in dev/test.
- normalized synthetic `question` key must not appear in dev/test.
- normalized synthetic `question + answer_synthetic` key must not appear in dev/test.
- synthetic answers must not duplicate each other.
- synthetic answers must not duplicate human test answers.
- `source_split` must be `train`.

Any dev/test hit blocks the row. Synthetic rows must never be added to dev/test.
"""


def check_rows(input_path: Path) -> list[dict[str, Any]]:
    keys = split_key_sets()
    seen_answers: set[str] = set()
    out = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer_key = sha1_text(normalize_text(row.get("answer_synthetic")))
            duplicate_synthetic = answer_key in seen_answers
            seen_answers.add(answer_key)
            q_key = question_key(row.get("question"))
            synthetic_qa = qa_key(row.get("question"), row.get("answer_synthetic"))
            checks = {
                "source_question_key_in_dev": row.get("source_question_key") in keys["dev"]["source_question_key"],
                "source_question_key_in_test": row.get("source_question_key") in keys["test"]["source_question_key"],
                "source_triple_key_in_dev": row.get("source_triple_key") in keys["dev"]["source_triple_key"],
                "source_triple_key_in_test": row.get("source_triple_key") in keys["test"]["source_triple_key"],
                "synthetic_question_key_in_dev": q_key in keys["dev"]["question_key"],
                "synthetic_question_key_in_test": q_key in keys["test"]["question_key"],
                "synthetic_qa_key_in_dev": synthetic_qa in keys["dev"]["qa_key"],
                "synthetic_qa_key_in_test": synthetic_qa in keys["test"]["qa_key"],
                "duplicate_synthetic_answer": duplicate_synthetic,
                "duplicate_with_human_test_answer": answer_key in keys["test"]["answer_key"],
            }
            leakage = any(checks.values())
            out.append(
                {
                    "synthetic_id": row.get("synthetic_id", ""),
                    **checks,
                    "leakage_status": "BLOCKED" if leakage else "PASS",
                    "notes": "generated synthetic leakage check; source must be train-only",
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args, _ = parser.parse_known_args()
    ensure_generation_dirs()
    write_text(output_path("leakage_check_plan.md"), leakage_plan_text())
    if args.input_jsonl:
        rows = check_rows(args.input_jsonl)
    else:
        rows = [
            {
                "synthetic_id": "",
                "source_question_key_in_dev": "",
                "source_question_key_in_test": "",
                "source_triple_key_in_dev": "",
                "source_triple_key_in_test": "",
                "synthetic_question_key_in_dev": "",
                "synthetic_question_key_in_test": "",
                "synthetic_qa_key_in_dev": "",
                "synthetic_qa_key_in_test": "",
                "duplicate_synthetic_answer": "",
                "duplicate_with_human_test_answer": "",
                "leakage_status": "DRY_RUN",
                "notes": "No generated JSONL supplied; wrote leakage plan only.",
            }
        ]
    write_table("generated_synthetic_leakage_check.csv", rows, FIELDS)
    print("Wrote leakage_check_plan.md and generated_synthetic_leakage_check.csv")


if __name__ == "__main__":
    main()
