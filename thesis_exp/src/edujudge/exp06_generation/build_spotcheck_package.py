"""Build manual spotcheck artifacts for Exp6-3 mini-batch generated candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ensure_mini_batch_dirs
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import mini_filtered_path, mini_spotcheck_path, read_jsonl_if_exists
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text
from thesis_exp.src.edujudge.utils.text_norm import truncate_text


SPOTCHECK_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "source_record_id",
    "language",
    "metric_canonical",
    "target_label_5",
    "error_type",
    "question_preview",
    "question",
    "answer_synthetic",
    "rubric_text",
    "rationale_for_label",
    "expected_failure_against_rubric",
    "manual_quality_ok",
    "manual_label_plausible",
    "manual_notes",
    "review_natural_plausible",
    "review_label_plausible",
    "review_error_type_aligned",
    "review_no_artifact_phrase",
    "review_no_copy_or_leakage_concern",
    "review_keep_for_pilot",
    "reviewer_notes",
]


FORM_FIELDS = [
    "synthetic_id",
    "manual_quality_ok",
    "manual_label_plausible",
    "manual_notes",
    "review_natural_plausible",
    "review_label_plausible",
    "review_error_type_aligned",
    "review_no_artifact_phrase",
    "review_no_copy_or_leakage_concern",
    "review_keep_for_pilot",
    "reviewer_notes",
]


def sample_rows(input_path: Path | None = None) -> list[dict[str, Any]]:
    input_path = input_path or mini_filtered_path("filtered_synthetic_candidates.jsonl")
    rows = read_jsonl_if_exists(input_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "source_record_id": row.get("source_record_id", ""),
                "language": row.get("language", ""),
                "metric_canonical": row.get("metric_canonical", ""),
                "target_label_5": row.get("target_label_5", ""),
                "error_type": row.get("error_type", ""),
                "question_preview": truncate_text(row.get("question", ""), 1200),
                "question": truncate_text(row.get("question", ""), 1200),
                "answer_synthetic": row.get("answer_synthetic", ""),
                "rubric_text": truncate_text(row.get("rubric_text", ""), 1200),
                "rationale_for_label": row.get("rationale_for_label", ""),
                "expected_failure_against_rubric": row.get("expected_failure_against_rubric", ""),
                "manual_quality_ok": "",
                "manual_label_plausible": "",
                "manual_notes": "",
                "review_natural_plausible": "",
                "review_label_plausible": "",
                "review_error_type_aligned": "",
                "review_no_artifact_phrase": "",
                "review_no_copy_or_leakage_concern": "",
                "review_keep_for_pilot": "",
                "reviewer_notes": "",
            }
        )
    return out


def write_guidelines(num_samples: int) -> None:
    text = f"""# Exp6-3 Mini-batch Spotcheck Guidelines

Samples to review: **{num_samples}**

Review every generated row before any full generation or training decision.

For each row, confirm:

- The answer is natural and plausible in the requested language.
- The target 1-5 label is plausible for the named metric and rubric.
- The error type is visible enough to justify the target label.
- The answer does not mention scoring, hidden instructions, data creation, or experiment design.
- The answer does not copy the source answer.
- There is no dev/test leakage concern from the source or answer text.

Rows that fail any required check must not be used for training. Labels remain
`synthetic_design` pseudo-labels, not human labels.
"""
    write_text(mini_spotcheck_path("spotcheck_guidelines.md"), text)


def build_package(input_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_mini_batch_dirs()
    rows = sample_rows(input_path)
    write_csv(mini_spotcheck_path("spotcheck_samples.csv"), rows, SPOTCHECK_FIELDS)
    form_rows = [{field: row.get(field, "") for field in FORM_FIELDS} for row in rows]
    if not form_rows:
        form_rows = []
    write_csv(mini_spotcheck_path("spotcheck_form_template.csv"), form_rows, FORM_FIELDS)
    write_guidelines(len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args = parser.parse_args()
    rows = build_package(args.input_jsonl)
    print(f"Wrote spotcheck package with {len(rows)} samples to {relpath(mini_spotcheck_path('spotcheck_samples.csv'))}")


if __name__ == "__main__":
    main()
