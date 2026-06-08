"""Write Exp6-2 train-only synthetic generation plan reports."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    DEFAULT_GENERATION_MODEL,
    EXP06_GENERATION_OUTPUT_DIR,
    EXP06_GENERATION_PROMPTS_DIR,
    EXP06_GENERATION_SAMPLES_DIR,
    TARGET_LABEL_COUNTS_PER_METRIC_LANGUAGE,
    ensure_generation_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.build_error_type_plan import main as run_error_plan
from thesis_exp.src.edujudge.exp06_generation.build_generation_prompts import main as run_prompts
from thesis_exp.src.edujudge.exp06_generation.check_generated_synthetic_leakage import main as run_leakage_plan
from thesis_exp.src.edujudge.exp06_generation.common import read_context_report, read_table, table_path
from thesis_exp.src.edujudge.exp06_generation.filter_generated_synthetic import main as run_filter_plan
from thesis_exp.src.edujudge.exp06_generation.sample_train_sources_for_synthetic import main as run_sampling
from thesis_exp.src.edujudge.exp06_generation.validate_generation_schema import main as run_schema
from thesis_exp.src.edujudge.utils.io import THESIS_DIR, md_table, relpath, write_text


def run_pipeline() -> None:
    run_sampling()
    run_error_plan()
    run_prompts()
    run_schema()
    run_filter_plan()
    run_leakage_plan()


def summarize() -> dict[str, Any]:
    sampling = read_table("train_source_sampling_plan.csv")
    error_types = read_table("error_type_plan.csv")
    target = read_table("generation_target_matrix.csv")
    diagnostic = read_table("synthetic_only_diagnostic_target_matrix.csv")
    filtering = read_table("filtering_rules.csv")
    selected = [row for row in sampling if str(row.get("selected_for_generation", "")).lower() == "true"]
    target_total = sum(int(row.get("target_count", 0)) for row in target)
    diagnostic_total = sum(int(row.get("target_count", 0)) for row in diagnostic)
    label_counts = Counter(row.get("target_label_5") for row in target for _ in range(int(row.get("target_count", 0))))
    return {
        "sampling": sampling,
        "selected": selected,
        "error_types": error_types,
        "target": target,
        "diagnostic": diagnostic,
        "filtering": filtering,
        "target_total": target_total,
        "diagnostic_total": diagnostic_total,
        "target_label_counts": dict(label_counts),
        "exp60": read_context_report(THESIS_DIR / "outputs" / "exp06_synthetic_low_score" / "report.md"),
        "exp61": read_context_report(THESIS_DIR / "outputs" / "exp06_synthetic_low_score" / "processed_excel_audit" / "report.md"),
    }


def write_report(summary: dict[str, Any]) -> None:
    report = f"""# Exp6-2 Train-only Synthetic Low-score Generation Plan

## Scope

This scaffold prepares train-only low-score synthetic generation with planned model
`{DEFAULT_GENERATION_MODEL}`. It does **not** call APIs, generate synthetic data, train models, or
modify Exp0-Exp5 results.

## Why New Generation Is Needed

Exp6-0 found existing synthetic/sampled data mostly blocked by dev/test leakage, judge-output risk,
or unclear provenance. Exp6-1 found processed Excel candidates have model/pseudo labels and only 8
low-score train-only rows. Therefore a new train-only generation plan is needed before any Exp6
augmentation training.

## Source Sampling

Source rows come only from `paper_like_triple_seed42/train.jsonl`. Dev/test questions are not used.

- Sampling plan rows: **{len(summary["sampling"])}**
- Selected train source anchors: **{len(summary["selected"])}**
- Selected anchors cover metrics/languages from train only.

{md_table(summary["selected"], ["source_record_id", "metric_canonical", "language", "current_label_5", "planned_error_types", "planned_target_labels", "notes"], max_rows=24)}

## Error Types

{md_table(summary["error_types"], ["error_type", "target_label_range", "applicable_metrics", "expected_risk"], max_rows=20)}

## Generation Target Matrix

First low-score augmentation target: **{summary["target_total"]}** rows.

Target label counts: `{summary["target_label_counts"]}`

The first matrix emphasizes labels 1/2 with a small label-3 boundary set. For D1/D4 synthetic-only
diagnostics, `synthetic_only_diagnostic_target_matrix.csv` provides an optional full-score matrix
with labels 1-5, but it is not part of the first low-score augmentation generation.

## Prompt Templates

- `{relpath(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer.md")}`
- `{relpath(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer_en.md")}`
- `{relpath(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer_zh.md")}`
- `{relpath(EXP06_GENERATION_SAMPLES_DIR / "dry_run_prompts.jsonl")}`

Prompts instruct the model to generate natural flawed answers without mentioning synthetic or
intentional wrongness. They request JSON output but do not call any API.

## Schema, Filtering, Leakage

- `{relpath(EXP06_GENERATION_OUTPUT_DIR / "synthetic_schema.md")}`
- `{relpath(table_path("filtering_rules.csv"))}`
- `{relpath(EXP06_GENERATION_OUTPUT_DIR / "leakage_check_plan.md")}`

Required checks include valid JSON, non-empty answer, length bounds, language match, no explicit
intentional-wrong phrasing, no copied original answer, valid label/metric/rubric, source split train,
deduplication, and no dev/test leakage.

## Recommendation

Generation can start only after human review approves the prompt templates, target matrix, API
budget, and leakage/filtering workflow. Training still cannot start until generated rows pass schema,
filtering, dedup, and dev/test leakage checks.
"""
    write_text(EXP06_GENERATION_OUTPUT_DIR / "report.md", report)


def write_review_package(summary: dict[str, Any]) -> None:
    review = f"""# Exp6-2 Generation Plan Review Package

Can generation start? **YES, after human review and API approval**

Can training start? **NO**

Required manual confirmations:

- Approve `{DEFAULT_GENERATION_MODEL}` or replace it with another generation model.
- Confirm API budget and rate limits.
- Review prompt templates for accidental dev/test leakage or label ambiguity.
- Confirm first-run target: {summary["target_total"]} low-score/boundary rows.
- Confirm optional D1/D4 full-score diagnostic should be generated separately.
- Confirm generated data remains train-only and pseudo-labeled.

API/model needed: `{DEFAULT_GENERATION_MODEL}`.

Estimated generation count: **{summary["target_total"]}** for first low-score augmentation; optional
diagnostic matrix count is **{summary["diagnostic_total"]}**.

Risks:

- Synthetic pseudo labels may not match human scoring.
- Low-score answers may be too artificial or too easy.
- Full-score synthetic-only diagnostic can introduce model-style distribution bias.
- Any dev/test overlap must block affected rows.

Next step: run a reviewed dry-run prompt sample, then implement an approved API runner, generate a
small batch, and audit generated rows before training.
"""
    write_text(EXP06_GENERATION_OUTPUT_DIR / "review_package.md", review)


def write_notion_summary(summary: dict[str, Any]) -> None:
    notion = f"""# Exp6-2 Generation Plan Summary

- API called: NO
- Synthetic data generated: NO
- Generation model planned: {DEFAULT_GENERATION_MODEL}
- Source split: train only
- Sampling plan rows: {len(summary["sampling"])}
- Selected source anchors: {len(summary["selected"])}
- First target generation count: {summary["target_total"]}
- Optional D1/D4 diagnostic target count: {summary["diagnostic_total"]}
- Error types: {', '.join(row['error_type'] for row in summary['error_types'])}
- Generation can start after human review: YES
- Training can start now: NO
"""
    write_text(EXP06_GENERATION_OUTPUT_DIR / "notion_exp06_generation_plan_summary.md", notion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pipeline", action="store_true")
    args = parser.parse_args()
    ensure_generation_dirs()
    if args.run_pipeline:
        run_pipeline()
    summary = summarize()
    write_report(summary)
    write_review_package(summary)
    write_notion_summary(summary)
    print(f"Wrote generation plan report to {EXP06_GENERATION_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
