"""Write Exp6-3 mini-batch generation dry-run/audit reports."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_GENERATION_SPLIT_MODE,
    EXP06_MINI_BATCH_OUTPUT_DIR,
    MINI_BATCH_TOTAL_TARGET,
    ensure_mini_batch_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.build_mini_batch_generation_plan import build_all_split_mode_outputs, build_plan_outputs
from thesis_exp.src.edujudge.exp06_generation.build_spotcheck_package import build_package
from thesis_exp.src.edujudge.exp06_generation.check_mini_batch_leakage import run_leakage
from thesis_exp.src.edujudge.exp06_generation.filter_generated_low_score import filter_candidates
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    count_jsonl_lines,
    mini_filtered_path,
    mini_generated_path,
    mini_leakage_path,
    mini_prompt_path,
    mini_report_path,
    mini_spotcheck_path,
    mini_table_path,
    path_list_for_report,
    read_mini_table,
    target_counts,
)
from thesis_exp.src.edujudge.exp06_generation.normalize_generated_low_score import normalize
from thesis_exp.src.edujudge.exp06_generation.readability_check_generation import run_readability
from thesis_exp.src.edujudge.exp06_generation.run_mini_batch_generation import run as run_generation
from thesis_exp.src.edujudge.exp06_generation.sanity_check_mini_batch_generation import run_sanity
from thesis_exp.src.edujudge.utils.io import md_table, read_csv, relpath, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


def run_pipeline(mode: str = "dry_run", max_items: int = MINI_BATCH_TOTAL_TARGET) -> None:
    ensure_mini_batch_dirs()
    build_all_split_mode_outputs()
    try:
        run_generation(mode=mode, max_items=max_items)
    except SystemExit:
        # Generation mode can be blocked by missing model/key. Continue audit output
        # generation so the user gets BLOCKED_NO_GENERATION plus review artifacts.
        pass
    normalize()
    filter_candidates()
    run_leakage()
    build_package()
    run_sanity()
    run_readability()


def read_csv_count(path: Path) -> int:
    rows = read_mini_table(path.name) if path.parent == mini_table_path(path.name).parent else []
    return len(rows)


def summarize() -> dict[str, Any]:
    targets = read_mini_table("mini_batch_target_matrix.csv")
    sources = read_mini_table("mini_batch_source_selection.csv")
    filter_report = read_mini_table("filter_report.csv")
    sanity = read_mini_table("sanity_check_mini_batch_generation.csv")
    readability = read_mini_table("readability_check_generation.csv")
    raw_count = count_jsonl_lines(mini_generated_path("raw_generations.jsonl"))
    normalized_count = count_jsonl_lines(mini_generated_path("normalized_synthetic_candidates.jsonl"))
    filtered_count = count_jsonl_lines(mini_filtered_path("filtered_synthetic_candidates.jsonl"))
    prompt_count = count_jsonl_lines(mini_prompt_path("mini_batch_prompts.jsonl"))
    blocked_report = mini_report_path("BLOCKED_NO_GENERATION.md").exists()
    blocked_source_report = mini_report_path("BLOCKED_NO_TRAIN_ONLY_SOURCE.md").exists()
    dry_run_report = mini_report_path("DRY_RUN_NO_GENERATION.md").exists()
    if blocked_source_report and not prompt_count:
        mode = "BLOCKED_NO_TRAIN_ONLY_SOURCE"
    elif blocked_report and not raw_count:
        mode = "BLOCKED_NO_GENERATION"
    else:
        mode = "DRY_RUN" if dry_run_report and not raw_count else "GENERATED"
    leakage_status = "DRY_RUN_NO_GENERATED_ROWS"
    leakage_summary_path = mini_leakage_path("leakage_summary.csv")
    if leakage_summary_path.exists():
        import csv

        with leakage_summary_path.open("r", encoding="utf-8", newline="") as handle:
            leakage_rows = list(csv.DictReader(handle))
        blocked = next((row for row in leakage_rows if row.get("check_name") == "blocked_leakage_rows"), {})
        if blocked.get("count") and blocked.get("count") != "0":
            leakage_status = "BLOCKED"
        elif filtered_count:
            leakage_status = "PASS"
    else:
        leakage_rows = []
    label_counts = Counter(row.get("target_label_5") for row in targets)
    lang_counts = Counter(row.get("language") for row in targets)
    metric_count = len({row.get("metric_canonical") for row in targets if row.get("metric_canonical")})
    error_count = len({row.get("error_type") for row in targets if row.get("error_type")})
    source_leaks = [
        row
        for row in sources
        if stringify(row.get("source_question_in_dev")).lower() == "true"
        or stringify(row.get("source_question_in_test")).lower() == "true"
        or stringify(row.get("source_triple_in_dev")).lower() == "true"
        or stringify(row.get("source_triple_in_test")).lower() == "true"
    ]
    split_diag_path = EXP06_MINI_BATCH_OUTPUT_DIR / "split_mode_diagnostics" / "split_mode_source_diagnostics.csv"
    split_diagnostics = read_csv(split_diag_path) if split_diag_path.exists() else []
    return {
        "targets": targets,
        "sources": sources,
        "filter_report": filter_report,
        "sanity": sanity,
        "readability": readability,
        "raw_count": raw_count,
        "normalized_count": normalized_count,
        "filtered_count": filtered_count,
        "prompt_count": prompt_count,
        "mode": mode,
        "leakage_status": leakage_status,
        "leakage_rows": leakage_rows,
        "label_counts": dict(label_counts),
        "language_counts": dict(lang_counts),
        "metric_coverage": metric_count,
        "error_type_coverage": error_count,
        "source_leaks": source_leaks,
        "sanity_failures": [row for row in sanity if row.get("status") == "FAIL"],
        "readability_failures": [row for row in readability if row.get("status") == "FAIL"],
        "readability_warnings": [row for row in readability if row.get("status") == "WARN"],
        "blocked_source_report": blocked_source_report,
        "default_generation_split_mode": DEFAULT_GENERATION_SPLIT_MODE,
        "split_diagnostics": split_diagnostics,
    }


def write_report(summary: dict[str, Any]) -> None:
    source_status_counts = Counter(row.get("selection_status") for row in summary["sources"])
    report = f"""# Exp6-3 Mini-batch Synthetic Low-score Generation + Audit

## Scope

This mini-batch pipeline prepares a 24-item train-only generation pilot for `deepseek-v4-pro`.
Default execution is dry-run. It does not train models, modify Exp0-Exp5, call an API in dry-run, or
create synthetic answers in dry-run.

Exp6-3b adds generation split modes. The formal mode is `question_disjoint_formal`, backed by
`question_seed42`, because the paper-like split is not question-disjoint.

## Current Mode

- Mode: **{summary["mode"]}**
- Default generation split mode: **{summary["default_generation_split_mode"]}**
- API called by this run: **{"YES" if summary["raw_count"] > 0 else "NO"}**
- Synthetic answers generated by this run: **{"YES" if summary["raw_count"] > 0 else "NO"}**
- Planned rows: **{len(summary["targets"])}**
- Prompt rows: **{summary["prompt_count"]}**
- Raw generation rows: **{summary["raw_count"]}**
- Normalized candidate rows: **{summary["normalized_count"]}**
- Filter-passed rows: **{summary["filtered_count"]}**

## Mini-batch Target Design

- Target label distribution: `{summary["label_counts"]}`
- Language distribution: `{summary["language_counts"]}`
- Metric coverage: **{summary["metric_coverage"]}**
- Error type coverage: **{summary["error_type_coverage"]}**
- Source selection statuses: `{dict(source_status_counts)}`
- Dev/test source overlap rows selected: **{len(summary["source_leaks"])}**
- Source selection complete: **{sum(1 for row in summary["sources"] if row.get("source_record_id"))}/{len(summary["targets"])}**
- Source blocker diagnostics: `{relpath(mini_table_path("train_source_blocker_diagnostics.csv"))}`

## Split Mode Diagnostics

{md_table(summary["split_diagnostics"], ["mode", "train_rows", "dev_rows", "test_rows", "train_dev_shared_questions", "train_test_shared_questions", "strict_eligible_source_rows", "risk_level", "allowed_for_training"], max_rows=10)}

Interpretation:

- `paper_like_strict` is blocked for leakage-free generation because strict source count is 0.
- `paper_like_triple_pilot` can produce prompts, but it is high-risk question-overlap debugging only.
- `question_disjoint_formal` is the formal Exp6 source mode.
- Results from `question_disjoint_formal` should be compared against baselines rerun on
  `question_seed42`, not directly against Exp2-Exp5 paper-like metrics.

{md_table(summary["targets"], ["synthetic_plan_id", "target_label_5", "language", "metric_canonical", "error_type"], max_rows=24)}

## Prompt Hardening

Mini-batch prompts ask for a natural, plausible answer candidate and require JSON fields
`answer_synthetic`, `target_label_5`, `error_type`, `rationale_for_label`, and
`expected_failure_against_rubric`. The requested answer text must not mention scoring, hidden
instructions, data creation, experiment design, or answer design.

Dry-run prompt path:

- `{relpath(mini_prompt_path("mini_batch_prompts.jsonl"))}`

Full-score diagnostic prompt templates were added but not executed:

{path_list_for_report([
    Path("thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_score_controlled_answer.md"),
    Path("thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_score_controlled_answer_en.md"),
    Path("thesis_exp/outputs/exp06_synthetic_low_score/generation_plan/prompt_templates/generate_score_controlled_answer_zh.md"),
])}

## Filtering, Leakage, Spotcheck

- Leakage status: **{summary["leakage_status"]}**
- Filter report: `{relpath(mini_table_path("filter_report.csv"))}`
- Leakage report: `{relpath(mini_leakage_path("leakage_report.md"))}`
- Spotcheck samples: `{relpath(mini_spotcheck_path("spotcheck_samples.csv"))}`
- Spotcheck is required before any full generation decision.

## Sanity and Readability

- Sanity failures: **{len(summary["sanity_failures"])}**
- Readability failures: **{len(summary["readability_failures"])}**
- Readability warnings: **{len(summary["readability_warnings"])}**

## Conclusion

Full 384-row generation can start: **NO** until the 24-row mini-batch is actually generated, filtered,
leakage-checked, and manually spotchecked.

Mini-batch generation can start: **{"YES, after prompt/API approval" if summary["prompt_count"] == 24 else "NO"}**

Exp6 training can start: **NO**. This stage only prepares and audits the mini-batch pilot.
"""
    write_text(mini_report_path("mini_batch_generation_report.md"), report)


def write_review_package(summary: dict[str, Any]) -> None:
    generated = summary["raw_count"] > 0
    passed = summary["filtered_count"]
    full_start = "YES" if generated and passed and summary["leakage_status"] == "PASS" else "NO"
    review = f"""# Exp6-3 Mini-batch Review Package

- Can full 384 generation start? **{full_start}**
- Can mini-batch generation start? **{"YES, after human prompt review" if summary["prompt_count"] == 24 else "NO"}**
- Can Exp6 training start? **NO**
- Mode: **{summary["mode"]}**
- Default generation split mode: **{summary["default_generation_split_mode"]}**
- Planned rows: **{len(summary["targets"])}**
- Generated raw rows: **{summary["raw_count"]}**
- Filter-passed rows: **{summary["filtered_count"]}**
- Leakage status: **{summary["leakage_status"]}**
- Spotcheck required: **YES**
- Labels are human-confirmed: **NO**
- Label status: **synthetic_design pseudo-label only**

Blockers:

- Full generation remains blocked until the 24-row mini-batch has real generated answers.
- `paper_like_triple_pilot` is not allowed for training.
- `question_disjoint_formal` can proceed to 24-row mini-batch generation after human review.
- Training remains blocked until generated rows pass filtering, leakage, and manual spotcheck.
- Any dev/test leakage must block affected rows.

Next step:

Review `{relpath(mini_prompt_path("mini_batch_prompts.jsonl"))}`.
If prompts are approved, run question-disjoint mini-batch generation with
`EXP6_RUN_GENERATION=1`, `GENERATION_MODEL`, and a valid key or endpoint, capped at 24 rows. Then
rerun normalization, filtering, leakage, spotcheck, sanity, and readability checks.
"""
    write_text(mini_report_path("mini_batch_review_package.md"), review)


def write_notion_summary(summary: dict[str, Any]) -> None:
    notion = f"""# Exp6-3 Mini-batch Summary

- API called: NO in dry-run
- Synthetic data generated: {"YES" if summary["raw_count"] > 0 else "NO"}
- Mini-batch planned count: {len(summary["targets"])}
- Target label distribution: {summary["label_counts"]}
- Language distribution: {summary["language_counts"]}
- Metric coverage: {summary["metric_coverage"]}
- Error type coverage: {summary["error_type_coverage"]}
- Dry-run prompt path: {relpath(mini_prompt_path("mini_batch_prompts.jsonl"))}
- Leakage status: {summary["leakage_status"]}
- Formal split mode: {summary["default_generation_split_mode"]}
- Mini-batch generation can start: {"YES after approval" if summary["prompt_count"] == 24 else "NO"}
- Full 384 generation can start: NO until mini-batch generation + audit + spotcheck pass
- Exp6 training can start: NO
"""
    write_text(mini_report_path("notion_exp06_mini_batch_summary.md"), notion)


def write_all_reports() -> dict[str, Any]:
    summary = summarize()
    write_report(summary)
    write_review_package(summary)
    write_notion_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--mode", choices=["dry_run", "generate"], default="dry_run")
    parser.add_argument("--max-items", "--max_items", dest="max_items", type=int, default=MINI_BATCH_TOTAL_TARGET)
    args = parser.parse_args()
    ensure_mini_batch_dirs()
    if args.run_pipeline:
        run_pipeline(mode=args.mode, max_items=args.max_items)
    summary = write_all_reports()
    print(
        f"Wrote Exp6-3 mini-batch reports to {mini_report_path('mini_batch_generation_report.md').parent}; "
        f"mode={summary['mode']} planned={len(summary['targets'])} generated={summary['raw_count']} "
        f"passed={summary['filtered_count']}"
    )


if __name__ == "__main__":
    main()
