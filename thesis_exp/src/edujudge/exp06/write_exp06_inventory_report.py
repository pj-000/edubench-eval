"""Write Exp6 synthetic low-score inventory report and review package."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_OUTPUT_DIR, EXP06_TABLES_DIR, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.analyze_synthetic_distribution import main as run_distribution
from thesis_exp.src.edujudge.exp06.check_synthetic_leakage import main as run_leakage
from thesis_exp.src.edujudge.exp06.inventory_synthetic_sources import main as run_inventory
from thesis_exp.src.edujudge.exp06.normalize_synthetic_candidates import main as run_normalize
from thesis_exp.src.edujudge.exp06.profile_synthetic_schema import main as run_profile
from thesis_exp.src.edujudge.exp06.common import read_csv_rows
from thesis_exp.src.edujudge.utils.io import md_table, relpath, write_text


def int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def run_pipeline() -> None:
    run_inventory()
    run_profile()
    run_normalize()
    run_leakage()
    run_distribution()


def summarize() -> dict[str, Any]:
    inventory = read_csv_rows(EXP06_TABLES_DIR / "synthetic_source_inventory.csv")
    candidates = read_csv_rows(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv")
    leakage = read_csv_rows(EXP06_TABLES_DIR / "synthetic_leakage_summary.csv")
    recs = read_csv_rows(EXP06_TABLES_DIR / "synthetic_filter_recommendation.csv")
    role_counts = Counter(row.get("likely_role", "unknown") for row in inventory)
    risk_counts = Counter(row.get("risk_level", "unknown") for row in inventory)
    total_labels = sum(1 for row in candidates if row.get("target_label_5"))
    total_low = sum(1 for row in candidates if row.get("target_label_5") in {"1", "2"})
    error_rows = sum(1 for row in candidates if row.get("error_type"))
    dev_test_leak_sources = [row for row in leakage if str(row.get("any_dev_test_leakage", "")).lower() == "true"]
    possible = [row for row in recs if row.get("recommended_use") == "POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION"]
    high_risk = [row for row in recs if row.get("recommended_use") == "HIGH_RISK_REVIEW_ONLY"]
    blocked = [row for row in recs if row.get("recommended_use") == "BLOCKED_OR_REVIEW_ONLY"]
    return {
        "inventory": inventory,
        "candidates": candidates,
        "leakage": leakage,
        "recommendations": recs,
        "role_counts": role_counts,
        "risk_counts": risk_counts,
        "total_candidates": len(candidates),
        "total_labels": total_labels,
        "total_low": total_low,
        "error_rows": error_rows,
        "dev_test_leak_sources": dev_test_leak_sources,
        "possible": possible,
        "high_risk": high_risk,
        "blocked": blocked,
    }


def status_yes_no(condition: bool) -> str:
    return "YES" if condition else "NO"


def write_report(summary: dict[str, Any]) -> None:
    possible = summary["possible"]
    blocked = summary["blocked"]
    high_risk = summary["high_risk"]
    dev_test_leak = summary["dev_test_leak_sources"]
    can_start = bool(possible) and not dev_test_leak

    role_rows = [{"likely_role": key, "num_sources": value} for key, value in summary["role_counts"].most_common()]
    risk_rows = [{"risk_level": key, "num_sources": value} for key, value in summary["risk_counts"].most_common()]

    report = f"""# Exp6 Synthetic Low-Score Data Inventory / Audit

## Scope

This audit scans existing synthetic, sampled, augmented, model-judge, and synthesis-script artifacts
only. It does not train models, call APIs, generate synthetic data, modify Exp0-Exp5 results, or add
synthetic rows to train/dev/test.

## Source Inventory

Inventory rows: **{len(summary["inventory"])}**

{md_table(role_rows, ["likely_role", "num_sources"], max_rows=20)}

{md_table(risk_rows, ["risk_level", "num_sources"], max_rows=20)}

## Normalized Candidate Overview

Normalized audit candidate rows: **{summary["total_candidates"]}**

Rows with `target_label_5`: **{summary["total_labels"]}**

Low-score candidate rows (`target_label_5` in 1/2): **{summary["total_low"]}**

Rows with `error_type`: **{summary["error_rows"]}**

## Required Questions

1. What synthetic / sampled sources exist?

The requested roots were scanned recursively where applicable. See
`tables/synthetic_source_inventory.csv` and `tables/synthetic_schema_profile.csv`. Main categories
are sampled SFT wrappers (`sampled_merge_*`, `human_sampled_eval_sft_criteria_test.json`),
`edu-data-synthesis-main` synthesis/eval data and scripts, `deepseek_output` / `qwen_output`, and
model judge outputs such as `groupby_metric_*`, `merge_model_metric.jsonl`, and
`deepseek-r1_merged.jsonl`.

2. Which can be used for Exp6?

Only sources listed as `POSSIBLE_FILTERED_TRAIN_ONLY_AFTER_MANUAL_CONFIRMATION` in
`tables/synthetic_filter_recommendation.csv` are possible candidates, and only for train-side
augmentation after manual label provenance review. Synthetic data must never be used as dev/test.

{md_table(possible, ["source_file", "candidate_rows", "target_label_5_rows", "low_score_rows", "leakage_risk", "recommended_use"], max_rows=20)}

3. Which cannot be used directly?

`groupby_metric_*_eval_*`, `merge_model_metric.jsonl`, and `deepseek-r1_merged.jsonl` are
model/judge outputs, not human labels. `sampled_merge_50_new.json` and
`sampled_merge_50_new_swift.json` are default HIGH risk. `human_sampled_eval_sft_criteria_test.json`
is treated as a test-style SFT sample and is not a direct train/dev/test source.

{md_table(blocked + high_risk, ["source_file", "likely_role", "candidate_rows", "low_score_rows", "recommended_use", "blocked_reasons"], max_rows=30)}

4. Is there dev/test leakage risk?

Dev/test leakage source count: **{len(dev_test_leak)}**. Exact details are in
`tables/synthetic_leakage_details.csv`. Any source with dev/test overlap is blocked until manual
review removes or quarantines the overlapping rows.

{md_table(dev_test_leak, ["source_file", "total_candidates", "question_key_in_dev", "question_key_in_test", "triple_key_in_dev", "triple_key_in_test", "normalized_qa_in_dev", "normalized_qa_in_test", "leakage_risk"], max_rows=30)}

5. Is `target_label_5` available?

Yes for rows where an official EduBench metric score could be parsed. These labels are still not
assumed human labels; model/judge/SFT provenance remains recorded in `normalization_status`.

6. Is `error_type` available?

Only sparsely. `error_type` is available for **{summary["error_rows"]}** normalized rows.

7. Can we do filtered synthetic?

{status_yes_no(bool(possible))}, as a candidate design only: filter to train-only rows, remove all
dev/test overlaps, require `target_label_5`, prefer low-score rows, block judge-only sources, and
manually confirm label provenance.

8. Can we do synthetic-only?

Not as a thesis-quality final model yet. A synthetic-only diagnostic may be run only after manual
confirmation and must still evaluate on unchanged human dev/test; it should be labeled diagnostic,
not comparable as a replacement for human training.

9. Can we do human + synthetic mix?

Potentially yes after filtering and manual confirmation. The human dev/test split must remain
unchanged; synthetic rows can enter train only.

10. Recommended next training matrix

Training should not start until manual confirmations are done. Proposed first matrix after approval:

| run | train data | synthetic filter | dev/test |
| --- | --- | --- | --- |
| E6-H0 | existing human train only | none | unchanged human dev/test |
| E6-F1 | human train + filtered synthetic low-score | no dev/test overlap, `target_label_5` in 1/2, non-judge source | unchanged human dev/test |
| E6-F2 | human train + filtered synthetic low-score + matched metric cap | F1 plus per-metric cap to avoid distribution distortion | unchanged human dev/test |
| E6-D1 | synthetic-only diagnostic | same filter as F1 | unchanged human dev/test |

## Artifact Index

- `{relpath(EXP06_TABLES_DIR / "synthetic_source_inventory.csv")}`
- `{relpath(EXP06_TABLES_DIR / "synthetic_schema_profile.csv")}`
- `{relpath(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv")}`
- `{relpath(EXP06_TABLES_DIR / "synthetic_leakage_summary.csv")}`
- `{relpath(EXP06_TABLES_DIR / "synthetic_filter_recommendation.csv")}`
"""
    write_text(EXP06_OUTPUT_DIR / "report.md", report)


def write_review_package(summary: dict[str, Any]) -> None:
    possible = summary["possible"]
    blocked = summary["blocked"]
    high_risk = summary["high_risk"]
    dev_test_leak = summary["dev_test_leak_sources"]
    can_start = bool(possible) and not dev_test_leak
    review = f"""# Exp6 Review Package

Can Exp6 training start? **{status_yes_no(can_start)}**

## Recommended Usable Sources

{md_table(possible, ["source_file", "candidate_rows", "target_label_5_rows", "low_score_rows", "leakage_risk", "recommended_use"], max_rows=30)}

## Blocked Sources

{md_table(blocked + high_risk, ["source_file", "likely_role", "candidate_rows", "blocked_reasons"], max_rows=40)}

## Leakage Status

Dev/test leakage source count: **{len(dev_test_leak)}**.

{md_table(dev_test_leak, ["source_file", "total_candidates", "leakage_risk", "notes"], max_rows=30)}

## Label Reliability Status

Labels are **not accepted as human labels by default**. Parsed `target_label_5` exists for
**{summary["total_labels"]}** rows, but provenance is model/SFT/synthetic unless manually confirmed.

## Error Type Availability

Rows with `error_type`: **{summary["error_rows"]}**.

## Proposed First Training Matrix

1. E6-H0: human-only reference, unchanged dev/test.
2. E6-F1: human + filtered low-score synthetic, train only.
3. E6-F2: human + filtered low-score synthetic with per-metric and per-language caps.
4. E6-D1: synthetic-only diagnostic, never as final replacement.

## Required Manual Confirmations

- Confirm which non-judge sources have trustworthy label provenance.
- Remove or quarantine every dev/test overlap listed in leakage details.
- Confirm `sampled_merge_*` labels and source-question provenance before any train-only use.
- Confirm no synthetic rows are introduced into dev/test.
- Confirm synthetic mix ratio and per-metric caps before training.
"""
    write_text(EXP06_OUTPUT_DIR / "review_package.md", review)


def write_notion_summary(summary: dict[str, Any]) -> None:
    possible = summary["possible"]
    dev_test_leak = summary["dev_test_leak_sources"]
    can_start = bool(possible) and not dev_test_leak
    notion = f"""# Exp6 Synthetic Inventory Summary

- Inventory complete: YES
- Normalized candidate rows: {summary["total_candidates"]}
- Rows with target_label_5: {summary["total_labels"]}
- Low-score rows: {summary["total_low"]}
- Rows with error_type: {summary["error_rows"]}
- Dev/test leakage sources: {len(dev_test_leak)}
- Can Exp6 training start: {status_yes_no(can_start)}
- Recommended source count: {len(possible)}
- Rule: synthetic can be train-only after filtering and manual confirmation; never dev/test.
"""
    write_text(EXP06_OUTPUT_DIR / "notion_exp06_inventory_summary.md", notion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pipeline", action="store_true", help="run inventory/profile/normalize/leakage/distribution before writing reports")
    args = parser.parse_args()
    ensure_exp06_dirs()
    if args.run_pipeline:
        run_pipeline()
    summary = summarize()
    write_report(summary)
    write_review_package(summary)
    write_notion_summary(summary)
    print(f"Wrote Exp6 report package to {EXP06_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
