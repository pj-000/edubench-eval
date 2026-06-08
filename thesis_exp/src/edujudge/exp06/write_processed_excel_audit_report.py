"""Write Exp6-1 processed Excel audit report and review package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06 import PROCESSED_EXCEL_AUDIT_DIR, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.audit_processed_excel_sources import main as run_schema
from thesis_exp.src.edujudge.exp06.build_processed_excel_train_only_candidates import main as run_candidates
from thesis_exp.src.edujudge.exp06.compare_processed_excel_en_zh_merge import main as run_compare
from thesis_exp.src.edujudge.exp06.confirm_processed_excel_label_provenance import main as run_provenance
from thesis_exp.src.edujudge.exp06.processed_excel_common import output_path, read_csv_rows, summarize_low_score
from thesis_exp.src.edujudge.utils.io import md_table, relpath, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


def run_pipeline() -> None:
    run_schema()
    run_compare()
    run_provenance()
    run_candidates()


def int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def summarize() -> dict[str, Any]:
    comparison = read_csv_rows(output_path("processed_excel_source_comparison.csv"))
    schema = read_csv_rows(output_path("processed_excel_schema_profile.csv"))
    provenance = read_csv_rows(output_path("processed_excel_label_provenance.csv"))
    dedup = read_csv_rows(output_path("processed_excel_dedup_report.csv"))
    overlap = read_csv_rows(output_path("processed_excel_train_dev_test_overlap.csv"))
    train = read_csv_rows(output_path("processed_excel_train_only_candidates.csv"))
    low = read_csv_rows(output_path("processed_excel_low_score_candidates.csv"))
    final = read_csv_rows(output_path("processed_excel_final_recommendation.csv"))
    large = read_csv_rows(output_path("processed_excel_exp60_large_detail_crosscheck.csv"))
    merge_equal = any(str(row.get("merge_equals_en_plus_zh", "")).lower() == "true" for row in comparison)
    human_confirmed = all(row.get("provenance_status") == "human_confirmed" for row in provenance) if provenance else False
    any_dev_test = any(str(row.get("any_dev_test_overlap", "")).lower() == "true" for row in overlap)
    can_start = (
        bool(final)
        and all(row.get("usable_for_exp6_training") == "YES" for row in final if row.get("source_file", "").endswith("processed_excel_data_1.jsonl"))
        and human_confirmed
        and not any_dev_test
    )
    return {
        "comparison": comparison,
        "schema": schema,
        "provenance": provenance,
        "dedup": dedup,
        "overlap": overlap,
        "train": train,
        "low": low,
        "final": final,
        "large": large,
        "merge_equal": merge_equal,
        "human_confirmed": human_confirmed,
        "any_dev_test": any_dev_test,
        "can_start": can_start,
        "low_summary": summarize_low_score(low),
        "provenance_counts": Counter(row.get("provenance_status", "unknown") for row in provenance),
    }


def yn(value: bool) -> str:
    return "YES" if value else "NO"


def write_report(summary: dict[str, Any]) -> None:
    low_summary = summary["low_summary"]
    report = f"""# Exp6-1 Processed Excel Label Provenance / Dedup / Train-only Confirmation

## Scope

This audit only inspects `deepseek_output/processed_excel_data_1*`. It does not train models, call
APIs, generate synthetic data, modify Exp0-Exp5 results, or add synthetic rows to any train/dev/test
split.

## Main Findings

- `processed_excel_data_1.jsonl` duplicate of `_en + _zh`: **{yn(summary["merge_equal"])}**
- Labels human-confirmed: **{yn(summary["human_confirmed"])}**
- Any exact dev/test overlap: **{yn(summary["any_dev_test"])}**
- Train-only candidate rows after canonical dedup: **{len(summary["train"])}**
- Low-score train-only candidate rows: **{len(summary["low"])}**
- Can Exp6 training start with processed Excel data now: **{yn(summary["can_start"])}**

## Source Comparison

{md_table(summary["comparison"], ["source_a", "source_b", "records_a", "records_b", "score_rows_a", "score_rows_b", "record_overlap", "question_overlap", "triple_overlap", "qa_overlap", "candidate_overlap", "merge_equals_en_plus_zh", "recommended_non_duplicate_use"], max_rows=10)}

## Schema Profile

{md_table(summary["schema"], ["source_file", "num_records", "score_rows", "top_level_fields", "nested_score_fields", "score_count_distribution", "language_distribution"], max_rows=10)}

## Label Provenance

The files contain `scores[].score` and `scores[].reason`, plus a top-level `model` field. They do
not contain a human/reviewer/annotator marker. Therefore labels are not human-confirmed and should be
treated as model/pseudo labels unless manually traced back to a human-reviewed Excel source.

{md_table(summary["provenance"], ["source_file", "score_fields", "evidence_for_human_label", "evidence_for_model_label", "evidence_for_pseudo_label", "provenance_status", "confidence", "notes"], max_rows=10)}

## Dedup

{md_table(summary["dedup"], ["scope", "source_file", "total_candidate_rows", "unique_candidate_keys", "duplicate_candidate_groups", "duplicate_candidate_rows", "dedup_recommendation"], max_rows=10)}

## Leakage

Exact train/dev/test checks were run using normalized question, question+answer, and
question+answer+metric keys. The candidate source has no exact dev/test overlap after canonical
dedup.

{md_table(summary["final"], ["source_file", "usable_for_exp6_training", "recommended_use", "label_provenance_status", "train_only_rows", "low_score_rows", "duplicate_risk", "leakage_risk", "next_action"], max_rows=10)}

## Low-Score Candidate Summary

```json
{json.dumps(low_summary, ensure_ascii=False, indent=2, sort_keys=True)}
```

## Recommendation

Do **not** start full Exp6 training from `processed_excel_data_1*` yet. The canonical merged source
has train-only rows after exact leakage checks, but only **{len(summary["low"])}** low-score rows and
the labels are not human-confirmed. At most, use it as a tiny pseudo-label pilot after manual
approval. For a meaningful low-score augmentation experiment, obtain or generate additional
train-only low-score data with explicit provenance and no dev/test overlap.

## Files

- `{relpath(output_path("processed_excel_source_comparison.csv"))}`
- `{relpath(output_path("processed_excel_label_provenance.csv"))}`
- `{relpath(output_path("processed_excel_train_only_candidates.csv"))}`
- `{relpath(output_path("processed_excel_low_score_candidates.csv"))}`
- `{relpath(output_path("processed_excel_final_recommendation.csv"))}`
"""
    write_text(PROCESSED_EXCEL_AUDIT_DIR / "report.md", report)


def write_review_package(summary: dict[str, Any]) -> None:
    enough = len(summary["low"]) >= 50
    review = f"""# Exp6-1 Processed Excel Review Package

Can Exp6 training start with processed_excel_data_1*? **{yn(summary["can_start"])}**

Are labels human-confirmed? **{"YES" if summary["human_confirmed"] else "NO"}**

Is processed_excel_data_1.jsonl duplicate of en+zh? **{yn(summary["merge_equal"])}**

How many train-only candidates remain? **{len(summary["train"])}**

How many low-score candidates remain? **{len(summary["low"])}**

Is this enough for a meaningful augmentation experiment? **{yn(enough)}**

Should we use it as pilot only? **YES**, only after manual approval as pseudo-label data.

Do we need to generate new train-only synthetic low-score data? **YES**, if Exp6 needs meaningful
low-score augmentation; this audit did not generate any new data.

## Final Recommendation

{md_table(summary["final"], ["source_file", "usable_for_exp6_training", "recommended_use", "reason", "label_provenance_status", "train_only_rows", "low_score_rows", "duplicate_risk", "leakage_risk", "next_action"], max_rows=10)}

## Required Manual Confirmations

- Trace `scores[].score` back to a human-reviewed Excel process, or keep it marked as model/pseudo label.
- Use either merged file or en+zh split files, never both.
- Keep processed Excel candidates train-only; do not add to dev/test.
- Do not run a full Exp6 mix until there are enough low-score rows and provenance is explicit.
"""
    write_text(PROCESSED_EXCEL_AUDIT_DIR / "review_package.md", review)


def write_notion_summary(summary: dict[str, Any]) -> None:
    notion = f"""# Exp6-1 Processed Excel Summary

- Label provenance status: {dict(summary["provenance_counts"])}
- Human-confirmed labels: {yn(summary["human_confirmed"])}
- Merged equals en+zh: {yn(summary["merge_equal"])}
- Exact dev/test leakage: {yn(summary["any_dev_test"])}
- Train-only candidates after canonical dedup: {len(summary["train"])}
- Low-score candidates: {len(summary["low"])}
- Can Exp6 training start: {yn(summary["can_start"])}
- Recommendation: pilot-only pseudo-label review; not enough for meaningful low-score augmentation.
"""
    write_text(PROCESSED_EXCEL_AUDIT_DIR / "notion_exp06_processed_excel_summary.md", notion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pipeline", action="store_true")
    args = parser.parse_args()
    ensure_exp06_dirs()
    if args.run_pipeline:
        run_pipeline()
    summary = summarize()
    write_report(summary)
    write_review_package(summary)
    write_notion_summary(summary)
    print(f"Wrote processed Excel audit report to {PROCESSED_EXCEL_AUDIT_DIR}")


if __name__ == "__main__":
    main()
