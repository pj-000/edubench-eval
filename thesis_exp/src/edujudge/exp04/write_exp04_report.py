"""Write Exp4 report and review package."""

from __future__ import annotations

import argparse
import csv
import math
from typing import Any

from thesis_exp.src.edujudge.exp04 import EXP04_OUTPUT_DIR, EXP04_TABLES_DIR, OBJECTIVE_IDS, ensure_exp04_dirs
from thesis_exp.src.edujudge.exp04.collect_exp04_results import collect_exp04_results
from thesis_exp.src.edujudge.utils.io import relpath, write_text


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def status_mark(status: str) -> str:
    return "PASS" if status in {"completed", "reused_exp03_a4", "eval_only"} else "PENDING"


def read_status_csv(path: Any) -> list[dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def overall_from_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "PENDING"
    return "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL"


def objective_setup_status(rows: list[dict[str, str]], objective_id: str) -> str:
    if objective_id == "O1_classification":
        checks = ["O1 can reuse Exp3 A4 artifacts"]
    elif objective_id == "O2_regression_smoothl1":
        checks = ["train O2 human_mean_5 target", "dev O2 human_mean_5 target", "test O2 human_mean_5 target"]
    else:
        checks = ["train O3 ordinal target shape", "dev O3 ordinal target shape", "test O3 ordinal target shape"]
    relevant = [row for row in rows if row.get("check") in checks]
    if not relevant:
        return "PENDING"
    return "PASS" if all(row.get("status") == "PASS" for row in relevant) else "FAIL"


def summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| objective | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | severe_error_rate | low_to_high_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['objective_id']} | {row['status']} | {fmt(row.get('test_accuracy'))} | "
            f"{fmt(row.get('test_MAE_label'))} | {fmt(row.get('test_MAE_expected'))} | "
            f"{fmt(row.get('test_qwk'))} | {fmt(row.get('test_kendall_tau'))} | "
            f"{fmt(row.get('test_severe_error_rate'))} | {fmt(row.get('test_low_to_high_rate'))} |"
        )
    return "\n".join(lines)


def write_review_package(rows: list[dict[str, Any]] | None = None) -> None:
    ensure_exp04_dirs()
    rows = rows if rows is not None else collect_exp04_results()
    setup_rows = read_status_csv(EXP04_TABLES_DIR / "sanity_check_exp04_setup.csv")
    readability_rows = read_status_csv(EXP04_TABLES_DIR / "readability_check_exp04.csv")
    setup_status = overall_from_rows(setup_rows)
    readability_status = overall_from_rows(readability_rows)
    can_smoke = setup_status == "PASS" and readability_status == "PASS"
    blockers = []
    if setup_status != "PASS":
        blockers.append(f"setup sanity is {setup_status}")
    if readability_status != "PASS":
        blockers.append(f"readability is {readability_status}")
    if not blockers:
        blockers.append("none for server smoke; formal O2/O3 remains gated on smoke PASS")

    status_by_objective = {row.get("objective_id"): row.get("status") for row in rows}
    setup_by_objective = {objective_id: objective_setup_status(setup_rows, objective_id) for objective_id in OBJECTIVE_IDS}
    review = f"""# Exp4 Review Package

Can server smoke test start? **{"YES" if can_smoke else "NO"}**

Can formal O2/O3 training start? **NO until smoke PASS**

## Setup Status

| item | status | notes |
| --- | --- | --- |
| O1 reuse status | {setup_by_objective["O1_classification"]} | run status: {status_by_objective.get("O1_classification", "pending")} |
| O2 regression setup status | {setup_by_objective["O2_regression_smoothl1"]} | uses `human_mean_5`, SmoothL1Loss |
| O3 ordinal setup status | {setup_by_objective["O3_ordinal"]} | uses four BCE threshold labels |
| readability status | {readability_status} | `readability_check_exp04` |
| setup sanity status | {setup_status} | `sanity_check_exp04_setup` |

## Main Results

{summary_table(rows)}

## Remaining Blockers

{chr(10).join(f"- {blocker}" for blocker in blockers)}

## Files

- `{relpath(EXP04_OUTPUT_DIR / "report.md")}`
- `{relpath(EXP04_OUTPUT_DIR / "sanity_check_exp04_setup.md")}`
- `{relpath(EXP04_OUTPUT_DIR / "readability_check_exp04.md")}`
- `{relpath(EXP04_TABLES_DIR / "target_objective_summary.csv")}`
"""
    write_text(EXP04_OUTPUT_DIR / "review_package.md", review)


def pick_best(rows: list[dict[str, Any]], key: str, lower_is_better: bool) -> str:
    candidates = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            candidates.append((number, row["objective_id"]))
    if not candidates:
        return "NA"
    value, objective_id = min(candidates) if lower_is_better else max(candidates)
    return f"{objective_id} ({fmt(value)})"


def write_exp04_report() -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    rows = collect_exp04_results()
    completed = [row for row in rows if row.get("status") in {"completed", "reused_exp03_a4", "eval_only"}]
    all_ready = len(completed) == len(rows)

    report = f"""# Exp4 Target Objective Comparison

Exp4 fixes the Exp3 A4 input template and compares how the 1-5 educational
score label should be modeled during training.

Exp4 is not another input ablation. The input text is fixed to:

`question + answer + metric + rubric + metadata`

The compared objectives are:

- O1 classification: reuse Exp3 A4 5-class CE result.
- O2 regression: train one continuous-score head with SmoothL1Loss on `human_mean_5`.
- O3 ordinal classification: train four threshold logits with BCEWithLogitsLoss.

## Main Results

{summary_table(rows)}

## Current Best

- Best MAE_label: {pick_best(rows, "test_MAE_label", lower_is_better=True)}
- Best MAE_expected: {pick_best(rows, "test_MAE_expected", lower_is_better=True)}
- Best Accuracy: {pick_best(rows, "test_accuracy", lower_is_better=False)}
- Lowest severe_error_rate: {pick_best(rows, "test_severe_error_rate", lower_is_better=True)}
- Lowest low_to_high_rate: {pick_best(rows, "test_low_to_high_rate", lower_is_better=True)}

## Interpretation Guide

Classification treats 1, 2, 3, 4, and 5 as unrelated classes. It is a strong
baseline but does not encode that confusing 2 with 3 is less severe than
confusing 2 with 5.

Regression treats the target as a continuous human mean score. The first round
uses SmoothL1Loss because it is less dominated by outlier residuals than MSE
while still optimizing distance in score space.

Ordinal classification treats the score as an ordered level. It predicts four
threshold events: score greater than 1, 2, 3, and 4. The first round does not
force monotonic correction; it reports monotonic violation rate so that a
follow-up correction is only added if violations are material.

## Output Files

- Summary table: `{relpath(EXP04_TABLES_DIR / "target_objective_summary.csv")}`
- Low-score table: `{relpath(EXP04_TABLES_DIR / "target_objective_low_score.csv")}`
- Per-bin table: `{relpath(EXP04_TABLES_DIR / "target_objective_per_bin.csv")}`
- Prediction copies: `{relpath(EXP04_OUTPUT_DIR / "predictions")}`
- Array copies: `{relpath(EXP04_OUTPUT_DIR / "arrays")}`
"""
    write_text(EXP04_OUTPUT_DIR / "report.md", report)

    write_review_package(rows)

    notion_summary = f"""# Exp4 Notion Summary

Question: with A4 input fixed, should 1-5 labels be modeled as classification,
regression, or ordinal classification?

{summary_table(rows)}

Recommended reading:

- Use Accuracy for exact label match.
- Use MAE_label and MAE_expected for score-distance errors.
- Use severe_error_rate and low_to_high_rate for educational risk.
"""
    write_text(EXP04_OUTPUT_DIR / "notion_exp04_summary.md", notion_summary)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Exp4 report and review package.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = write_exp04_report()
    print(f"Exp4 report written for {len(rows)} objectives.")
    print(f"Output: {relpath(EXP04_OUTPUT_DIR / 'report.md')}")


if __name__ == "__main__":
    main()
