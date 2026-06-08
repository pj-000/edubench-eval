"""Write Exp5 L1 report and review package."""

from __future__ import annotations

import argparse
import csv
import math
from typing import Any

from thesis_exp.src.edujudge.exp05 import EXP05_OUTPUT_DIR, EXP05_TABLES_DIR, L1_RUN_ID, ensure_exp05_dirs
from thesis_exp.src.edujudge.exp05.collect_exp05_results import collect_exp05_results
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


def read_rows(path: Any) -> list[dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def overall_status(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "PENDING"
    return "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL"


def summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| loss | status | Accuracy | MAE_label | MAE_expected | QWK | Kendall tau | low_to_high_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('loss_id')} | {row.get('status')} | {fmt(row.get('test_accuracy'))} | "
            f"{fmt(row.get('test_MAE_label'))} | {fmt(row.get('test_MAE_expected'))} | "
            f"{fmt(row.get('test_qwk'))} | {fmt(row.get('test_kendall_tau'))} | "
            f"{fmt(row.get('test_low_to_high_rate'))} |"
        )
    return "\n".join(lines)


def class_weights_table() -> str:
    rows = read_rows(EXP05_TABLES_DIR / "class_weights.csv")
    if not rows:
        return "_Class weights have not been generated yet._"
    lines = [
        "| label_5 | train_count | raw_weight | clipped_weight |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('label_5')} | {row.get('train_count')} | "
            f"{fmt(row.get('raw_weight'))} | {fmt(row.get('clipped_weight'))} |"
        )
    return "\n".join(lines)


def write_review_package(rows: list[dict[str, Any]] | None = None) -> None:
    ensure_exp05_dirs()
    rows = rows if rows is not None else collect_exp05_results()
    setup = overall_status(read_rows(EXP05_TABLES_DIR / "sanity_check_exp05_setup.csv"))
    readability = overall_status(read_rows(EXP05_TABLES_DIR / "readability_check_exp05.csv"))
    smoke_rows = read_rows(EXP05_TABLES_DIR / "sanity_check_exp05_outputs_smoke.csv")
    smoke = overall_status(smoke_rows) if smoke_rows else "PENDING"
    l1 = next((row for row in rows if row.get("loss_id") == L1_RUN_ID), {})
    l1_status = str(l1.get("status") or "pending")
    can_smoke = setup == "PASS" and readability == "PASS"
    can_formal = can_smoke and smoke == "PASS"
    review = f"""# Exp5 L1 Review Package

Can L1 smoke start? **{"YES" if can_smoke else "NO"}**

Can L1 formal training start? **{"YES" if can_formal else "NO"}**

Can L2 start? **NO until L1 reviewed**

## Class Weights Summary

{class_weights_table()}

## Status

| item | status |
| --- | --- |
| Setup sanity status | {setup} |
| Readability status | {readability} |
| Smoke status | {smoke} |
| Formal training status | {l1_status} |

## Main Results

{summary_table(rows)}

## Remaining Blockers

- L1 formal training should start only after smoke output sanity passes.
- L2/L3/L4 are intentionally out of scope for this patch.

## Files

- `{relpath(EXP05_OUTPUT_DIR / "report.md")}`
- `{relpath(EXP05_OUTPUT_DIR / "sanity_check_exp05_setup.md")}`
- `{relpath(EXP05_OUTPUT_DIR / "readability_check_exp05.md")}`
- `{relpath(EXP05_TABLES_DIR / "loss_ablation_summary.csv")}`
"""
    write_text(EXP05_OUTPUT_DIR / "review_package.md", review)


def write_exp05_report() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    rows = collect_exp05_results()
    report = f"""# Exp5 Low-Score Loss Ablation

Exp5 studies whether the low-score overestimation problem is partly caused by
class imbalance in the train split.

Exp4 already selected A4 plus ordinal classification as the strongest overall
scoring setup. However, the Exp4 O3 ordinal baseline still overestimates some
low-score samples. L1 is a control experiment for class imbalance, not a new
main method.

L1 keeps the input, objective, data split, model backbone, and checkpoint
selection comparable to Exp4 O3. The only change is the sample weight:

`weight_c = clip(N / (5 * N_c), 0.5, 3.0)`

The weighted loss is:

`sum_i(w_i * L_i_ord) / sum_i(w_i)`

L1 does not explicitly punish the direction where low scores are predicted as
high scores. That asymmetric error will be studied later in L2/L3.

## Class Weights

{class_weights_table()}

## Main Results

{summary_table(rows)}

## Output Files

- Summary table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_summary.csv")}`
- Low-score table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_low_score.csv")}`
- Per-bin table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_per_bin.csv")}`
- Delta table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_delta_vs_L0.csv")}`
"""
    write_text(EXP05_OUTPUT_DIR / "report.md", report)
    write_text(
        EXP05_OUTPUT_DIR / "notion_exp05_l1_summary.md",
        f"""# Exp5 L1 Notion Summary

Question: are low-score failures mainly caused by label imbalance?

L1 reuses Exp4 O3 ordinal as L0 and trains only a weighted ordinal variant.

{summary_table(rows)}

Class weights come from the train split only.

{class_weights_table()}
""",
    )
    write_review_package(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Exp5 L1 report.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = write_exp05_report()
    print(f"Exp5 report written for {len(rows)} rows.")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / 'report.md')}")


if __name__ == "__main__":
    main()
