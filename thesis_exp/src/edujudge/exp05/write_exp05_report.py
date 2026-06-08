"""Write Exp5 low-score loss report and review package."""

from __future__ import annotations

import argparse
import csv
import math
from typing import Any

from thesis_exp.src.edujudge.exp05 import EXP05_OUTPUT_DIR, EXP05_TABLES_DIR, L1_RUN_ID, L2_RUN_CONFIGS, ensure_exp05_dirs
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


def status_from_table(path: Any) -> str:
    rows = read_rows(path)
    return overall_status(rows)


def l2_toy_status() -> str:
    rows = read_rows(EXP05_TABLES_DIR / "l2_toy_loss_checks.csv")
    return overall_status(rows)


def l2_formal_status(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("status") or "pending") for row in rows if row.get("loss_id") in L2_RUN_CONFIGS]
    if not statuses:
        return "pending"
    if all(status == "completed" for status in statuses):
        return "completed"
    if any(status == "completed" for status in statuses):
        return "partial"
    return "pending"


def write_review_package(rows: list[dict[str, Any]] | None = None) -> None:
    ensure_exp05_dirs()
    rows = rows if rows is not None else collect_exp05_results()
    setup = status_from_table(EXP05_TABLES_DIR / "sanity_check_exp05_setup.csv")
    readability = status_from_table(EXP05_TABLES_DIR / "readability_check_exp05.csv")
    l1_smoke_rows = read_rows(EXP05_TABLES_DIR / "sanity_check_exp05_outputs_smoke.csv")
    l1_smoke = overall_status(l1_smoke_rows) if l1_smoke_rows else "PENDING"
    l2_smoke_rows = read_rows(EXP05_TABLES_DIR / "sanity_check_exp05_outputs_l2_smoke.csv")
    l2_smoke = overall_status(l2_smoke_rows) if l2_smoke_rows else "PENDING"
    toy = l2_toy_status()
    l1 = next((row for row in rows if row.get("loss_id") == L1_RUN_ID), {})
    l1_status = str(l1.get("status") or "pending")
    l2_status = l2_formal_status(rows)
    can_l1_smoke = setup == "PASS" and readability == "PASS"
    can_l1_formal = can_l1_smoke and l1_smoke == "PASS"
    can_l2_smoke = setup == "PASS" and readability == "PASS" and toy == "PASS"
    can_l2_formal = can_l2_smoke and l2_smoke == "PASS"
    if l2_status == "completed":
        blockers = [
            "none for L2; formal training completed and setup/output/readability checks passed",
            "L3/L4 remain intentionally out of scope until L2 is reviewed",
        ]
    elif can_l2_smoke:
        blockers = [
            "run L2 smoke before formal L2 training",
            "L3/L4 remain intentionally out of scope until L2 is reviewed",
        ]
    else:
        blockers = [
            "fix setup/readability/L2 toy checks before L2 smoke",
            "L3/L4 remain intentionally out of scope until L2 is reviewed",
        ]
    review = f"""# Exp5 Review Package

Can L1 smoke start? **{"YES" if can_l1_smoke else "NO"}**

Can L1 formal training start? **{"YES" if can_l1_formal else "NO"}**

Can L2 smoke start? **{"YES" if can_l2_smoke else "NO"}**

Can L2 formal training start? **{"YES" if can_l2_formal else "NO"}**

Can L3 start? **NO until L2 reviewed**

## Class Weights Summary

{class_weights_table()}

## Status

| item | status |
| --- | --- |
| Setup sanity status | {setup} |
| Readability status | {readability} |
| L2 toy loss check status | {toy} |
| L1 smoke status | {l1_smoke} |
| L1 formal status | {l1_status} |
| L2 smoke status | {l2_smoke} |
| L2 formal status | {l2_status} |

## Main Results

{summary_table(rows)}

## Remaining Blockers

{chr(10).join(f"- {blocker}" for blocker in blockers)}

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
high scores.

## L2 Asymmetric Low-Score Overestimation Loss

L2 tests a different hypothesis: if true low-score samples are predicted too
high, explicitly penalize that overestimation during ordinal training. L2 does
not use class weights, does not change the A4 input, does not change the data
split, and does not add high-score preservation or threshold suppression.

For ordinal probabilities `p_t = sigmoid(z_t)`, the continuous prediction is:

`s_hat = 1 + sum_t p_t`

The L2 penalty is:

`I(y <= 2) * (max(s_hat - y - margin, 0) / 4)^2`

The final loss is:

`mean_i(L_i_ord + lambda_low * P_i_low)`

The first two variants are:

- L2a: `lambda_low=0.3`, `margin=0.0`
- L2b: `lambda_low=0.5`, `margin=0.0`

If L2 lowers `low_to_high_rate` but hurts high-score metrics such as `Acc@5`
or `high_to_mid_or_low_rate`, the follow-up L4 experiment should test high-score
preservation.

## Class Weights

{class_weights_table()}

## Main Results

{summary_table(rows)}

## Output Files

- Summary table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_summary.csv")}`
- Low-score table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_low_score.csv")}`
- Per-bin table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_per_bin.csv")}`
- Delta table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_delta_vs_L0.csv")}`
- Delta vs L1 table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_delta_vs_L1.csv")}`
- Tradeoff table: `{relpath(EXP05_TABLES_DIR / "loss_ablation_tradeoff.csv")}`
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
    write_text(
        EXP05_OUTPUT_DIR / "notion_exp05_l2_summary.md",
        f"""# Exp5 L2 Notion Summary

问题：在不改变 A4 输入、不改变 ordinal objective、不使用 class weights 的前提下，
如果专门惩罚真实低分样本被预测得过高，能否进一步降低 low-to-high rate？

L2 和 L1 的区别：

- L1 处理类别不均衡，使用 class weights。
- L2 不使用 class weights，只加入低分高估非对称惩罚。
- L2a 使用 lambda_low=0.3，margin=0.0。
- L2b 使用 lambda_low=0.5，margin=0.0。

当前汇总：

{summary_table(rows)}

解读要点：

- 如果 L2 的 low_to_high_rate 低于 L1，说明方向性惩罚有额外作用。
- 如果 L2 同时损伤 Acc@5 或 high-score 指标，后续需要 L4 high-score preservation。
- L3/L4 暂不实现，等 L2 结果审阅后再做。
""",
    )
    write_review_package(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Exp5 low-score loss ablation report.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = write_exp05_report()
    print(f"Exp5 report written for {len(rows)} rows.")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / 'report.md')}")


if __name__ == "__main__":
    main()
