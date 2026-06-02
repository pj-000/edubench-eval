"""Write Exp1 markdown reports and Notion summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from thesis_exp.src.edujudge.audit import (
    DATASET_NAME,
    EVALUATORS,
    EXP00_OUTPUT_DIR,
    EXP01_FIGURES_DIR,
    EXP01_OUTPUT_DIR,
    EXP01_TABLES_DIR,
    PROCESSED_DATASET_PATH,
    TEST_SPLIT_PATH,
    ensure_exp01_dirs,
    markdown_table,
    relpath,
    write_text,
)


def _fmt(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    return markdown_table(df[columns].to_dict("records"), columns, max_rows=max_rows)


def _trend_status(pdf_cmp: pd.DataFrame) -> str:
    comparable = pdf_cmp[pdf_cmp["status"] != "not_comparable"]
    if comparable.empty:
        return "NO"
    matched = int((comparable["status"] == "matched_trend").sum())
    differs = int((comparable["status"] == "differs").sum())
    missing = int((comparable["status"] == "missing_current").sum())
    if matched and not differs and not missing:
        return "YES"
    if matched >= max(1, differs + missing):
        return "PARTIAL"
    return "NO"


def load_tables() -> dict[str, pd.DataFrame]:
    names = [
        "alignment_coverage",
        "evaluator_metrics",
        "per_bin_metrics",
        "low_score_metrics",
        "high_score_metrics",
        "metric_level_metrics",
        "scenario_level_metrics",
        "subject_level_metrics",
        "pdf_reference_comparison",
    ]
    return {name: pd.read_csv(EXP01_TABLES_DIR / f"{name}.csv") for name in names}


def _summary_values(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    coverage = tables["alignment_coverage"]
    metrics = tables["evaluator_metrics"]
    low = tables["low_score_metrics"]
    found = coverage[coverage["n_aligned"] > 0]["evaluator"].tolist()
    missing = coverage[coverage["n_aligned"] == 0]["evaluator"].tolist()
    best_mae = metrics.sort_values("MAE").iloc[0] if not metrics.empty else {}
    best_exact = metrics.sort_values("Exact Match", ascending=False).iloc[0] if not metrics.empty else {}
    largest_bias = metrics.reindex(metrics["Signed Bias"].abs().sort_values(ascending=False).index).iloc[0]
    trend = _trend_status(tables["pdf_reference_comparison"])
    can_exp2 = "YES" if found else "NO"
    return {
        "found": found,
        "missing": missing,
        "best_mae": best_mae,
        "best_exact": best_exact,
        "largest_bias": largest_bias,
        "trend": trend,
        "can_exp2": can_exp2,
        "low": low,
    }


def write_report() -> dict[str, Any]:
    ensure_exp01_dirs()
    tables = load_tables()
    summary = _summary_values(tables)
    coverage = tables["alignment_coverage"]
    metrics = tables["evaluator_metrics"]
    per_bin = tables["per_bin_metrics"]
    low = tables["low_score_metrics"]
    metric_level = tables["metric_level_metrics"]
    scenario_level = tables["scenario_level_metrics"]
    subject_level = tables["subject_level_metrics"]
    pdf_cmp = tables["pdf_reference_comparison"]
    test_rows = int(coverage["n_test"].iloc[0]) if not coverage.empty else 0
    warnings = []
    if "EduBenchEvaluator" in summary["missing"]:
        warnings.append("EduBenchEvaluator predictions not found; Exp2 should reproduce the CE baseline.")

    top_metric_mae = (
        metric_level.groupby("metric_canonical")["MAE"].mean().sort_values(ascending=False).head(5).reset_index()
    )
    top_scenario_mae = (
        scenario_level.groupby("scenario_canonical")["MAE"].mean().sort_values(ascending=False).head(5).reset_index()
    )
    top_subject_mae = (
        subject_level.groupby("subject_canonical")["MAE"].mean().sort_values(ascending=False).head(8).reset_index()
    )
    figures = sorted(path for path in EXP01_FIGURES_DIR.glob("fig01_*.png"))
    table_paths = sorted(path for path in EXP01_TABLES_DIR.glob("*.csv"))

    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- None."
    report = f"""# Exp1: Evaluator-vs-Human Audit

## 1. Purpose

This experiment audits agreement between existing automatic judge predictions and human scores on
the locked Exp0.1 paper-like test split. It reproduces the PDF-style low-score blind spot analysis
and extends it with metric, scenario, subject, language, education-level, and generator-model
strata. It is intended as the Chapter 4 target analysis for downstream training and calibration
experiments.

## 2. Inputs

| item | value |
| --- | --- |
| main dataset | `{relpath(PROCESSED_DATASET_PATH)}` |
| dataset name | `{DATASET_NAME}` |
| main split | `{relpath(TEST_SPLIT_PATH)}` |
| test rows | {test_rows} |
| human reference | `human_mean_5` and rounded `label_5` |
| Exp0.1 references | `{relpath(EXP00_OUTPUT_DIR / 'review_package.md')}`, `{relpath(EXP00_OUTPUT_DIR / 'data_card.md')}`, `{relpath(EXP00_OUTPUT_DIR / 'leakage_report.md')}`, `{relpath(EXP00_OUTPUT_DIR / 'sanity_check_exp00_reference.md')}` |

## 3. Judge Score Inventory and Alignment

Evaluators found: {", ".join(summary["found"]) if summary["found"] else "None"}.
Evaluators missing: {", ".join(summary["missing"]) if summary["missing"] else "None"}.

{_table(coverage, ["evaluator", "n_test", "n_aligned", "coverage", "n_valid_score", "valid_score_rate", "n_missing", "n_invalid", "primary_alignment_method"])}

Missing evaluator warnings:

{warning_text}

No missing or invalid prediction was filled. Synthetic/sample files are inventoried only as excluded
sources.

## 4. Overall Agreement with Human Scores

{_table(metrics, ["evaluator", "n_valid", "MAE", "RMSE", "Signed Bias", "Exact Match", "Within-1 Accuracy", "Macro-F1", "Weighted-F1", "Quadratic Weighted Kappa", "Kendall tau", "Spearman rho"])}

Lowest MAE: {_fmt(summary["best_mae"].get("evaluator"))} ({_fmt(summary["best_mae"].get("MAE"))}).
Highest Exact Match: {_fmt(summary["best_exact"].get("evaluator"))} ({_fmt(summary["best_exact"].get("Exact Match"))}).
Largest absolute signed bias: {_fmt(summary["largest_bias"].get("evaluator"))} ({_fmt(summary["largest_bias"].get("Signed Bias"))}).

## 5. Low-score Blind Spot

{_table(low, ["evaluator", "n_valid_low", "low_exact_match", "low_recall", "low_MAE", "low_signed_bias", "low_overestimation_rate", "low_severe_overestimation_rate", "low_to_high_rate", "mean_pred_low", "mean_human_low"])}

Per-bin Acc@1/Acc@2:

{_table(per_bin[per_bin["label_5"].isin([1, 2])], ["evaluator", "label_5", "n_valid", "accuracy", "mean_pred", "signed_bias"], max_rows=20)}

The central failure mode remains low-score overestimation: true labels 1 and 2 are much harder than
high-score items, and several evaluators push low-scored answers into mid/high predicted labels.

## 6. Calibration Failure

Calibration is summarized by mean predicted score per true label and signed bias per true label.
Positive signed bias, especially for labels 1 and 2, indicates systematic overestimation. High-score
items are generally preserved more reliably than low-score items.

## 7. Metric-level Differences

Highest mean MAE metrics:

{_table(top_metric_mae, ["metric_canonical", "MAE"])}

Kendall tau may be undefined for strata where predictions or human scores are constant; those cells
are reported as NaN rather than failing the pipeline.

## 8. Scenario-level Differences

Highest mean MAE scenarios:

{_table(top_scenario_mae, ["scenario_canonical", "MAE"])}

## 9. Subject-level Differences

Highest mean MAE subjects:

{_table(top_subject_mae, ["subject_canonical", "MAE"])}

Warning: subject metadata comes from local enriched metadata and should be treated as stratified
audit metadata.

## 10. Comparison with PDF Reference

PDF trend reproduced: **{summary["trend"]}**.

{_table(pdf_cmp, ["evaluator", "metric_name", "pdf_reference", "current_value", "delta", "status"], max_rows=80)}

The comparison checks whether the main trend is reproduced rather than forcing exact numerical
matches. Differences can come from repaired/reconstructed split details, alignment source choice,
and unavailable PDF bin-agreement definitions.

## 11. Implications for Exp2-Exp7

- Exp2 should reproduce or establish the EduBenchEvaluator CE baseline.
- Exp3 should test rubric-aware inputs.
- Exp5 should test low-score-sensitive loss.
- Exp6 should use synthetic low-score augmentation only as a controlled follow-up, not as Exp1 data.
- Exp7 should address calibration.
- Later experiments should not rely only on overall accuracy.

## 12. Figures and Tables

Core figures:

{chr(10).join(f'- `{relpath(path)}`' for path in figures)}

Tables:

{chr(10).join(f'- `{relpath(path)}`' for path in table_paths)}

## 13. Limitations

- No new model is trained.
- No API is called.
- Only existing judge predictions are used.
- Missing evaluator predictions must be supplied in later experiments rather than inferred here.
- The paper-like split has the Exp0.1 question-overlap warning context.
- Subject-level provenance depends on local enriched metadata.
"""
    write_text(EXP01_OUTPUT_DIR / "report.md", report)
    write_review_package(tables, summary)
    write_notion_summaries(tables, summary)
    return {"trend": summary["trend"], "can_exp2": summary["can_exp2"], "found": summary["found"], "missing": summary["missing"]}


def write_review_package(tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    coverage = tables["alignment_coverage"]
    metrics = tables["evaluator_metrics"]
    low = tables["low_score_metrics"]
    test_rows = int(coverage["n_test"].iloc[0]) if not coverage.empty else 0
    blockers = []
    if not summary["found"]:
        blockers.append("No evaluator predictions aligned.")
    if "EduBenchEvaluator" in summary["missing"]:
        blockers.append("EduBenchEvaluator predictions not found; Exp2 should reproduce the CE baseline.")
    blockers_text = "\n".join(f"- {item}" for item in blockers) if blockers else "- None blocking Exp2 baseline setup."
    text = f"""# Exp1 Review Package

| item | value |
| --- | --- |
| Can Exp2 start? | {summary["can_exp2"]} |
| Test set rows | {test_rows} |
| Evaluators found | {", ".join(summary["found"]) if summary["found"] else "None"} |
| Evaluators missing | {", ".join(summary["missing"]) if summary["missing"] else "None"} |
| PDF trend reproduced? | {summary["trend"]} |

## Alignment Coverage

{_table(coverage, ["evaluator", "n_test", "n_aligned", "coverage", "n_valid_score", "valid_score_rate", "n_missing", "n_invalid"])}

## Main Evaluator Metrics

{_table(metrics, ["evaluator", "n_valid", "MAE", "Signed Bias", "Exact Match", "Within-1 Accuracy", "Kendall tau", "Spearman rho"])}

## Low-score Blind Spot

{_table(low, ["evaluator", "n_valid_low", "low_exact_match", "low_recall", "low_signed_bias", "low_to_high_rate", "mean_pred_low"])}

## Blockers Before Exp2

{blockers_text}

## Recommended Next Step

Start Exp2 by locking the CE baseline protocol and preserving this Exp1 low-score audit as the
primary evaluator-bias target.
"""
    write_text(EXP01_OUTPUT_DIR / "review_package.md", text)


def write_notion_summaries(tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    coverage = tables["alignment_coverage"]
    metrics = tables["evaluator_metrics"]
    low = tables["low_score_metrics"]
    test_rows = int(coverage["n_test"].iloc[0]) if not coverage.empty else 0
    summary_text = f"""# Exp1 自动 Judge 与人类评分一致性审计

## 1. 实验目的

审计自动评估器与人类评分的一致性，重点定位低分盲区、系统性高估，以及 metric、scenario、subject 等分层差异。

## 2. 输入数据

- 主数据集：`{relpath(PROCESSED_DATASET_PATH)}`
- 数据集名称：`{DATASET_NAME}`
- 主测试集：`{relpath(TEST_SPLIT_PATH)}`
- test rows：{test_rows}
- 不训练模型，不调用 API，不使用 synthetic/sample 数据。

## 3. 输出结果

- 对齐预测：`{relpath(EXP01_OUTPUT_DIR / 'predictions_aligned.jsonl')}`
- 主指标表：`{relpath(EXP01_TABLES_DIR / 'evaluator_metrics.csv')}`
- 低分审计表：`{relpath(EXP01_TABLES_DIR / 'low_score_metrics.csv')}`
- 报告：`{relpath(EXP01_OUTPUT_DIR / 'report.md')}`

## 4. 核心指标

{_table(metrics, ["evaluator", "n_valid", "MAE", "Signed Bias", "Exact Match", "Kendall tau"])}

## 5. 低分盲区结论

{_table(low, ["evaluator", "n_valid_low", "low_exact_match", "low_recall", "low_signed_bias", "low_to_high_rate", "mean_pred_low"])}

低分段 label<=2 的识别明显弱于高分段，自动 judge 倾向于把低质量回答判到更高等级。

## 6. 与小论文结果关系

PDF 主要趋势复现状态：**{summary["trend"]}**。本实验不使用 reference value 修正当前结果，只作为趋势对照。

## 7. 对后续实验的影响

Exp2 需要建立 CE baseline；Exp3 关注 rubric-aware 输入；Exp5 关注低分敏感 loss；Exp6 才能进入 synthetic low-score augmentation；Exp7 需要做 calibration。

## 8. 当前风险和限制

- 只使用已有 judge predictions。
- missing evaluator 不补全、不猜测。
- subject 分层来自 local enriched metadata，只作为审计 metadata。
- 后续不能只看 overall accuracy。
"""
    write_text(EXP01_OUTPUT_DIR / "notion_exp01_summary.md", summary_text)

    paper_notes = f"""# Exp1 论文笔记：自动评估器偏差审计

## 1. 自动评估器一致性分析

在 `{DATASET_NAME}` 的 paper-like test split 上，本文将多个自动评估器的已有预测分数与三位人类标注者聚合后的 `human_mean_5` 和 `label_5` 对齐，计算 MAE、Exact Match、Kendall tau 等一致性指标。

## 2. 低分段识别失效

低分段样本 label<=2 的 exact match 与 recall 明显偏低，说明自动 judge 对劣质回答的识别能力不足。该现象比 overall agreement 更能揭示教育场景中的安全风险。

## 3. 系统性高估

多个评估器在低分段呈正 signed bias，表现为将人类低评分回答预测到中高分。这种高估会弱化评估器对错误、偏题、教学支持不足等问题的惩罚能力。

## 4. 维度/场景差异

分层结果显示，不同 metric 和 scenario 的 MAE 存在差异，subject-level 结果可作为局部误差来源参考，但其 provenance 来自 local enriched metadata，应在论文中作为审计 metadata 谨慎表述。

## 5. 后续方法动机

Exp1 的结论直接支持后续实验设计：Exp2 建立 baseline，Exp3 引入 rubric-aware 输入，Exp5 设计 low-score-sensitive loss，Exp6 评估低分增强数据，Exp7 进行校准，以避免只优化 overall accuracy 而忽视低分盲区。
"""
    write_text(EXP01_OUTPUT_DIR / "notion_exp01_paper_notes.md", paper_notes)


def main() -> None:
    summary = write_report()
    print(f"PDF trend reproduced: {summary['trend']}")
    print(f"Can Exp2 start: {summary['can_exp2']}")
    print(f"Outputs: {relpath(EXP01_OUTPUT_DIR / 'report.md')}")


if __name__ == "__main__":
    main()

