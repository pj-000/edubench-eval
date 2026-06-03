"""Write Exp3 reports, review package, Notion notes, and checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import EXP03_OUTPUT_DIR, EXP03_REPORTS_DIR, EXP03_TABLES_DIR, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.compute_input_ablation_metrics import compute_input_ablation_metrics
from thesis_exp.src.edujudge.exp03.readability_check_exp03 import run_readability_check
from thesis_exp.src.edujudge.exp03.rubric_quality_audit import audit_rubric_quality, overall_status as rubric_quality_status
from thesis_exp.src.edujudge.exp03.rubric_repair import (
    RAW_MODE,
    REPAIR_TRACE_PATH,
    default_rubric_mode,
)
from thesis_exp.src.edujudge.exp03.templates import TEMPLATE_SPECS
from thesis_exp.src.edujudge.utils.io import md_table, read_csv, relpath, write_text


def read_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def status_for(summary: list[dict[str, Any]], template_name: str) -> str:
    row = next((item for item in summary if item.get("template_name") == template_name), None)
    return str(row.get("status")) if row else "pending"


def sanity_status() -> str:
    rows = read_rows(EXP03_TABLES_DIR / "sanity_check_exp03_setup.csv")
    if not rows:
        return "NOT_RUN"
    return "PASS" if all(row.get("status") in {"PASS", "WARN"} for row in rows) else "FAIL"


def smoke_status() -> str:
    report = EXP03_OUTPUT_DIR / "smoke_test" / "smoke_test_report.md"
    if not report.exists():
        return "NOT_RUN"
    text = report.read_text(encoding="utf-8")
    if "Overall status: **PASS**" in text:
        return "PASS"
    if "Overall status: **FAIL**" in text:
        return "FAIL"
    return "UNKNOWN"


def current_rubric_quality_status() -> str:
    rows = read_rows(EXP03_TABLES_DIR / "rubric_quality_audit.csv")
    if not rows:
        rows = audit_rubric_quality()
    return rubric_quality_status(rows)


def readability_status() -> str:
    rows = read_rows(EXP03_TABLES_DIR / "readability_check_exp03.csv")
    if not rows:
        return "NOT_RUN"
    return "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL"


def special_zh_rubric_issue() -> str:
    rows = read_rows(EXP03_TABLES_DIR / "rubric_quality_audit.csv")
    for row in rows:
        metrics = {row.get("metric_a"), row.get("metric_b")}
        if row.get("language") == "zh" and metrics == {
            "Scenario Element Integration",
            "Instruction Following & Task Completion",
        }:
            return str(row.get("severity"))
    return "MISSING"


def truncation_warning(summary: list[dict[str, Any]]) -> str:
    warnings = []
    for row in summary:
        value = row.get("truncation_rate")
        if value in (None, ""):
            continue
        rate = float(value)
        if rate > 0.05:
            warnings.append(f"{row.get('template_name')}={rate:.4f}")
    return ", ".join(warnings) if warnings else "none"


def report_template_table() -> str:
    rows = []
    for template_name, spec in TEMPLATE_SPECS.items():
        rows.append(
            {
                "template_name": template_name,
                "ablation_id": spec.ablation_id,
                "input_fields": ", ".join(spec.input_fields),
                "excluded": ", ".join(spec.excluded_from_text),
                "default": spec.formal_training_default,
            }
        )
    return md_table(rows, ["ablation_id", "template_name", "input_fields", "default"], max_rows=10)


def summary_table(summary: list[dict[str, Any]]) -> str:
    return md_table(
        summary,
        [
            "ablation_id",
            "template_name",
            "status",
            "test_accuracy",
            "test_MAE_label",
            "test_kendall_tau",
            "test_low_to_high_rate",
            "mean_token_length",
            "truncation_rate",
        ],
        max_rows=10,
    )


def low_score_table() -> str:
    rows = read_rows(EXP03_TABLES_DIR / "input_ablation_low_score_comparison.csv")
    return md_table(
        rows,
        ["ablation_id", "template_name", "status", "test_acc_at_1", "test_acc_at_2", "test_low_to_high_rate"],
        max_rows=10,
    )


def human_confirmation_needed(rubric_mode: str) -> str:
    return "NO"


def server_smoke_line(quality_status: str) -> str:
    if quality_status == "ERROR":
        return "NO until rubric quality issue reviewed."
    return "YES."


def formal_training_line(quality_status: str, rubric_mode: str, smoke: str) -> str:
    if quality_status == "ERROR":
        return "NO until rubric quality issue reviewed."
    if rubric_mode == RAW_MODE:
        return "NO until corrected rubric mode is selected."
    if smoke != "PASS":
        return "NO until server smoke test PASS."
    return "YES after server smoke test and rubric quality audit PASS/WARNING reviewed."


def write_report(summary: list[dict[str, Any]], quality_status: str, rubric_mode: str, formatting_status: str) -> None:
    rubric_rows = read_rows(EXP03_TABLES_DIR / "rubric_source_audit.csv")
    coverage = "NA"
    if rubric_rows:
        total = sum(int(float(row.get("n_rows") or 0)) for row in rubric_rows)
        covered = sum(int(float(row.get("n_rows") or 0)) for row in rubric_rows if float(row.get("coverage") or 0) == 1.0)
        coverage = f"{covered}/{total}"
    lines = [
        "# Exp3: Rubric-aware Input Ablation",
        "",
        "## 1. Purpose",
        "Exp3 is an input ablation experiment. It tests whether adding question, metric, rubric, and",
        "metadata fields improves agreement between a local education scoring model and human scores.",
        "",
        "## 2. Relation to Exp2",
        "Exp2 is the Q+A+metric Cross Entropy baseline. Exp3 keeps the same 5-class CE objective,",
        "model family, split, and checkpoint selection. A2 is the Exp2-compatible template and is",
        "reused by default instead of retrained.",
        "",
        "## 3. Input Templates",
        report_template_table(),
        "",
        "A4 intentionally excludes generator_model, answer_model, human labels, and chain-of-thought.",
        "",
        "## 4. Rubric Source and Quality Audit",
        f"Rubric coverage: {coverage}.",
        f"Rubric mode: **{rubric_mode}**.",
        f"Rubric quality status: **{quality_status}**.",
        f"Special zh SEI vs IFTC check: **{special_zh_rubric_issue()}**.",
        f"Human confirmation needed: **{human_confirmation_needed(rubric_mode)}**.",
        "Raw rubric text is read from the split row field. The audit shows it is constant within",
        "each metric/language group, so Exp3 treats it as metric-level rubric description, not",
        "sample-specific human annotation. The active rubric mode may override known defective",
        "metric/language rows before A3/A4 prompts are built.",
        "",
        f"Source audit: `{relpath(EXP03_REPORTS_DIR / 'rubric_source_audit.md')}`",
        f"Quality audit: `{relpath(EXP03_REPORTS_DIR / 'rubric_quality_audit.md')}`",
        f"Repair trace: `{relpath(REPAIR_TRACE_PATH)}`",
        "",
        "## 5. Dataset and Training Setup",
        "The locked Exp0.1 paper-like triple split is used: train=2654, dev=664, test=2218. Labels",
        "remain label_5 in {1,2,3,4,5}, mapped to class indices 0..4 for CE training. Test is used",
        "only after selecting the best checkpoint on dev Exact Match.",
        "",
        "## 6. Smoke Test Results",
        f"Smoke test status: **{smoke_status()}**.",
        f"Server smoke can start: **{server_smoke_line(quality_status)}**",
        "",
        "## 7. Available Ablation Results",
        summary_table(summary),
        "",
        "## 8. Low-score Analysis",
        low_score_table(),
        "",
        "The key Exp3 low-score metric is low_to_high_rate: the fraction of true label 1/2 samples",
        "predicted as 4/5. A3/A4 should be judged partly by whether rubric or metadata reduces this",
        "overestimation without collapsing high-score accuracy.",
        "",
        "## 9. Metric-level Analysis",
        "Metric-level delta tables are written after available runs are collected. The A3 - A2 table",
        "is the main place to inspect which dimensions benefit from rubric-aware input.",
        "",
        "## 10. Token Length and Truncation",
        f"Token truncation warning: {truncation_warning(summary)}.",
        f"Output formatting status: **{formatting_status}**.",
        "Token lengths are estimated when no tokenizer is available and recomputed with the model",
        "tokenizer when supplied during dataset building or training.",
        "",
        "## 11. Implications for Exp4-Exp7",
        "If A3 improves low-score recognition without material truncation, Exp4-Exp7 should use A3",
        "as the default input. If A4 helps only in distribution-specific slices, metadata should be",
        "reported as useful but potentially distribution-dependent.",
        "",
        "## 12. Limitations",
        "- Exp3 changes only inputs, not loss or model architecture.",
        "- Metadata may introduce distribution dependence.",
        "- Subject metadata provenance is derived from Exp0 alignment and should be described cautiously.",
        "- Question-split robustness remains a later validation target.",
    ]
    write_text(EXP03_OUTPUT_DIR / "report.md", "\n".join(lines))


def write_review_package(summary: list[dict[str, Any]], quality_status: str, rubric_mode: str, formatting_status: str) -> None:
    sanity = sanity_status()
    smoke = smoke_status()
    a3 = status_for(summary, "A3_question_answer_metric_rubric")
    a4 = status_for(summary, "A4_question_answer_metric_rubric_metadata")
    exp4_ready = "YES" if a3 == "completed" and a4 == "completed" and smoke in {"PASS", "NOT_RUN"} else "PENDING"
    lines = [
        "# Exp3 Review Package",
        "",
        f"- Can server smoke test start? **{server_smoke_line(quality_status)}**",
        f"- Can formal A3/A4 training start? **{formal_training_line(quality_status, rubric_mode, smoke)}**",
        f"- Can Exp4 start? **{exp4_ready}**",
        f"- Templates implemented: {', '.join(TEMPLATE_SPECS)}",
        "- Rubric source status: metric/language-level rubric from split row field",
        f"- Rubric mode: {rubric_mode}",
        f"- Rubric quality audit status: {quality_status}",
        f"- Special zh SEI vs IFTC rubric check: {special_zh_rubric_issue()}",
        f"- Human confirmation needed: {human_confirmation_needed(rubric_mode)}",
        f"- Output formatting passed: {formatting_status}",
        f"- A2 Exp2 reuse status: {status_for(summary, 'A2_question_answer_metric')}",
        f"- Smoke test status: {smoke}",
        f"- Token truncation warning: {truncation_warning(summary)}",
        f"- Sanity setup status: {sanity}",
        "",
        "## Pending Blockers",
    ]
    blockers = []
    if sanity != "PASS":
        blockers.append("sanity_check_exp03_setup has not passed")
    if quality_status == "ERROR":
        blockers.append("rubric_quality_audit has ERROR rows that must be reviewed before formal training")
    if formatting_status != "PASS":
        blockers.append("output formatting/readability check has not passed")
    if smoke != "PASS":
        blockers.append("server smoke test pending")
    if a3 != "completed":
        blockers.append("formal A3 training pending")
    if a4 != "completed":
        blockers.append("formal A4 training pending")
    lines.extend([f"- {item}" for item in blockers] or ["- none"])
    write_text(EXP03_OUTPUT_DIR / "review_package.md", "\n".join(lines))


def write_notion_notes(summary: list[dict[str, Any]], quality_status: str, rubric_mode: str) -> None:
    lines = [
        "# Exp3 输入信息消融实验总结",
        "",
        "## 实验目的",
        "Exp3 用来验证 rubric 和 metadata 是否能改善教育评分模型与人类评分的一致性，尤其关注低分识别。",
        "",
        "## 输入模板",
        "- A0: answer only",
        "- A1: question + answer",
        "- A2: question + answer + metric，复用 Exp2 CE baseline",
        "- A3: question + answer + metric + rubric",
        "- A4: question + answer + metric + rubric + scenario/subject/education_level/language",
        "",
        "## 数据",
        "固定使用 Exp0.1 paper_like_triple_seed42 split：train=2654，dev=664，test=2218。",
        "",
        "## 当前状态",
        summary_table(summary),
        "",
        "## 训练前硬化检查",
        f"- rubric mode: {rubric_mode}",
        f"- rubric quality audit: {quality_status}",
        f"- zh Scenario Element Integration vs Instruction Following & Task Completion: {special_zh_rubric_issue()}",
        f"- human confirmation needed: {human_confirmation_needed(rubric_mode)}",
        "- 正式训练前需要先完成服务器 smoke test，并人工审阅 rubric quality audit 的 ERROR/WARNING。",
        "",
        "## 后续训练计划",
        "第一轮正式训练只跑 A3 和 A4；A0/A1 资源不足时后补，A2 默认不重训。",
    ]
    write_text(EXP03_OUTPUT_DIR / "notion_exp03_summary.md", "\n".join(lines))

    paper_lines = [
        "# Exp3 论文笔记：Rubric-aware Input Ablation",
        "",
        "Exp3 将教育评分任务从普通文本分类扩展为 rubric-conditioned scoring。与 Exp2 相比，Exp3",
        "不改变模型结构和 CE objective，而是系统比较 answer、question、metric、rubric 和 metadata",
        "对人类一致性的影响。",
        "",
        "A2 对应 Exp2 的 Q+A+metric baseline，因此可以直接复用 Exp2 正式结果。A3/A4 是核心设置，",
        "用于检验显式 rubric 是否能降低低分样本被高估的问题，以及 metadata 是否进一步带来收益。",
        "",
        "论文中需要谨慎说明：当前 rubric 来源于 split row field，但在 metric/language 组内重复，",
        "应表述为 metric-level rubric description，而不是 sample-specific human annotation。训练前",
        "rubric quality audit 发现的跨 metric 重复或高度相似 rubrics 需要单独说明和处理。",
    ]
    write_text(EXP03_OUTPUT_DIR / "notion_exp03_paper_notes.md", "\n".join(paper_lines))


def write_exp03_report() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    summary = compute_input_ablation_metrics()
    rubric_mode = default_rubric_mode()
    quality_rows = audit_rubric_quality()
    quality_status = rubric_quality_status(quality_rows)
    formatting_status = readability_status()
    write_report(summary, quality_status, rubric_mode, formatting_status)
    write_review_package(summary, quality_status, rubric_mode, formatting_status)
    write_notion_notes(summary, quality_status, rubric_mode)
    final_readability_rows = run_readability_check()
    final_formatting_status = "PASS" if all(row.get("status") == "PASS" for row in final_readability_rows) else "FAIL"
    if final_formatting_status != formatting_status:
        write_report(summary, quality_status, rubric_mode, final_formatting_status)
        write_review_package(summary, quality_status, rubric_mode, final_formatting_status)
        write_notion_notes(summary, quality_status, rubric_mode)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Exp3 report files.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    write_exp03_report()
    print(f"Exp3 reports written to {relpath(EXP03_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
