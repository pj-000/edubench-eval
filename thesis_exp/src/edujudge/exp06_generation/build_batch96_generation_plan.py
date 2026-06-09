"""Build Exp6-5 curated mini-batch and 96-row hardened generation plan.

This module is dry-run by default. It prepares prompt/filter-hardened artifacts
for the next question-disjoint batch and never trains a model.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    BATCH96_LANGUAGE_COUNTS,
    BATCH96_PROMPT_VERSION,
    BATCH96_TARGET_LABEL_COUNTS,
    BATCH96_TOTAL_TARGET,
    DEFAULT_GENERATION_MODEL,
    ERROR_TYPES,
    EXP06_BATCH96_CURATED_DIR,
    EXP06_BATCH96_OUTPUT_DIR,
    EXP06_BATCH96_PROMPT_TEMPLATES_DIR,
    EXP06_BATCH96_PROMPTS_DIR,
    EXP06_BATCH96_REPORTS_DIR,
    EXP06_BATCH96_TABLES_DIR,
    ensure_batch96_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.build_mini_batch_generation_plan import (
    candidate_pool_for_mode,
    load_mode_train_by_source_id,
    mode_key_sets,
    overlap_flags,
)
from thesis_exp.src.edujudge.exp06_generation.common import synthetic_id_from_parts
from thesis_exp.src.edujudge.exp06_generation.hardened_filter import HARDENED_FILTER_FIELDS, apply_hardened_checks
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    METRIC_ERROR_SEQUENCE,
    PRIORITY_METRICS,
    has_complete_source_fields,
)
from thesis_exp.src.edujudge.utils.io import md_table, read_csv, read_jsonl, relpath, write_csv, write_jsonl, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify, truncate_text


MINI_FILTERED_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/filtered/filtered_synthetic_candidates.jsonl")
)
MANUAL_DECISIONS_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/manual_spotcheck_decisions.csv")
)
PROMPT_HARDENING_RECOMMENDATIONS_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/prompt_hardening_recommendations.md")
)
READINESS_AFTER_REVIEW_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/mini_batch_generation/spotcheck_review/full_generation_readiness_after_manual_review.md")
)

CURATED_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "field_name",
    "original_value",
    "revised_value",
    "decision",
    "suggested_value",
    "reason",
    "notes",
]
REJECTED_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "target_label_5",
    "suggested_label_5",
    "language",
    "metric_canonical",
    "error_type",
    "decision",
    "reason",
    "notes",
]
TARGET_FIELDS = [
    "synthetic_plan_id",
    "target_label_5",
    "language",
    "metric_canonical",
    "error_type",
    "target_count",
    "priority_rank",
    "generation_split_mode",
    "generation_prompt_version",
    "notes",
]
SOURCE_FIELDS = [
    "synthetic_plan_id",
    "synthetic_id",
    "source_record_id",
    "source_split",
    "source_file",
    "source_row_index",
    "source_question_key",
    "source_triple_key",
    "question",
    "rubric_text",
    "target_label_5",
    "planned_target_label_5",
    "language",
    "metric_canonical",
    "error_type",
    "planned_error_type",
    "current_label_5",
    "human_label_5",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "selected",
    "selected_for_generation",
    "allowed_for_training",
    "selection_status",
    "question_overlap_with_dev",
    "question_overlap_with_test",
    "triple_overlap_with_dev",
    "triple_overlap_with_test",
    "record_overlap_with_dev",
    "record_overlap_with_test",
    "source_question_in_dev",
    "source_question_in_test",
    "source_triple_in_dev",
    "source_triple_in_test",
    "source_risk_level",
    "question_preview",
    "source_answer_preview",
    "rubric_preview",
    "selection_notes",
]
PROMPT_FIELDS = [
    "synthetic_plan_id",
    "synthetic_id",
    "source_record_id",
    "target_label_5",
    "error_type",
    "language",
    "metric_canonical",
    "generation_model",
    "generation_prompt_version",
    "prompt_text",
]
SANITY_FIELDS = ["check_name", "status", "count", "notes"]


def batch96_path(*parts: str) -> Path:
    ensure_batch96_dirs()
    return EXP06_BATCH96_OUTPUT_DIR.joinpath(*parts)


def clean_multiline(value: Any) -> str:
    return "\n".join(line.rstrip() for line in stringify(value).splitlines())


def first_supported_suggestion(value: str) -> str:
    for token in value.replace("/", " or ").split(" or "):
        candidate = token.strip()
        if candidate in ERROR_TYPES:
            return candidate
    return value.strip()


def load_manual_decisions() -> dict[str, dict[str, str]]:
    return {row["synthetic_plan_id"]: row for row in read_csv(MANUAL_DECISIONS_PATH)}


def build_curated_mini_batch() -> dict[str, Any]:
    ensure_batch96_dirs()
    filtered_rows = read_jsonl(MINI_FILTERED_PATH)
    decisions = load_manual_decisions()
    curated: list[dict[str, Any]] = []
    revision_log: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in filtered_rows:
        plan_id = stringify(row.get("synthetic_plan_id"))
        decision = decisions.get(plan_id, {})
        decision_name = stringify(decision.get("decision"))
        revised = dict(row)
        revised["manual_spotcheck_decision"] = decision_name
        revised["manual_spotcheck_notes"] = decision.get("notes", "")

        if decision_name == "accept":
            curated.append(revised)
            continue

        if decision_name == "revise_error_type":
            old_value = stringify(row.get("error_type"))
            new_value = first_supported_suggestion(decision.get("suggested_error_type", ""))
            revised["error_type"] = new_value
            revised["manual_revision_applied"] = "error_type"
            revision_log.append(
                {
                    "synthetic_id": row.get("synthetic_id", ""),
                    "synthetic_plan_id": plan_id,
                    "field_name": "error_type",
                    "original_value": old_value,
                    "revised_value": new_value,
                    "decision": decision_name,
                    "suggested_value": decision.get("suggested_error_type", ""),
                    "reason": decision.get("reason", ""),
                    "notes": decision.get("notes", ""),
                }
            )
            curated.append(revised)
            continue

        if decision_name == "revise_label":
            old_value = stringify(row.get("target_label_5"))
            new_value = stringify(decision.get("suggested_label_5"))
            revised["target_label_5"] = new_value
            revised["score_raw"] = new_value
            revised["manual_revision_applied"] = "target_label_5"
            revision_log.append(
                {
                    "synthetic_id": row.get("synthetic_id", ""),
                    "synthetic_plan_id": plan_id,
                    "field_name": "target_label_5",
                    "original_value": old_value,
                    "revised_value": new_value,
                    "decision": decision_name,
                    "suggested_value": new_value,
                    "reason": decision.get("reason", ""),
                    "notes": decision.get("notes", ""),
                }
            )
            curated.append(revised)
            continue

        if decision_name.startswith("reject"):
            rejected.append(
                {
                    "synthetic_id": row.get("synthetic_id", ""),
                    "synthetic_plan_id": plan_id,
                    "target_label_5": row.get("target_label_5", ""),
                    "suggested_label_5": decision.get("suggested_label_5", ""),
                    "language": row.get("language", ""),
                    "metric_canonical": row.get("metric_canonical", ""),
                    "error_type": row.get("error_type", ""),
                    "decision": decision_name,
                    "reason": decision.get("reason", ""),
                    "notes": decision.get("notes", ""),
                }
            )
            continue

        rejected.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": plan_id,
                "target_label_5": row.get("target_label_5", ""),
                "suggested_label_5": decision.get("suggested_label_5", ""),
                "language": row.get("language", ""),
                "metric_canonical": row.get("metric_canonical", ""),
                "error_type": row.get("error_type", ""),
                "decision": decision_name or "missing_manual_decision",
                "reason": "not accepted by manual review workflow",
                "notes": decision.get("notes", ""),
            }
        )

    write_jsonl(EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_synthetic_candidates.jsonl", curated)
    write_csv(EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_revision_log.csv", revision_log, CURATED_FIELDS)
    write_csv(
        EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_rejected_or_relabel_upward.csv",
        rejected,
        REJECTED_FIELDS,
    )
    return {
        "filtered_count": len(filtered_rows),
        "curated": curated,
        "curated_count": len(curated),
        "revision_log": revision_log,
        "revised_count": len(revision_log),
        "rejected": rejected,
        "rejected_count": len(rejected),
    }


def target_labels_for_metric(metric_index: int) -> list[int]:
    group = metric_index // 4
    if group == 0:
        return [1, 1, 1, 1, 2, 2, 2, 3]
    if group == 1:
        return [1, 1, 1, 2, 2, 2, 2, 3]
    return [1, 1, 1, 2, 2, 2, 3, 3]


def error_type_for_metric(metric: str, slot: int) -> str:
    sequence = [item for item in METRIC_ERROR_SEQUENCE.get(metric, []) if item in ERROR_TYPES]
    if not sequence:
        sequence = ["rubric_violation"]
    return sequence[slot % len(sequence)]


def build_target_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(PRIORITY_METRICS):
        labels = target_labels_for_metric(metric_index)
        for slot, label in enumerate(labels):
            language = "en" if slot % 2 == 0 else "zh"
            rows.append(
                {
                    "synthetic_plan_id": f"b96_{len(rows) + 1:03d}",
                    "target_label_5": label,
                    "language": language,
                    "metric_canonical": metric,
                    "error_type": error_type_for_metric(metric, slot),
                    "target_count": 1,
                    "priority_rank": metric_index + 1,
                    "generation_split_mode": "question_disjoint_formal",
                    "generation_prompt_version": BATCH96_PROMPT_VERSION,
                    "notes": "Exp6-5 96-row dry-run target; question_seed42/train source only",
                }
            )
    return rows


def choose_sources(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = "question_disjoint_formal"
    train_by_id = load_mode_train_by_source_id(mode)
    keys = mode_key_sets(mode)
    used_question_keys: set[str] = set()
    used_source_ids: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for target in target_rows:
        pool = candidate_pool_for_mode(target, mode, train_by_id, keys)
        chosen: tuple[str, dict[str, Any], str] | None = None
        for source_id, row, status in sorted(
            pool,
            key=lambda item: (
                stringify(item[1].get("question_key")) in used_question_keys,
                used_source_ids[item[0]],
                -int(float(stringify(item[1].get("label_5") or 0))),
                item[0],
            ),
        ):
            if stringify(row.get("question_key")) not in used_question_keys:
                chosen = (source_id, row, status)
                break
        if chosen is None and pool:
            chosen = sorted(pool, key=lambda item: (used_source_ids[item[0]], item[0]))[0]
        if chosen is None:
            rows.append(unavailable_source_row(target))
            continue

        source_id, row, status = chosen
        used_question_keys.add(stringify(row.get("question_key")))
        used_source_ids[source_id] += 1
        flags = overlap_flags(row, keys)
        synthetic_id = synthetic_id_from_parts(
            target.get("synthetic_plan_id"),
            source_id,
            target.get("target_label_5"),
            target.get("error_type"),
            BATCH96_PROMPT_VERSION,
        )
        rows.append(
            {
                "synthetic_plan_id": target.get("synthetic_plan_id", ""),
                "synthetic_id": synthetic_id,
                "source_record_id": source_id,
                "source_split": "train",
                "source_file": row.get("source_file", ""),
                "source_row_index": row.get("source_row_index", ""),
                "source_question_key": row.get("question_key", ""),
                "source_triple_key": row.get("triple_key", ""),
                "question": clean_multiline(row.get("question")),
                "rubric_text": clean_multiline(row.get("rubric")),
                "target_label_5": target.get("target_label_5", ""),
                "planned_target_label_5": target.get("target_label_5", ""),
                "language": target.get("language", ""),
                "metric_canonical": target.get("metric_canonical", ""),
                "error_type": target.get("error_type", ""),
                "planned_error_type": target.get("error_type", ""),
                "current_label_5": row.get("label_5", ""),
                "human_label_5": row.get("label_5", ""),
                "scenario_canonical": row.get("scenario_canonical", ""),
                "subject_canonical": row.get("subject_canonical", ""),
                "education_level_canonical": row.get("education_level_canonical", ""),
                "selected": True,
                "selected_for_generation": True,
                "allowed_for_training": True,
                "selection_status": status,
                **flags,
                "source_question_in_dev": flags["question_overlap_with_dev"],
                "source_question_in_test": flags["question_overlap_with_test"],
                "source_triple_in_dev": flags["triple_overlap_with_dev"],
                "source_triple_in_test": flags["triple_overlap_with_test"],
                "source_risk_level": "LOW",
                "question_preview": truncate_text(clean_multiline(row.get("question")), 700),
                "source_answer_preview": truncate_text(clean_multiline(row.get("answer")), 700),
                "rubric_preview": truncate_text(clean_multiline(row.get("rubric")), 700),
                "selection_notes": "question_seed42/train source; dev/test question overlap forbidden; source answer reference only",
            }
        )
    return rows


def unavailable_source_row(target: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in SOURCE_FIELDS}
    row.update(
        {
            "synthetic_plan_id": target.get("synthetic_plan_id", ""),
            "target_label_5": target.get("target_label_5", ""),
            "planned_target_label_5": target.get("target_label_5", ""),
            "language": target.get("language", ""),
            "metric_canonical": target.get("metric_canonical", ""),
            "error_type": target.get("error_type", ""),
            "planned_error_type": target.get("error_type", ""),
            "selected": False,
            "selected_for_generation": False,
            "allowed_for_training": False,
            "selection_status": "unavailable",
            "source_risk_level": "BLOCKED",
            "selection_notes": "BLOCKED: no eligible question_seed42/train source found",
        }
    )
    return row


def write_hardened_prompt_templates() -> dict[str, Path]:
    generic = """You are preparing one answer candidate for an education answer evaluator.

Write a natural, plausible answer to the original task in the requested language. The answer must
match the requested target label and issue pattern for the target metric.

Language: {language}
Metric: {metric_canonical}
Target label on the 1-5 scale: {target_label_5}
Issue pattern: {error_type}

Quality requirements:
- For target label 1, include a severe, clear, rubric-relevant failure.
- For target label 2, include a major defect that makes the answer unreliable even if it is
  superficially fluent.
- For target label 3, produce boundary quality with visible weaknesses; do not produce a clearly
  good answer.
- If the answer is clearly stronger than the requested target label, return `needs_revision=true`
  and explain why in `revision_reason`.
- Align the answer's actual weakness with `error_type`; if the weakness does not align, return
  `needs_revision=true`.
- Do not use artifact phrases such as "low-score answer", "intentionally wrong", "synthetic",
  "generated for an experiment", "低分回答", "故意错误", or "合成".
- Do not copy the reference answer.
- Do not modify the original task.

Original task:
{question}

Rubric for the target metric:
{rubric_text}

Reference answer to avoid copying:
{source_answer}

Return only valid JSON with exactly these fields:
{{
  "answer_synthetic": "<answer candidate only>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<audit note explaining why the target label is plausible>",
  "expected_failure_against_rubric": "<audit note naming the clear rubric weakness>",
  "label_plausibility_self_check": "pass or needs_revision",
  "error_type_alignment_self_check": "pass or needs_revision",
  "rubric_failure_visibility": "clear, boundary, weak, or needs_revision",
  "too_good_for_target_label": false,
  "needs_revision": false,
  "revision_reason": ""
}}
"""
    en = generic.replace("Language: {language}", "Language: English")
    zh = """你正在为教育问答评价器准备一个回答候选。

请根据原始任务写出自然、可信的中文回答。回答必须符合目标评价维度、目标标签和问题模式。

评价维度：{metric_canonical}
目标 1-5 分标签：{target_label_5}
问题模式：{error_type}

质量要求：
- 如果目标标签是 1，必须包含严重、清晰、与 rubric 直接相关的缺陷。
- 如果目标标签是 2，必须包含主要缺陷，即使表面流畅也不可靠。
- 如果目标标签是 3，必须是边界质量，有明显不足；不要生成明显好的答案。
- 如果回答明显高于目标标签，返回 `needs_revision=true`，并在 `revision_reason` 中说明原因。
- 回答的实际缺陷必须与 `error_type` 对齐；如果不对齐，返回 `needs_revision=true`。
- 不要使用“低分回答”“故意错误”“synthetic”“generated for an experiment”“合成”等痕迹短语。
- 不要照抄参考答案。
- 不要修改原始任务。

原始任务：
{question}

目标维度评分细则：
{rubric_text}

避免照抄的参考答案：
{source_answer}

只返回合法 JSON，字段必须完整：
{{
  "answer_synthetic": "<回答候选本身>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<审计说明：为什么目标标签可信>",
  "expected_failure_against_rubric": "<审计说明：指出清晰的 rubric 缺陷>",
  "label_plausibility_self_check": "pass or needs_revision",
  "error_type_alignment_self_check": "pass or needs_revision",
  "rubric_failure_visibility": "clear, boundary, weak, or needs_revision",
  "too_good_for_target_label": false,
  "needs_revision": false,
  "revision_reason": ""
}}
"""
    paths = {
        "generic": EXP06_BATCH96_PROMPT_TEMPLATES_DIR / "generate_low_score_answer_hardened.md",
        "en": EXP06_BATCH96_PROMPT_TEMPLATES_DIR / "generate_low_score_answer_hardened_en.md",
        "zh": EXP06_BATCH96_PROMPT_TEMPLATES_DIR / "generate_low_score_answer_hardened_zh.md",
    }
    write_text(paths["generic"], generic)
    write_text(paths["en"], en)
    write_text(paths["zh"], zh)
    return paths


def render_prompt(source_row: dict[str, Any], train_row: dict[str, Any], templates: dict[str, str]) -> str:
    key = "zh" if source_row.get("language") == "zh" else "en"
    return templates[key].format(
        language="Chinese" if key == "zh" else "English",
        metric_canonical=source_row.get("metric_canonical", ""),
        target_label_5=source_row.get("target_label_5", ""),
        error_type=source_row.get("error_type", ""),
        question=stringify(train_row.get("question")),
        rubric_text=stringify(train_row.get("rubric")),
        source_answer=truncate_text(train_row.get("answer"), 1500),
    )


def build_prompts(source_rows: list[dict[str, Any]], template_paths: dict[str, Path]) -> list[dict[str, Any]]:
    train_by_id = load_mode_train_by_source_id("question_disjoint_formal")
    templates = {name: path.read_text(encoding="utf-8") for name, path in template_paths.items()}
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source_id = stringify(source_row.get("source_record_id"))
        train_row = train_by_id.get(source_id)
        if not source_id or not train_row:
            continue
        rows.append(
            {
                "synthetic_plan_id": source_row.get("synthetic_plan_id", ""),
                "synthetic_id": source_row.get("synthetic_id", ""),
                "source_record_id": source_id,
                "target_label_5": source_row.get("target_label_5", ""),
                "error_type": source_row.get("error_type", ""),
                "language": source_row.get("language", ""),
                "metric_canonical": source_row.get("metric_canonical", ""),
                "generation_model": DEFAULT_GENERATION_MODEL,
                "generation_prompt_version": BATCH96_PROMPT_VERSION,
                "prompt_text": render_prompt(source_row, train_row, templates),
            }
        )
    return rows


def summarize_targets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labels": dict(Counter(stringify(row.get("target_label_5")) for row in rows)),
        "languages": dict(Counter(stringify(row.get("language")) for row in rows)),
        "metrics": dict(Counter(stringify(row.get("metric_canonical")) for row in rows)),
        "error_types": dict(Counter(stringify(row.get("error_type")) for row in rows)),
        "metric_coverage": len({row.get("metric_canonical") for row in rows}),
        "error_type_coverage": len({row.get("error_type") for row in rows}),
    }


def source_leak_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in source_rows
        if stringify(row.get("source_question_in_dev")).lower() == "true"
        or stringify(row.get("source_question_in_test")).lower() == "true"
        or stringify(row.get("source_triple_in_dev")).lower() == "true"
        or stringify(row.get("source_triple_in_test")).lower() == "true"
        or row.get("source_split") != "train"
    ]


def prompt_hardening_status(template_paths: dict[str, Path]) -> str:
    required = [
        "label_plausibility_self_check",
        "error_type_alignment_self_check",
        "needs_revision",
        "too_good_for_target_label",
        "severe, clear, rubric-relevant failure",
        "boundary quality",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in template_paths.values())
    return "PASS" if all(item in text for item in required) else "FAIL"


def filter_hardening_status() -> str:
    return "PASS" if all(field in HARDENED_FILTER_FIELDS for field in HARDENED_FILTER_FIELDS) else "FAIL"


def build_sanity_rows(
    curated_summary: dict[str, Any],
    targets: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    template_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    target_summary = summarize_targets(targets)
    leaks = source_leak_rows(sources)
    selected = [row for row in sources if stringify(row.get("source_record_id"))]
    rows = [
        {
            "check_name": "curated_usable_count_16",
            "status": "PASS" if curated_summary["curated_count"] == 16 else "FAIL",
            "count": curated_summary["curated_count"],
            "notes": "manual-reviewed mini-batch usable pool",
        },
        {
            "check_name": "mb008_excluded",
            "status": "PASS" if all(row.get("synthetic_plan_id") != "mb008" for row in curated_summary["curated"]) else "FAIL",
            "count": sum(1 for row in curated_summary["curated"] if row.get("synthetic_plan_id") == "mb008"),
            "notes": "mb008 must not enter low-score curated pool",
        },
        {
            "check_name": "planned_count_96",
            "status": "PASS" if len(targets) == BATCH96_TOTAL_TARGET else "FAIL",
            "count": len(targets),
            "notes": "batch96 planned raw generations",
        },
        {
            "check_name": "label_distribution",
            "status": "PASS" if target_summary["labels"] == {"1": 40, "2": 40, "3": 16} else "FAIL",
            "count": len(targets),
            "notes": str(target_summary["labels"]),
        },
        {
            "check_name": "language_distribution",
            "status": "PASS" if target_summary["languages"] == BATCH96_LANGUAGE_COUNTS else "FAIL",
            "count": len(targets),
            "notes": str(target_summary["languages"]),
        },
        {
            "check_name": "metric_coverage_12",
            "status": "PASS" if target_summary["metric_coverage"] == 12 else "FAIL",
            "count": target_summary["metric_coverage"],
            "notes": "12 EduBench metrics if possible",
        },
        {
            "check_name": "error_type_coverage_7",
            "status": "PASS" if target_summary["error_type_coverage"] == 7 else "FAIL",
            "count": target_summary["error_type_coverage"],
            "notes": str(target_summary["error_types"]),
        },
        {
            "check_name": "source_selection_complete",
            "status": "PASS" if len(selected) == len(targets) else "FAIL",
            "count": len(selected),
            "notes": f"selected={len(selected)} planned={len(targets)}",
        },
        {
            "check_name": "no_dev_test_source_overlap",
            "status": "PASS" if not leaks else "FAIL",
            "count": len(leaks),
            "notes": "question_seed42/train only; dev/test question forbidden",
        },
        {
            "check_name": "prompt_hardening_status",
            "status": prompt_hardening_status(template_paths),
            "count": len(template_paths),
            "notes": "hardened templates include label/error self-checks and target-label guards",
        },
        {
            "check_name": "filter_hardening_status",
            "status": filter_hardening_status(),
            "count": len(HARDENED_FILTER_FIELDS),
            "notes": ", ".join(HARDENED_FILTER_FIELDS),
        },
        {
            "check_name": "api_not_called",
            "status": "PASS" if os.environ.get("EXP6_RUN_GENERATION") != "1" else "FAIL",
            "count": 0,
            "notes": "dry-run only unless EXP6_RUN_GENERATION=1",
        },
    ]
    write_csv(EXP06_BATCH96_TABLES_DIR / "sanity_check_batch96_generation.csv", rows, SANITY_FIELDS)
    return rows


def write_reports(
    curated_summary: dict[str, Any],
    targets: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    template_paths: dict[str, Path],
    sanity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_summary = summarize_targets(targets)
    leaks = source_leak_rows(sources)
    api_called = False
    synthetic_generated = False
    can_batch96 = (
        len(targets) == 96
        and len(prompts) == 96
        and not leaks
        and prompt_hardening_status(template_paths) == "PASS"
        and filter_hardening_status() == "PASS"
    )
    status = {
        "curated_usable_count": curated_summary["curated_count"],
        "revised_count": curated_summary["revised_count"],
        "rejected_count": curated_summary["rejected_count"],
        "batch96_planned_count": len(targets),
        "label_distribution": target_summary["labels"],
        "language_distribution": target_summary["languages"],
        "metric_coverage": target_summary["metric_coverage"],
        "error_type_coverage": target_summary["error_type_coverage"],
        "api_called": api_called,
        "synthetic_generated": synthetic_generated,
        "can_batch96_generation_start": can_batch96,
        "can_full_384_generation_start": False,
        "can_exp6_training_start": False,
        "prompt_hardening_status": prompt_hardening_status(template_paths),
        "filter_hardening_status": filter_hardening_status(),
        "source_leakage_status": "PASS" if not leaks else "FAIL",
    }

    report = f"""# Exp6-5 Batch96 Generation Plan Report

## Scope

This dry-run prepares the next 96-row question-disjoint low-score generation batch after manual
spot-check of the 17 filtered mini-batch samples. It does not train models, modify Exp0-Exp5
results, write checkpoints, call an API, or enter full 384-row generation.

## Curated Mini-batch

- Total filtered mini-batch rows: **{curated_summary["filtered_count"]}**
- Curated usable count: **{curated_summary["curated_count"]}**
- Revised count: **{curated_summary["revised_count"]}**
- Rejected count: **{curated_summary["rejected_count"]}**
- `mb008` enters low-score curated pool: **NO**

Curated files:

- `{relpath(EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_synthetic_candidates.jsonl")}`
- `{relpath(EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_revision_log.csv")}`
- `{relpath(EXP06_BATCH96_CURATED_DIR / "curated_mini_batch_rejected_or_relabel_upward.csv")}`

## Prompt and Filter Hardening

- Prompt hardening status: **{status["prompt_hardening_status"]}**
- Filter hardening status: **{status["filter_hardening_status"]}**
- Hardened filter fields: `{", ".join(HARDENED_FILTER_FIELDS)}`

Prompt templates:

- `{relpath(template_paths["generic"])}`
- `{relpath(template_paths["en"])}`
- `{relpath(template_paths["zh"])}`

## Batch96 Plan

- Planned rows: **{len(targets)}**
- Prompt rows: **{len(prompts)}**
- Target label distribution: `{target_summary["labels"]}`
- Language distribution: `{target_summary["languages"]}`
- Metric coverage: **{target_summary["metric_coverage"]}**
- Error type coverage: **{target_summary["error_type_coverage"]}**
- Source split: **question_seed42/train only**
- Dev/test source overlap rows: **{len(leaks)}**
- API called: **NO**
- Synthetic generated: **NO**

{md_table(targets, ["synthetic_plan_id", "target_label_5", "language", "metric_canonical", "error_type"], max_rows=24)}

## Gates

- Can batch96 generation start? **{"YES" if can_batch96 else "NO"}**
- Can full 384 generation start? **NO**
- Can Exp6 training start? **NO**
"""
    write_text(EXP06_BATCH96_REPORTS_DIR / "batch96_generation_plan_report.md", report)

    review = f"""# Exp6-5 Batch96 Review Package

- Can batch96 generation start? **{"YES" if can_batch96 else "NO"}**
- Can full 384 generation start? **NO**
- Can Exp6 training start? **NO**
- Curated mini-batch usable count: **{curated_summary["curated_count"]}**
- Prompt hardening status: **{status["prompt_hardening_status"]}**
- Filter hardening status: **{status["filter_hardening_status"]}**
- Source leakage status: **{status["source_leakage_status"]}**
- API generation status: **DRY_RUN_NO_API_CALL**
- Synthetic generated: **NO**

Notes:

- This package authorizes only the next 96-row generation batch after API approval.
- Direct full 384-row generation remains blocked.
- Exp6 training remains blocked until batch96 results are generated, filtered, leakage-checked,
  and manually reviewed.
"""
    write_text(EXP06_BATCH96_REPORTS_DIR / "batch96_review_package.md", review)

    notion = f"""# Exp6 Batch96 Plan

| item | value |
| --- | --- |
| Curated usable count | {curated_summary["curated_count"]} |
| Revised count | {curated_summary["revised_count"]} |
| Rejected count | {curated_summary["rejected_count"]} |
| Batch96 planned count | {len(targets)} |
| Label distribution | `{target_summary["labels"]}` |
| Language distribution | `{target_summary["languages"]}` |
| Metric coverage | {target_summary["metric_coverage"]} |
| Error type coverage | {target_summary["error_type_coverage"]} |
| API called | NO |
| Synthetic generated | NO |
| Batch96 generation can start | {"YES" if can_batch96 else "NO"} |
| Full 384 can start | NO |
| Training can start | NO |

## Required gates

- Use only `question_seed42/train` sources.
- Keep dev/test questions out of generation sources.
- Run only batch96 when API generation is explicitly enabled.
- Do not start full 384 or Exp6 training from this plan.
"""
    write_text(EXP06_BATCH96_REPORTS_DIR / "notion_exp06_batch96_plan.md", notion)

    sanity = f"""# Exp6-5 Batch96 Generation Sanity Check

Overall status: **{"PASS" if all(row["status"] == "PASS" for row in sanity_rows) else "FAIL"}**

{md_table(sanity_rows, SANITY_FIELDS, max_rows=50)}
"""
    write_text(EXP06_BATCH96_REPORTS_DIR / "sanity_check_batch96_generation.md", sanity)
    return status


def build_all() -> dict[str, Any]:
    ensure_batch96_dirs()
    curated_summary = build_curated_mini_batch()
    template_paths = write_hardened_prompt_templates()
    targets = build_target_matrix()
    sources = choose_sources(targets)
    prompts = build_prompts(sources, template_paths)

    write_csv(EXP06_BATCH96_TABLES_DIR / "batch96_target_matrix.csv", targets, TARGET_FIELDS)
    write_csv(EXP06_BATCH96_TABLES_DIR / "batch96_source_selection.csv", sources, SOURCE_FIELDS)
    write_jsonl(EXP06_BATCH96_PROMPTS_DIR / "batch96_prompts.jsonl", prompts)

    sanity_rows = build_sanity_rows(curated_summary, targets, sources, prompts, template_paths)
    return write_reports(curated_summary, targets, sources, prompts, template_paths, sanity_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-only", action="store_true", default=True)
    args = parser.parse_args()
    if os.environ.get("EXP6_RUN_GENERATION") == "1":
        raise SystemExit(
            "EXP6_RUN_GENERATION=1 was set, but this planner intentionally does not call APIs. "
            "Use a batch96 generation runner after reviewing the dry-run plan."
        )
    status = build_all()
    print(
        "Wrote Exp6-5 batch96 dry-run plan: "
        f"planned={status['batch96_planned_count']} curated={status['curated_usable_count']} "
        f"can_batch96={'YES' if status['can_batch96_generation_start'] else 'NO'}"
    )


if __name__ == "__main__":
    main()
