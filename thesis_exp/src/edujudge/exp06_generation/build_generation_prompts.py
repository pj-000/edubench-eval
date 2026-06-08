"""Build dry-run prompt templates for train-only low-score generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    DEFAULT_GENERATION_MODEL,
    EXP06_GENERATION_PROMPTS_DIR,
    EXP06_GENERATION_SAMPLES_DIR,
    PROMPT_VERSION,
    ensure_generation_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.common import (
    load_split,
    planned_error_types,
    read_table,
    source_record_id,
    synthetic_id_from_parts,
)
from thesis_exp.src.edujudge.utils.io import write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify, truncate_text


BASE_TEMPLATE = """You are generating audit-only synthetic training data for an education answer evaluator.

Generate a plausible but flawed student/assistant answer for the given educational task.

Constraints:
- Use the same language as the original question: {language}.
- Target metric: {metric_canonical}.
- Target 5-point pseudo-label: {target_label_5}.
- Error type: {error_type}.
- Use the rubric to make the answer score around the target label for this metric.
- Keep the answer natural and realistic.
- Do not mention that the answer is intentionally flawed, synthetic, low-score, or generated for an experiment.
- Do not copy the original answer if one is visible in the task.
- Do not change the original question.

Original question:
{question}

Rubric:
{rubric_text}

Return only valid JSON with this schema:
{{
  "answer_synthetic": "<natural flawed answer>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "brief_design_note": "<short note for audit logs, not part of training text>"
}}
"""

ZH_TEMPLATE = """你正在为教育问答评价器生成仅供审计的 train-only synthetic 数据。

请根据给定教育任务生成一个自然、可信但有缺陷的回答。

约束：
- 使用与原始问题一致的语言：{language}。
- 目标评价维度：{metric_canonical}。
- 目标 5 分制伪标签：{target_label_5}。
- 错误类型：{error_type}。
- 根据 rubric 控制回答质量，使其大约符合目标分数。
- 回答要自然，不要显得刻意错误。
- 不要提到“故意错误”“synthetic”“低分”“实验生成”等元信息。
- 不要照抄原始回答。
- 不要修改原始问题。

原始问题：
{question}

评分细则：
{rubric_text}

只返回合法 JSON：
{{
  "answer_synthetic": "<自然但有缺陷的回答>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "brief_design_note": "<审计用简短说明，不进入训练文本>"
}}
"""

EN_TEMPLATE = BASE_TEMPLATE


def write_templates() -> None:
    ensure_generation_dirs()
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer.md", BASE_TEMPLATE)
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer_zh.md", ZH_TEMPLATE)
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_low_score_answer_en.md", EN_TEMPLATE)


def render_prompt(row: dict[str, Any], target_label: int, error_type: str) -> str:
    template = ZH_TEMPLATE if row.get("language") == "zh" else EN_TEMPLATE
    return template.format(
        language=row.get("language", ""),
        metric_canonical=row.get("metric_canonical", ""),
        target_label_5=target_label,
        error_type=error_type,
        question=row.get("question", ""),
        rubric_text=row.get("rubric", "") or row.get("rubric_text", ""),
    )


def build_dry_run_prompts(max_prompts: int = 24) -> list[dict[str, Any]]:
    sampling = read_table("train_source_sampling_plan.csv")
    selected_ids = {row["source_record_id"] for row in sampling if str(row.get("selected_for_generation", "")).lower() == "true"}
    train_by_id = {source_record_id(row): row for row in load_split("train")}
    prompts = []
    for source_id in sorted(selected_ids):
        row = train_by_id.get(source_id)
        if not row:
            continue
        errors = planned_error_types(row.get("metric_canonical", ""))
        for idx, target_label in enumerate([1, 2]):
            error_type = errors[idx % len(errors)]
            synthetic_id = synthetic_id_from_parts(source_id, row.get("triple_key"), target_label, error_type, PROMPT_VERSION)
            prompt = render_prompt(row, target_label, error_type)
            prompts.append(
                {
                    "synthetic_id": synthetic_id,
                    "dry_run": True,
                    "generation_model": DEFAULT_GENERATION_MODEL,
                    "generation_prompt_version": PROMPT_VERSION,
                    "source_record_id": source_id,
                    "source_question_key": row.get("question_key", ""),
                    "source_triple_key": row.get("triple_key", ""),
                    "metric_canonical": row.get("metric_canonical", ""),
                    "language": row.get("language", ""),
                    "target_label_5": target_label,
                    "error_type": error_type,
                    "prompt": prompt,
                }
            )
            if len(prompts) >= max_prompts:
                return prompts
    return prompts


def main() -> None:
    ensure_generation_dirs()
    write_templates()
    prompts = build_dry_run_prompts()
    path = EXP06_GENERATION_SAMPLES_DIR / "dry_run_prompts.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in prompts:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote prompt templates and {len(prompts)} dry-run prompts")


if __name__ == "__main__":
    main()
