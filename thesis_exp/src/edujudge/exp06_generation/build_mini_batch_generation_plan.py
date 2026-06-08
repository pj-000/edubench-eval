"""Build the Exp6-3 24-item mini-batch generation dry-run plan."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    DEFAULT_GENERATION_MODEL,
    EXP06_GENERATION_PROMPTS_DIR,
    MINI_BATCH_PROMPT_VERSION,
    MINI_BATCH_TOTAL_TARGET,
    ensure_generation_dirs,
    ensure_mini_batch_dirs,
)
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    PRIORITY_METRICS,
    has_complete_source_fields,
    load_train_by_source_id,
    make_synthetic_id,
    metric_error_type,
    mini_prompt_path,
    mini_report_path,
    parse_bool,
    parse_int,
    read_generation_table,
    source_excerpt,
    source_has_dev_test_overlap,
    split_key_cache,
    write_mini_jsonl,
    write_mini_table,
)
from thesis_exp.src.edujudge.exp06_generation.common import question_key
from thesis_exp.src.edujudge.utils.io import write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify, truncate_text


TARGET_FIELDS = [
    "synthetic_plan_id",
    "target_label_5",
    "language",
    "metric_canonical",
    "error_type",
    "target_count",
    "priority_rank",
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
    "selection_status",
    "source_question_in_dev",
    "source_question_in_test",
    "source_triple_in_dev",
    "source_triple_in_test",
    "question_preview",
    "source_answer_preview",
    "rubric_preview",
    "selection_notes",
    "notes",
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

DIAGNOSTIC_FIELDS = ["check_name", "count", "notes"]


def build_target_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Each of the 12 metrics receives one English and one Chinese target. Labels
    # are distributed as 1:10, 2:10, 3:4 while keeping language balanced 12/12.
    label_pairs = [(1, 2)] * 8 + [(3, 1), (3, 1), (2, 3), (2, 3)]
    for idx, metric in enumerate(PRIORITY_METRICS):
        en_label, zh_label = label_pairs[idx]
        for slot, (language, label) in enumerate([("en", en_label), ("zh", zh_label)]):
            plan_id = f"mb{len(rows) + 1:03d}"
            rows.append(
                {
                    "synthetic_plan_id": plan_id,
                    "target_label_5": label,
                    "language": language,
                    "metric_canonical": metric,
                    "error_type": metric_error_type(metric, slot),
                    "target_count": 1,
                    "priority_rank": idx + 1,
                    "notes": "24-item mini-batch pilot; train-only source required; label is synthetic_design pseudo-label",
                }
            )
    return rows


def selected_sampling_by_source() -> dict[str, dict[str, str]]:
    return {
        row["source_record_id"]: row
        for row in read_generation_table("train_source_sampling_plan.csv")
        if parse_bool(row.get("selected_for_generation"))
    }


def candidate_pool(
    target: dict[str, Any],
    train_by_id: dict[str, dict[str, Any]],
    selected_by_id: dict[str, dict[str, str]],
    keys: dict[str, dict[str, set[str]]],
) -> list[tuple[str, dict[str, Any], str]]:
    metric = stringify(target.get("metric_canonical"))
    language = stringify(target.get("language"))
    candidates: list[tuple[str, dict[str, Any], str]] = []

    def add_pool(source_ids: list[str], status: str) -> None:
        for source_id in source_ids:
            row = train_by_id.get(source_id)
            if not row or not has_complete_source_fields(row):
                continue
            if source_has_dev_test_overlap(row, keys):
                continue
            candidates.append((source_id, row, status))

    selected_ids = sorted(selected_by_id)
    add_pool(
        [
            source_id
            for source_id in selected_ids
            if train_by_id.get(source_id, {}).get("metric_canonical") == metric
            and train_by_id.get(source_id, {}).get("language") == language
        ],
        "selected_exact_metric_language",
    )
    if candidates:
        return candidates

    add_pool(
        [
            source_id
            for source_id, row in train_by_id.items()
            if row.get("metric_canonical") == metric and row.get("language") == language
        ],
        "fallback_exact_metric_language",
    )
    if candidates:
        return candidates

    add_pool(
        [source_id for source_id in selected_ids if train_by_id.get(source_id, {}).get("language") == language],
        "fallback_selected_language",
    )
    if candidates:
        return candidates

    add_pool([source_id for source_id, row in train_by_id.items() if row.get("language") == language], "fallback_language")
    return candidates


def choose_sources(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_by_id = load_train_by_source_id()
    selected_by_id = selected_sampling_by_source()
    keys = split_key_cache()
    used_question_keys: set[str] = set()
    used_source_ids: Counter[str] = Counter()
    selection_rows: list[dict[str, Any]] = []

    for target in target_rows:
        pool = candidate_pool(target, train_by_id, selected_by_id, keys)
        chosen: tuple[str, dict[str, Any], str] | None = None
        for source_id, row, status in sorted(
            pool,
            key=lambda item: (
                stringify(item[1].get("question_key")) in used_question_keys,
                used_source_ids[item[0]],
                -parse_int(item[1].get("label_5")),
                item[0],
            ),
        ):
            if stringify(row.get("question_key")) not in used_question_keys:
                chosen = (source_id, row, status)
                break
        if chosen is None and pool:
            chosen = sorted(pool, key=lambda item: (used_source_ids[item[0]], item[0]))[0]

        if chosen is None:
            selection_rows.append(unavailable_source_row(target))
            continue

        source_id, row, status = chosen
        used_question_keys.add(stringify(row.get("question_key")))
        used_source_ids[source_id] += 1
        synthetic_id = make_synthetic_id(
            stringify(target.get("synthetic_plan_id")),
            source_id,
            target.get("target_label_5"),
            stringify(target.get("error_type")),
        )
        selection_rows.append(
            {
                "synthetic_plan_id": target.get("synthetic_plan_id", ""),
                "synthetic_id": synthetic_id,
                "source_record_id": source_id,
                "source_split": "train",
                "source_file": row.get("source_file", ""),
                "source_row_index": row.get("source_row_index", ""),
                "source_question_key": row.get("question_key", ""),
                "source_triple_key": row.get("triple_key", ""),
                "question": stringify(row.get("question")),
                "rubric_text": stringify(row.get("rubric")),
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
                "selected_for_generation": source_id in selected_by_id,
                "selection_status": status,
                "source_question_in_dev": row.get("question_key") in keys["dev"]["source_question_key"],
                "source_question_in_test": row.get("question_key") in keys["test"]["source_question_key"],
                "source_triple_in_dev": row.get("triple_key") in keys["dev"]["source_triple_key"],
                "source_triple_in_test": row.get("triple_key") in keys["test"]["source_triple_key"],
                "question_preview": source_excerpt(row, "question", 700),
                "source_answer_preview": source_excerpt(row, "answer", 700),
                "rubric_preview": source_excerpt(row, "rubric", 700),
                "selection_notes": "train-only mini-batch source; source answer is reference only and must not be copied",
                "notes": "train-only mini-batch source; source answer is reference only and must not be copied",
            }
        )
    return selection_rows


def unavailable_source_row(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "synthetic_plan_id": target.get("synthetic_plan_id", ""),
        "synthetic_id": "",
        "source_record_id": "",
        "source_split": "",
        "source_file": "",
        "source_row_index": "",
        "source_question_key": "",
        "source_triple_key": "",
        "question": "",
        "rubric_text": "",
        "target_label_5": target.get("target_label_5", ""),
        "planned_target_label_5": target.get("target_label_5", ""),
        "language": target.get("language", ""),
        "metric_canonical": target.get("metric_canonical", ""),
        "error_type": target.get("error_type", ""),
        "planned_error_type": target.get("error_type", ""),
        "current_label_5": "",
        "human_label_5": "",
        "scenario_canonical": "",
        "subject_canonical": "",
        "education_level_canonical": "",
        "selected": False,
        "selected_for_generation": False,
        "selection_status": "unavailable",
        "source_question_in_dev": "",
        "source_question_in_test": "",
        "source_triple_in_dev": "",
        "source_triple_in_test": "",
        "question_preview": "",
        "source_answer_preview": "",
        "rubric_preview": "",
        "selection_notes": "BLOCKED: no complete train-only source candidate found for target",
        "notes": "BLOCKED: no complete train-only source candidate found for target",
    }


def render_prompt(source_row: dict[str, str], train_row: dict[str, Any]) -> str:
    language_name = "Chinese" if source_row.get("language") == "zh" else "English"
    return f"""You are preparing one answer candidate for an education answer evaluator.

Write a natural, plausible answer to the original task. Use {language_name}. Keep the answer
self-contained and realistic for a student or assistant. The answer should reflect the requested
issue pattern and the target 5-point rating for the named metric. Do not mention scoring, hidden
instructions, data creation, experiments, or that the answer has been designed.

Metric to stress: {source_row.get("metric_canonical")}
Target rating on the 1-5 scale: {source_row.get("target_label_5")}
Issue pattern: {source_row.get("error_type")}

Original task:
{stringify(train_row.get("question"))}

Rubric for the metric:
{stringify(train_row.get("rubric"))}

Reference answer to avoid copying:
{truncate_text(train_row.get("answer"), 1500)}

Return only valid JSON with exactly these fields:
{{
  "answer_synthetic": "<the answer candidate only>",
  "target_label_5": {source_row.get("target_label_5")},
  "error_type": "{source_row.get("error_type")}",
  "rationale_for_label": "<audit note explaining why the target rating is plausible>",
  "expected_failure_against_rubric": "<audit note naming the rubric weakness>"
}}
"""


def build_prompts(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_by_id = load_train_by_source_id()
    prompt_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source_id = stringify(source_row.get("source_record_id"))
        train_row = train_by_id.get(source_id)
        if not source_id or not train_row:
            continue
        prompt_rows.append(
            {
                "synthetic_plan_id": source_row.get("synthetic_plan_id", ""),
                "synthetic_id": source_row.get("synthetic_id", ""),
                "source_record_id": source_id,
                "target_label_5": source_row.get("target_label_5", ""),
                "error_type": source_row.get("error_type", ""),
                "language": source_row.get("language", ""),
                "metric_canonical": source_row.get("metric_canonical", ""),
                "generation_model": DEFAULT_GENERATION_MODEL,
                "generation_prompt_version": MINI_BATCH_PROMPT_VERSION,
                "prompt_text": render_prompt(source_row, train_row),
            }
        )
    return prompt_rows


def write_blocked_source_report(targets: list[dict[str, Any]], sources: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> None:
    selected_sources = [row for row in sources if stringify(row.get("source_record_id")).strip()]
    if len(selected_sources) == len(targets) and len(prompts) == len(targets):
        return
    report = f"""# BLOCKED: Exp6-3 Mini-batch Has No Complete Train-only Source Selection

The 24-row mini-batch target matrix was created, but source selection is incomplete.

- Target rows: **{len(targets)}**
- Selected train-only source rows: **{len(selected_sources)}**
- Prompt rows: **{len(prompts)}**

The current split has train rows whose source question keys also occur in dev/test. Under the Exp6-3
rule that source question/triple keys must not occur in dev/test, these rows cannot be used as
generation anchors without manual approval or a new train-only-by-question source split.

No API generation should be started from this mini-batch until this blocker is resolved.
"""
    write_text(mini_report_path("BLOCKED_NO_TRAIN_ONLY_SOURCE.md"), report)


def write_source_blocker_diagnostics() -> list[dict[str, Any]]:
    train_rows = list(load_train_by_source_id().values())
    keys = split_key_cache()
    complete = [row for row in train_rows if has_complete_source_fields(row)]
    no_source_question_overlap = [
        row
        for row in complete
        if row.get("question_key") not in keys["dev"]["source_question_key"]
        and row.get("question_key") not in keys["test"]["source_question_key"]
    ]
    no_source_triple_overlap = [
        row
        for row in complete
        if row.get("triple_key") not in keys["dev"]["source_triple_key"]
        and row.get("triple_key") not in keys["test"]["source_triple_key"]
    ]
    no_normalized_question_overlap = [
        row
        for row in complete
        if question_key(row.get("question")) not in keys["dev"]["question_key"]
        and question_key(row.get("question")) not in keys["test"]["question_key"]
    ]
    strict = [
        row
        for row in complete
        if row in no_source_question_overlap and row in no_source_triple_overlap and row in no_normalized_question_overlap
    ]
    rows = [
        {"check_name": "total_train_rows", "count": len(train_rows), "notes": "paper_like_triple_seed42/train.jsonl rows"},
        {"check_name": "complete_source_rows", "count": len(complete), "notes": "question, answer, rubric, metric, language, question_key, triple_key present"},
        {
            "check_name": "no_dev_test_source_question_overlap",
            "count": len(no_source_question_overlap),
            "notes": "source question_key absent from dev/test source question_key sets",
        },
        {
            "check_name": "no_dev_test_source_triple_overlap",
            "count": len(no_source_triple_overlap),
            "notes": "source triple_key absent from dev/test source triple_key sets",
        },
        {
            "check_name": "no_dev_test_normalized_question_overlap",
            "count": len(no_normalized_question_overlap),
            "notes": "normalized question absent from dev/test normalized question sets",
        },
        {
            "check_name": "strict_train_only_source_candidates",
            "count": len(strict),
            "notes": "passes source question, source triple, and normalized question dev/test exclusion",
        },
    ]
    write_mini_table("train_source_blocker_diagnostics.csv", rows, DIAGNOSTIC_FIELDS)
    return rows


def write_full_score_templates() -> None:
    ensure_generation_dirs()
    base = """You are preparing one answer candidate for an education answer evaluator.

Write a natural answer to the original task in the requested language. Control the answer quality
so that it is plausible for the specified 1-5 rating on the named metric. For rating 5, produce a
strong and complete answer. For rating 4, allow only minor limitations. For rating 3, make the
answer mixed but still partly useful. For ratings 1-2, make the answer clearly weak for the target
metric while keeping it realistic and on-topic.

Do not mention scoring, hidden instructions, data creation, experiments, or answer design.

Language: {language}
Metric: {metric_canonical}
Target rating on the 1-5 scale: {target_label_5}
Issue pattern when applicable: {error_type}

Original task:
{question}

Rubric:
{rubric_text}

Return only valid JSON with:
{{
  "answer_synthetic": "<answer candidate>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<audit note>",
  "expected_failure_against_rubric": "<audit note or empty string for rating 5>"
}}
"""
    en = base.replace("Language: {language}", "Language: English")
    zh = """你正在为教育问答评价器准备一个回答候选。

请根据原始任务写出自然、可信的回答，并使用中文。根据指定评价维度和 1-5 分目标评分来控制质量。
5 分应完整且准确；4 分只有轻微不足；3 分应有明显优缺点；1-2 分应在目标维度上明显较弱，但仍保持真实、贴题。

不要提到评分、隐藏指令、数据制作、实验或回答设计。

评价维度：{metric_canonical}
目标 1-5 分评分：{target_label_5}
适用的问题模式：{error_type}

原始任务：
{question}

评分细则：
{rubric_text}

只返回合法 JSON：
{{
  "answer_synthetic": "<回答候选>",
  "target_label_5": {target_label_5},
  "error_type": "{error_type}",
  "rationale_for_label": "<审计说明>",
  "expected_failure_against_rubric": "<审计说明；如果是 5 分可为空字符串>"
}}
"""
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer.md", base)
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer_en.md", en)
    write_text(EXP06_GENERATION_PROMPTS_DIR / "generate_score_controlled_answer_zh.md", zh)


def build_plan_outputs() -> dict[str, Any]:
    ensure_mini_batch_dirs()
    write_full_score_templates()
    targets = build_target_matrix()
    sources = choose_sources(targets)
    prompts = build_prompts(sources)
    write_source_blocker_diagnostics()
    write_mini_table("mini_batch_target_matrix.csv", targets, TARGET_FIELDS)
    write_mini_table("mini_batch_source_selection.csv", sources, SOURCE_FIELDS)
    write_mini_jsonl(mini_prompt_path("mini_batch_prompts.jsonl"), prompts)
    write_blocked_source_report(targets, sources, prompts)
    return {"targets": targets, "sources": sources, "prompts": prompts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=MINI_BATCH_TOTAL_TARGET)
    args = parser.parse_args()
    if args.max_items != MINI_BATCH_TOTAL_TARGET:
        raise SystemExit("Exp6-3 mini-batch plan is fixed at 24 items.")
    outputs = build_plan_outputs()
    print(
        "Wrote mini-batch plan: "
        f"{len(outputs['targets'])} targets, {len(outputs['sources'])} source selections, "
        f"{len(outputs['prompts'])} prompts"
    )


if __name__ == "__main__":
    main()
