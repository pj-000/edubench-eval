"""Run or block Exp6-9 topup-1 synthetic generation and audit.

This workflow is generation-gated by ``EXP6_RUN_GENERATION=1``. Without that
environment variable and an API key/local endpoint, it writes blocked audit
artifacts and never fabricates synthetic outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import time
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import (
    DEFAULT_GENERATION_MODEL,
    ERROR_TYPES,
    EXP06_BATCH96_OUTPUT_DIR,
    EXP06_GENERATION_SRC_DIR,
)
from thesis_exp.src.edujudge.exp06_generation.build_mini_batch_generation_plan import (
    load_mode_train_by_source_id,
    mode_key_sets,
    overlap_flags,
    source_allowed_for_mode,
)
from thesis_exp.src.edujudge.exp06_generation.common import qa_key, question_key, synthetic_id_from_parts
from thesis_exp.src.edujudge.exp06_generation.hardened_filter import (
    HARDENED_FILTER_FIELDS,
    apply_hardened_checks,
)
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    METRIC_ERROR_SEQUENCE,
    PRIORITY_METRICS,
    answer_hash,
)
from thesis_exp.src.edujudge.exp06_generation.run_batch96_generation import (
    FILTER_DETAIL_FIELDS,
    FILTER_REPORT_FIELDS,
    LEAKAGE_DETAIL_FIELDS,
    RAW_FIELDS,
    READABILITY_FIELDS,
    STATIC_FIELDS,
    append_check,
    distribution,
    parse_raw_response,
    read_csv_if_exists,
    read_jsonl_if_exists,
    required_filter_reasons,
    spotcheck_selection,
)
from thesis_exp.src.edujudge.exp06_generation.run_mini_batch_generation import (
    generation_env,
    now_iso,
    post_chat_completion,
    validate_generated_content,
)
from thesis_exp.src.edujudge.utils.io import md_table, read_jsonl, relpath, write_csv, write_jsonl, write_text
from thesis_exp.src.edujudge.utils.text_norm import detect_language_from_text, normalize_text, stringify, truncate_text


TOPUP1_OUTPUT_DIR = Path("thesis_exp/outputs/exp06_synthetic_low_score/topup_generation/topup1")
TOPUP1_TABLES_DIR = TOPUP1_OUTPUT_DIR / "tables"
TOPUP1_PROMPTS_DIR = TOPUP1_OUTPUT_DIR / "prompts"
TOPUP1_GENERATED_DIR = TOPUP1_OUTPUT_DIR / "generated"
TOPUP1_FILTERED_DIR = TOPUP1_OUTPUT_DIR / "filtered"
TOPUP1_LEAKAGE_DIR = TOPUP1_OUTPUT_DIR / "leakage"
TOPUP1_SPOTCHECK_DIR = TOPUP1_OUTPUT_DIR / "spotcheck"
TOPUP1_REPORTS_DIR = TOPUP1_OUTPUT_DIR / "reports"

BATCH96_CURATED_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/curated_review")
    / "curated_batch96_synthetic_candidates.jsonl"
)
BATCH96_TOPUP_PLAN_PATH = (
    Path("thesis_exp/outputs/exp06_synthetic_low_score/batch96_generation/curated_review")
    / "batch96_topup_plan_after_review.csv"
)
BATCH96_PROMPT_TEMPLATE_DIR = EXP06_BATCH96_OUTPUT_DIR / "prompt_templates"

TOPUP1_PROMPT_VERSION = "exp06_topup1_thinking_hardened_v2"
TOPUP1_TARGET_LABEL_COUNTS = {"1": 75, "2": 68, "3": 17}
TOPUP1_LANGUAGE_COUNTS = {"en": 80, "zh": 80}
TOPUP1_TOTAL_TARGET = sum(TOPUP1_TARGET_LABEL_COUNTS.values())
FINAL_TARGET_LABEL_COUNTS = {"1": 168, "2": 168, "3": 48}

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
    "batch96_source_question_reuse_count",
    "topup1_source_question_reuse_count",
    "source_reuse_status",
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
SPOTCHECK_FIELDS = [
    "synthetic_id",
    "synthetic_plan_id",
    "source_record_id",
    "language",
    "metric_canonical",
    "target_label_5",
    "error_type",
    "question_preview",
    "question",
    "answer_synthetic",
    "rubric_text",
    "rationale_for_label",
    "expected_failure_against_rubric",
    "label_plausibility_status",
    "error_type_alignment_status",
    "rubric_failure_visibility",
    "too_good_for_target_label",
    "manual_review_required",
    "manual_quality_ok",
    "manual_label_plausible",
    "manual_notes",
    "review_natural_plausible",
    "review_label_plausible",
    "review_error_type_aligned",
    "review_no_artifact_phrase",
    "review_no_copy_or_leakage_concern",
    "review_keep_for_topup1",
    "reviewer_notes",
]


def ensure_topup1_dirs() -> None:
    for path in [
        TOPUP1_OUTPUT_DIR,
        TOPUP1_TABLES_DIR,
        TOPUP1_PROMPTS_DIR,
        TOPUP1_GENERATED_DIR,
        TOPUP1_FILTERED_DIR,
        TOPUP1_LEAKAGE_DIR,
        TOPUP1_SPOTCHECK_DIR,
        TOPUP1_REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def table_path(name: str) -> Path:
    ensure_topup1_dirs()
    return TOPUP1_TABLES_DIR / name


def prompt_path() -> Path:
    ensure_topup1_dirs()
    return TOPUP1_PROMPTS_DIR / "topup1_prompts.jsonl"


def generated_path(name: str) -> Path:
    ensure_topup1_dirs()
    return TOPUP1_GENERATED_DIR / name


def filtered_path(name: str) -> Path:
    ensure_topup1_dirs()
    return TOPUP1_FILTERED_DIR / name


def leakage_path(name: str) -> Path:
    ensure_topup1_dirs()
    return TOPUP1_LEAKAGE_DIR / name


def report_path(name: str) -> Path:
    ensure_topup1_dirs()
    return TOPUP1_REPORTS_DIR / name


def clean_multiline(value: Any) -> str:
    return "\n".join(line.rstrip() for line in stringify(value).splitlines())


def build_target_matrix() -> list[dict[str, Any]]:
    labels_remaining = dict(TOPUP1_TARGET_LABEL_COUNTS)
    langs = ["en" if index % 2 == 0 else "zh" for index in range(TOPUP1_TOTAL_TARGET)]
    metric_counts: Counter[str] = Counter()
    error_counts_by_metric: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for index in range(TOPUP1_TOTAL_TARGET):
        label = max(
            labels_remaining,
            key=lambda item: (labels_remaining[item] / TOPUP1_TARGET_LABEL_COUNTS[item], item),
        )
        labels_remaining[label] -= 1
        metric = PRIORITY_METRICS[index % len(PRIORITY_METRICS)]
        sequence = [item for item in METRIC_ERROR_SEQUENCE.get(metric, []) if item in ERROR_TYPES] or ["rubric_violation"]
        error_type = sequence[error_counts_by_metric[metric] % len(sequence)]
        error_counts_by_metric[metric] += 1
        metric_counts[metric] += 1
        rows.append(
            {
                "synthetic_plan_id": f"topup1_{index + 1:03d}",
                "target_label_5": label,
                "language": langs[index],
                "metric_canonical": metric,
                "error_type": error_type,
                "target_count": 1,
                "priority_rank": PRIORITY_METRICS.index(metric) + 1,
                "generation_split_mode": "question_disjoint_formal",
                "generation_prompt_version": TOPUP1_PROMPT_VERSION,
                "notes": "Exp6-9 topup-1 raw generation target; question_seed42/train only",
            }
        )
    return rows


def batch96_question_reuse_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    if not BATCH96_CURATED_PATH.exists():
        return counts
    for row in read_jsonl(BATCH96_CURATED_PATH):
        key = stringify(row.get("source_question_key"))
        if key:
            counts[key] += 1
    return counts


def choose_sources(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mode = "question_disjoint_formal"
    train_by_id = load_mode_train_by_source_id(mode)
    keys = mode_key_sets(mode)
    batch96_question_counts = batch96_question_reuse_counts()
    topup_question_counts: Counter[str] = Counter()
    used_source_ids: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for target in target_rows:
        metric = stringify(target.get("metric_canonical"))
        language = stringify(target.get("language"))
        exact = [
            (source_id, row, "exact_metric_language")
            for source_id, row in train_by_id.items()
            if row.get("metric_canonical") == metric
            and row.get("language") == language
            and source_allowed_for_mode(row, mode, keys)
        ]
        pool = exact or [
            (source_id, row, "fallback_language_only_metric_unavailable")
            for source_id, row in train_by_id.items()
            if row.get("language") == language and source_allowed_for_mode(row, mode, keys)
        ]
        if not pool:
            rows.append(unavailable_source_row(target))
            continue

        source_id, train_row, status = sorted(
            pool,
            key=lambda item: (
                batch96_question_counts[stringify(item[1].get("question_key"))] > 0,
                topup_question_counts[stringify(item[1].get("question_key"))],
                used_source_ids[item[0]],
                -int(float(stringify(item[1].get("label_5") or 0))),
                item[0],
            ),
        )[0]
        question_key_value = stringify(train_row.get("question_key"))
        topup_question_counts[question_key_value] += 1
        used_source_ids[source_id] += 1
        flags = overlap_flags(train_row, keys)
        synthetic_id = synthetic_id_from_parts(
            target.get("synthetic_plan_id"),
            source_id,
            target.get("target_label_5"),
            target.get("error_type"),
            TOPUP1_PROMPT_VERSION,
        )
        batch96_reuse = batch96_question_counts[question_key_value]
        topup_reuse = topup_question_counts[question_key_value]
        rows.append(
            {
                "synthetic_plan_id": target.get("synthetic_plan_id", ""),
                "synthetic_id": synthetic_id,
                "source_record_id": source_id,
                "source_split": "train",
                "source_file": train_row.get("source_file", ""),
                "source_row_index": train_row.get("source_row_index", ""),
                "source_question_key": question_key_value,
                "source_triple_key": train_row.get("triple_key", ""),
                "question": clean_multiline(train_row.get("question")),
                "rubric_text": clean_multiline(train_row.get("rubric")),
                "target_label_5": target.get("target_label_5", ""),
                "planned_target_label_5": target.get("target_label_5", ""),
                "language": target.get("language", ""),
                "metric_canonical": target.get("metric_canonical", ""),
                "error_type": target.get("error_type", ""),
                "planned_error_type": target.get("error_type", ""),
                "current_label_5": train_row.get("label_5", ""),
                "human_label_5": train_row.get("label_5", ""),
                "scenario_canonical": train_row.get("scenario_canonical", ""),
                "subject_canonical": train_row.get("subject_canonical", ""),
                "education_level_canonical": train_row.get("education_level_canonical", ""),
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
                "batch96_source_question_reuse_count": batch96_reuse,
                "topup1_source_question_reuse_count": topup_reuse,
                "source_reuse_status": (
                    "new_question_not_used_by_batch96"
                    if batch96_reuse == 0 and topup_reuse == 1
                    else "question_reused_recorded"
                ),
                "question_preview": truncate_text(clean_multiline(train_row.get("question")), 700),
                "source_answer_preview": truncate_text(clean_multiline(train_row.get("answer")), 700),
                "rubric_preview": truncate_text(clean_multiline(train_row.get("rubric")), 700),
                "selection_notes": "question_seed42/train source; batch96 source question reuse minimized and recorded",
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
            "source_reuse_status": "unavailable",
            "selection_notes": "BLOCKED: no eligible question_seed42/train source found",
        }
    )
    return row


def thinking_low_score_guard(language: str) -> str:
    if language == "zh":
        return """

Thinking-mode low-label guard:
- 推理模式会倾向于把答案修好；不要让最终 `answer_synthetic` 变成好答案。
- `answer_synthetic` 本身必须保留可见的 rubric-relevant 缺陷，而不只是审计字段说明有缺陷。
- 目标标签 1：答案必须有严重实质性失败，例如核心事实/推理/场景/指令明显错误；不能只是格式差、语气普通或细节不足。
- 目标标签 2：答案必须有主要缺陷，使其不可靠；不能是最终结论正确且只有轻微遗漏。
- 目标标签 3：答案必须是边界质量，有清楚可扣分的弱点；不能明显达到 4 分或 5 分。
- 如果你发现最终答案已经明显高于目标标签，必须设置 `needs_revision=true`，并说明 `revision_reason`。
- 最终答案不要提及标签、评分、实验、生成、低分、故意错误或 synthetic 身份。
"""
    return """

Thinking-mode low-label guard:
- Thinking mode tends to repair answers; do not let the final `answer_synthetic` become a good answer.
- The `answer_synthetic` itself must keep a visible rubric-relevant failure, not merely describe a failure in the audit fields.
- Target label 1: include a severe substantive failure, such as a clearly wrong core fact, reasoning path, scenario use, or instruction response; do not make it only a formatting, tone, or detail issue.
- Target label 2: include a major defect that makes the answer unreliable; do not provide a correct final answer with only minor omissions.
- Target label 3: keep it at boundary quality with a clear weakness a human reviewer would mark down; do not let it reach obvious label 4 or 5 quality.
- If the final answer is clearly stronger than the target label, set `needs_revision=true` and explain `revision_reason`.
- The final answer must not mention labels, scoring, experiments, generation, low-quality intent, or synthetic identity.
"""


def render_prompt(source_row: dict[str, Any], train_row: dict[str, Any], templates: dict[str, str]) -> str:
    key = "zh" if source_row.get("language") == "zh" else "en"
    rendered = templates[key].format(
        language="Chinese" if key == "zh" else "English",
        metric_canonical=source_row.get("metric_canonical", ""),
        target_label_5=source_row.get("target_label_5", ""),
        error_type=source_row.get("error_type", ""),
        question=stringify(train_row.get("question")),
        rubric_text=stringify(train_row.get("rubric")),
        source_answer=truncate_text(train_row.get("answer"), 1500),
    )
    return rendered + thinking_low_score_guard(key)


def build_prompts(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_by_id = load_mode_train_by_source_id("question_disjoint_formal")
    template_paths = {
        "en": BATCH96_PROMPT_TEMPLATE_DIR / "generate_low_score_answer_hardened_en.md",
        "zh": BATCH96_PROMPT_TEMPLATE_DIR / "generate_low_score_answer_hardened_zh.md",
    }
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
                "generation_prompt_version": TOPUP1_PROMPT_VERSION,
                "prompt_text": render_prompt(source_row, train_row, templates),
            }
        )
    return rows


def build_plan_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_topup1_dirs()
    targets = build_target_matrix()
    sources = choose_sources(targets)
    prompts = build_prompts(sources)
    write_csv(table_path("topup1_target_matrix.csv"), targets, TARGET_FIELDS)
    write_csv(table_path("topup1_source_selection.csv"), sources, SOURCE_FIELDS)
    write_jsonl(prompt_path(), prompts)
    return targets, sources, prompts


def scan_topup1_for_secret_markers() -> int:
    hits = 0
    key_prefix = "sk" + "-"
    if not TOPUP1_OUTPUT_DIR.exists():
        return 0
    for path in TOPUP1_OUTPUT_DIR.rglob("*"):
        if not path.is_file() or path.stat().st_size > 10_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if key_prefix in text or "Bearer" + " " in text:
            hits += 1
    return hits


def tracked_weight_files() -> list[str]:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "thesis_exp/artifacts", "thesis_exp/outputs/exp06_synthetic_low_score"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    suffixes = (".pt", ".bin", ".safetensors", ".ckpt")
    return [
        name
        for name in result.stdout.splitlines()
        if name.endswith(suffixes) or "checkpoint" in name.lower() or "weight" in name.lower()
    ]


def static_checks() -> tuple[bool, list[dict[str, Any]]]:
    ensure_topup1_dirs()
    rows: list[dict[str, Any]] = []
    compile_failures: list[str] = []
    for path in sorted(EXP06_GENERATION_SRC_DIR.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_failures.append(f"{path.name}: {exc.msg}")
    append_check(rows, "py_compile_exp06_generation", not compile_failures, len(compile_failures), "; ".join(compile_failures))

    prompts = read_jsonl_if_exists(prompt_path())
    targets = read_csv_if_exists(table_path("topup1_target_matrix.csv"))
    sources = read_csv_if_exists(table_path("topup1_source_selection.csv"))
    label_counts = Counter(stringify(row.get("target_label_5")) for row in targets)
    language_counts = Counter(stringify(row.get("language")) for row in targets)
    metric_coverage = len({row.get("metric_canonical") for row in targets if row.get("metric_canonical")})
    error_coverage = len({row.get("error_type") for row in targets if row.get("error_type")})
    source_question_overlap = sum(
        1 for row in sources if row.get("source_question_in_dev") == "True" or row.get("source_question_in_test") == "True"
    )
    source_triple_overlap = sum(
        1 for row in sources if row.get("source_triple_in_dev") == "True" or row.get("source_triple_in_test") == "True"
    )
    non_train_sources = sum(1 for row in sources if row.get("source_split") != "train")
    batch96_reuse_rows = sum(1 for row in sources if int(stringify(row.get("batch96_source_question_reuse_count") or 0)) > 0)

    append_check(rows, "topup1_prompts_exists", prompt_path().exists(), int(prompt_path().exists()), relpath(prompt_path()))
    append_check(rows, "prompt_count_160", len(prompts) == TOPUP1_TOTAL_TARGET, len(prompts), "expected 160")
    append_check(rows, "label_distribution_75_68_17", dict(label_counts) == TOPUP1_TARGET_LABEL_COUNTS, len(targets), str(dict(label_counts)))
    append_check(rows, "language_distribution_80_80", dict(language_counts) == TOPUP1_LANGUAGE_COUNTS, len(targets), str(dict(language_counts)))
    append_check(rows, "metric_coverage_12", metric_coverage == 12, metric_coverage, "12 metrics")
    append_check(rows, "error_type_coverage_7", error_coverage == 7, error_coverage, "7 error types")
    append_check(rows, "all_sources_from_question_seed42_train", non_train_sources == 0 and len(sources) == TOPUP1_TOTAL_TARGET, non_train_sources, "source_split must be train")
    append_check(rows, "no_source_question_overlap_dev_test", source_question_overlap == 0, source_question_overlap, "dev/test question forbidden")
    append_check(rows, "no_source_triple_overlap_dev_test", source_triple_overlap == 0, source_triple_overlap, "dev/test triple forbidden")
    append_check(rows, "source_reuse_count_recorded", all("batch96_source_question_reuse_count" in row for row in sources), batch96_reuse_rows, "rows reusing Batch96 source questions")
    secret_hits = scan_topup1_for_secret_markers()
    append_check(rows, "api_key_not_written_to_topup1_logs", secret_hits == 0, secret_hits, "scanned secret-like key markers")
    weights = tracked_weight_files()
    append_check(rows, "no_checkpoint_or_weights_tracked", not weights, len(weights), "; ".join(weights[:10]))
    write_csv(table_path("static_checks_topup1_generation.csv"), rows, STATIC_FIELDS)
    return all(row["status"] == "PASS" for row in rows), rows


def write_blocked_report(reason: str, static_rows: list[dict[str, Any]]) -> None:
    text = f"""# BLOCKED: Exp6-9 Topup-1 Generation Did Not Run

Generation was not executed and no synthetic rows were fabricated.

Reason: **{reason}**

Required environment:

- `EXP6_RUN_GENERATION=1`
- `GENERATION_MODEL=deepseek-v4-pro`
- an API key environment variable, unless a local endpoint is configured and needs no key

Static checks:

{md_table(static_rows, STATIC_FIELDS, max_rows=30)}

The runner is capped at exactly 160 prompt rows and never logs API keys.
"""
    write_text(report_path("BLOCKED_NO_GENERATION.md"), text)


def generation_allowed(static_ok: bool) -> tuple[bool, str]:
    if not static_ok:
        return False, "static checks failed"
    if os.environ.get("EXP6_RUN_GENERATION") != "1":
        return False, "EXP6_RUN_GENERATION is not set to 1"
    env = generation_env()
    if not (os.getenv("GENERATION_MODEL") or DEFAULT_GENERATION_MODEL):
        return False, "missing GENERATION_MODEL"
    if not env["api_key"] and not env["endpoint"]:
        return False, "missing API key or local GENERATION_ENDPOINT"
    return True, "generation enabled"


def raw_rows_are_complete(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == TOPUP1_TOTAL_TARGET and all(row.get("generation_status") == "generated" for row in rows)


def run_generation(request_timeout: int = 240, retries: int = 2, resume: bool = True) -> list[dict[str, Any]]:
    rows = read_jsonl_if_exists(prompt_path())[:TOPUP1_TOTAL_TARGET]
    env = generation_env()
    model = os.getenv("GENERATION_MODEL") or env["model"] or DEFAULT_GENERATION_MODEL
    endpoint = env["endpoint"] or "https://api.deepseek.com/chat/completions"
    api_key = env["api_key"]
    raw_path = generated_path("raw_generations.jsonl")
    existing_by_plan: dict[str, dict[str, Any]] = {}
    if resume and raw_path.exists():
        for row in read_jsonl_if_exists(raw_path):
            plan_id = stringify(row.get("synthetic_plan_id"))
            if plan_id:
                existing_by_plan[plan_id] = row
    plan_order = [stringify(row.get("synthetic_plan_id")) for row in rows]
    out_by_plan = {plan_id: row for plan_id, row in existing_by_plan.items() if plan_id in set(plan_order)}

    for index, row in enumerate(rows, start=1):
        plan_id = stringify(row.get("synthetic_plan_id"))
        existing = out_by_plan.get(plan_id)
        if existing and existing.get("generation_status") == "generated":
            try:
                validate_generated_content(stringify(existing.get("raw_response")))
                print(f"topup1_generation_progress row={index}/160 status=skipped_existing", flush=True)
                continue
            except Exception:
                pass
        status = "failed"
        raw_response = ""
        error_message = ""
        for attempt in range(retries + 1):
            try:
                raw_response = post_chat_completion(
                    endpoint,
                    api_key,
                    model,
                    stringify(row.get("prompt_text")),
                    timeout=request_timeout,
                )
                validate_generated_content(raw_response)
                status = "generated"
                error_message = ""
                break
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                OSError,
                ValueError,
            ) as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(2**attempt)
        out_by_plan[plan_id] = {
            "synthetic_plan_id": plan_id,
            "synthetic_id": row.get("synthetic_id", ""),
            "source_record_id": row.get("source_record_id", ""),
            "raw_prompt": row.get("prompt_text", ""),
            "raw_response": raw_response,
            "generation_model": model,
            "generation_status": status,
            "error_message": error_message,
            "created_at": now_iso(),
        }
        write_jsonl(raw_path, [out_by_plan[plan_id] for plan_id in plan_order if plan_id in out_by_plan])
        print(f"topup1_generation_progress row={index}/160 status={status}", flush=True)
    output = [out_by_plan[plan_id] for plan_id in plan_order if plan_id in out_by_plan]
    write_jsonl(raw_path, output)
    return output


def normalize_raw_rows() -> list[dict[str, Any]]:
    raw_rows = read_jsonl_if_exists(generated_path("raw_generations.jsonl"))
    sources_by_plan = {
        row.get("synthetic_plan_id", ""): row for row in read_csv_if_exists(table_path("topup1_source_selection.csv"))
    }
    prompts_by_plan = {row.get("synthetic_plan_id", ""): row for row in read_jsonl_if_exists(prompt_path())}
    train_by_id = load_mode_train_by_source_id("question_disjoint_formal")
    normalized: list[dict[str, Any]] = []
    for raw in raw_rows[:TOPUP1_TOTAL_TARGET]:
        plan_id = stringify(raw.get("synthetic_plan_id"))
        source_meta = sources_by_plan.get(plan_id, {})
        prompt_meta = prompts_by_plan.get(plan_id, {})
        source_id = stringify(raw.get("source_record_id") or source_meta.get("source_record_id"))
        train_row = train_by_id.get(source_id, {})
        parsed, parse_status = parse_raw_response(stringify(raw.get("raw_response")))
        notes = [parse_status]
        plan_label = stringify(source_meta.get("target_label_5") or prompt_meta.get("target_label_5"))
        plan_error = stringify(source_meta.get("error_type") or prompt_meta.get("error_type"))
        returned_label = stringify(parsed.get("target_label_5"))
        returned_error = stringify(parsed.get("error_type"))
        if returned_label and returned_label != plan_label:
            notes.append(f"returned_label_mismatch:{returned_label}; using_plan_label:{plan_label}")
        if returned_error and returned_error != plan_error:
            notes.append(f"returned_error_type_mismatch:{returned_error}; using_plan_error:{plan_error}")
        answer = stringify(parsed.get("answer_synthetic"))
        if not answer:
            notes.append("missing_answer_synthetic")
        normalized.append(
            {
                "synthetic_id": raw.get("synthetic_id") or source_meta.get("synthetic_id") or prompt_meta.get("synthetic_id", ""),
                "synthetic_plan_id": plan_id,
                "source_record_id": source_id,
                "source_question_key": source_meta.get("source_question_key") or train_row.get("question_key", ""),
                "source_triple_key": source_meta.get("source_triple_key") or train_row.get("triple_key", ""),
                "source_split": source_meta.get("source_split") or ("train" if train_row else ""),
                "question": train_row.get("question", source_meta.get("question", "")),
                "source_answer": train_row.get("answer", ""),
                "answer_synthetic": answer,
                "metric_canonical": source_meta.get("metric_canonical") or train_row.get("metric_canonical", ""),
                "rubric_text": train_row.get("rubric", source_meta.get("rubric_text", "")),
                "language": source_meta.get("language") or train_row.get("language", ""),
                "scenario_canonical": train_row.get("scenario_canonical", source_meta.get("scenario_canonical", "")),
                "subject_canonical": train_row.get("subject_canonical", source_meta.get("subject_canonical", "")),
                "education_level_canonical": train_row.get("education_level_canonical", source_meta.get("education_level_canonical", "")),
                "target_label_5": plan_label,
                "label_source": "synthetic_design",
                "label_provenance": "pseudo_label",
                "error_type": plan_error,
                "generation_model": raw.get("generation_model", ""),
                "generation_prompt_version": TOPUP1_PROMPT_VERSION,
                "rationale_for_label": stringify(parsed.get("rationale_for_label")),
                "expected_failure_against_rubric": stringify(parsed.get("expected_failure_against_rubric")),
                "label_plausibility_self_check": stringify(parsed.get("label_plausibility_self_check")),
                "error_type_alignment_self_check": stringify(parsed.get("error_type_alignment_self_check")),
                "rubric_failure_visibility": stringify(parsed.get("rubric_failure_visibility")),
                "too_good_for_target_label": parsed.get("too_good_for_target_label", ""),
                "needs_revision": parsed.get("needs_revision", ""),
                "revision_reason": stringify(parsed.get("revision_reason")),
                "raw_generation": raw.get("raw_response", ""),
                "generation_status": raw.get("generation_status", ""),
                "normalization_status": "pass" if parse_status == "parsed_json" and answer else "fail",
                "normalization_notes": "; ".join(notes),
            }
        )
    write_jsonl(generated_path("normalized_synthetic_candidates.jsonl"), normalized)
    return normalized


def filter_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl_if_exists(generated_path("normalized_synthetic_candidates.jsonl"))
    test_rows = read_jsonl(Path("thesis_exp/data/splits/question_seed42/test.jsonl"))
    human_test_answer_keys = {answer_hash(row.get("answer")) for row in test_rows}
    seen_answers: set[str] = set()
    passed_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for row in rows:
        reasons = required_filter_reasons(row, seen_answers, human_test_answer_keys)
        hardened = apply_hardened_checks(row)
        if hardened["artifact_phrase_status"] == "fail":
            reasons.append("artifact_phrase")
        if stringify(row.get("target_label_5")) in {"1", "2"} and hardened["rubric_failure_visibility"] == "missing_clear_failure":
            reasons.append("missing_clear_rubric_failure")
        if stringify(row.get("target_label_5")) == "3" and hardened["too_good_for_target_label"] == "manual_review_required":
            reasons.append("too_good_for_target_label")
        row.update(hardened)
        row["filter_status"] = "pass" if not reasons else "fail"
        row["filter_reasons"] = reasons
        for reason in reasons:
            reason_counter[reason.split(":")[0]] += 1
        if not reasons:
            passed_rows.append(row)
        detail_rows.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "passed_filter": not reasons,
                "filter_reasons": "; ".join(reasons),
                "filter_stage": "topup1_required_plus_hardened_filters",
                "filter_status": row["filter_status"],
                "failure_reasons": "; ".join(reasons),
                **{field: hardened.get(field, "") for field in HARDENED_FILTER_FIELDS},
                "hardened_filter_reasons": hardened.get("hardened_filter_reasons", ""),
                "notes": "Exp6-9 topup1 hardened filter",
            }
        )
    report_rows = [
        {"check_name": "total_normalized_rows", "status": "INFO", "count": len(rows), "notes": "input normalized rows"},
        {"check_name": "passed_filter_rows", "status": "PASS" if passed_rows else "INFO", "count": len(passed_rows), "notes": "rows written to filtered candidates"},
        {"check_name": "failed_filter_rows", "status": "FAIL" if len(detail_rows) - len(passed_rows) else "PASS", "count": len(detail_rows) - len(passed_rows), "notes": "rows blocked by filters"},
    ]
    if not rows:
        report_rows.append({"check_name": "generation_presence", "status": "BLOCKED_NO_GENERATION", "count": 0, "notes": "no raw generations available"})
    for reason, count in sorted(reason_counter.items()):
        report_rows.append({"check_name": f"failure_reason:{reason}", "status": "FAIL", "count": count, "notes": "filter failure reason count"})
    write_jsonl(filtered_path("filtered_synthetic_candidates.jsonl"), passed_rows)
    write_csv(table_path("filter_report.csv"), report_rows, FILTER_REPORT_FIELDS)
    write_csv(table_path("filter_failure_details.csv"), detail_rows, FILTER_DETAIL_FIELDS)
    return passed_rows, report_rows, detail_rows


def leakage_check() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl_if_exists(filtered_path("filtered_synthetic_candidates.jsonl"))
    keys = mode_key_sets("question_disjoint_formal")
    dev_rows = read_jsonl(Path("thesis_exp/data/splits/question_seed42/dev.jsonl"))
    test_rows = read_jsonl(Path("thesis_exp/data/splits/question_seed42/test.jsonl"))
    keys["dev"]["qa_key"] = {qa_key(row.get("question"), row.get("answer")) for row in dev_rows}
    keys["test"]["qa_key"] = {qa_key(row.get("question"), row.get("answer")) for row in test_rows}
    keys["test"]["answer_key"] = {answer_hash(row.get("answer")) for row in test_rows}
    seen_answers: set[str] = set()
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        answer_key = answer_hash(row.get("answer_synthetic"))
        duplicate_answer = answer_key in seen_answers
        seen_answers.add(answer_key)
        synthetic_question = question_key(row.get("question"))
        synthetic_qa = qa_key(row.get("question"), row.get("answer_synthetic"))
        checks = {
            "source_question_key_in_dev": row.get("source_question_key") in keys["dev"]["question_key"],
            "source_question_key_in_test": row.get("source_question_key") in keys["test"]["question_key"],
            "source_triple_key_in_dev": row.get("source_triple_key") in keys["dev"]["triple_key"],
            "source_triple_key_in_test": row.get("source_triple_key") in keys["test"]["triple_key"],
            "synthetic_question_key_in_dev": synthetic_question in keys["dev"]["norm_question_key"],
            "synthetic_question_key_in_test": synthetic_question in keys["test"]["norm_question_key"],
            "synthetic_qa_key_in_dev": synthetic_qa in keys["dev"]["qa_key"],
            "synthetic_qa_key_in_test": synthetic_qa in keys["test"]["qa_key"],
            "duplicate_synthetic_answer": duplicate_answer,
            "duplicate_with_human_test_answer": answer_key in keys["test"]["answer_key"],
            "source_split_not_train": row.get("source_split") != "train",
        }
        blocked = any(checks.values())
        detail_rows.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "source_record_id": row.get("source_record_id", ""),
                **checks,
                "leakage_status": "BLOCKED" if blocked else "PASS",
                "notes": "Exp6-9 topup1 leakage and dedup check",
            }
        )
    blocked_rows = [row for row in detail_rows if row["leakage_status"] == "BLOCKED"]
    summary_rows = [
        {"check_name": "total_filtered_rows", "status": "INFO", "count": len(rows), "notes": "input rows after hardened filter"},
        {"check_name": "passed_leakage_rows", "status": "PASS" if not blocked_rows else "INFO", "count": len(rows) - len(blocked_rows), "notes": "rows without leakage/dedup hits"},
        {"check_name": "blocked_leakage_rows", "status": "FAIL" if blocked_rows else "PASS", "count": len(blocked_rows), "notes": "any blocked row blocks training use"},
    ]
    if not rows:
        summary_rows.append({"check_name": "generation_presence", "status": "BLOCKED_NO_GENERATION", "count": 0, "notes": "no filtered rows available"})
    write_csv(leakage_path("leakage_summary.csv"), summary_rows, STATIC_FIELDS)
    write_csv(leakage_path("leakage_details.csv"), detail_rows, LEAKAGE_DETAIL_FIELDS)
    write_leakage_report(detail_rows)
    return summary_rows, detail_rows


def write_leakage_report(detail_rows: list[dict[str, Any]]) -> None:
    blocked = sum(1 for row in detail_rows if row.get("leakage_status") == "BLOCKED")
    status = "BLOCKED_NO_GENERATION" if not detail_rows else ("BLOCKED" if blocked else "PASS")
    text = f"""# Exp6-9 Topup-1 Leakage Report

Status: **{status}**

- Filtered rows checked: **{len(detail_rows)}**
- Blocked rows: **{blocked}**
- Leakage summary: `{relpath(leakage_path("leakage_summary.csv"))}`
- Leakage details: `{relpath(leakage_path("leakage_details.csv"))}`

Checks:

- source_question_key not in dev/test
- source_triple_key not in dev/test
- synthetic question not in dev/test
- synthetic question+answer not in dev/test
- answer_synthetic not duplicate with human test answer
- duplicate synthetic answer
"""
    write_text(leakage_path("leakage_report.md"), text)


def build_spotcheck_package() -> list[dict[str, Any]]:
    rows = read_jsonl_if_exists(filtered_path("filtered_synthetic_candidates.jsonl"))
    sample = spotcheck_selection(rows)
    out: list[dict[str, Any]] = []
    for row in sample:
        out.append(
            {
                "synthetic_id": row.get("synthetic_id", ""),
                "synthetic_plan_id": row.get("synthetic_plan_id", ""),
                "source_record_id": row.get("source_record_id", ""),
                "language": row.get("language", ""),
                "metric_canonical": row.get("metric_canonical", ""),
                "target_label_5": row.get("target_label_5", ""),
                "error_type": row.get("error_type", ""),
                "question_preview": truncate_text(row.get("question", ""), 1200),
                "question": truncate_text(row.get("question", ""), 1200),
                "answer_synthetic": row.get("answer_synthetic", ""),
                "rubric_text": truncate_text(row.get("rubric_text", ""), 1200),
                "rationale_for_label": row.get("rationale_for_label", ""),
                "expected_failure_against_rubric": row.get("expected_failure_against_rubric", ""),
                **{field: row.get(field, "") for field in HARDENED_FILTER_FIELDS},
                "manual_quality_ok": "",
                "manual_label_plausible": "",
                "manual_notes": "",
                "review_natural_plausible": "",
                "review_label_plausible": "",
                "review_error_type_aligned": "",
                "review_no_artifact_phrase": "",
                "review_no_copy_or_leakage_concern": "",
                "review_keep_for_topup1": "",
                "reviewer_notes": "",
            }
        )
    write_csv(TOPUP1_SPOTCHECK_DIR / "spotcheck_samples.csv", out, SPOTCHECK_FIELDS)
    guideline = f"""# Exp6-9 Topup-1 Spotcheck Guidelines

Samples to review: **{len(out)}**

If filtered samples are 80 or fewer, review all. Otherwise review this stratified sample,
covering labels, languages, error types, and metrics.

Synthetic labels remain `synthetic_design` pseudo-labels, not human labels. Exp6 training
remains blocked until the full filtered pool is complete and reviewed.
"""
    write_text(TOPUP1_SPOTCHECK_DIR / "spotcheck_guidelines.md", guideline)
    return out


def readability_check() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(TOPUP1_OUTPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = relpath(path)
        data = path.read_bytes()
        rows.append({"check_name": "file_size_under_10mb", "path": rel, "status": "PASS" if len(data) <= 10_000_000 else "FAIL", "details": f"bytes={len(data)}"})
        rows.append({"check_name": "lf_endings", "path": rel, "status": "PASS" if b"\r" not in data else "FAIL", "details": "LF/no CR newline" if b"\r" not in data else "CR newline found"})
        if path.suffix == ".jsonl":
            try:
                rows.append({"check_name": "jsonl_loadable", "path": rel, "status": "PASS", "details": f"records={len(read_jsonl(path))}"})
            except Exception as exc:
                rows.append({"check_name": "jsonl_loadable", "path": rel, "status": "FAIL", "details": f"{type(exc).__name__}: {exc}"})
        if path.suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    count = sum(1 for _ in csv.reader(handle))
                rows.append({"check_name": "csv_readable", "path": rel, "status": "PASS", "details": f"rows={count}"})
            except Exception as exc:
                rows.append({"check_name": "csv_readable", "path": rel, "status": "FAIL", "details": f"{type(exc).__name__}: {exc}"})
    write_csv(table_path("readability_check_topup1_generation.csv"), rows, READABILITY_FIELDS)
    return rows


def manual_review_required_count(rows: list[dict[str, Any]]) -> int:
    return sum(stringify(row.get("manual_review_required")).lower() in {"true", "yes", "1", "required"} for row in rows)


def write_generation_reports(static_rows: list[dict[str, Any]], generation_reason: str) -> dict[str, Any]:
    raw_rows = read_jsonl_if_exists(generated_path("raw_generations.jsonl"))
    normalized_rows = read_jsonl_if_exists(generated_path("normalized_synthetic_candidates.jsonl"))
    filtered_rows = read_jsonl_if_exists(filtered_path("filtered_synthetic_candidates.jsonl"))
    leakage_summary = read_csv_if_exists(leakage_path("leakage_summary.csv"))
    blocked_leakage = next((row for row in leakage_summary if row.get("check_name") == "blocked_leakage_rows"), {})
    leakage_status = "BLOCKED_NO_GENERATION" if not filtered_rows else ("BLOCKED" if stringify(blocked_leakage.get("count")) not in {"", "0"} else "PASS")
    dist = distribution(filtered_rows)
    static_ok = all(row["status"] == "PASS" for row in static_rows)
    api_called = bool(raw_rows)
    generated_count = sum(1 for row in raw_rows if row.get("generation_status") == "generated")
    manual_count = manual_review_required_count(filtered_rows)
    can_topup2_start = bool(filtered_rows) and leakage_status == "PASS"
    status = {
        "static_checks_status": "PASS" if static_ok else "FAIL",
        "api_called": api_called,
        "generated_count": generated_count,
        "normalized_count": len(normalized_rows),
        "filtered_pass_count": len(filtered_rows),
        "leakage_status": leakage_status,
        "label_distribution_after_filter": dist["labels"],
        "language_distribution_after_filter": dist["languages"],
        "metric_coverage_after_filter": dist["metric_coverage"],
        "error_type_coverage_after_filter": dist["error_type_coverage"],
        "manual_review_required_count": manual_count,
        "can_topup2_start": can_topup2_start,
        "can_training_start": False,
    }
    report = f"""# Exp6-9 Topup-1 Synthetic Generation Report

## Status

- Static checks status: **{status["static_checks_status"]}**
- API called: **{"YES" if api_called else "NO"}**
- API generation status: **{generation_reason}**
- Planned count: **160**
- Generated count: **{generated_count}**
- Normalized count: **{len(normalized_rows)}**
- Filtered pass count: **{len(filtered_rows)}**
- Leakage status: **{leakage_status}**

## Filtered Distribution

- Label distribution: `{dist["labels"]}`
- Language distribution: `{dist["languages"]}`
- Metric coverage: **{dist["metric_coverage"]}**
- Error type coverage: **{dist["error_type_coverage"]}**
- Error type distribution: `{dist["error_types"]}`
- Manual review required count: **{manual_count}**

## Gates

- Can topup2 start: **{"YES" if can_topup2_start else "NO"}**
- Direct full 384 generation can start: **NO**
- Exp6 training can start: **NO**
"""
    write_text(report_path("topup1_generation_report.md"), report)
    review = f"""# Exp6-9 Topup-1 Review Package

- Planned count: **160**
- Generated count: **{generated_count}**
- Normalized count: **{len(normalized_rows)}**
- Filtered pass count: **{len(filtered_rows)}**
- Leakage status: **{leakage_status}**
- Label distribution after filter: `{dist["labels"]}`
- Language distribution after filter: `{dist["languages"]}`
- Metric coverage after filter: **{dist["metric_coverage"]}**
- Error type coverage after filter: **{dist["error_type_coverage"]}**
- Manual review required count: **{manual_count}**
- Spotcheck required: **YES**
- Can topup2 start: **{"YES" if can_topup2_start else "NO"}**
- Exp6 training can start: **NO**

Synthetic labels remain `synthetic_design` pseudo-labels and must not be treated as human labels.
"""
    write_text(report_path("topup1_review_package.md"), review)
    notion = f"""# Exp6 Topup-1 Generation Summary

| item | value |
| --- | --- |
| Static checks | {status["static_checks_status"]} |
| API called | {"YES" if api_called else "NO"} |
| Generated count | {generated_count} |
| Normalized count | {len(normalized_rows)} |
| Filtered pass count | {len(filtered_rows)} |
| Leakage status | {leakage_status} |
| Label distribution after filter | `{dist["labels"]}` |
| Language distribution after filter | `{dist["languages"]}` |
| Metric coverage after filter | {dist["metric_coverage"]} |
| Error type coverage after filter | {dist["error_type_coverage"]} |
| Manual review required count | {manual_count} |
| Spotcheck package | `{relpath(TOPUP1_SPOTCHECK_DIR / "spotcheck_samples.csv")}` |
| Topup2 can start | {"YES" if can_topup2_start else "NO"} |
| Training can start | NO |
"""
    write_text(report_path("notion_exp06_topup1_summary.md"), notion)
    sanity_rows = [
        {"check_name": "static_checks_status", "status": status["static_checks_status"], "count": len(static_rows), "notes": "pre-generation checks"},
        {"check_name": "api_called", "status": "INFO", "count": int(api_called), "notes": generation_reason},
        {"check_name": "generated_count", "status": "PASS" if generated_count in {0, TOPUP1_TOTAL_TARGET} else "FAIL", "count": generated_count, "notes": "0 only allowed for blocked generation"},
        {"check_name": "normalized_count", "status": "PASS" if len(normalized_rows) in {0, TOPUP1_TOTAL_TARGET} else "FAIL", "count": len(normalized_rows), "notes": "normalized rows"},
        {"check_name": "filtered_pass_count", "status": "INFO" if filtered_rows else "BLOCKED_NO_GENERATION", "count": len(filtered_rows), "notes": "filtered rows"},
        {"check_name": "leakage_status", "status": "PASS" if leakage_status in {"PASS", "BLOCKED_NO_GENERATION"} else "FAIL", "count": len(filtered_rows), "notes": leakage_status},
        {"check_name": "full_384_generation_blocked", "status": "PASS", "count": 0, "notes": "NO"},
        {"check_name": "exp6_training_blocked", "status": "PASS", "count": 0, "notes": "NO"},
    ]
    write_csv(table_path("sanity_check_topup1_generation.csv"), sanity_rows, STATIC_FIELDS)
    text = f"""# Exp6-9 Topup-1 Sanity Check

Overall status: **{"PASS" if all(row["status"] in {"PASS", "INFO", "BLOCKED_NO_GENERATION"} for row in sanity_rows) else "FAIL"}**

{md_table(sanity_rows, STATIC_FIELDS, max_rows=30)}
"""
    write_text(report_path("sanity_check_topup1_generation.md"), text)
    return status


def run_workflow(request_timeout: int = 240) -> dict[str, Any]:
    ensure_topup1_dirs()
    build_plan_artifacts()
    static_ok, static_rows = static_checks()
    allowed, reason = generation_allowed(static_ok)
    raw_rows: list[dict[str, Any]] = []
    if allowed:
        raw_rows = run_generation(request_timeout=request_timeout, retries=2)
        reason = "API_CALLED" if raw_rows_are_complete(raw_rows) else "API_CALLED_INCOMPLETE"
    else:
        write_blocked_report(reason, static_rows)
    normalize_raw_rows()
    filter_rows()
    leakage_check()
    build_spotcheck_package()
    readability_check()
    return write_generation_reports(static_rows, reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-timeout", type=int, default=240)
    args = parser.parse_args()
    status = run_workflow(request_timeout=args.request_timeout)
    print(
        "Exp6-9 topup1 audit complete: "
        f"static={status['static_checks_status']} api_called={'YES' if status['api_called'] else 'NO'} "
        f"generated={status['generated_count']} normalized={status['normalized_count']} "
        f"filtered={status['filtered_pass_count']} leakage={status['leakage_status']}"
    )


if __name__ == "__main__":
    main()
