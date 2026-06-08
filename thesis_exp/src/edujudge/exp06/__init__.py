"""Exp6 synthetic low-score data inventory and audit utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import REPO_ROOT, THESIS_DIR


EXP06_NAME = "exp06_synthetic_low_score"
EXP06_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP06_NAME
EXP06_TABLES_DIR = EXP06_OUTPUT_DIR / "tables"
EXP06_REPORTS_DIR = EXP06_OUTPUT_DIR / "reports"
EXP06_SAMPLES_DIR = EXP06_OUTPUT_DIR / "samples"

EXP06_SRC_DIR = THESIS_DIR / "src" / "edujudge" / "exp06"
EXP0_SPLIT_DIR = THESIS_DIR / "data" / "splits" / "paper_like_triple_seed42"

REQUESTED_SOURCES = [
    "sampled_merge_50_new.json",
    "sampled_merge_50_new_swift.json",
    "edu-data-synthesis-main",
    "edu-data-synthesis-main.zip",
    "human_sampled_eval_sft_criteria_test.json",
    "deepseek_output",
    "qwen_output",
    "deepseek-r1_merged.jsonl",
    "merge_model_metric.jsonl",
    "groupby_metric_qwq_eval_en.jsonl",
    "groupby_metric_qwq_eval_zh.jsonl",
    "groupby_metric_r1_eval_en.jsonl",
    "groupby_metric_r1_eval_zh.jsonl",
    "groupby_metric_v3_eval_en.jsonl",
    "groupby_metric_v3_eval_zh.jsonl",
]

LIKELY_ROLES = [
    "human_scored",
    "synthetic_candidate",
    "sampled_augmented",
    "model_judge_output",
    "generation_script",
    "unknown",
]

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "BLOCKED"]

INVENTORY_FIELDS = [
    "source_path",
    "exists",
    "file_type",
    "num_records",
    "likely_role",
    "contains_question",
    "contains_answer",
    "contains_metric",
    "contains_label",
    "contains_score",
    "contains_rubric",
    "contains_reasoning",
    "contains_error_type",
    "contains_source_question_id",
    "contains_language",
    "contains_synthetic_marker",
    "usable_for_exp06",
    "risk_level",
    "notes",
]

SCHEMA_PROFILE_FIELDS = [
    "source_path",
    "exists",
    "file_type",
    "num_records",
    "likely_role",
    "all_keys",
    "nested_keys",
    "question_like_fields",
    "answer_like_fields",
    "metric_like_fields",
    "label_score_fields",
    "rubric_fields",
    "reason_rationale_fields",
    "source_id_fields",
    "generation_method_fields",
    "language_fields",
    "error_type_fields",
    "sample_rows",
    "profile_error",
]

NORMALIZED_FIELDS = [
    "synthetic_id",
    "source_file",
    "source_row_index",
    "question",
    "answer",
    "metric_raw",
    "metric_canonical",
    "rubric_text",
    "language",
    "target_label_5",
    "score_raw",
    "score_scale_detected",
    "reasoning",
    "error_type",
    "generation_method",
    "source_question_key",
    "source_triple_key",
    "is_synthetic",
    "normalization_status",
    "normalization_notes",
]


def ensure_exp06_dirs() -> None:
    for path in [
        EXP06_OUTPUT_DIR,
        EXP06_TABLES_DIR,
        EXP06_REPORTS_DIR,
        EXP06_SAMPLES_DIR,
        EXP06_SRC_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def requested_source_paths() -> list[Path]:
    return [REPO_ROOT / source for source in REQUESTED_SOURCES]
