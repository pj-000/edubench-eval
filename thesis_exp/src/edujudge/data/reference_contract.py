"""Reference contract constants for Exp 0.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


CONTRACT_PATH = THESIS_DIR / "configs" / "reference_contract.yaml"

EXPECTED_SCENARIOS = [
    "Question Answering",
    "Error Correction",
    "Idea Provision",
    "Personalized Learning Support",
    "Emotional Support",
    "Question Generation",
    "Automatic Grading",
    "Teaching Material Generation",
    "Personalized Content Creation",
]

SCENARIO_ALIASES = {
    "Question Answering": ["Q&A", "QA", "Problem Solving", "Question Answering", "problem_solving", "答疑", "回答问题"],
    "Error Correction": ["EC", "Error Correction", "error_correction", "纠错"],
    "Idea Provision": ["IP", "Idea Provision", "idea_provision", "根据学生画像给出建议"],
    "Personalized Learning Support": ["PLS", "Personalized Learning Support", "personalized_learning_support", "根据学生画像设计学习路径"],
    "Emotional Support": ["ES", "Emotional Support", "psychological_support", "emotional_support", "学生心理健康判断与建议"],
    "Question Generation": ["QG", "Question Generation", "question_generation", "根据知识点生成问题"],
    "Automatic Grading": ["AG", "Automatic Grading", "Assignment Judging", "automatic_grading", "判题"],
    "Teaching Material Generation": ["TMG", "Teaching Material Generation", "teaching_material_generation", "教学素材生成"],
    "Personalized Content Creation": ["PCC", "Personalized Content Creation", "personalized_content_creation", "个性化的学习内容或任务"],
}

EXPECTED_METRIC_GROUPS = {
    "Scenario Adaptability": [
        "Instruction Following & Task Completion",
        "Role & Tone Consistency",
        "Content Relevance & Scope Control",
        "Scenario Element Integration",
    ],
    "Factual & Reasoning Accuracy": [
        "Basic Factual Accuracy",
        "Domain Knowledge Accuracy",
        "Reasoning Process Rigor",
        "Error Identification & Correction Precision",
    ],
    "Pedagogical Application": [
        "Clarity, Simplicity & Inspiration",
        "Motivation, Guidance & Positive Feedback",
        "Personalization, Adaptation & Learning Support",
        "Higher-Order Thinking & Skill Development",
    ],
}

EXPECTED_METRICS = [metric for metrics in EXPECTED_METRIC_GROUPS.values() for metric in metrics]

EXPECTED_SUBJECTS = [
    "Applied Economics",
    "Aquaculture",
    "Automation",
    "Basic Medicine",
    "Biology",
    "Business Administration",
    "Chemistry",
    "Chinese",
    "Clinical Medicine",
    "Computer Science",
    "Crop Science",
    "English",
    "General Pedagogy",
    "Geography",
    "History",
    "Law",
    "Literature and Art",
    "Mathematics",
    "Military Science",
    "Physical Education",
    "Physics",
    "Psychology",
    "Public Administration",
    "Sociology",
    "Theoretical Economics",
]

EXPECTED_EDUCATION_LEVELS = ["Elementary School", "Middle School", "High School", "Undergraduate", "Master", "PhD"]
EXPECTED_LANGUAGES = ["en", "zh"]
EXPECTED_GENERATOR_MODELS = ["deepseek-r1", "qwen-max", "qwen2.5-7b-instruct", "deepseek-v3", "qwen2.5-14b-instruct"]

PDF_AUDIT_TOTAL = 5536
PDF_TRAIN_POOL_ROWS = 3318
PDF_HELDOUT_TEST_ROWS = 2218
DATASET_NAME = "edubench_audit_human_scored_subset"


def reference_summary() -> dict[str, Any]:
    return {
        "contract_path": str(CONTRACT_PATH),
        "dataset_name": DATASET_NAME,
        "official_scenarios": EXPECTED_SCENARIOS,
        "official_metrics": EXPECTED_METRICS,
        "official_subjects": EXPECTED_SUBJECTS,
        "expected_total_scored_items": PDF_AUDIT_TOTAL,
        "expected_train_pool_rows": PDF_TRAIN_POOL_ROWS,
        "expected_heldout_test_rows": PDF_HELDOUT_TEST_ROWS,
    }
