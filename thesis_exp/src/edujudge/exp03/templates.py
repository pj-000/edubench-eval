"""Input templates for Exp3 rubric-aware input ablations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import clean_answer_text, clean_question_text
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import make_prompt as make_exp02_prompt
from thesis_exp.src.edujudge.exp03 import TEMPLATE_NAMES
from thesis_exp.src.edujudge.utils.text_norm import stringify


PREDICTION_INSTRUCTION = "Predict the human-aligned educational quality score from 1 to 5."


@dataclass(frozen=True)
class TemplateSpec:
    template_name: str
    ablation_id: str
    description: str
    input_fields: tuple[str, ...]
    excluded_from_text: tuple[str, ...]
    reuse_exp02: bool = False
    formal_training_default: str = "pending"


TEMPLATE_SPECS: dict[str, TemplateSpec] = {
    "A0_answer_only": TemplateSpec(
        template_name="A0_answer_only",
        ablation_id="A0",
        description="Answer only.",
        input_fields=("answer",),
        excluded_from_text=(
            "question",
            "metric_canonical",
            "rubric",
            "scenario_canonical",
            "subject_canonical",
            "education_level_canonical",
            "language",
            "generator_model",
            "answer_model",
            "human scores",
        ),
        formal_training_default="disabled_initial",
    ),
    "A1_question_answer": TemplateSpec(
        template_name="A1_question_answer",
        ablation_id="A1",
        description="Question plus answer.",
        input_fields=("question", "answer"),
        excluded_from_text=(
            "metric_canonical",
            "rubric",
            "scenario_canonical",
            "subject_canonical",
            "education_level_canonical",
            "language",
            "generator_model",
            "answer_model",
            "human scores",
        ),
        formal_training_default="disabled_initial",
    ),
    "A2_question_answer_metric": TemplateSpec(
        template_name="A2_question_answer_metric",
        ablation_id="A2",
        description="Question plus answer plus metric; strict Exp2 CE baseline input.",
        input_fields=("question", "answer", "metric_canonical"),
        excluded_from_text=(
            "rubric",
            "scenario_canonical",
            "subject_canonical",
            "education_level_canonical",
            "language",
            "generator_model",
            "answer_model",
            "human scores",
        ),
        reuse_exp02=True,
        formal_training_default="reuse_exp02",
    ),
    "A3_question_answer_metric_rubric": TemplateSpec(
        template_name="A3_question_answer_metric_rubric",
        ablation_id="A3",
        description="Question plus answer plus metric plus rubric.",
        input_fields=("question", "answer", "metric_canonical", "rubric_text"),
        excluded_from_text=(
            "scenario_canonical",
            "subject_canonical",
            "education_level_canonical",
            "language",
            "generator_model",
            "answer_model",
            "human scores",
        ),
        formal_training_default="core_formal",
    ),
    "A4_question_answer_metric_rubric_metadata": TemplateSpec(
        template_name="A4_question_answer_metric_rubric_metadata",
        ablation_id="A4",
        description="Question plus answer plus metric plus rubric plus scenario/subject/education/language metadata.",
        input_fields=(
            "scenario_canonical",
            "subject_canonical",
            "education_level_canonical",
            "language",
            "question",
            "answer",
            "metric_canonical",
            "rubric_text",
        ),
        excluded_from_text=("generator_model", "answer_model", "human scores"),
        formal_training_default="core_formal",
    ),
}


def get_template_spec(template_name: str) -> TemplateSpec:
    if template_name not in TEMPLATE_SPECS:
        raise ValueError(f"Unknown template_name={template_name!r}. Expected one of {TEMPLATE_NAMES}.")
    return TEMPLATE_SPECS[template_name]


def rubric_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(stringify(item).strip() for item in value if stringify(item).strip()).strip()
    if isinstance(value, dict):
        lines = []
        for key in sorted(value):
            item = stringify(value[key]).strip()
            if item:
                lines.append(f"{key}: {item}")
        return "\n".join(lines).strip()
    return stringify(value).strip()


def require_rubric_text(row: dict[str, Any]) -> str:
    rubric_text = stringify(row.get("rubric_text")).strip() or rubric_to_text(row.get("rubric"))
    if not rubric_text:
        ident = row.get("record_id") or row.get("id") or "<unknown>"
        raise ValueError(f"Missing rubric_text for record_id={ident}")
    return rubric_text


def _section(label: str, value: Any) -> str:
    return f"{label}:\n{stringify(value).strip()}"


def make_prompt(row: dict[str, Any], template_name: str) -> str:
    """Render one Exp3 input template.

    A2 delegates to Exp2's prompt builder so the baseline input remains exactly
    equivalent to the prior CE baseline.
    """
    get_template_spec(template_name)
    if template_name == "A2_question_answer_metric":
        return make_exp02_prompt(row)

    question = clean_question_text(row.get("question"))
    answer = clean_answer_text(row.get("answer"))
    metric = stringify(row.get("metric_canonical")).strip()
    rubric_text = require_rubric_text(row) if "rubric" in template_name else ""

    if template_name == "A0_answer_only":
        sections = [_section("Answer", answer), PREDICTION_INSTRUCTION]
    elif template_name == "A1_question_answer":
        sections = [_section("Question", question), _section("Answer", answer), PREDICTION_INSTRUCTION]
    elif template_name == "A3_question_answer_metric_rubric":
        sections = [
            _section("Question", question),
            _section("Answer", answer),
            _section("Evaluation Dimension", metric),
            _section("Rubric", rubric_text),
            PREDICTION_INSTRUCTION,
        ]
    elif template_name == "A4_question_answer_metric_rubric_metadata":
        sections = [
            _section("Scenario", row.get("scenario_canonical")),
            _section("Subject", row.get("subject_canonical")),
            _section("Education Level", row.get("education_level_canonical")),
            _section("Language", row.get("language")),
            _section("Question", question),
            _section("Answer", answer),
            _section("Evaluation Dimension", metric),
            _section("Rubric", rubric_text),
            PREDICTION_INSTRUCTION,
        ]
    else:
        raise AssertionError(f"Unhandled template_name={template_name}")
    return "\n\n".join(sections)


def estimate_token_length(text: str) -> int:
    """Cheap fallback when no tokenizer is available."""
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def template_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for template_name in TEMPLATE_NAMES:
        spec = get_template_spec(template_name)
        row = asdict(spec)
        row["input_fields"] = ", ".join(spec.input_fields)
        row["excluded_from_text"] = ", ".join(spec.excluded_from_text)
        rows.append(row)
    return rows
