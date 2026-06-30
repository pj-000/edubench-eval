"""Taxonomy helpers for Exp17-D1 hidden failure audit.

This module intentionally has no model or transformer dependencies. It only
validates human annotation labels used by the Exp17-D1 diagnostic workflow.
"""

from __future__ import annotations

from typing import Any


PRIMARY_FAILURE_MODES = [
    "format_violation",
    "task_constraint_violation",
    "factual_or_rubric_mismatch",
    "surface_fluent_but_hidden_defect",
    "missing_key_point",
    "insufficient_evidence",
    "possible_label_conflict",
    "other",
    "unclear",
]

LLM_OR_MODEL_OVERSCORING_OPTIONS = [
    "yes",
    "no",
    "unclear",
]

RUBRIC_LINK_LEVELS = [
    "explicit_rubric_clause",
    "implicit_task_constraint",
    "inferred_from_context",
    "not_rubric_linked",
    "unclear",
]

TRAINABILITY_OPTIONS = [
    "strong_train_signal",
    "weak_train_signal",
    "format_auxiliary_signal",
    "pairwise_only",
    "downweight_or_exclude",
    "review_only",
    "unclear",
]

RECOMMENDED_TRAINING_USE = [
    "evidence_positive",
    "format_auxiliary",
    "pairwise_low",
    "downweight",
    "exclude",
    "review_only",
    "unclear",
]

CONFIDENCE_OPTIONS = [1, 2, 3, 4, 5]

BOOLEAN_OPTIONS = {"0", "1", "yes", "no", "true", "false", "y", "n", "unclear", ""}


def normalize_label_string(x: Any) -> str:
    """Normalize a manual label cell for enum comparison."""

    if x is None:
        return ""
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def _validate_enum(row: dict[str, Any], field: str, valid_values: list[str]) -> list[str]:
    value = normalize_label_string(row.get(field, ""))
    if not value:
        return []
    if value not in valid_values:
        return [f"{field}={row.get(field)!r} is not one of {valid_values}"]
    return []


def _validate_boolish(row: dict[str, Any], field: str) -> list[str]:
    value = normalize_label_string(row.get(field, ""))
    if value not in BOOLEAN_OPTIONS:
        return [f"{field}={row.get(field)!r} is not a yes/no/1/0/unclear value"]
    return []


def validate_annotation_row(row: dict[str, Any]) -> list[str]:
    """Return validation issues for one Exp17-D1 annotation row."""

    issues: list[str] = []
    issues += _validate_enum(row, "primary_failure_mode_manual", PRIMARY_FAILURE_MODES)
    issues += _validate_enum(row, "secondary_failure_mode_manual", PRIMARY_FAILURE_MODES)
    issues += _validate_enum(row, "rubric_link_level_manual", RUBRIC_LINK_LEVELS)
    issues += _validate_enum(
        row,
        "llm_or_model_over_scoring_pattern_manual",
        LLM_OR_MODEL_OVERSCORING_OPTIONS,
    )
    issues += _validate_enum(row, "trainability_manual", TRAINABILITY_OPTIONS)
    issues += _validate_enum(row, "recommended_training_use_manual", RECOMMENDED_TRAINING_USE)
    for field in [
        "is_surface_fluent_manual",
        "is_hidden_failure_manual",
        "is_format_or_task_constraint_manual",
        "possible_label_conflict_manual",
    ]:
        issues += _validate_boolish(row, field)

    confidence = normalize_label_string(row.get("confidence_manual", ""))
    if confidence:
        try:
            value = int(float(confidence))
        except Exception:
            issues.append(f"confidence_manual={row.get('confidence_manual')!r} is not an integer 1-5")
        else:
            if value not in CONFIDENCE_OPTIONS:
                issues.append(f"confidence_manual={row.get('confidence_manual')!r} is not in 1-5")
    return issues
