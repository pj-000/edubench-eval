"""Hardened Exp6 low-score candidate filter checks.

The checks here are deterministic and do not call external APIs. They are meant
to be layered on top of the existing required filters so future generated rows
carry explicit prompt-hardening review fields.
"""

from __future__ import annotations

from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ERROR_TYPES
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import ARTIFACT_PHRASES
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify


HARDENED_FILTER_FIELDS = [
    "label_plausibility_status",
    "error_type_alignment_status",
    "rubric_failure_visibility",
    "too_good_for_target_label",
    "artifact_phrase_status",
    "manual_review_required",
]


NEGATIVE_SELF_CHECK_VALUES = {
    "fail",
    "failed",
    "false",
    "no",
    "needs_revision",
    "revision_needed",
    "too_good",
    "unclear",
}


def boolish(value: Any) -> bool:
    return stringify(value).strip().lower() in {"true", "1", "yes", "y"}


def artifact_phrase_hits(answer: Any) -> list[str]:
    folded = normalize_text(answer)
    return [phrase for phrase in ARTIFACT_PHRASES if phrase in folded]


def self_check_is_negative(value: Any) -> bool:
    return stringify(value).strip().lower() in NEGATIVE_SELF_CHECK_VALUES


def has_clear_failure(row: dict[str, Any]) -> bool:
    evidence = " ".join(
        [
            stringify(row.get("expected_failure_against_rubric")),
            stringify(row.get("rationale_for_label")),
            stringify(row.get("label_plausibility_self_check")),
            stringify(row.get("rubric_failure_visibility")),
        ]
    ).strip()
    return len(evidence) >= 20


def apply_hardened_checks(row: dict[str, Any]) -> dict[str, str]:
    target_label = stringify(row.get("target_label_5"))
    answer = stringify(row.get("answer_synthetic"))
    error_type = stringify(row.get("error_type"))
    reasons: list[str] = []

    hits = artifact_phrase_hits(answer)
    artifact_phrase_status = "fail" if hits else "pass"
    if hits:
        reasons.append("artifact_phrase")

    needs_revision = boolish(row.get("needs_revision")) or boolish(row.get("too_good_for_target_label"))
    if self_check_is_negative(row.get("label_plausibility_self_check")):
        needs_revision = True
        reasons.append("label_plausibility_self_check")
    if self_check_is_negative(row.get("error_type_alignment_self_check")):
        needs_revision = True
        reasons.append("error_type_alignment_self_check")

    clear_failure = has_clear_failure(row)
    if target_label in {"1", "2"}:
        rubric_failure_visibility = "clear" if clear_failure else "missing_clear_failure"
        if not clear_failure:
            reasons.append("missing_clear_rubric_failure")
    elif target_label == "3":
        rubric_failure_visibility = "boundary_or_visible" if clear_failure else "manual_review_required"
        if not clear_failure:
            reasons.append("missing_boundary_failure_evidence")
    else:
        rubric_failure_visibility = "invalid_target_label"
        reasons.append("invalid_target_label")

    too_good = needs_revision or self_check_is_negative(row.get("rubric_failure_visibility"))
    too_good_for_target_label = "manual_review_required" if too_good else "not_detected"
    label_plausibility_status = "manual_review_required" if too_good else "pass"

    if error_type in ERROR_TYPES:
        error_type_alignment_status = "manual_review_required" if self_check_is_negative(row.get("error_type_alignment_self_check")) else "pass"
    else:
        error_type_alignment_status = "fail"
        reasons.append("invalid_error_type")

    manual_review_required = "yes" if reasons or too_good else "no"

    return {
        "label_plausibility_status": label_plausibility_status,
        "error_type_alignment_status": error_type_alignment_status,
        "rubric_failure_visibility": rubric_failure_visibility,
        "too_good_for_target_label": too_good_for_target_label,
        "artifact_phrase_status": artifact_phrase_status,
        "manual_review_required": manual_review_required,
        "hardened_filter_reasons": "; ".join(dict.fromkeys(reasons)),
    }
