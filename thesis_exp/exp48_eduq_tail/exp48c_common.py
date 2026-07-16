"""Shared paths and validation helpers for the Exp48C pointwise audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import MODULE, read_jsonl
from .exp48b_common import OUT as EXP48B_OUT

OUT = MODULE / "outputs/exp48c_rubric_only_audit"
PRIVATE = OUT / "private"
FAMILIES = EXP48B_OUT / "private/generated_families/exp48b_constructed_families.jsonl"
EXP48B_MAPPING = EXP48B_OUT / "private/verifier_packets/exp48b_private_answer_mapping.jsonl"
EXP48B_DECISION = EXP48B_OUT / "decision/exp48b_single_verifier_pilot_decision.json"
EXP48B_METRICS = EXP48B_OUT / "tables/exp48b_single_verifier_score_metrics.csv"
EXP48B_PROTOCOL = EXP48B_OUT / "configs/exp48b_protocol_lock.json"

VERIFIERS = ("codex", "qwen")
PACKET_SEEDS = {"codex": 4831, "qwen": 4832}
FORBIDDEN_FIELDS = {
    "family_id", "intended_score", "answer_id", "metric_contract", "score_program",
    "source_question", "source_question_key", "base_answer", "source_span",
    "replacement_span", "rubric_grounded_reason", "edit", "answers", "ranking",
    "D2", "D3", "H4",
}
PACKET_FIELDS = {
    "packet_id", "anonymous_answer_id", "metric", "language",
    "synthetic_question", "rubric_levels", "answer",
}
OUTPUT_FIELDS = {
    "packet_id", "anonymous_answer_id", "target_scope_confirmed",
    "most_plausible_score", "score_range", "confidence", "rubric_level_quote",
    "answer_evidence_spans", "missing_requirement_reason", "needs_adjudication",
    "rubric_grounded_reason", "verifier_provenance",
}


def packet_path(verifier: str) -> Path:
    return PRIVATE / f"pointwise_packets_{verifier}/exp48c_{verifier}_packets_36.jsonl"


def output_path(verifier: str) -> Path:
    return PRIVATE / f"{verifier}_outputs/exp48c_{verifier}_rubric_only_outputs.jsonl"


def mapping_path() -> Path:
    return PRIVATE / "private_answer_mapping/exp48c_private_answer_mapping.jsonl"


def load_mapping(verifier: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(mapping_path())
    return rows if verifier is None else [row for row in rows if row["verifier"] == verifier]


def validate_pointwise_output(output: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    """Validate structure and exact-substring evidence without changing a score."""
    errors: list[str] = []
    missing = OUTPUT_FIELDS - set(output)
    if missing:
        errors.append(f"missing_fields:{','.join(sorted(missing))}")
    if output.get("packet_id") != packet["packet_id"]:
        errors.append("packet_id_mismatch")
    if output.get("anonymous_answer_id") != packet["anonymous_answer_id"]:
        errors.append("anonymous_answer_id_mismatch")
    if output.get("target_scope_confirmed") is not True:
        errors.append("target_scope_not_confirmed")
    score = output.get("most_plausible_score")
    if not isinstance(score, int) or isinstance(score, bool) or score not in range(1, 6):
        errors.append("invalid_score")
    score_range = output.get("score_range")
    if not (
        isinstance(score_range, list) and len(score_range) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in score_range)
        and 1 <= score_range[0] <= score_range[1] <= 5
        and score_range[1] - score_range[0] <= 2
    ):
        errors.append("invalid_score_range")
    elif isinstance(score, int) and not (score_range[0] <= score <= score_range[1]):
        errors.append("score_outside_range")
    confidence = output.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        errors.append("invalid_confidence")
    if confidence == "low" and output.get("needs_adjudication") is not True:
        errors.append("low_confidence_requires_adjudication")
    if not isinstance(output.get("needs_adjudication"), bool):
        errors.append("invalid_needs_adjudication")
    quote = output.get("rubric_level_quote")
    rubric = str(packet["rubric_levels"].get(str(score), ""))
    if not isinstance(quote, str) or not quote.strip() or quote not in rubric:
        errors.append("rubric_quote_not_exact_substring")
    spans = output.get("answer_evidence_spans")
    if not isinstance(spans, list) or any(not isinstance(span, str) for span in spans):
        errors.append("invalid_evidence_spans")
    else:
        for index, span in enumerate(spans):
            if span and span not in packet["answer"]:
                errors.append(f"evidence_span_{index}_not_exact_substring")
    missing_reason = output.get("missing_requirement_reason")
    if missing_reason is not None and (not isinstance(missing_reason, str) or not missing_reason.strip()):
        errors.append("invalid_missing_requirement_reason")
    if isinstance(spans, list) and not any(spans) and not missing_reason:
        errors.append("empty_evidence_requires_missing_reason")
    if not isinstance(output.get("rubric_grounded_reason"), str) or not output["rubric_grounded_reason"].strip():
        errors.append("rubric_grounded_reason_empty")
    provenance = output.get("verifier_provenance")
    if not isinstance(provenance, dict) or any(not str(provenance.get(key, "")).strip() for key in ("verifier_id", "model_family", "model_version", "session_id")):
        errors.append("incomplete_provenance")
    return sorted(set(errors))


def read_protocol() -> dict[str, Any]:
    return json.loads(EXP48B_PROTOCOL.read_text(encoding="utf-8"))
