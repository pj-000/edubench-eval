"""Shared metric-specific protocol utilities for Exp48B."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import MODULE, ROOT, normalize_text

OUT = MODULE / "outputs/exp48b_metric_rubric_local_edit_pilot"
PRIVATE = OUT / "private"
TRAIN = ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
TARGET_SCORES = (2, 3, 4)
VERIFIER_STATES = {"entailed", "contradicted", "absent", "unclear"}


def parse_rubric_levels(value: Any) -> dict[int, str]:
    """Parse the original metric rubric without rewriting its score descriptors."""
    items = value if isinstance(value, list) else [value]
    levels: dict[int, str] = {}
    for item in items:
        match = re.match(r"^\s*([1-5])\s*[:：]\s*(.+?)\s*$", str(item), flags=re.DOTALL)
        if not match:
            raise ValueError(f"Unparseable rubric item: {item!r}")
        levels[int(match.group(1))] = match.group(2).strip()
    if set(levels) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Rubric must define levels 1-5 exactly, got {sorted(levels)}")
    return levels


def apply_v2_score(states: dict[str, str]) -> int | None:
    """Apply the locked monotonic v2 program to metric-specific defect assertions."""
    if set(states) != {"D2", "D3", "H4"} or set(states.values()) - VERIFIER_STATES:
        return None
    if "unclear" in states.values() or "absent" in states.values():
        return None
    if states["D2"] == "entailed":
        return 2
    if states["D2"] == "contradicted" and states["D3"] == "entailed":
        return 3
    if all(states[key] == "contradicted" for key in ("D2", "D3")) and states["H4"] == "entailed":
        return 4
    return None


def construct_answers(plan: dict[str, Any]) -> list[dict[str, Any]]:
    base = str(plan["base_answer"])
    outputs = []
    for score in TARGET_SCORES:
        if score == 4:
            text = base
            edit = None
        else:
            edit = plan[f"score{score}_edit"]
            source = str(edit["source_span"])
            replacement = str(edit["replacement_span"])
            if base.count(source) != 1:
                raise ValueError(f"score{score} source_span must occur exactly once in base answer")
            text = base.replace(source, replacement, 1)
        outputs.append({"answer_id": f"{plan['family_id']}_s{score}", "intended_score": score, "text": text, "edit": edit})
    return outputs


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "family_id", "synthetic_question_key", "metric", "language", "synthetic_question",
        "rubric_levels", "metric_contract", "base_answer", "score2_edit", "score3_edit",
    }
    missing = required - set(plan)
    if missing:
        return [f"missing_fields:{','.join(sorted(missing))}"]
    levels = plan.get("rubric_levels", {})
    if set(str(key) for key in levels) != {"1", "2", "3", "4", "5"}:
        errors.append("rubric_levels_must_cover_1_to_5")
    contract = plan.get("metric_contract", {})
    if set(contract) != {"D2", "D3", "H4"}:
        errors.append("metric_contract_must_define_D2_D3_H4")
    else:
        expected_levels = {"D2": 2, "D3": 3, "H4": 4}
        for key, level in expected_levels.items():
            item = contract.get(key, {})
            if int(item.get("rubric_level", -1)) != level:
                errors.append(f"{key}_rubric_level_mismatch")
            if not str(item.get("assertion", "")).strip() or not str(item.get("rubric_quote", "")).strip():
                errors.append(f"{key}_contract_incomplete")
    base = str(plan.get("base_answer", ""))
    if not base.strip():
        errors.append("base_answer_empty")
    spans: list[str] = []
    for score in (2, 3):
        edit = plan.get(f"score{score}_edit", {})
        source = str(edit.get("source_span", ""))
        replacement = str(edit.get("replacement_span", ""))
        if not source or base.count(source) != 1:
            errors.append(f"score{score}_source_span_not_unique")
        if not replacement or normalize_text(source) == normalize_text(replacement):
            errors.append(f"score{score}_replacement_invalid")
        if source and replacement:
            ratio = len(normalize_text(replacement)) / max(1, len(normalize_text(source)))
            if not 0.8 <= ratio <= 1.2:
                errors.append(f"score{score}_replacement_length_ratio:{ratio:.3f}")
        if not str(edit.get("rubric_grounded_reason", "")).strip():
            errors.append(f"score{score}_reason_missing")
        spans.append(source)
    if len(spans) == 2 and spans[0] == spans[1]:
        errors.append("score2_and_score3_must_edit_different_spans")
    try:
        answers = construct_answers(plan)
    except ValueError as exc:
        errors.append(str(exc))
        answers = []
    if answers:
        lengths = [max(1, len(normalize_text(row["text"]))) for row in answers]
        if max(lengths) / min(lengths) > 1.2:
            errors.append("answer_length_ratio_above_1p2")
        if len({normalize_text(row["text"]) for row in answers}) != 3:
            errors.append("constructed_answers_not_unique")
        if any(re.search(r"(?:target|intended|目标|预期)\s*(?:score|分数|得分)?\s*[:=]?\s*[234]\b", row["text"], flags=re.I) for row in answers):
            errors.append("score_leakage_in_answer")
    return sorted(set(errors))


def exhaustive_monotonicity_audit() -> dict[str, Any]:
    """Enumerate all binary D2/D3/H4 states and assert failure cannot raise score."""
    binary = ("entailed", "contradicted")
    rows = []
    for d2 in binary:
        for d3 in binary:
            for h4 in binary:
                states = {"D2": d2, "D3": d3, "H4": h4}
                rows.append({**states, "score": apply_v2_score(states)})
    checks = {
        "critical_failure_always_score2": all(row["score"] == 2 for row in rows if row["D2"] == "entailed"),
        "score3_requires_no_critical_and_moderate_defect": apply_v2_score({"D2": "contradicted", "D3": "entailed", "H4": "contradicted"}) == 3,
        "score4_requires_no_defects_and_high_contract": apply_v2_score({"D2": "contradicted", "D3": "contradicted", "H4": "entailed"}) == 4,
        "no_default_score": apply_v2_score({"D2": "contradicted", "D3": "contradicted", "H4": "contradicted"}) is None,
    }
    return {"program_version": "eduq_tail_v2_metric_specific", "state_rows": rows, "checks": checks, "pass": all(checks.values())}
