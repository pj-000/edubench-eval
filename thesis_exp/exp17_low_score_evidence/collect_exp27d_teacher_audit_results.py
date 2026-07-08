"""Collect lightweight Exp27D teacher-audit v4 API re-pilot summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.validate_exp27d_teacher_audit import (  # noqa: E402
    DEFAULT_EXP27C_PACKETS,
    validate,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42")
DEFAULT_EXP27C_DECISION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "decision/exp27c_teacher_audit_v3_api_repilot_decision.json"
)
PROVIDERS = ["qwen", "deepseek"]
STAGES = ["blind", "audit"]
DERIVED_RISK_RULE_VERSION = "exp27d_v4_rule_20260708"
DERIVED_RISK_RULES = [
    "answer_key_uncertainty in {possible_answer_key_issue, insufficient_reference, rubric_ambiguous} -> unclear",
    "score_cap <= 2 -> high",
    "teacher_score <= 2 and failure_bucket=hidden_or_missing_failure and surface_plausibility in {high, medium} -> high",
    "teacher_score <= 2 and failure_bucket=hidden_or_missing_failure and surface_plausibility=low -> medium",
    "teacher_score <= 2 and failure_bucket=visible_failure -> medium",
    "teacher_score == 3 and failure_bucket=hidden_or_missing_failure and surface_plausibility=high -> medium",
    "teacher_score >= 4 and failure_bucket=no_failure -> low",
    "failure_bucket=unclear or confidence=low -> unclear",
    "otherwise -> low",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def label_region(label: int | None) -> str:
    if label is None:
        return "unknown"
    if label <= 2:
        return "low"
    if label == 3:
        return "mid"
    return "high"


def jaccard(a: Any, b: Any) -> float:
    sa = set(a) if isinstance(a, list) else set()
    sb = set(b) if isinstance(b, list) else set()
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def failure_bucket(blind: dict[str, Any]) -> str:
    visibility = as_text(blind.get("failure_visibility"))
    failures = blind.get("major_failures") if isinstance(blind.get("major_failures"), list) else []
    if visibility == "no_major_failure" or failures == ["no_major_failure"]:
        return "no_failure"
    if visibility == "explicit_failure":
        return "visible_failure"
    if visibility in {"hidden_failure", "missing_required_content"}:
        return "hidden_or_missing_failure"
    return "unclear"


def derived_overestimation_risk(blind: dict[str, Any]) -> str:
    score = as_int(blind.get("teacher_score"))
    cap = as_int(blind.get("score_cap"))
    bucket = failure_bucket(blind)
    surface = as_text(blind.get("surface_plausibility"))
    answer_key = as_text(blind.get("answer_key_uncertainty"))
    confidence = as_text(blind.get("confidence"))
    if answer_key in {"possible_answer_key_issue", "insufficient_reference", "rubric_ambiguous"}:
        return "unclear"
    if cap is not None and cap <= 2:
        return "high"
    if score is not None and score <= 2 and bucket == "hidden_or_missing_failure":
        return "high" if surface in {"high", "medium"} else "medium"
    if score is not None and score <= 2 and bucket == "visible_failure":
        return "medium"
    if score is not None and score == 3 and bucket == "hidden_or_missing_failure" and surface == "high":
        return "medium"
    if score is not None and score >= 4 and bucket == "no_failure":
        return "low"
    if bucket == "unclear" or confidence == "low":
        return "unclear"
    return "low"


def load_packets(out_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    packets = read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
    refs = read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
    exp27c_ids = {
        str(row.get("sample_id"))
        for row in read_jsonl(DEFAULT_EXP27C_PACKETS)
        if row.get("sample_id")
    }
    return (
        {str(row["sample_id"]): row for row in packets},
        {str(row["sample_id"]): row for row in refs},
        exp27c_ids,
    )


def load_outputs(out_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for provider in PROVIDERS:
        for stage in STAGES:
            path = out_dir / "annotations" / "parsed" / provider / f"exp27d_{stage}_outputs.jsonl"
            outputs[(provider, stage)] = read_jsonl(path)
    return outputs


def parsed_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid:
            out[sid] = row
    return out


def schema_success(row: dict[str, Any]) -> bool:
    return bool(row.get("parsed")) and not row.get("parse_error") and not row.get("schema_errors")


def evidence_valid(blind: dict[str, Any], packet: dict[str, Any]) -> tuple[bool, bool, bool]:
    def norm(text: str) -> str:
        import re

        return re.sub(r"\s+", " ", text).strip()

    evidence = blind.get("evidence_span")
    answer = str(packet.get("teacher_input", {}).get("answer", ""))
    non_null = isinstance(evidence, str) and bool(evidence.strip())
    substring_valid = bool(non_null and (evidence in answer or norm(evidence) in norm(answer)))
    null_missing_reason_valid = (
        evidence is None
        and blind.get("evidence_type") == "missing_required_content"
        and blind.get("missing_evidence_reason") is not None
    )
    not_applicable_valid = evidence is None and blind.get("evidence_type") == "not_applicable"
    return non_null, substring_valid, null_missing_reason_valid or not_applicable_valid


def cap_compatible(a: Any, b: Any) -> bool:
    ia = as_int(a)
    ib = as_int(b)
    if ia is None and ib is None:
        return True
    if ia is None or ib is None:
        return False
    return abs(ia - ib) <= 1


def training_use_compatible(a: str, b: str) -> bool:
    rank = {"exclude": 0, "review_only": 0, "low_weight": 1, "high_weight": 2}
    if a == b:
        return True
    if a not in rank or b not in rank:
        return False
    return abs(rank[a] - rank[b]) <= 1


def validation_annotation_key(error: str) -> str:
    if ":" not in error:
        return ""
    head = error.split(":", 1)[0]
    return head if "/" in head and "[" in head and "]" in head else ""


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(1 for row in rows if row.get(field)) / len(rows) if rows else 0.0


def collect(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    packets_by_id, refs_by_id, exp27c_ids = load_packets(out_dir)
    validation_summary = validate(out_dir, args.exp27c_packets, with_annotations=True)
    validation_issues = read_csv_rows(out_dir / "tables" / "exp27d_v4_validation_issues.csv")
    outputs = load_outputs(out_dir)

    api_summary_rows: list[dict[str, Any]] = []
    total_annotation_rows = 0
    repair_attempt_rows = 0
    repair_success_rows = 0
    repair_changed_teacher_score_count = 0
    repair_changed_major_failures_count = 0
    repair_changed_score_cap_count = 0
    repair_changed_failure_visibility_count = 0
    repair_changed_judgement_rows = 0
    repair_audit_rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for stage in STAGES:
            rows = outputs[(provider, stage)]
            parsed = sum(1 for row in rows if row.get("parsed") and not row.get("parse_error"))
            schema_ok = sum(1 for row in rows if schema_success(row))
            stage_repair_attempt_rows = sum(1 for row in rows if int(row.get("schema_repair_attempts") or 0) > 0)
            stage_repair_success_rows = sum(
                1 for row in rows if int(row.get("schema_repair_attempts") or 0) > 0 and schema_success(row)
            )
            stage_changed_teacher_score = sum(1 for row in rows if row.get("repair_changed_teacher_score") is True)
            stage_changed_major_failures = sum(1 for row in rows if row.get("repair_changed_major_failures") is True)
            stage_changed_score_cap = sum(1 for row in rows if row.get("repair_changed_score_cap") is True)
            stage_changed_failure_visibility = sum(
                1 for row in rows if row.get("repair_changed_failure_visibility") is True
            )
            for row in rows:
                repair_audit_rows.append(
                    {
                        "provider": provider,
                        "stage": stage,
                        "sample_id": row.get("sample_id", ""),
                        "schema_repair_attempts": row.get("schema_repair_attempts", 0),
                        "repair_success": bool(int(row.get("schema_repair_attempts") or 0) > 0 and schema_success(row)),
                        "repair_changed_teacher_score": row.get("repair_changed_teacher_score", False),
                        "repair_changed_major_failures": row.get("repair_changed_major_failures", False),
                        "repair_changed_score_cap": row.get("repair_changed_score_cap", False),
                        "repair_changed_failure_visibility": row.get("repair_changed_failure_visibility", False),
                    }
                )
                if any(
                    row.get(key) is True
                    for key in [
                        "repair_changed_teacher_score",
                        "repair_changed_major_failures",
                        "repair_changed_score_cap",
                        "repair_changed_failure_visibility",
                    ]
                ):
                    repair_changed_judgement_rows += 1
            total_annotation_rows += len(rows)
            repair_attempt_rows += stage_repair_attempt_rows
            repair_success_rows += stage_repair_success_rows
            repair_changed_teacher_score_count += stage_changed_teacher_score
            repair_changed_major_failures_count += stage_changed_major_failures
            repair_changed_score_cap_count += stage_changed_score_cap
            repair_changed_failure_visibility_count += stage_changed_failure_visibility
            api_summary_rows.append(
                {
                    "provider": provider,
                    "stage": stage,
                    "rows": len(rows),
                    "parse_success_rate": parsed / len(rows) if rows else 0.0,
                    "schema_validation_success_rate": schema_ok / len(rows) if rows else 0.0,
                    "parse_error_count": sum(1 for row in rows if row.get("parse_error")),
                    "schema_error_count": sum(1 for row in rows if row.get("schema_errors")),
                    "schema_repair_attempt_rows": stage_repair_attempt_rows,
                    "schema_repair_attempt_rate": stage_repair_attempt_rows / len(rows) if rows else 0.0,
                    "repair_success_rows": stage_repair_success_rows,
                    "repair_success_rate": stage_repair_success_rows / stage_repair_attempt_rows
                    if stage_repair_attempt_rows
                    else 0.0,
                    "repair_changed_teacher_score_count": stage_changed_teacher_score,
                    "repair_changed_major_failures_count": stage_changed_major_failures,
                    "repair_changed_score_cap_count": stage_changed_score_cap,
                    "repair_changed_failure_visibility_count": stage_changed_failure_visibility,
                }
            )

    q_blind = parsed_by_id(outputs[("qwen", "blind")])
    d_blind = parsed_by_id(outputs[("deepseek", "blind")])
    common_blind = sorted(set(q_blind) & set(d_blind))
    score_rows: list[dict[str, Any]] = []
    raw_risk_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    derived_risk_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    jaccards: list[float] = []
    for sid in common_blind:
        q_parsed = q_blind[sid].get("parsed") or {}
        d_parsed = d_blind[sid].get("parsed") or {}
        qb = q_parsed.get("blind") or {}
        db = d_parsed.get("blind") or {}
        q_score = as_int(qb.get("teacher_score"))
        d_score = as_int(db.get("teacher_score"))
        label = as_int(refs_by_id.get(sid, {}).get("original_score"))
        packet = packets_by_id.get(sid, {})
        region = label_region(label)
        q_bucket = failure_bucket(qb)
        d_bucket = failure_bucket(db)
        q_derived = derived_overestimation_risk(qb)
        d_derived = derived_overestimation_risk(db)
        score_exact = q_score == d_score
        score_adjacent = q_score is not None and d_score is not None and abs(q_score - d_score) <= 1
        score_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "pilot_group": refs_by_id.get(sid, {}).get("pilot_group", ""),
                "paired_exp27c_row": sid in exp27c_ids,
                "language": packet.get("source_meta", {}).get("language", ""),
                "qwen_score": q_score,
                "deepseek_score": d_score,
                "score_exact": score_exact,
                "score_adjacent": score_adjacent,
            }
        )
        jf = jaccard(qb.get("major_failures"), db.get("major_failures"))
        jaccards.append(jf)
        raw_risk_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "qwen_failure_visibility": qb.get("failure_visibility", ""),
                "deepseek_failure_visibility": db.get("failure_visibility", ""),
                "failure_visibility_exact": qb.get("failure_visibility") == db.get("failure_visibility"),
                "qwen_raw_overestimation_risk": qb.get("overestimation_risk", ""),
                "deepseek_raw_overestimation_risk": db.get("overestimation_risk", ""),
                "raw_overestimation_risk_exact": qb.get("overestimation_risk") == db.get("overestimation_risk"),
                "qwen_evidence_type": qb.get("evidence_type", ""),
                "deepseek_evidence_type": db.get("evidence_type", ""),
                "evidence_type_exact": qb.get("evidence_type") == db.get("evidence_type"),
                "major_failures_jaccard": jf,
            }
        )
        bucket_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "qwen_failure_bucket": q_bucket,
                "deepseek_failure_bucket": d_bucket,
                "failure_bucket_exact": q_bucket == d_bucket,
            }
        )
        derived_risk_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "qwen_derived_overestimation_risk": q_derived,
                "deepseek_derived_overestimation_risk": d_derived,
                "derived_overestimation_risk_exact": q_derived == d_derived,
                "qwen_surface_plausibility": qb.get("surface_plausibility", ""),
                "deepseek_surface_plausibility": db.get("surface_plausibility", ""),
            }
        )
        cap_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "qwen_score_cap": qb.get("score_cap", ""),
                "deepseek_score_cap": db.get("score_cap", ""),
                "score_cap_exact": qb.get("score_cap") == db.get("score_cap"),
                "score_cap_compatible": cap_compatible(qb.get("score_cap"), db.get("score_cap")),
            }
        )
        if not score_adjacent or q_bucket != d_bucket or q_derived != d_derived:
            conflict_rows.append(
                {
                    "sample_id": sid,
                    "label": label,
                    "label_region": region,
                    "qwen_score": q_score,
                    "deepseek_score": d_score,
                    "score_adjacent": score_adjacent,
                    "qwen_failure_bucket": q_bucket,
                    "deepseek_failure_bucket": d_bucket,
                    "failure_bucket_match": q_bucket == d_bucket,
                    "qwen_derived_risk": q_derived,
                    "deepseek_derived_risk": d_derived,
                    "derived_risk_match": q_derived == d_derived,
                }
            )
    for provider in PROVIDERS:
        for row in outputs[(provider, "blind")]:
            sid = str(row.get("sample_id") or "")
            parsed = row.get("parsed") or {}
            blind = parsed.get("blind") or {}
            packet = packets_by_id.get(sid, {})
            non_null, substring_valid, null_or_na_valid = evidence_valid(blind, packet)
            label = as_int(refs_by_id.get(sid, {}).get("original_score"))
            evidence_rows.append(
                {
                    "provider": provider,
                    "sample_id": sid,
                    "label": label,
                    "label_region": label_region(label),
                    "evidence_non_null": non_null,
                    "evidence_substring_valid": substring_valid,
                    "evidence_null_with_missing_reason_or_na_valid": null_or_na_valid,
                    "evidence_valid_or_null_with_reason": substring_valid or null_or_na_valid,
                    "evidence_type": blind.get("evidence_type", ""),
                    "missing_evidence_reason": blind.get("missing_evidence_reason", ""),
                }
            )

    q_audit = parsed_by_id(outputs[("qwen", "audit")])
    d_audit = parsed_by_id(outputs[("deepseek", "audit")])
    common_audit = sorted(set(q_audit) & set(d_audit))
    audit_rows: list[dict[str, Any]] = []
    for sid in common_audit:
        qa = (q_audit[sid].get("parsed") or {}).get("audit") or {}
        da = (d_audit[sid].get("parsed") or {}).get("audit") or {}
        label = as_int(refs_by_id.get(sid, {}).get("original_score"))
        audit_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": label_region(label),
                "qwen_teacher_score": qa.get("teacher_score", ""),
                "deepseek_teacher_score": da.get("teacher_score", ""),
                "qwen_label_quality": qa.get("label_quality", ""),
                "deepseek_label_quality": da.get("label_quality", ""),
                "label_quality_exact": qa.get("label_quality") == da.get("label_quality"),
                "qwen_training_use": qa.get("recommended_training_use", ""),
                "deepseek_training_use": da.get("recommended_training_use", ""),
                "training_use_compatible": training_use_compatible(
                    as_text(qa.get("recommended_training_use")),
                    as_text(da.get("recommended_training_use")),
                ),
                "qwen_needs_human_review": qa.get("needs_human_review", ""),
                "deepseek_needs_human_review": da.get("needs_human_review", ""),
                "needs_human_review_match": qa.get("needs_human_review") == da.get("needs_human_review"),
                "qwen_label_noise_type": qa.get("label_noise_type", ""),
                "deepseek_label_noise_type": da.get("label_noise_type", ""),
                "qwen_hard_conflict": qa.get("hard_conflict", ""),
                "deepseek_hard_conflict": da.get("hard_conflict", ""),
                "hard_conflict_any": qa.get("hard_conflict") is True or da.get("hard_conflict") is True,
            }
        )

    parse_success_rate = mean([row["parse_success_rate"] for row in api_summary_rows]) if api_summary_rows else 0.0
    schema_success_rate = mean([row["schema_validation_success_rate"] for row in api_summary_rows]) if api_summary_rows else 0.0
    hard_error_count = int(validation_summary.get("hard_errors", 0))
    soft_error_count = int(validation_summary.get("soft_errors", 0))
    warning_count = int(validation_summary.get("warnings", 0))
    total_rows_for_errors = max(1, sum(row["rows"] for row in api_summary_rows))
    hard_error_rate = hard_error_count / total_rows_for_errors
    hard_annotation_keys = {
        validation_annotation_key(as_text(row.get("error")))
        for row in validation_issues
        if as_text(row.get("severity")) == "hard" and validation_annotation_key(as_text(row.get("error")))
    }
    usable_annotation_rate = (
        (total_annotation_rows - len(hard_annotation_keys)) / total_annotation_rows
        if total_annotation_rows
        else 0.0
    )
    schema_repair_attempt_rate = repair_attempt_rows / total_annotation_rows if total_annotation_rows else 0.0
    repair_success_rate = repair_success_rows / repair_attempt_rows if repair_attempt_rows else 0.0
    score_exact = rate(score_rows, "score_exact")
    score_adjacent = rate(score_rows, "score_adjacent")
    failure_bucket_exact = rate(bucket_rows, "failure_bucket_exact")
    raw_risk_exact = rate(raw_risk_rows, "raw_overestimation_risk_exact")
    derived_risk_exact = rate(derived_risk_rows, "derived_overestimation_risk_exact")
    evidence_type_exact = rate(raw_risk_rows, "evidence_type_exact")
    evidence_valid_rate = rate(evidence_rows, "evidence_valid_or_null_with_reason")
    label_quality_exact = rate(audit_rows, "label_quality_exact")
    training_use_compatible_rate = rate(audit_rows, "training_use_compatible")
    needs_review_match = rate(audit_rows, "needs_human_review_match")
    high_audit = [row for row in audit_rows if row.get("label_region") == "high"]
    high_control_hard_conflict_rate = rate(high_audit, "hard_conflict_any")

    decision_pass = (
        parse_success_rate >= 0.99
        and schema_success_rate >= 0.99
        and usable_annotation_rate >= 0.95
        and hard_error_rate <= 0.03
        and score_adjacent >= 0.95
        and score_exact >= 0.60
        and failure_bucket_exact >= 0.80
        and derived_risk_exact >= 0.75
        and evidence_valid_rate >= 0.95
        and training_use_compatible_rate >= 0.95
        and high_control_hard_conflict_rate <= 0.10
        and repair_changed_judgement_rows == 0
    )
    recommendation = "proceed_to_361_teacher_audit" if decision_pass else "revise_schema_or_prompt_before_361"

    stratified_rows: list[dict[str, Any]] = []
    for group_name, group_rows in [
        ("all", score_rows),
        ("paired_exp27c_60", [row for row in score_rows if row.get("paired_exp27c_row")]),
        ("new20", [row for row in score_rows if not row.get("paired_exp27c_row")]),
        ("low", [row for row in score_rows if row.get("label_region") == "low"]),
        ("mid", [row for row in score_rows if row.get("label_region") == "mid"]),
        ("high", [row for row in score_rows if row.get("label_region") == "high"]),
    ]:
        stratified_rows.append(
            {
                "group": group_name,
                "n": len(group_rows),
                "score_exact_agreement": rate(group_rows, "score_exact"),
                "score_adjacent_agreement": rate(group_rows, "score_adjacent"),
            }
        )
    for lang in sorted({row.get("language", "") for row in score_rows}):
        group_rows = [row for row in score_rows if row.get("language") == lang]
        stratified_rows.append(
            {
                "group": f"language={lang}",
                "n": len(group_rows),
                "score_exact_agreement": rate(group_rows, "score_exact"),
                "score_adjacent_agreement": rate(group_rows, "score_adjacent"),
            }
        )

    hard_rows = [row for row in validation_issues if as_text(row.get("severity")) == "hard"]
    soft_rows = [row for row in validation_issues if as_text(row.get("severity")) in {"soft", "warning"}]
    issue_counter = Counter(
        (as_text(row.get("severity")) or "unknown", as_text(row.get("kind")) or "unknown")
        for row in validation_issues
    )
    error_breakdown_rows = [
        {"severity": severity, "error_kind": kind, "count": count}
        for (severity, kind), count in sorted(issue_counter.items())
    ]

    exp27c_decision = read_json(args.exp27c_decision)
    paired_rows = [row for row in score_rows if row.get("paired_exp27c_row")]
    paired_bucket = [row for row in bucket_rows if row.get("sample_id") in exp27c_ids]
    paired_derived = [row for row in derived_risk_rows if row.get("sample_id") in exp27c_ids]
    paired_comparison_rows = [
        {
            "metric": "teacher_score_adjacent_agreement",
            "exp27c_v3": exp27c_decision.get("teacher_score_adjacent_agreement", ""),
            "exp27d_v4": rate(paired_rows, "score_adjacent"),
        },
        {
            "metric": "teacher_score_exact_agreement",
            "exp27c_v3": exp27c_decision.get("teacher_score_exact_agreement", ""),
            "exp27d_v4": rate(paired_rows, "score_exact"),
        },
        {
            "metric": "failure_bucket_agreement",
            "exp27c_v3": "",
            "exp27d_v4": rate(paired_bucket, "failure_bucket_exact"),
        },
        {
            "metric": "derived_overestimation_risk_agreement",
            "exp27c_v3": "",
            "exp27d_v4": rate(paired_derived, "derived_overestimation_risk_exact"),
        },
        {
            "metric": "hard_validation_error_rate",
            "exp27c_v3": exp27c_decision.get("hard_validation_error_rate", ""),
            "exp27d_v4": hard_error_rate,
        },
        {
            "metric": "usable_annotation_rate",
            "exp27c_v3": exp27c_decision.get("usable_annotation_rate", ""),
            "exp27d_v4": usable_annotation_rate,
        },
    ]

    write_csv(out_dir / "tables" / "exp27d_v4_api_repilot_summary.csv", api_summary_rows + stratified_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_cross_provider_score_agreement.csv", score_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_cross_provider_risk_agreement.csv", raw_risk_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_failure_bucket_agreement.csv", bucket_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_derived_risk_agreement.csv", derived_risk_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_evidence_span_validation.csv", evidence_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_score_cap_consistency.csv", cap_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_label_audit_agreement.csv", audit_rows)
    write_csv(
        out_dir / "tables" / "exp27d_v4_validation_error_breakdown.csv",
        error_breakdown_rows or [{"severity": "", "error_kind": "", "count": 0}],
    )
    write_csv(out_dir / "tables" / "exp27d_v4_hard_semantic_errors.csv", hard_rows or [{"severity": "", "kind": "", "error": ""}])
    write_csv(out_dir / "tables" / "exp27d_v4_soft_semantic_warnings.csv", soft_rows or [{"severity": "", "kind": "", "error": ""}])
    write_csv(out_dir / "tables" / "exp27d_v4_schema_repair_audit.csv", repair_audit_rows or [{"provider": "", "stage": "", "sample_id": ""}])
    write_csv(out_dir / "tables" / "exp27d_v4_teacher_conflict_cases_light.csv", conflict_rows or [{"sample_id": ""}])
    write_csv(out_dir / "tables" / "exp27d_v4_paired_v3_v4_comparison.csv", paired_comparison_rows)

    decision = {
        "recommendation": recommendation,
        "derived_risk_rule_version": DERIVED_RISK_RULE_VERSION,
        "derived_risk_rules": DERIVED_RISK_RULES,
        "parse_success_rate": parse_success_rate,
        "schema_validation_success_rate": schema_success_rate,
        "hard_validation_error_count": hard_error_count,
        "hard_validation_error_rate": hard_error_rate,
        "soft_validation_error_count": soft_error_count,
        "warning_count": warning_count,
        "usable_annotation_rate": usable_annotation_rate,
        "schema_repair_attempt_rows": repair_attempt_rows,
        "schema_repair_attempt_rate": schema_repair_attempt_rate,
        "repair_success_rows": repair_success_rows,
        "repair_success_rate": repair_success_rate,
        "repair_changed_teacher_score_count": repair_changed_teacher_score_count,
        "repair_changed_major_failures_count": repair_changed_major_failures_count,
        "repair_changed_score_cap_count": repair_changed_score_cap_count,
        "repair_changed_failure_visibility_count": repair_changed_failure_visibility_count,
        "repair_changed_judgement_count": repair_changed_judgement_rows,
        "teacher_score_exact_agreement": score_exact,
        "teacher_score_adjacent_agreement": score_adjacent,
        "failure_bucket_agreement": failure_bucket_exact,
        "raw_overestimation_risk_exact_agreement": raw_risk_exact,
        "derived_overestimation_risk_agreement": derived_risk_exact,
        "evidence_type_exact_agreement": evidence_type_exact,
        "major_failures_jaccard_mean": mean(jaccards) if jaccards else 0.0,
        "evidence_span_valid_or_null_with_reason_rate": evidence_valid_rate,
        "label_quality_exact_agreement": label_quality_exact,
        "recommended_training_use_compatible_agreement": training_use_compatible_rate,
        "needs_human_review_agreement": needs_review_match,
        "high_control_hard_conflict_rate": high_control_hard_conflict_rate,
        "proceed_to_361": decision_pass,
    }
    write_json(out_dir / "decision" / "exp27d_teacher_audit_v4_api_repilot_decision.json", decision)

    report = [
        "# Exp27D Teacher Audit V4 API Re-Pilot Report",
        "",
        "This report summarizes parsed teacher outputs only. Raw API and full parsed text stay ignored.",
        "",
        "## Decision",
        "",
        f"- recommendation: `{recommendation}`",
        f"- proceed_to_361: {decision_pass}",
        "",
        "## Core Metrics",
        "",
        f"- parse_success_rate: {parse_success_rate:.4f}",
        f"- schema_validation_success_rate: {schema_success_rate:.4f}",
        f"- hard_validation_error_count: {hard_error_count}",
        f"- hard_validation_error_rate: {hard_error_rate:.4f}",
        f"- usable_annotation_rate: {usable_annotation_rate:.4f}",
        f"- repair_changed_judgement_count: {repair_changed_judgement_rows}",
        f"- teacher_score_exact_agreement: {score_exact:.4f}",
        f"- teacher_score_adjacent_agreement: {score_adjacent:.4f}",
        f"- failure_bucket_agreement: {failure_bucket_exact:.4f}",
        f"- derived_overestimation_risk_agreement: {derived_risk_exact:.4f}",
        f"- evidence_span_valid_or_null_with_reason_rate: {evidence_valid_rate:.4f}",
        f"- recommended_training_use_compatible_agreement: {training_use_compatible_rate:.4f}",
        f"- high_control_hard_conflict_rate: {high_control_hard_conflict_rate:.4f}",
        "",
        "## Derived Risk Rule",
        "",
        f"- version: `{DERIVED_RISK_RULE_VERSION}`",
        *[f"- {rule}" for rule in DERIVED_RISK_RULES],
        "",
        "## Thresholds",
        "",
        "- parse >= 0.99",
        "- schema >= 0.99",
        "- usable >= 0.95",
        "- hard error rate <= 0.03",
        "- adjacent score agreement >= 0.95",
        "- exact score agreement >= 0.60",
        "- failure bucket agreement >= 0.80",
        "- derived risk agreement >= 0.75",
        "- evidence valid >= 0.95",
        "- training-use compatible >= 0.95",
        "- high-control hard conflict <= 0.10",
        "- repair_changed_judgement_count == 0",
        "",
        "## Guardrails",
        "",
        "- no training",
        "- no GPU",
        "- no test labels",
        "- D1/dev labels are not used",
        "- raw API outputs remain under ignored `annotations/raw_api/`",
        "- full parsed teacher text remains under ignored `annotations/parsed/`",
    ]
    write_text(out_dir / "reports" / "exp27d_teacher_audit_v4_api_repilot_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp27D teacher-audit v4 API re-pilot summaries.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exp27c-packets", type=Path, default=DEFAULT_EXP27C_PACKETS)
    parser.add_argument("--exp27c-decision", type=Path, default=DEFAULT_EXP27C_DECISION)
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
