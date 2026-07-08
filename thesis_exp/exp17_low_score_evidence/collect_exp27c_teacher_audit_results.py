"""Collect lightweight Exp27C teacher-audit v3 API re-pilot summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.validate_exp27c_teacher_audit import validate  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42")
PROVIDERS = ["qwen", "deepseek"]
STAGES = ["blind", "audit"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def load_packets(out_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packets = read_jsonl(out_dir / "packets" / "exp27c_v3_repilot_blind_packets.jsonl")
    refs = read_jsonl(out_dir / "packets" / "exp27c_v3_repilot_audit_reference_private.jsonl")
    return {str(row["sample_id"]): row for row in packets}, {str(row["sample_id"]): row for row in refs}


def load_outputs(out_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for provider in PROVIDERS:
        for stage in STAGES:
            path = out_dir / "annotations" / "parsed" / provider / f"exp27c_{stage}_outputs.jsonl"
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
    return non_null, substring_valid, null_missing_reason_valid


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


def classify_validation_error(error: str) -> tuple[str, str]:
    provider_stage = error.split(":", 1)[0].split("[", 1)[0] if error else "unknown"
    if "low score with failure should use non-null score_cap" in error:
        kind = "missing_score_cap_for_low_failure"
    elif "no_major_failure must use overestimation_risk=low" in error:
        kind = "no_major_failure_risk_not_low"
    elif "teacher_reason restates score field" in error:
        kind = "reason_restates_score"
    elif "schema:" in error:
        kind = "schema_error"
    elif "evidence_span is not exact substring" in error:
        kind = "evidence_span_not_substring"
    elif "gap>=2 should use needs_human_review=true" in error:
        kind = "audit_gap_review_flag"
    else:
        kind = "other"
    return provider_stage, kind


def validation_annotation_key(error: str) -> str:
    if ":" not in error:
        return ""
    head = error.split(":", 1)[0]
    return head if "/" in head and "[" in head and "]" in head else ""


def collect(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    packets_by_id, refs_by_id = load_packets(out_dir)
    validation_summary = validate(out_dir, args.exp27a_agreement, with_annotations=True)
    validation_errors = read_csv_rows(out_dir / "tables" / "exp27c_v3_validation_errors.csv")
    validation_issues = read_csv_rows(out_dir / "tables" / "exp27c_v3_validation_issues.csv")
    outputs = load_outputs(out_dir)

    api_summary_rows: list[dict[str, Any]] = []
    total_annotation_rows = 0
    repair_attempt_rows = 0
    repair_success_rows = 0
    repair_changed_teacher_score_count = 0
    repair_changed_major_failures_count = 0
    for provider in PROVIDERS:
        for stage in STAGES:
            rows = outputs[(provider, stage)]
            parsed = sum(1 for row in rows if row.get("parsed") and not row.get("parse_error"))
            schema_ok = sum(1 for row in rows if schema_success(row))
            stage_repair_attempt_rows = sum(1 for row in rows if int(row.get("schema_repair_attempts") or 0) > 0)
            stage_repair_success_rows = sum(
                1 for row in rows if int(row.get("schema_repair_attempts") or 0) > 0 and schema_success(row)
            )
            stage_repair_changed_teacher_score_count = sum(
                1 for row in rows if row.get("repair_changed_teacher_score") is True
            )
            stage_repair_changed_major_failures_count = sum(
                1 for row in rows if row.get("repair_changed_major_failures") is True
            )
            total_annotation_rows += len(rows)
            repair_attempt_rows += stage_repair_attempt_rows
            repair_success_rows += stage_repair_success_rows
            repair_changed_teacher_score_count += stage_repair_changed_teacher_score_count
            repair_changed_major_failures_count += stage_repair_changed_major_failures_count
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
                    "repair_changed_teacher_score_count": stage_repair_changed_teacher_score_count,
                    "repair_changed_major_failures_count": stage_repair_changed_major_failures_count,
                }
            )

    q_blind = parsed_by_id(outputs[("qwen", "blind")])
    d_blind = parsed_by_id(outputs[("deepseek", "blind")])
    common_blind = sorted(set(q_blind) & set(d_blind))
    score_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
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
        score_exact = q_score == d_score
        score_adjacent = q_score is not None and d_score is not None and abs(q_score - d_score) <= 1
        score_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "pilot_group": refs_by_id.get(sid, {}).get("pilot_group", ""),
                "language": packet.get("source_meta", {}).get("language", ""),
                "qwen_score": q_score,
                "deepseek_score": d_score,
                "score_exact": score_exact,
                "score_adjacent": score_adjacent,
            }
        )
        jf = jaccard(qb.get("major_failures"), db.get("major_failures"))
        jaccards.append(jf)
        risk_rows.append(
            {
                "sample_id": sid,
                "label": label,
                "label_region": region,
                "qwen_failure_visibility": qb.get("failure_visibility", ""),
                "deepseek_failure_visibility": db.get("failure_visibility", ""),
                "failure_visibility_exact": qb.get("failure_visibility") == db.get("failure_visibility"),
                "qwen_overestimation_risk": qb.get("overestimation_risk", ""),
                "deepseek_overestimation_risk": db.get("overestimation_risk", ""),
                "overestimation_risk_exact": qb.get("overestimation_risk") == db.get("overestimation_risk"),
                "qwen_evidence_type": qb.get("evidence_type", ""),
                "deepseek_evidence_type": db.get("evidence_type", ""),
                "evidence_type_exact": qb.get("evidence_type") == db.get("evidence_type"),
                "major_failures_jaccard": jf,
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
        if not score_adjacent or qb.get("failure_visibility") != db.get("failure_visibility") or qb.get("overestimation_risk") != db.get("overestimation_risk"):
            conflict_rows.append(
                {
                    "sample_id": sid,
                    "label": label,
                    "label_region": region,
                    "qwen_score": q_score,
                    "deepseek_score": d_score,
                    "score_adjacent": score_adjacent,
                    "failure_visibility_match": qb.get("failure_visibility") == db.get("failure_visibility"),
                    "overestimation_risk_match": qb.get("overestimation_risk") == db.get("overestimation_risk"),
                    "evidence_type_match": qb.get("evidence_type") == db.get("evidence_type"),
                }
            )

    evidence_rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for row in outputs[(provider, "blind")]:
            sid = str(row.get("sample_id") or "")
            parsed = row.get("parsed") or {}
            blind = parsed.get("blind") or {}
            packet = packets_by_id.get(sid, {})
            non_null, substring_valid, null_missing_reason_valid = evidence_valid(blind, packet)
            label = as_int(refs_by_id.get(sid, {}).get("original_score"))
            evidence_rows.append(
                {
                    "provider": provider,
                    "sample_id": sid,
                    "label": label,
                    "label_region": label_region(label),
                    "evidence_non_null": non_null,
                    "evidence_substring_valid": substring_valid,
                    "evidence_null_with_missing_reason_valid": null_missing_reason_valid,
                    "evidence_valid_or_null_with_reason": substring_valid or null_missing_reason_valid or blind.get("evidence_type") == "not_applicable",
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
            }
        )

    def rate(rows: list[dict[str, Any]], field: str) -> float:
        return sum(1 for row in rows if row.get(field)) / len(rows) if rows else 0.0

    parse_success_rate = mean([row["parse_success_rate"] for row in api_summary_rows]) if api_summary_rows else 0.0
    schema_success_rate = mean([row["schema_validation_success_rate"] for row in api_summary_rows]) if api_summary_rows else 0.0
    semantic_error_count = int(validation_summary.get("errors", 0))
    semantic_error_rate = semantic_error_count / max(1, sum(row["rows"] for row in api_summary_rows))
    hard_error_count = int(validation_summary.get("hard_errors", 0))
    soft_error_count = int(validation_summary.get("soft_errors", 0))
    warning_count = int(validation_summary.get("warnings", 0))
    hard_error_rate = hard_error_count / max(1, sum(row["rows"] for row in api_summary_rows))
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
    failure_visibility_exact = rate(risk_rows, "failure_visibility_exact")
    overestimation_risk_exact = rate(risk_rows, "overestimation_risk_exact")
    evidence_type_exact = rate(risk_rows, "evidence_type_exact")
    evidence_valid_rate = rate(evidence_rows, "evidence_valid_or_null_with_reason")
    label_quality_exact = rate(audit_rows, "label_quality_exact")
    training_use_compatible_rate = rate(audit_rows, "training_use_compatible")
    needs_review_match = rate(audit_rows, "needs_human_review_match")
    high_audit = [row for row in audit_rows if row.get("label_region") == "high"]
    high_conflicts = sum(
        1
        for row in high_audit
        if row.get("qwen_label_quality") == "suspected_conflict"
        or row.get("deepseek_label_quality") == "suspected_conflict"
    )
    high_control_suspected_conflict_rate = high_conflicts / len(high_audit) if high_audit else 0.0

    decision_pass = (
        parse_success_rate >= 0.99
        and schema_success_rate >= 0.98
        and usable_annotation_rate >= 0.90
        and hard_error_rate <= 0.05
        and score_adjacent >= 0.95
        and score_exact >= 0.60
        and failure_visibility_exact >= 0.75
        and overestimation_risk_exact >= 0.70
        and evidence_valid_rate >= 0.95
        and training_use_compatible_rate >= 0.90
    )
    recommendation = "proceed_to_361_teacher_audit" if decision_pass else "revise_schema_or_prompt_before_361"

    stratified_rows: list[dict[str, Any]] = []
    for group_name, group_rows in [
        ("all", score_rows),
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

    error_counter: Counter[tuple[str, str, str]] = Counter()
    if validation_issues:
        for row in validation_issues:
            severity = as_text(row.get("severity")) or "unknown"
            kind = as_text(row.get("kind")) or "unknown"
            error = as_text(row.get("error"))
            if error:
                provider_stage, _ = classify_validation_error(error)
                error_counter[(provider_stage, severity, kind)] += 1
    else:
        for row in validation_errors:
            error = as_text(row.get("error"))
            if error:
                provider_stage, kind = classify_validation_error(error)
                error_counter[(provider_stage, "unknown", kind)] += 1
    error_breakdown_rows = [
        {"provider_stage": provider_stage, "severity": severity, "error_kind": kind, "count": count}
        for (provider_stage, severity, kind), count in sorted(error_counter.items())
    ]

    write_csv(out_dir / "tables" / "exp27c_v3_api_repilot_summary.csv", api_summary_rows + stratified_rows)
    write_csv(out_dir / "tables" / "exp27c_v3_cross_provider_score_agreement.csv", score_rows)
    write_csv(out_dir / "tables" / "exp27c_v3_cross_provider_risk_agreement.csv", risk_rows)
    write_csv(out_dir / "tables" / "exp27c_v3_evidence_span_validation.csv", evidence_rows)
    write_csv(out_dir / "tables" / "exp27c_v3_score_cap_consistency.csv", cap_rows)
    write_csv(out_dir / "tables" / "exp27c_v3_label_audit_agreement.csv", audit_rows)
    write_csv(
        out_dir / "tables" / "exp27c_v3_semantic_validation_errors.csv",
        validation_errors or [{"error": ""}],
    )
    write_csv(
        out_dir / "tables" / "exp27c_v3_validation_error_breakdown.csv",
        error_breakdown_rows or [{"provider_stage": "", "severity": "", "error_kind": "", "count": 0}],
    )
    write_csv(out_dir / "tables" / "exp27c_v3_teacher_conflict_cases_light.csv", conflict_rows or [{"sample_id": ""}])

    decision = {
        "recommendation": recommendation,
        "parse_success_rate": parse_success_rate,
        "schema_validation_success_rate": schema_success_rate,
        "semantic_validation_error_count": semantic_error_count,
        "semantic_validation_error_rate": semantic_error_rate,
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
        "teacher_score_exact_agreement": score_exact,
        "teacher_score_adjacent_agreement": score_adjacent,
        "failure_visibility_exact_agreement": failure_visibility_exact,
        "overestimation_risk_exact_agreement": overestimation_risk_exact,
        "evidence_type_exact_agreement": evidence_type_exact,
        "major_failures_jaccard_mean": mean(jaccards) if jaccards else 0.0,
        "evidence_span_valid_or_null_with_reason_rate": evidence_valid_rate,
        "label_quality_exact_agreement": label_quality_exact,
        "recommended_training_use_compatible_agreement": training_use_compatible_rate,
        "needs_human_review_agreement": needs_review_match,
        "high_control_suspected_conflict_rate": high_control_suspected_conflict_rate,
        "proceed_to_361": decision_pass,
    }
    write_json(out_dir / "decision" / "exp27c_teacher_audit_v3_api_repilot_decision.json", decision)
    report = [
        "# Exp27C Teacher Audit V3 API Re-Pilot Report",
        "",
        "This report summarizes the 60-row dual-teacher v3 re-pilot. It reads parsed outputs only.",
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
        f"- semantic_validation_error_count: {semantic_error_count}",
        f"- hard_validation_error_count: {hard_error_count}",
        f"- hard_validation_error_rate: {hard_error_rate:.4f}",
        f"- soft_validation_error_count: {soft_error_count}",
        f"- warning_count: {warning_count}",
        f"- usable_annotation_rate: {usable_annotation_rate:.4f}",
        f"- schema_repair_attempt_rate: {schema_repair_attempt_rate:.4f}",
        f"- repair_success_rate: {repair_success_rate:.4f}",
        f"- repair_changed_teacher_score_count: {repair_changed_teacher_score_count}",
        f"- repair_changed_major_failures_count: {repair_changed_major_failures_count}",
        f"- teacher_score_exact_agreement: {score_exact:.4f}",
        f"- teacher_score_adjacent_agreement: {score_adjacent:.4f}",
        f"- failure_visibility_exact_agreement: {failure_visibility_exact:.4f}",
        f"- overestimation_risk_exact_agreement: {overestimation_risk_exact:.4f}",
        f"- evidence_type_exact_agreement: {evidence_type_exact:.4f}",
        f"- major_failures_jaccard_mean: {decision['major_failures_jaccard_mean']:.4f}",
        f"- evidence_span_valid_or_null_with_reason_rate: {evidence_valid_rate:.4f}",
        f"- label_quality_exact_agreement: {label_quality_exact:.4f}",
        f"- recommended_training_use_compatible_agreement: {training_use_compatible_rate:.4f}",
        f"- needs_human_review_agreement: {needs_review_match:.4f}",
        f"- high_control_suspected_conflict_rate: {high_control_suspected_conflict_rate:.4f}",
        "",
        "## Validation Error Breakdown",
        "",
    ]
    if error_breakdown_rows:
        report.extend(
            f"- {row['provider_stage']} / {row['severity']} / {row['error_kind']}: {row['count']}"
            for row in error_breakdown_rows
        )
    else:
        report.append("- no validation errors")
    report.extend(
        [
            "",
            "## Guardrails",
            "",
            "- no training",
            "- no GPU",
            "- no test labels",
            "- raw API outputs remain under ignored `annotations/raw_api/`",
            "- full parsed teacher text remains under ignored `annotations/parsed/`",
        ]
    )
    write_text(out_dir / "reports" / "exp27c_teacher_audit_v3_api_repilot_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp27C teacher-audit v3 API re-pilot summaries.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--exp27a-agreement",
        type=Path,
        default=Path(
            "thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42/"
            "tables/exp27a_teacher_blind_cross_provider_agreement.csv"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
