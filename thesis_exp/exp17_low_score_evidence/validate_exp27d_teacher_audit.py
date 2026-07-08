"""Validate Exp27D teacher-audit v4 packets and optional annotations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42")
DEFAULT_EXP27C_PACKETS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27c_teacher_audit_v3_seed42/"
    "packets/exp27c_v3_repilot_blind_packets.jsonl"
)

FORBIDDEN_BLIND_KEYS = {
    "original_score",
    "label_5",
    "human_1",
    "human_2",
    "human_3",
    "human_1_raw",
    "human_2_raw",
    "human_3_raw",
    "human_reason",
    "recovered_reason",
    "reason_manual",
}
FORBIDDEN_BLIND_FAILURES = {"possible_label_conflict", "answer_key_or_reference_mismatch"}
SEVERE_FAILURES = {
    "missing_key_point",
    "missing_personalization",
    "missing_scenario_integration",
    "rubric_mismatch",
    "factual_error",
    "insufficient_reasoning",
    "insufficient_evidence",
    "partial_or_incomplete",
    "task_constraint_violation",
    "format_violation",
    "irrelevant_or_off_topic",
    "unsupported_claim",
}


def normalized_contains(needle: str, haystack: str) -> bool:
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    return bool(needle and (needle in haystack or norm(needle) in norm(haystack)))


def classify_issue(error: str) -> tuple[str, str]:
    if "schema:" in error or "parsed is not object" in error or "parse_error=" in error:
        return "hard", "schema_or_parse_error"
    if "evidence_span is not exact or normalized substring" in error:
        return "hard", "evidence_span_invalid"
    if "requires" in error or "must use" in error or "missing" in error:
        if "teacher_reason restates score field" in error:
            return "soft", "reason_restates_score"
        if "score=3 no_major_failure risk" in error:
            return "warning", "score3_no_major_failure_risk"
        return "hard", "semantic_hard_rule"
    if "teacher_reason restates score field" in error:
        return "soft", "reason_restates_score"
    if "gap>=2" in error:
        return "hard", "audit_gap_rule"
    return "warning", "semantic_warning"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def score_region(score: int) -> str:
    if score <= 2:
        return "low"
    if score == 3:
        return "mid"
    return "high"


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(walk_keys(child))
    return keys


def load_schemas(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    blind_schema = read_json(out_dir / "schema" / "exp27d_teacher_blind_schema_v4.json")
    audit_schema = read_json(out_dir / "schema" / "exp27d_teacher_audit_schema_v4.json")
    return blind_schema, audit_schema


def validate_packet_files(
    out_dir: Path,
    exp27c_packets: Path,
    errors: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    packets = read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
    audit_ref = read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
    exp27c_rows = read_jsonl(exp27c_packets)
    exp27c_ids = {str(row.get("sample_id")) for row in exp27c_rows if row.get("sample_id")}
    audit_by_id = {str(row.get("sample_id")): row for row in audit_ref}

    if len(packets) != 80:
        errors.append(f"packets: expected exactly 80 rows, got {len(packets)}")
    if len(exp27c_ids) != 60:
        errors.append(f"exp27c_packets: expected 60 unique ids, got {len(exp27c_ids)}")

    seen: set[str] = set()
    for idx, packet in enumerate(packets):
        sid = as_text(packet.get("sample_id"))
        if not sid:
            errors.append(f"packet[{idx}]: missing sample_id")
        if sid in seen:
            errors.append(f"packet[{idx}]: duplicate sample_id {sid}")
        seen.add(sid)
        if packet.get("split") != "train":
            errors.append(f"packet[{idx}]: split is not train")
        forbidden_keys = FORBIDDEN_BLIND_KEYS & set(walk_keys(packet))
        if forbidden_keys:
            errors.append(f"packet[{idx}]: blind packet leaks forbidden keys {sorted(forbidden_keys)}")
        teacher_input = packet.get("teacher_input")
        if not isinstance(teacher_input, dict):
            errors.append(f"packet[{idx}]: missing teacher_input")
        else:
            for key in ["question", "answer", "metric", "rubric", "metadata"]:
                if not as_text(teacher_input.get(key)).strip():
                    errors.append(f"packet[{idx}]: empty teacher_input.{key}")
        ref = audit_by_id.get(sid)
        if ref is None:
            errors.append(f"packet[{idx}]: missing audit reference")
        elif ref.get("split") != "train":
            errors.append(f"packet[{idx}]: audit reference split is not train")

    missing_exp27c = sorted(exp27c_ids - seen)
    if missing_exp27c:
        errors.append(f"packets: missing Exp27C paired ids {missing_exp27c[:5]}")

    ref_ids = set(audit_by_id)
    extra_refs = sorted(ref_ids - seen)
    if extra_refs:
        errors.append(f"audit_reference: contains ids not in blind packets {extra_refs[:5]}")

    return packets, audit_ref, exp27c_ids


def validate_leakage(out_dir: Path, errors: list[str]) -> None:
    rows = read_csv_rows(out_dir / "tables" / "exp27d_v4_leakage_audit.csv")
    seen = {row.get("check"): int(row.get("count") or 0) for row in rows}
    for key in ["dev_sample_overlap", "dev_question_overlap", "test_sample_overlap", "test_question_overlap", "test_label_read"]:
        if seen.get(key, 999) != 0:
            errors.append(f"leakage: {key}={seen.get(key)}")


def validate_blind_object(blind: dict[str, Any], packet: dict[str, Any] | None, errors: list[str], where: str) -> None:
    score = blind.get("teacher_score")
    if isinstance(score, int):
        expected = score_region(score)
        if blind.get("score_region") != expected:
            errors.append(f"{where}: score_region={blind.get('score_region')!r} does not match teacher_score={score}")
    reason = as_text(blind.get("teacher_reason")).strip()
    if re.search(r"\b(score|评分|得分)\s*[:=]?\s*[1-5]\b", reason, flags=re.IGNORECASE):
        errors.append(f"{where}: teacher_reason restates score field")
    failures = blind.get("major_failures")
    if isinstance(failures, list):
        bad_failures = sorted(set(failures) & FORBIDDEN_BLIND_FAILURES)
        if bad_failures:
            errors.append(f"{where}: forbidden blind major_failures {bad_failures}")
        if "no_major_failure" in failures:
            if set(failures) != {"no_major_failure"}:
                errors.append(f"{where}: no_major_failure mixed with other failures")
            if blind.get("score_cap") is not None:
                errors.append(f"{where}: no_major_failure must use score_cap=null")
            if blind.get("failure_visibility") != "no_major_failure":
                errors.append(f"{where}: no_major_failure must use failure_visibility=no_major_failure")
            if isinstance(score, int):
                if score <= 2:
                    errors.append(f"{where}: low score must not use no_major_failure")
                elif score == 3 and blind.get("overestimation_risk") not in {"low", "medium"}:
                    errors.append(f"{where}: score=3 no_major_failure risk should be low or medium")
                elif score >= 4 and blind.get("overestimation_risk") != "low":
                    errors.append(f"{where}: high no_major_failure must use overestimation_risk=low")
        elif set(failures) & SEVERE_FAILURES and blind.get("failure_visibility") == "no_major_failure":
            errors.append(f"{where}: severe failures must not use failure_visibility=no_major_failure")
    evidence = blind.get("evidence_span")
    if evidence is not None and not isinstance(evidence, str):
        errors.append(f"{where}: evidence_span must be string or null")
    if packet is not None and isinstance(evidence, str) and evidence.strip():
        answer = as_text(packet.get("teacher_input", {}).get("answer"))
        if not normalized_contains(evidence, answer):
            errors.append(f"{where}: evidence_span is not exact or normalized substring of answer")
        if blind.get("evidence_type") == "missing_required_content":
            errors.append(f"{where}: missing_required_content should use evidence_span=null")
    if evidence is None and blind.get("evidence_type") == "verbatim_answer_span":
        errors.append(f"{where}: verbatim_answer_span requires non-null evidence_span")
    if blind.get("evidence_type") == "missing_required_content" and blind.get("missing_evidence_reason") is None:
        errors.append(f"{where}: missing_required_content requires missing_evidence_reason")
    cap = blind.get("score_cap")
    visibility = blind.get("failure_visibility")
    if isinstance(score, int):
        if score <= 2 and visibility != "no_major_failure":
            if cap is None:
                errors.append(f"{where}: low score with failure should use non-null score_cap")
            elif isinstance(cap, int) and cap > 3:
                errors.append(f"{where}: low score_cap should be <=3")
        if score >= 4 and "no_major_failure" in (failures or []) and cap is not None:
            errors.append(f"{where}: high no_major_failure should use score_cap=null")


def validate_audit_object(audit: dict[str, Any], errors: list[str], where: str) -> None:
    original = audit.get("original_score")
    teacher = audit.get("teacher_score")
    if original is None:
        errors.append(f"{where}: audit missing original_score")
    if teacher is None:
        errors.append(f"{where}: audit missing teacher_score")
    if isinstance(original, int) and isinstance(teacher, int):
        gap = abs(original - teacher)
        if gap == 0 and audit.get("score_agreement") != "exact":
            errors.append(f"{where}: gap=0 should use score_agreement=exact")
        if gap == 1 and audit.get("score_agreement") not in {"adjacent", "unclear"}:
            errors.append(f"{where}: gap=1 should use score_agreement=adjacent or unclear")
        if gap >= 2:
            if audit.get("score_agreement") != "conflict":
                errors.append(f"{where}: gap>=2 should use score_agreement=conflict")
            if audit.get("needs_human_review") is not True:
                errors.append(f"{where}: gap>=2 should use needs_human_review=true")
            if audit.get("hard_conflict") is not True:
                errors.append(f"{where}: gap>=2 should use hard_conflict=true")
    if audit.get("hard_conflict") is True and audit.get("recommended_training_use") == "high_weight":
        errors.append(f"{where}: hard_conflict should not use recommended_training_use=high_weight")


def validate_annotation_file(
    provider: str,
    stage: str,
    path: Path,
    packet_by_id: dict[str, dict[str, Any]],
    blind_schema: dict[str, Any],
    audit_schema: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    schema = blind_schema if stage == "blind" else audit_schema
    validator = jsonschema.Draft202012Validator(schema)
    for idx, row in enumerate(read_jsonl(path)):
        sid = as_text(row.get("sample_id"))
        parsed = row.get("parsed")
        parse_error = as_text(row.get("parse_error"))
        where = f"{provider}/{stage}[{idx}]"
        if parse_error:
            errors.append(f"{where}: parse_error={parse_error[:160]}")
        if not isinstance(parsed, dict):
            errors.append(f"{where}: parsed is not object")
            continue
        for err in validator.iter_errors(parsed):
            errors.append(f"{where}: schema: {err.message}")
        if parsed.get("sample_id") != sid:
            errors.append(f"{where}: sample_id mismatch")
        packet = packet_by_id.get(sid)
        row_out: dict[str, Any] = {
            "provider": provider,
            "stage": stage,
            "sample_id": sid,
            "parse_error": bool(parse_error),
        }
        if stage == "blind":
            blind = parsed.get("blind") if isinstance(parsed.get("blind"), dict) else {}
            validate_blind_object(blind, packet, errors, where)
            row_out.update(
                {
                    "teacher_score": blind.get("teacher_score", ""),
                    "score_region": blind.get("score_region", ""),
                    "failure_visibility": blind.get("failure_visibility", ""),
                    "overestimation_risk": blind.get("overestimation_risk", ""),
                    "surface_plausibility": blind.get("surface_plausibility", ""),
                    "evidence_type": blind.get("evidence_type", ""),
                    "missing_evidence_reason": blind.get("missing_evidence_reason", ""),
                    "major_failures": ";".join(as_text(item) for item in blind.get("major_failures", []))
                    if isinstance(blind.get("major_failures"), list)
                    else as_text(blind.get("major_failures")),
                }
            )
        else:
            audit = parsed.get("audit") if isinstance(parsed.get("audit"), dict) else {}
            validate_audit_object(audit, errors, where)
            if not as_text(parsed.get("blind_annotation_id")).strip():
                errors.append(f"{where}: missing blind_annotation_id")
            if not as_text(parsed.get("blind_annotation_hash")).strip():
                errors.append(f"{where}: missing blind_annotation_hash")
            row_out.update(
                {
                    "teacher_score": audit.get("teacher_score", ""),
                    "original_score": audit.get("original_score", ""),
                    "score_agreement": audit.get("score_agreement", ""),
                    "label_quality": audit.get("label_quality", ""),
                    "label_noise_type": audit.get("label_noise_type", ""),
                    "recommended_training_use": audit.get("recommended_training_use", ""),
                    "sample_weight_suggestion": audit.get("sample_weight_suggestion", ""),
                    "needs_human_review": audit.get("needs_human_review", ""),
                    "strictness_disagreement": audit.get("strictness_disagreement", ""),
                    "hard_conflict": audit.get("hard_conflict", ""),
                }
            )
        out_rows.append(row_out)
    return out_rows


def validate(out_dir: Path, exp27c_packets: Path = DEFAULT_EXP27C_PACKETS, with_annotations: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    packets, audit_ref, exp27c_ids = validate_packet_files(out_dir, exp27c_packets, errors)
    validate_leakage(out_dir, errors)
    blind_schema, audit_schema = load_schemas(out_dir)
    packet_by_id = {str(packet.get("sample_id")): packet for packet in packets if packet.get("sample_id")}
    provider_rows: list[dict[str, Any]] = []
    if with_annotations:
        ann_dir = out_dir / "annotations" / "parsed"
        for provider_dir in sorted(ann_dir.glob("*")) if ann_dir.exists() else []:
            if not provider_dir.is_dir():
                continue
            provider = provider_dir.name
            for stage in ["blind", "audit"]:
                path = provider_dir / f"exp27d_{stage}_outputs.jsonl"
                if path.exists():
                    provider_rows.extend(
                        validate_annotation_file(provider, stage, path, packet_by_id, blind_schema, audit_schema, errors)
                    )

    audit_by_id = {str(row.get("sample_id")): row for row in audit_ref}
    dist = Counter(
        (
            packet.get("pilot_group", ""),
            audit_by_id.get(str(packet.get("sample_id")), {}).get("original_score", ""),
            packet.get("source_meta", {}).get("language", ""),
        )
        for packet in packets
    )
    write_csv(
        out_dir / "tables" / "exp27d_v4_validation_packet_distribution.csv",
        [
            {"pilot_group": group, "label": label, "language": lang, "count": count}
            for (group, label, lang), count in sorted(dist.items())
        ],
    )
    if provider_rows:
        write_csv(out_dir / "tables" / "exp27d_v4_teacher_output_validation_rows.csv", provider_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_validation_errors.csv", [{"error": error} for error in errors[:500]])
    issue_rows = [
        {"severity": severity, "kind": kind, "error": error}
        for error in errors
        for severity, kind in [classify_issue(error)]
    ]
    severity_counts = Counter(row["severity"] for row in issue_rows)
    kind_counts = Counter((row["severity"], row["kind"]) for row in issue_rows)
    write_csv(out_dir / "tables" / "exp27d_v4_validation_issues.csv", issue_rows or [{"severity": "", "kind": "", "error": ""}])
    write_csv(
        out_dir / "tables" / "exp27d_v4_validation_issue_breakdown.csv",
        [
            {"severity": severity, "kind": kind, "count": count}
            for (severity, kind), count in sorted(kind_counts.items())
        ]
        or [{"severity": "", "kind": "", "count": 0}],
    )

    packet_ids = {str(packet.get("sample_id")) for packet in packets}
    hard_errors = severity_counts.get("hard", 0)
    soft_errors = severity_counts.get("soft", 0)
    warnings = severity_counts.get("warning", 0)
    summary = {
        "packets": len(packets),
        "annotation_rows": len(provider_rows),
        "errors": len(errors),
        "hard_errors": hard_errors,
        "soft_errors": soft_errors,
        "warnings": warnings,
        "schema_validation_passed": hard_errors == 0,
        "with_annotations": with_annotations,
        "exp27c_overlap_expected": len(exp27c_ids),
        "exp27c_overlap_present": len(exp27c_ids & packet_ids),
        "blind_packet_original_score_count": 0,
        "human_reason_in_prompt_count": 0,
        "test_label_read": False,
        "ready_for_v4_api_repilot": hard_errors == 0,
    }
    write_json(out_dir / "decision" / "exp27d_teacher_audit_v4_validation_decision.json", summary)
    report = [
        "# Exp27D Teacher Audit V4 Validation",
        "",
        f"- packets: {summary['packets']}",
        f"- annotation_rows: {summary['annotation_rows']}",
        f"- errors: {summary['errors']}",
        f"- hard_errors: {summary['hard_errors']}",
        f"- soft_errors: {summary['soft_errors']}",
        f"- warnings: {summary['warnings']}",
        f"- exp27c_overlap_present: {summary['exp27c_overlap_present']}/{summary['exp27c_overlap_expected']}",
        f"- schema_validation_passed: {summary['schema_validation_passed']}",
        f"- ready_for_v4_api_repilot: {summary['ready_for_v4_api_repilot']}",
        f"- test_label_read: {summary['test_label_read']}",
    ]
    if errors:
        report.extend(["", "## First Errors", ""])
        report.extend(f"- {error}" for error in errors[:80])
    write_text(out_dir / "reports" / "exp27d_teacher_audit_v4_validation_report.md", "\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp27D teacher-audit v4 artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--exp27c-packets", type=Path, default=DEFAULT_EXP27C_PACKETS)
    parser.add_argument("--with-annotations", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.out_dir, args.exp27c_packets, with_annotations=args.with_annotations),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
