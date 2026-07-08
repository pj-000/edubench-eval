"""Validate Exp27A teacher-audit packets and optional API annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp27a_teacher_audit_packets import (  # noqa: E402
    AGREEMENT_VALUES,
    CONFIDENCE_VALUES,
    FAILURE_TAGS,
    QUALITY_VALUES,
    RISK_FLAGS,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27a_teacher_audit_pilot_seed42")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def explains_absence(reason: str) -> bool:
    lowered = reason.lower()
    cues = [
        "missing",
        "absence",
        "absent",
        "no reasoning",
        "no explanation",
        "no evidence",
        "no personalization",
        "no reference",
        "no integration",
        "only the final answer",
        "only provides",
        "without any reasoning",
        "without reasoning",
        "generic",
        "templated",
        "缺失",
        "缺少",
        "没有任何",
        "没有推理",
        "没有解释",
        "没有证据",
        "没有展示",
        "没有体现",
        "没有个性化",
        "没有场景",
        "仅给出",
        "仅提供",
        "只给出",
        "只提供",
        "通用",
        "模板",
    ]
    return any(cue in lowered for cue in cues)


def validate_blind(parsed: dict[str, Any], packet: dict[str, Any] | None, errors: list[str], where: str) -> None:
    blind = parsed.get("blind")
    if not isinstance(blind, dict):
        errors.append(f"{where}: missing blind object")
        return
    score = blind.get("teacher_score")
    if not isinstance(score, int) or score < 1 or score > 5:
        errors.append(f"{where}: teacher_score invalid: {score!r}")
    reason = as_text(blind.get("teacher_reason")).strip()
    if len(reason) < 12:
        errors.append(f"{where}: teacher_reason too short")
    failures = blind.get("major_failures")
    if not isinstance(failures, list) or not failures:
        errors.append(f"{where}: major_failures must be non-empty list")
    else:
        unknown = [item for item in failures if item not in FAILURE_TAGS]
        if unknown:
            errors.append(f"{where}: unknown major_failures {unknown}")
    cap = blind.get("score_cap")
    if cap is not None and (not isinstance(cap, int) or cap < 1 or cap > 5):
        errors.append(f"{where}: score_cap invalid: {cap!r}")
    risk = blind.get("risk_flag")
    if risk not in RISK_FLAGS:
        errors.append(f"{where}: risk_flag invalid: {risk!r}")
    conf = blind.get("confidence")
    if conf not in CONFIDENCE_VALUES:
        errors.append(f"{where}: confidence invalid: {conf!r}")
    evidence = blind.get("evidence_span")
    if evidence is not None and not isinstance(evidence, str):
        errors.append(f"{where}: evidence_span must be string or null")
    if packet is not None and isinstance(evidence, str) and evidence.strip():
        answer = as_text(packet.get("teacher_input", {}).get("answer"))
        if evidence not in answer:
            errors.append(f"{where}: evidence_span is not exact substring of answer")
    if isinstance(failures, list) and failures == ["no_major_failure"]:
        if evidence is not None:
            errors.append(f"{where}: no_major_failure must use evidence_span=null")
        if cap is not None:
            errors.append(f"{where}: no_major_failure must use score_cap=null")
    if risk == "clean_high" and cap is not None:
        errors.append(f"{where}: clean_high must use score_cap=null")
    if (
        score is not None
        and isinstance(score, int)
        and score <= 2
        and evidence is None
        and conf == "high"
        and not explains_absence(reason)
    ):
        errors.append(f"{where}: low score with null evidence_span cannot have high confidence")


def validate_audit(parsed: dict[str, Any], packet: dict[str, Any] | None, errors: list[str], where: str) -> None:
    audit = parsed.get("audit")
    if not isinstance(audit, dict):
        errors.append(f"{where}: missing audit object")
        return
    score = audit.get("original_score")
    if score is not None and (not isinstance(score, int) or score < 1 or score > 5):
        errors.append(f"{where}: original_score invalid: {score!r}")
    if (
        packet is not None
        and "original_score_for_audit_only" in packet
        and score is not None
        and score != packet.get("original_score_for_audit_only")
    ):
        errors.append(f"{where}: original_score does not match packet")
    if audit.get("score_agreement") not in AGREEMENT_VALUES:
        errors.append(f"{where}: score_agreement invalid: {audit.get('score_agreement')!r}")
    if audit.get("label_quality") not in QUALITY_VALUES:
        errors.append(f"{where}: label_quality invalid: {audit.get('label_quality')!r}")
    if not isinstance(audit.get("needs_human_review"), bool):
        errors.append(f"{where}: needs_human_review must be bool")
    if len(as_text(audit.get("audit_reason")).strip()) < 8:
        errors.append(f"{where}: audit_reason too short")


def validate_parsed(parsed: Any, packet: dict[str, Any] | None, errors: list[str], where: str) -> None:
    if not isinstance(parsed, dict):
        errors.append(f"{where}: parsed is not object")
        return
    sid = parsed.get("sample_id")
    if not isinstance(sid, str) or not sid:
        errors.append(f"{where}: sample_id missing")
    if packet is not None and sid != packet.get("sample_id"):
        errors.append(f"{where}: sample_id mismatch")
    validate_blind(parsed, packet, errors, where)
    validate_audit(parsed, packet, errors, where)


def validate_packets(out_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    packets_path = out_dir / "packets" / "exp27a_pilot_blind_packets.jsonl"
    audit_ref_path = out_dir / "packets" / "exp27a_pilot_audit_reference_private.jsonl"
    packets = read_jsonl(packets_path)
    audit_ref_by_id = {str(row.get("sample_id")): row for row in read_jsonl(audit_ref_path)}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, packet in enumerate(packets):
        sid = packet.get("sample_id")
        if not isinstance(sid, str) or not sid:
            errors.append(f"packet[{idx}]: sample_id missing")
            continue
        if sid in seen:
            errors.append(f"packet[{idx}]: duplicate sample_id {sid}")
        seen.add(sid)
        if "original_score_for_audit_only" in packet:
            errors.append(f"packet[{idx}]: blind packet leaks original_score_for_audit_only")
        if packet.get("source_meta", {}).get("label_5") is not None:
            errors.append(f"packet[{idx}]: blind packet leaks source_meta.label_5")
        audit_ref = audit_ref_by_id.get(sid)
        if audit_ref is None:
            errors.append(f"packet[{idx}]: missing audit reference")
        teacher_input = packet.get("teacher_input")
        if not isinstance(teacher_input, dict):
            errors.append(f"packet[{idx}]: teacher_input missing")
        else:
            for key in ["question", "answer", "metric", "rubric", "metadata"]:
                if not as_text(teacher_input.get(key)).strip():
                    errors.append(f"packet[{idx}]: empty teacher_input.{key}")
        rows.append(
            {
                "sample_id": sid,
                "batch_id": packet.get("batch_id", ""),
                "pilot_group": packet.get("pilot_group", ""),
                "label": "" if audit_ref is None else audit_ref.get("original_score", ""),
                "language": packet.get("source_meta", {}).get("language", ""),
                "metric": packet.get("source_meta", {}).get("metric", ""),
            }
        )
    return packets, errors, rows


def validate_annotation_file(
    provider: str,
    stage: str,
    path: Path,
    packet_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_jsonl(path)
    errors: list[str] = []
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sid = as_text(row.get("sample_id"))
        packet = packet_by_id.get(sid)
        parse_error = as_text(row.get("parse_error"))
        parsed = row.get("parsed")
        if parse_error:
            errors.append(f"{provider}/{stage}[{idx}]: parse_error={parse_error[:120]}")
        validate_parsed(parsed, packet, errors, f"{provider}/{stage}[{idx}]")
        blind = parsed.get("blind") if isinstance(parsed, dict) and isinstance(parsed.get("blind"), dict) else {}
        audit = parsed.get("audit") if isinstance(parsed, dict) and isinstance(parsed.get("audit"), dict) else {}
        out_rows.append(
            {
                "provider": provider,
                "stage": stage,
                "sample_id": sid,
                "teacher_score": blind.get("teacher_score", ""),
                "major_failures": ";".join(as_text(item) for item in blind.get("major_failures", []))
                if isinstance(blind.get("major_failures"), list)
                else as_text(blind.get("major_failures")),
                "evidence_span": as_text(blind.get("evidence_span")),
                "score_cap": blind.get("score_cap", ""),
                "risk_flag": blind.get("risk_flag", ""),
                "confidence": blind.get("confidence", ""),
                "teacher_reason": as_text(blind.get("teacher_reason"))[:500],
                "score_agreement": audit.get("score_agreement", ""),
                "label_quality": audit.get("label_quality", ""),
                "needs_human_review": audit.get("needs_human_review", ""),
                "parse_error": bool(parse_error),
            }
        )
    return out_rows, errors


def validate(out_dir: Path) -> dict[str, Any]:
    packets, packet_errors, packet_rows = validate_packets(out_dir)
    packet_by_id = {str(packet.get("sample_id")): packet for packet in packets if packet.get("sample_id")}
    all_errors = list(packet_errors)
    provider_rows: list[dict[str, Any]] = []
    ann_dir = out_dir / "annotations"
    for provider_dir in sorted(ann_dir.glob("*")) if ann_dir.exists() else []:
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name
        for stage in ["blind", "audit"]:
            path = provider_dir / f"exp27a_{stage}_outputs.jsonl"
            if path.exists():
                rows, errors = validate_annotation_file(provider, stage, path, packet_by_id)
                provider_rows.extend(rows)
                all_errors.extend(errors)

    packet_distribution = Counter((row["pilot_group"], row["label"], row["language"]) for row in packet_rows)
    write_csv(
        out_dir / "tables" / "exp27a_validation_packet_distribution.csv",
        [
            {"pilot_group": group, "label": label, "language": lang, "count": count}
            for (group, label, lang), count in sorted(packet_distribution.items())
        ],
    )
    if provider_rows:
        write_csv(out_dir / "tables" / "exp27a_teacher_output_validation_rows.csv", provider_rows)
    error_rows = [{"error": error} for error in all_errors[:500]]
    write_csv(out_dir / "tables" / "exp27a_validation_errors.csv", error_rows)
    summary = {
        "packets": len(packets),
        "annotation_rows": len(provider_rows),
        "errors": len(all_errors),
        "parse_error_rows": sum(1 for row in provider_rows if row.get("parse_error")),
        "providers_seen": sorted({row["provider"] for row in provider_rows}),
        "stages_seen": sorted({row["stage"] for row in provider_rows}),
        "ready_for_teacher_pilot": len(packet_errors) == 0,
    }
    write_json(out_dir / "decision" / "exp27a_teacher_audit_validation_decision.json", summary)
    report = [
        "# Exp27A Teacher Audit Validation",
        "",
        f"- packets: {summary['packets']}",
        f"- annotation rows: {summary['annotation_rows']}",
        f"- errors: {summary['errors']}",
        f"- parse_error_rows: {summary['parse_error_rows']}",
        f"- ready_for_teacher_pilot: {summary['ready_for_teacher_pilot']}",
    ]
    if all_errors:
        report.extend(["", "## First Errors", ""])
        report.extend(f"- {error}" for error in all_errors[:80])
    write_text(out_dir / "reports" / "exp27a_teacher_audit_validation_report.md", "\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp27A teacher-audit packets/outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate(args.out_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
