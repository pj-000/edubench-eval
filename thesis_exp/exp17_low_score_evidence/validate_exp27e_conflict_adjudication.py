"""Validate Exp27E provider-bias and conflict-adjudication outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jsonschema


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27e_provider_bias_conflict_analysis_seed42"
)
DEFAULT_DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
REQUIRED_FILES = [
    "tables/exp27e_provider_vs_human_score_bias.csv",
    "tables/exp27e_provider_training_use_distribution.csv",
    "tables/exp27e_conflict_type_distribution.csv",
    "tables/exp27e_consensus_policy_simulation.csv",
    "tables/exp27e_leakage_audit.csv",
    "annotation/exp27e_gpt55_human_adjudication_queue.csv",
    "annotation/exp27e_gpt55_human_adjudication_packets.jsonl",
    "reports/exp27e_provider_bias_conflict_analysis_report.md",
    "decision/exp27e_provider_bias_conflict_analysis_decision.json",
]
FORBIDDEN_PACKET_KEYS = {
    "teacher_reason",
    "audit_reason",
    "raw_api_path",
    "response_meta",
    "thinking",
    "blind",
    "audit",
    "parsed",
}


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(path: Path) -> tuple[set[str], set[str]]:
    sample_ids: set[str] = set()
    question_keys: set[str] = set()
    for row in read_jsonl(path):
        sid = str(row.get("sample_id") or row.get("id") or row.get("record_id") or "")
        qkey = str(row.get("question_key") or row.get("question_id") or "")
        if sid:
            sample_ids.add(sid)
        if qkey:
            question_keys.add(qkey)
    return sample_ids, question_keys


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(walk_keys(child))
    return keys


def validate(out_dir: Path, dev_jsonl: Path = DEFAULT_DEV, test_jsonl: Path = DEFAULT_TEST) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_FILES:
        if not (out_dir / rel).exists():
            errors.append(f"missing required file: {rel}")

    decision = read_json(out_dir / "decision" / "exp27e_provider_bias_conflict_analysis_decision.json")
    for key in [
        "recommend_use_both_for_361",
        "recommended_primary_teacher_for_full_train",
        "recommend_selective_second_teacher",
        "recommend_gpt55_or_human_adjudication",
        "proceed_to_361_after_adjudication",
    ]:
        if key not in decision:
            errors.append(f"decision missing key: {key}")
    if decision.get("recommended_primary_teacher_for_full_train") not in {"qwen", "deepseek", "none_yet"}:
        errors.append("decision recommended_primary_teacher_for_full_train has invalid value")

    queue_rows = read_csv_rows(out_dir / "annotation" / "exp27e_gpt55_human_adjudication_queue.csv")
    packet_rows = read_jsonl(out_dir / "annotation" / "exp27e_gpt55_human_adjudication_packets.jsonl")
    queue_ids = {row.get("sample_id", "") for row in queue_rows if row.get("sample_id")}
    packet_ids = {str(row.get("sample_id") or "") for row in packet_rows if row.get("sample_id")}
    if queue_ids != packet_ids:
        errors.append(f"queue/packet sample_id mismatch: queue={len(queue_ids)} packet={len(packet_ids)}")
    if len(queue_rows) != len(packet_rows):
        errors.append(f"queue/packet row count mismatch: queue={len(queue_rows)} packet={len(packet_rows)}")
    top40_count = sum(1 for row in queue_rows if str(row.get("top40_for_manual_review", "")).lower() == "true")
    if queue_rows and top40_count != min(40, len(queue_rows)):
        errors.append(f"top40_for_manual_review count should be min(40, queue_size), got {top40_count}")
    for row in queue_rows:
        for forbidden in ["teacher_reason", "audit_reason", "raw_api_path", "response_meta"]:
            if forbidden in row:
                errors.append(f"queue contains forbidden field: {forbidden}")
    for idx, row in enumerate(packet_rows):
        forbidden_keys = FORBIDDEN_PACKET_KEYS & walk_keys(row)
        if forbidden_keys:
            errors.append(f"packet[{idx}] contains forbidden raw teacher keys: {sorted(forbidden_keys)}")
        if row.get("split") != "train":
            errors.append(f"packet[{idx}] split is not train")

    leak_rows = read_csv_rows(out_dir / "tables" / "exp27e_leakage_audit.csv")
    leak = {row.get("check"): int(row.get("count") or 0) for row in leak_rows}
    for key in [
        "packet_not_in_train_sample_count",
        "packet_not_in_train_question_count",
        "dev_sample_overlap",
        "dev_question_overlap",
        "test_sample_overlap",
        "test_question_overlap",
        "test_label_read",
    ]:
        if leak.get(key, 999) != 0:
            errors.append(f"leakage audit failed: {key}={leak.get(key)}")

    dev_ids, dev_qkeys = split_ids(dev_jsonl)
    test_ids, test_qkeys = split_ids(test_jsonl)
    packet_qkeys = {
        str(row.get("compact_teacher_disagreement_summary", {}).get("question_key") or "")
        for row in packet_rows
        if row.get("compact_teacher_disagreement_summary")
    }
    if packet_ids & dev_ids:
        errors.append("adjudication packets overlap dev sample ids")
    if packet_ids & test_ids:
        errors.append("adjudication packets overlap test sample ids")
    if packet_qkeys and packet_qkeys & dev_qkeys:
        errors.append("adjudication packets overlap dev question keys")
    if packet_qkeys and packet_qkeys & test_qkeys:
        errors.append("adjudication packets overlap test question keys")

    schema_path = Path("thesis_exp/exp17_low_score_evidence/schemas/exp27e_conflict_adjudication_schema.json")
    if schema_path.exists():
        try:
            jsonschema.Draft202012Validator.check_schema(read_json(schema_path))
        except jsonschema.SchemaError as exc:
            errors.append(f"adjudication schema invalid: {exc.message}")
    else:
        warnings.append("root adjudication schema file not found")

    return {
        "errors": len(errors),
        "warnings": len(warnings),
        "queue_rows": len(queue_rows),
        "top40_rows": top40_count,
        "packet_rows": len(packet_rows),
        "valid": not errors,
        "error_messages": errors[:50],
        "warning_messages": warnings[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp27E conflict-adjudication outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    args = parser.parse_args()
    summary = validate(args.out_dir, args.dev_jsonl, args.test_jsonl)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
