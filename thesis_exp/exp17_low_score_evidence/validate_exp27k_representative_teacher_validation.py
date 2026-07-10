"""Validate Exp27K packet completeness, blindness, and optional API outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_json  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27k_representative_teacher_validation_seed42"
)
FORBIDDEN_PACKET_KEYS = {
    "final_score",
    "final_reference_score",
    "reviewer_a_score",
    "reviewer_b_score",
    "adjudicated_score",
    "exp27j_silver_score",
    "calibrated_score",
}
PROVIDERS = ("qwen", "deepseek")
STAGES = ("blind", "audit")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flatten_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(flatten_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(flatten_keys(item))
    return keys


def schema_success(row: dict[str, Any]) -> bool:
    return isinstance(row.get("parsed"), dict) and not row.get("parse_error") and not row.get("schema_errors")


def load_output_rows(out_dir: Path, provider: str, stage: str) -> list[dict[str, Any]]:
    return read_jsonl(out_dir / "annotations" / "parsed" / provider / f"exp27d_{stage}_outputs.jsonl")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    packets_path = out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
    refs_path = out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl"
    coverage_path = out_dir / "tables" / "exp27k_coverage_and_leakage.csv"
    required_paths = [
        packets_path,
        refs_path,
        coverage_path,
        out_dir / "prompts" / "exp27d_blind_teacher_prompt_v4.md",
        out_dir / "prompts" / "exp27d_label_audit_prompt_v4.md",
        out_dir / "schema" / "exp27d_teacher_blind_schema_v4.json",
        out_dir / "schema" / "exp27d_teacher_audit_schema_v4.json",
    ]
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for path in required_paths:
        add(f"exists:{path.name}", path.exists(), str(path))
    if not all(path.exists() for path in required_paths):
        result = {"status": "FAIL", "errors": sum(not row["passed"] for row in checks), "api_complete": False}
        write_csv(out_dir / "tables" / "exp27k_validation_checks.csv", checks)
        write_json(out_dir / "decision" / "exp27k_validation_decision.json", result)
        return result

    packets = read_jsonl(packets_path)
    refs = read_jsonl(refs_path)
    packet_ids = [str(row.get("sample_id")) for row in packets]
    ref_ids = [str(row.get("sample_id")) for row in refs]
    add("packet_rows_equal_reference_rows", len(packets) == len(refs), f"packets={len(packets)} refs={len(refs)}")
    add("packet_ids_unique", len(packet_ids) == len(set(packet_ids)), f"rows={len(packet_ids)}")
    add("reference_ids_unique", len(ref_ids) == len(set(ref_ids)), f"rows={len(ref_ids)}")
    add("packet_reference_id_sets_match", set(packet_ids) == set(ref_ids), f"overlap={len(set(packet_ids) & set(ref_ids))}")

    forbidden_hits = sorted(set().union(*(flatten_keys(row) for row in packets)) & FORBIDDEN_PACKET_KEYS)
    add("no_silver_or_calibrated_fields_in_packets", not forbidden_hits, f"hits={forbidden_hits}")
    add(
        "packets_are_train_only",
        all(row.get("split") == "train" for row in packets),
        f"splits={sorted({str(row.get('split')) for row in packets})}",
    )
    add(
        "target_is_evaluator_output",
        all(row.get("target_to_score") == "evaluator_output_answer_field" for row in packets),
        f"rows={len(packets)}",
    )
    add(
        "audit_refs_contain_only_train_human_label_not_silver",
        all("original_score" in row and not (flatten_keys(row) & FORBIDDEN_PACKET_KEYS) for row in refs),
        f"rows={len(refs)}",
    )

    leakage = {row["check"]: int(row["count"]) for row in read_csv(coverage_path)}
    zero_checks = [
        "missing_rows_in_train",
        "inherited_exp27j_dev_sample_overlap",
        "inherited_exp27j_dev_question_overlap",
        "inherited_exp27j_test_sample_overlap",
        "inherited_exp27j_test_question_overlap",
        "silver_reference_fields_in_teacher_packet",
        "dev_test_files_opened_by_exp27k",
        "dev_label_used",
        "test_label_read",
    ]
    add("all_leakage_guards_zero", all(leakage.get(name) == 0 for name in zero_checks), str({k: leakage.get(k) for k in zero_checks}))
    add(
        "coverage_partition_is_120",
        leakage.get("prior_teacher_covered_rows", 0) + leakage.get("missing_teacher_rows", 0) == 120,
        f"prior={leakage.get('prior_teacher_covered_rows')} missing={leakage.get('missing_teacher_rows')}",
    )

    api_complete = True
    for provider in PROVIDERS:
        for stage in STAGES:
            rows = load_output_rows(out_dir, provider, stage)
            complete = len(rows) == len(packets) and all(schema_success(row) for row in rows)
            if not complete:
                api_complete = False
            add(
                f"api_complete:{provider}:{stage}",
                complete or args.allow_missing_api,
                f"rows={len(rows)}/{len(packets)} schema_ok={sum(schema_success(row) for row in rows)}",
            )

    errors = sum(not row["passed"] for row in checks)
    result = {
        "status": "PASS" if errors == 0 else "FAIL",
        "errors": errors,
        "packet_rows": len(packets),
        "api_complete": api_complete,
        "allow_missing_api": args.allow_missing_api,
        "proceed_to_analysis": api_complete,
        "proceed_to_training": False,
    }
    write_csv(out_dir / "tables" / "exp27k_validation_checks.csv", checks)
    write_json(out_dir / "decision" / "exp27k_validation_decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp27K representative teacher validation artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-missing-api", action="store_true")
    args = parser.parse_args()
    result = validate(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
