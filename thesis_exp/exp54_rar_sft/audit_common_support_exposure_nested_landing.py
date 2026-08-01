"""Validate the public common-support/nested-landing patch artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(report_path: Path, lock_path: Path, source_path: Path, protocol_path: Path, test_path: Path) -> str:
    report, lock = _read(report_path), _read(lock_path)
    for field, path in (
        ("public_report_sha256", report_path),
        ("analysis_source_sha256", source_path),
        ("protocol_sha256", protocol_path),
        ("test_source_sha256", test_path),
    ):
        if lock.get(field) != sha256_file(path):
            raise ValueError(f"hash mismatch: {field}")
    if report.get("schema_version") != "exp54-common-support-exposure-nested-landing-patch-v1":
        raise ValueError("report schema differs")
    if report.get("scientific_direction_label") != "D_NO_METHOD_DIRECTION_IDENTIFIED_ON_CURRENT_DATA":
        raise ValueError("scientific direction label differs")
    if report.get("decision") != "WRITE_THESIS_NO_NEW_METHOD_ON_CURRENT_DATA":
        raise ValueError("patch decision differs")
    scope = report.get("data_scope", {})
    for flag in ("test_accessed", "new_training", "new_api_calls", "gpu_used"):
        if scope.get(flag) is not False or lock.get(flag) is not False:
            raise ValueError(f"forbidden execution flag active: {flag}")
    inventory = report.get("inventory", {})
    if inventory.get("E0_identifiable") is not False or inventory.get("low_score_exposure_identifiable") is not False:
        raise ValueError("non-identifiable exposure was promoted")
    if report.get("decision_evidence", {}).get("new_method_authorized") is not False:
        raise ValueError("new method was authorized")
    for seed in ("42", "43", "44"):
        transport = report["per_seed"][seed]["nested_probability_transport"]
        if transport["cumulative_supported"]["identified"] is not False:
            raise ValueError("empty cumulative population promoted")
        if transport["direct_adjacent_supported"]["identified"] is not False:
            raise ValueError("empty adjacent population promoted")
        if abs(float(transport["all_low"]["probability_conservation_residual"])) > 1e-10:
            raise ValueError("probability transport does not conserve mass")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True).lower()
    for forbidden in ('"record_id"', '"question_key"', '"answer_key"', '"raw_output"', '"rationale"'):
        if forbidden in serialized:
            raise ValueError(f"row-level field leaked: {forbidden}")
    return "COMMON_SUPPORT_EXPOSURE_NESTED_LANDING_PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.report, args.lock, args.source, args.protocol, args.test_source))


if __name__ == "__main__":
    main()
