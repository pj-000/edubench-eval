"""Independently validate the public residual-risk report and its lock."""

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
    expected_hashes = {
        "public_report_sha256": sha256_file(report_path),
        "analysis_source_sha256": sha256_file(source_path),
        "protocol_sha256": sha256_file(protocol_path),
        "test_source_sha256": sha256_file(test_path),
    }
    for field, expected in expected_hashes.items():
        if lock.get(field) != expected:
            raise ValueError(f"lock mismatch: {field}")
    if report.get("schema_version") != "exp54-residual-risk-decomposition-v1":
        raise ValueError("report schema differs")
    gates = report.get("gate_results")
    if set(gates or {}) != {"A_EVALUATION_UNIT", "B_BOUNDARY_SUPPORT", "C_RELATIVE_TO_ABSOLUTE"}:
        raise ValueError("direction gates differ")
    passed = sorted(name for name, value in gates.items() if value)
    if sorted(report.get("passed_primary_gates", [])) != passed:
        raise ValueError("passed gate list differs")
    expected_direction = passed[0] if len(passed) == 1 else "D_STOP_METHOD_EXPANSION"
    if report.get("direction") != expected_direction or lock.get("direction") != expected_direction:
        raise ValueError("direction decision differs")
    scope = report.get("data_scope", {})
    for flag in ("test_accessed", "new_training", "new_api_calls", "gpu_used"):
        if scope.get(flag) is not False or lock.get(flag) is not False:
            raise ValueError(f"forbidden execution flag active: {flag}")
    if scope.get("train_rows") != 2654 or scope.get("dev_rows") != 664:
        raise ValueError("locked row counts differ")
    if lock.get("bootstrap_replicates") != 10000:
        raise ValueError("formal bootstrap budget differs")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True).lower()
    for forbidden in (
        '"record_id"',
        '"question_key"',
        '"answer_key"',
        '"raw_output"',
        '"rationale"',
        '"canonical_score_option_probabilities"',
    ):
        if forbidden in serialized:
            raise ValueError(f"row-level field leaked into public report: {forbidden}")
    if lock.get("row_level_artifacts_published") is not False:
        raise ValueError("row-level publication flag differs")
    return "RESIDUAL_RISK_DECOMPOSITION_PASS"


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
