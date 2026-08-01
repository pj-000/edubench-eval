"""Fail-closed audit for the crossed-metric feasibility artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.analyze_crossed_metric_conditional_fidelity_feasibility import (
    SCHEMA_VERSION,
    feasibility_report,
    read_jsonl,
    sha256_file,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(
    train_path: Path,
    dev_path: Path,
    protocol_path: Path,
    report_path: Path,
    lock_path: Path,
    source_path: Path,
    test_path: Path,
) -> str:
    protocol, report, lock = _read(protocol_path), _read(report_path), _read(lock_path)
    expected = feasibility_report(read_jsonl(train_path), read_jsonl(dev_path), protocol)
    if report != expected:
        raise ValueError("public report differs from independent reconstruction")
    for field, path in (
        ("train_sha256", train_path),
        ("dev_sha256", dev_path),
        ("protocol_sha256", protocol_path),
        ("analysis_source_sha256", source_path),
        ("test_source_sha256", test_path),
        ("public_report_sha256", report_path),
    ):
        if lock.get(field) != sha256_file(path):
            raise ValueError(f"hash mismatch: {field}")
    if report.get("schema_version") != SCHEMA_VERSION or lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema differs")
    if report.get("decision") != "NO_GO_INSUFFICIENT_CROSSED_METRIC_SUPPORT":
        raise ValueError("formal current-data result must remain NO-GO")
    if report["gate_results"]["dev_metric_pairs_with_at_least_20_shared_qa"] != 0:
        raise ValueError("unexpected eligible dev metric pair")
    if report["gate_results"]["jointly_eligible_metric_pairs"]["observed"] != 0:
        raise ValueError("unexpected jointly eligible metric pair")
    if report["interpretation"]["new_method_authorized"] is not False:
        raise ValueError("new method was authorized")
    for flag in ("new_training", "new_inference", "new_api_calls", "gpu_used", "test_accessed"):
        if report["data_scope"].get(flag) is not False or lock.get(flag) is not False:
            raise ValueError(f"forbidden execution flag active: {flag}")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True).lower()
    for forbidden in ('"record_id"', '"question_key"', '"answer_key"', '"question"', '"answer"'):
        if forbidden in serialized:
            raise ValueError(f"row-level field leaked: {forbidden}")
    return "CROSSED_METRIC_CONDITIONAL_FIDELITY_FEASIBILITY_PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.train, args.dev, args.protocol, args.report, args.lock, args.source, args.test_source))


if __name__ == "__main__":
    main()
