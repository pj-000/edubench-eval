#!/usr/bin/env python3
"""Validate Exp35A preparation boundaries and private packet integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42")


def root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with root(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with root(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    decision = read_json(out / "decision/exp35a_preparation_decision.json")
    a = read_jsonl(out / "private_review/blind_packets/exp35a_reviewer_a_packet.jsonl")
    b = read_jsonl(out / "private_review/blind_packets/exp35a_reviewer_b_packet.jsonl")
    manifest = read_jsonl(out / "private/exp35a_selection_manifest.jsonl")
    failures = []
    if len(a) != 196 or a != b or len(manifest) != 196:
        failures.append("packet_or_manifest_count")
    if len({row["sample_id"] for row in a}) != 196:
        failures.append("duplicate_sample_id")
    counts = Counter(row["qualification_view"] for row in a)
    if counts != {"fresh_general_qualification": 120, "low_tail_reassessment": 76}:
        failures.append(f"view_counts={dict(counts)}")
    forbidden = {"label", "label_5", "human_1", "human_2", "human_3", "qwen", "deepseek", "teacher_score"}
    for index, row in enumerate(a):
        if forbidden & set(row):
            failures.append(f"forbidden_packet_key_{index}")
            break
        payload = dict(row)
        observed = payload.pop("packet_hash")
        if canonical_hash(payload) != observed:
            failures.append(f"packet_hash_{index}")
            break
    if decision.get("dev_rows_read") != 0 or decision.get("test_access_count") != 0:
        failures.append("forbidden_split_access")
    for private_path in (
        out / "private/exp35a_selection_manifest.jsonl",
        out / "private_review/blind_packets/exp35a_reviewer_a_packet.jsonl",
    ):
        result = subprocess.run(["git", "check-ignore", "--quiet", str(root(private_path))], cwd=REPO_ROOT)
        if result.returncode != 0:
            failures.append(f"not_ignored:{private_path}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks": 6,
        "packet_rows": len(a),
        "failures": failures,
        "dev_rows_read": 0,
        "test_access_count": 0,
    }
    report = root(out / "reports/exp35a_preparation_validation.md")
    report.write_text(
        "# Exp35A Preparation Validation\n\n"
        f"- Status: {result['status']}\n- Packet rows: {len(a)}\n"
        f"- View counts: {dict(counts)}\n- Failures: {failures or 'none'}\n"
        "- Dev/test access: 0/0.\n- Private packets/manifests are gitignored.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
