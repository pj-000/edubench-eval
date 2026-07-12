#!/usr/bin/env python3
"""Validate final Exp35A review and qualification artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main() -> None:
    failures = []
    required = [
        OUT / "decision/exp35a_preparation_decision.json",
        OUT / "decision/exp35a_review_decision.json",
        OUT / "decision/exp35a_edudart_qualification_decision.json",
        OUT / "tables/exp35a_qualification_distribution.csv",
        OUT / "tables/exp35a_reviewer_agreement.csv",
        OUT / "tables/exp35a_edudart_qualification_metrics.csv",
        OUT / "reports/exp35a_preparation_report.md",
        OUT / "reports/exp35a_review_report.md",
        OUT / "reports/exp35a_edudart_qualification_report.md",
        OUT / "reports/exp35a_failure_diagnosis.md",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing:{path.relative_to(ROOT)}")
    preparation = load_json(required[0])
    review = load_json(required[1])
    qualification = load_json(required[2])
    if not review.get("review_gate_passed") or review.get("final_reference_rows") != 196:
        failures.append("review_gate_or_reference")
    if qualification.get("qualification_gate_passed"):
        failures.append("unexpected_qualification_pass")
    if qualification.get("generate_exp35b_train_supervision"):
        failures.append("unexpected_train_supervision_permission")
    for payload, name in ((preparation, "preparation"), (review, "review"), (qualification, "qualification")):
        if payload.get("dev_rows_read") != 0 or payload.get("test_access_count") != 0:
            failures.append(f"forbidden_split_access:{name}")
    private_counts = {
        "reviewer_a": line_count(OUT / "private_review/reviewer_filled/exp35a_reviewer_a_results.jsonl"),
        "reviewer_b": line_count(OUT / "private_review/reviewer_filled/exp35a_reviewer_b_results.jsonl"),
        "adjudicator": line_count(OUT / "private_review/adjudication_filled/exp35a_adjudicator_results.jsonl"),
        "silver_reference": len(list(csv.DictReader((OUT / "private/exp35a_model_reviewed_silver_reference.csv").open(encoding="utf-8")))),
    }
    if private_counts != {"reviewer_a": 196, "reviewer_b": 196, "adjudicator": 109, "silver_reference": 196}:
        failures.append(f"private_counts:{private_counts}")
    for path in (
        OUT / "private_review/reviewer_filled/exp35a_reviewer_a_results.jsonl",
        OUT / "private/exp35a_model_reviewed_silver_reference.csv",
        OUT / "private/exp35a_qualification_predictions.csv",
    ):
        result = subprocess.run(["git", "check-ignore", "--quiet", str(path)], cwd=ROOT)
        if result.returncode != 0:
            failures.append(f"not_ignored:{path.relative_to(ROOT)}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks": 7,
        "private_counts": private_counts,
        "qualification_gate_passed": qualification.get("qualification_gate_passed"),
        "exp35b_allowed": qualification.get("generate_exp35b_train_supervision"),
        "dev_rows_read": 0,
        "test_access_count": 0,
        "failures": failures,
    }
    report = OUT / "reports/exp35a_final_validation.md"
    report.write_text(
        "# Exp35A Final Validation\n\n"
        f"- Status: {result['status']}\n- Private counts: {private_counts}\n"
        f"- Qualification gate passed: {result['qualification_gate_passed']}\n"
        f"- Exp35B allowed: {result['exp35b_allowed']}\n"
        f"- Dev/test access: 0/0\n- Failures: {failures or 'none'}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
