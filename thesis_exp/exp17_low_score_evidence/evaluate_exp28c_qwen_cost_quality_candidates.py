"""Evaluate a cost-aware Qwen candidate against qualified Max-thinking."""

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

from thesis_exp.exp17_low_score_evidence.collect_exp28c_teacher_protocol_results import (
    metric_row,
    qwk,
    read_jsonl,
    structure_row,
)


REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28b_paper_train_annotation_inventory_seed42/"
    "private/exp28b_benchmark_reference_2654.jsonl"
)
MAX_OUTPUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/"
    "private/qwen/p0_holistic_zero_shot/sealed_qualification.jsonl"
)
PLUS_OUTPUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_qwen_plus_nonthinking_qualification_seed42/"
    "private/qwen/p0_holistic_zero_shot/sealed_qualification.jsonl"
)
PLUS_FIRST_PASS_SUMMARY = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_qwen_plus_nonthinking_qualification_seed42/"
    "decision/exp28c_qwen_plus_first_pass_summary.json"
)
OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_qwen_plus_nonthinking_qualification_seed42"
)

PRICE_CNY_PER_MILLION = {
    "qwen3.7-max": {"input": 12.0, "output": 36.0},
    "qwen3.7-plus": {"input": 2.4, "output": 9.6},
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def valid_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in rows
        if isinstance(row.get("annotation"), dict) and not row.get("schema_errors")
    }


def usage_summary(
    label: str, pricing_model: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    input_tokens = output_tokens = 0
    for row in rows:
        usage = (row.get("response_meta") or {}).get("usage") or {}
        input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    prices = PRICE_CNY_PER_MILLION[pricing_model]
    estimated_cost = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
    return {
        "model": label,
        "pricing_model": pricing_model,
        "rows": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_cny": estimated_cost,
        "estimated_cost_per_row_cny": estimated_cost / len(rows) if rows else "",
        "projected_2654_cost_cny": estimated_cost / len(rows) * 2654 if rows else "",
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.reference)
        if row.get("protocol_subset") == "sealed_qualification"
    }
    max_rows = read_jsonl(args.max_output)
    candidate_rows = read_jsonl(args.candidate_output)
    max_valid = valid_by_id(max_rows)
    candidate_valid = valid_by_id(candidate_rows)
    if len(references) != 120:
        raise ValueError(f"Expected 120 sealed qualification references, got {len(references)}")

    metrics = []
    structures = []
    for model, rows, valid in (
        ("qwen3.7-max-thinking", max_rows, max_valid),
        (args.candidate_label, candidate_rows, candidate_valid),
    ):
        aligned = [(reference, valid[sid]) for sid, reference in references.items() if sid in valid]
        metric = metric_row(model, "p0_holistic_zero_shot", "sealed_qualification", aligned)
        metric["model"] = model
        metrics.append(metric)
        structure = structure_row(model, "p0_holistic_zero_shot", "sealed_qualification", rows, 120)
        structure["model"] = model
        structures.append(structure)

    common = sorted(set(max_valid) & set(candidate_valid))
    max_scores = [int(max_valid[sid]["annotation"]["score"]) for sid in common]
    candidate_scores = [int(candidate_valid[sid]["annotation"]["score"]) for sid in common]
    agreement = {
        "n": len(common),
        "exact_agreement": sum(a == b for a, b in zip(max_scores, candidate_scores)) / len(common) if common else 0.0,
        "within1_agreement": sum(abs(a - b) <= 1 for a, b in zip(max_scores, candidate_scores)) / len(common) if common else 0.0,
        "score_MAE": sum(abs(a - b) for a, b in zip(max_scores, candidate_scores)) / len(common) if common else "",
        "score_QWK": qwk(max_scores, candidate_scores) if common else "",
    }
    costs = [
        usage_summary("qwen3.7-max-thinking", "qwen3.7-max", max_rows),
        usage_summary(args.candidate_label, args.candidate_model, candidate_rows),
    ]
    first_pass = json.loads(args.candidate_first_pass_summary.read_text(encoding="utf-8"))
    native_format_success_rate = int(first_pass["passed"]) / int(first_pass["packet_rows"])
    metric_by_model = {row["model"]: row for row in metrics}
    structure_by_model = {row["model"]: row for row in structures}
    baseline = metric_by_model["qwen3.7-max-thinking"]
    candidate = metric_by_model[args.candidate_label]
    checks = {
        "complete_120": int(candidate["n"]) == 120,
        "native_format_success_at_least_0p85": native_format_success_rate >= 0.85,
        "parse_at_least_0p98": float(structure_by_model[args.candidate_label]["parse_schema_success_rate"]) >= 0.98,
        "unanimous_mae_within_0p10": float(candidate["MAE_unanimous"]) <= float(baseline["MAE_unanimous"]) + 0.10,
        "qwk_within_0p05": float(candidate["QWK"]) >= float(baseline["QWK"]) - 0.05,
        "low_to_high_within_0p05": float(candidate["low_to_high_rate"] or 0.0) <= float(baseline["low_to_high_rate"] or 0.0) + 0.05,
        "high_to_low_within_0p03": float(candidate["high_to_low_rate"] or 0.0) <= float(baseline["high_to_low_rate"] or 0.0) + 0.03,
        "human_range_within_0p05": float(candidate["within_human_score_range_rate"]) >= float(baseline["within_human_score_range_rate"]) - 0.05,
        "max_exact_agreement_at_least_0p70": agreement["exact_agreement"] >= 0.70,
        "max_within1_agreement_at_least_0p90": agreement["within1_agreement"] >= 0.90,
    }
    passed = all(checks.values())
    decision = {
        "status": "USE_QWEN_CANDIDATE_PRIMARY" if passed else "KEEP_QWEN3_7_MAX_THINKING_PRIMARY",
        "candidate": args.candidate_label,
        "selected_primary_teacher": args.candidate_label if passed else "qwen3.7-max-thinking",
        "checks": checks,
        "agreement": agreement,
        "native_first_pass_format": {
            "passed": int(first_pass["passed"]),
            "failed": int(first_pass["failed"]),
            "success_rate": native_format_success_rate,
        },
        "quality_selection_precedes_cost_selection": True,
        "paper_dev_or_test_read": False,
        "reference_scope": "paper_train_only_sealed_qualification",
        "reason": (
            "The candidate passed all locked quality and agreement guards; use it as the cost-efficient primary teacher."
            if passed
            else "The candidate failed at least one locked quality or agreement guard; retain Max-thinking as primary teacher."
        ),
    }

    write_csv(args.out_dir / "tables" / "exp28c_qwen_candidate_quality_comparison.csv", metrics)
    write_csv(args.out_dir / "tables" / "exp28c_qwen_candidate_structure_comparison.csv", structures)
    write_csv(args.out_dir / "tables" / "exp28c_qwen_candidate_cost_comparison.csv", costs)
    write_csv(args.out_dir / "tables" / "exp28c_qwen_candidate_max_agreement.csv", [agreement])
    decision_path = args.out_dir / "decision" / "exp28c_qwen_candidate_qualification_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Exp28C Cost-Aware Qwen Candidate Qualification",
        "",
        "- scope: paper-train sealed qualification only (120 rows)",
        "- paper dev/test read: no",
        f"- decision: **{decision['status']}**",
        f"- selected primary teacher: `{decision['selected_primary_teacher']}`",
        "",
        "Quality guards were locked before collecting the candidate results. Cost is considered only",
        "after all quality and Max-agreement guards pass. Teacher outputs remain model annotations,",
        "not human review.",
    ]
    report_path = args.out_dir / "reports" / "exp28c_qwen_candidate_qualification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--max-output", type=Path, default=MAX_OUTPUT)
    parser.add_argument("--candidate-output", type=Path, default=PLUS_OUTPUT)
    parser.add_argument(
        "--candidate-first-pass-summary", type=Path, default=PLUS_FIRST_PASS_SUMMARY
    )
    parser.add_argument(
        "--candidate-model", choices=sorted(PRICE_CNY_PER_MILLION), default="qwen3.7-plus"
    )
    parser.add_argument("--candidate-label", default="qwen3.7-plus-nonthinking")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(collect(parse_args()), ensure_ascii=False, sort_keys=True))
