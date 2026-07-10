"""Ingest and audit the returned Exp27N selective adjudications.

The 54 completed rows remain private model-reviewed silver annotations. This
CPU-only step validates their target-aware schema, verifies evidence against
the blind packets, merges them with the 16 existing Exp27M reviews, and emits
only lightweight aggregate QC artifacts for version control.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import (  # noqa: E402
    bool_value,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


DEFAULT_EXP27I = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
DEFAULT_EXP27M = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27m_model_review_audit_policy_seed42"
)
DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42"
)
EXPECTED_FIELDS = {
    "sample_id",
    "target_scope_confirmed",
    "final_score_range",
    "final_score",
    "failure_bucket",
    "major_failures",
    "rubric_evidence",
    "evaluator_output_evidence",
    "missing_evidence_reason",
    "score_cap",
    "confidence",
    "training_use",
    "review_status",
}
FAILURE_BUCKETS = {
    "no_failure",
    "visible_failure",
    "hidden_or_missing_failure",
    "unclear",
}
CONFIDENCE_LEVELS = {"high", "medium", "low"}
TRAINING_USES = {"resolved_model_silver", "review_only"}

# One returned row paraphrased its supporting quote. The replacement is an
# exact sentence from the same evaluator output and supports the same score.
# No semantic annotation field is changed by this normalization.
EVIDENCE_REPAIRS = {
    "61725aede8ed2f1879dae36d9b4f8a751b64787b": (
        "By examining these five forces, businesses can gain insights into the "
        "strengths and weaknesses of their competitive position within an industry "
        "and make more informed strategic decisions."
    )
}


def require(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def score(value: Any) -> int:
    parsed = int(value)
    if isinstance(value, bool) or not 1 <= parsed <= 5:
        raise ValueError(f"Invalid ordinal score: {value!r}")
    return parsed


def evaluator_target(packet: dict[str, Any]) -> str:
    content = packet["messages"][1]["content"]
    start = "<EVALUATOR_OUTPUT_TO_SCORE>"
    end = "</EVALUATOR_OUTPUT_TO_SCORE>"
    if start not in content or end not in content:
        raise ValueError(f"Packet lacks target markers: {packet['sample_id']}")
    return content.split(start, 1)[1].rsplit(end, 1)[0]


def validate_and_normalize(
    row: dict[str, Any], packet: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    sid = str(row.get("sample_id") or "")
    if set(row) != EXPECTED_FIELDS:
        raise ValueError(f"Unexpected annotation fields for {sid}: {sorted(set(row) ^ EXPECTED_FIELDS)}")
    if row["target_scope_confirmed"] is not True or row["review_status"] != "completed":
        raise ValueError(f"Incomplete target-scope review: {sid}")

    final_range = row["final_score_range"]
    if (
        not isinstance(final_range, list)
        or len(final_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in final_range)
        or not 1 <= final_range[0] <= final_range[1] <= 5
    ):
        raise ValueError(f"Invalid final_score_range for {sid}: {final_range!r}")
    final_score = score(row["final_score"])
    if not final_range[0] <= final_score <= final_range[1]:
        raise ValueError(f"final_score outside range for {sid}")
    if row["failure_bucket"] not in FAILURE_BUCKETS:
        raise ValueError(f"Invalid failure bucket for {sid}: {row['failure_bucket']}")
    failures = row["major_failures"]
    if not isinstance(failures, list) or any(
        not isinstance(value, str) or not value.strip() for value in failures
    ):
        raise ValueError(f"Invalid major_failures for {sid}")
    if row["failure_bucket"] == "no_failure" and failures:
        raise ValueError(f"no_failure row has major_failures: {sid}")
    if not isinstance(row["rubric_evidence"], str) or not row["rubric_evidence"].strip():
        raise ValueError(f"Missing rubric_evidence for {sid}")
    if row["confidence"] not in CONFIDENCE_LEVELS:
        raise ValueError(f"Invalid confidence for {sid}: {row['confidence']}")
    if row["training_use"] not in TRAINING_USES:
        raise ValueError(f"Invalid training_use for {sid}: {row['training_use']}")
    if (
        row["failure_bucket"] == "unclear" or row["confidence"] == "low"
    ) and row["training_use"] != "review_only":
        raise ValueError(f"Uncertain row is not isolated as review_only: {sid}")
    score_cap = row["score_cap"]
    if score_cap is not None:
        score(score_cap)

    normalized = dict(row)
    target = evaluator_target(packet)
    evidence = row["evaluator_output_evidence"]
    repair_status = "exact"
    if evidence is None:
        if not isinstance(row["missing_evidence_reason"], str) or not row[
            "missing_evidence_reason"
        ].strip():
            raise ValueError(f"Null evaluator evidence lacks missing_evidence_reason: {sid}")
        repair_status = "missing_evidence_explained"
    elif not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"Invalid evaluator evidence for {sid}")
    elif evidence not in target:
        repaired = EVIDENCE_REPAIRS.get(sid)
        if not repaired or repaired not in target:
            raise ValueError(f"Evaluator evidence is not an exact source substring: {sid}")
        normalized["evaluator_output_evidence"] = repaired
        repair_status = "exact_substring_repair"
    if evidence is not None and row["missing_evidence_reason"] not in (None, ""):
        raise ValueError(f"Both evaluator evidence and missing reason are populated: {sid}")
    return normalized, repair_status


def quadratic_weighted_kappa(gold: list[int], pred: list[int]) -> float:
    if len(gold) != len(pred) or not gold:
        return math.nan
    n = 5
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    gold_hist = [0.0] * n
    pred_hist = [0.0] * n
    for left, right in zip(gold, pred):
        observed[left - 1][right - 1] += 1.0
        gold_hist[left - 1] += 1.0
        pred_hist[right - 1] += 1.0
    total = float(len(gold))
    numerator = denominator = 0.0
    for left in range(n):
        for right in range(n):
            weight = ((left - right) / (n - 1)) ** 2
            numerator += weight * observed[left][right] / total
            denominator += weight * (gold_hist[left] * pred_hist[right] / total**2)
    return 1.0 - numerator / denominator if denominator else 1.0


def comparison_row(
    population: str,
    source: str,
    rows: list[dict[str, Any]],
    audited: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference = [score(row["final_score"]) for row in rows]
    candidate = [score(audited[row["sample_id"]][source]) for row in rows]
    distances = [abs(left - right) for left, right in zip(reference, candidate)]
    return {
        "population": population,
        "source": source,
        "n": len(rows),
        "MAE_to_model_review_silver": sum(distances) / len(distances),
        "QWK_to_model_review_silver": quadratic_weighted_kappa(reference, candidate),
        "exact_rate": sum(value == 0 for value in distances) / len(distances),
        "within_one_rate": sum(value <= 1 for value in distances) / len(distances),
        "severe_gap_rate": sum(value >= 2 for value in distances) / len(distances),
    }


def ingest(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = args.out_dir / "packets" / "exp27n_selective_adjudication_packets_54.jsonl"
    existing_path = args.exp27m_dir / "private" / "exp27m_model_review_reference_107.jsonl"
    audited_path = args.exp27i_dir / "data" / "exp27i_teacher_audited_361_calibrated_train.jsonl"
    for path, purpose in (
        (args.annotations, "returned Exp27N annotations"),
        (packet_path, "Exp27N blind packet"),
        (existing_path, "existing Exp27M reviews"),
        (audited_path, "361-row teacher audit"),
    ):
        require(path, purpose)

    packets = read_jsonl(packet_path)
    packet_by_id = {str(row["sample_id"]): row for row in packets}
    annotations = read_jsonl(args.annotations)
    if len(annotations) != 54 or len({str(row.get("sample_id")) for row in annotations}) != 54:
        raise ValueError("Returned Exp27N file must contain 54 unique rows")
    if {str(row["sample_id"]) for row in annotations} != set(packet_by_id):
        raise ValueError("Returned Exp27N sample IDs do not exactly match the blind packet")

    normalized: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    for row in annotations:
        clean_row, repair_status = validate_and_normalize(row, packet_by_id[str(row["sample_id"])])
        normalized.append(clean_row)
        repair_counts[repair_status] += 1
    normalized.sort(key=lambda row: row["sample_id"])

    audited_rows = read_jsonl(audited_path)
    audited = {str(row["sample_id"]): row for row in audited_rows}
    if len(audited) != 361:
        raise ValueError("Exp27I audit must contain 361 unique rows")
    required_ids = {
        sid
        for sid, row in audited.items()
        if abs(score(row["qwen_score"]) - score(row["original_human_score"])) >= 2
        or bool_value(row.get("target_issue_flag"))
    }
    existing = [
        row for row in read_jsonl(existing_path) if str(row["sample_id"]) in required_ids
    ]
    if len(existing) != 16:
        raise ValueError(f"Expected 16 existing required reviews, found {len(existing)}")

    consolidated: list[dict[str, Any]] = []
    for row in existing:
        use = (
            "review_only"
            if row["final_bucket"] == "unclear" or row["final_confidence"] == "low"
            else "resolved_model_silver"
        )
        consolidated.append(
            {
                "sample_id": row["sample_id"],
                "final_score_range": row["final_score_range"],
                "final_score": row["final_score"],
                "failure_bucket": row["final_bucket"],
                "confidence": row["final_confidence"],
                "training_use": use,
                "resolution_source": "exp27m_existing_model_review",
                "reference_status": "model_review_silver_not_human_gold",
            }
        )
    for row in normalized:
        consolidated.append(
            {
                **row,
                "resolution_source": "exp27n_single_gpt56pro_selective_adjudication",
                "reference_status": "model_review_silver_not_human_gold",
            }
        )
    consolidated.sort(key=lambda row: row["sample_id"])
    if len(consolidated) != 70 or {row["sample_id"] for row in consolidated} != required_ids:
        raise ValueError("The 16 existing and 54 returned reviews do not cover all 70 required rows")

    private_dir = args.out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = private_dir / "exp27n_gpt56pro_selective_adjudication_54.raw.jsonl"
    normalized_path = private_dir / "exp27n_gpt56pro_selective_adjudication_54.normalized.jsonl"
    consolidated_path = private_dir / "exp27n_adjudication_reference_70.jsonl"
    if args.annotations.resolve() != raw_copy.resolve():
        shutil.copyfile(args.annotations, raw_copy)
    write_jsonl(normalized_path, normalized)
    write_jsonl(consolidated_path, consolidated)

    new_resolved = [row for row in normalized if row["training_use"] == "resolved_model_silver"]
    all_resolved = [row for row in consolidated if row["training_use"] == "resolved_model_silver"]
    all_review_only = [row for row in consolidated if row["training_use"] == "review_only"]
    completion_rows = [
        {"item": "blind_packet_rows", "count": len(packets)},
        {"item": "returned_rows", "count": len(annotations)},
        {"item": "returned_unique_sample_ids", "count": len({row["sample_id"] for row in normalized})},
        {"item": "schema_valid_rows", "count": len(normalized)},
        {"item": "exact_evidence_rows", "count": repair_counts["exact"]},
        {"item": "missing_evidence_explained_rows", "count": repair_counts["missing_evidence_explained"]},
        {"item": "exact_substring_repair_rows", "count": repair_counts["exact_substring_repair"]},
        {"item": "new_resolved_model_silver", "count": len(new_resolved)},
        {"item": "new_review_only", "count": len(normalized) - len(new_resolved)},
        {"item": "existing_required_reviews", "count": len(existing)},
        {"item": "all_required_reviews_completed", "count": len(consolidated)},
        {"item": "all_required_resolved_model_silver", "count": len(all_resolved)},
        {"item": "all_required_review_only", "count": len(all_review_only)},
    ]
    distribution_rows: list[dict[str, Any]] = []
    for dimension in ("final_score", "failure_bucket", "confidence", "training_use"):
        counts = Counter(str(row[dimension]) for row in normalized)
        distribution_rows.extend(
            {"population": "returned_54", "dimension": dimension, "value": key, "count": value}
            for key, value in sorted(counts.items())
        )
    source_rows = []
    for population, rows in (("all_returned", normalized), ("resolved_only", new_resolved)):
        for source in (
            "original_human_score",
            "qwen_score",
            "deepseek_score",
            "calibrated_score",
        ):
            source_rows.append(comparison_row(population, source, rows, audited))
    qc_rows = [
        {"check": "exact_packet_id_alignment", "status": "PASS", "count": 54},
        {"check": "schema_and_cross_field_validation", "status": "PASS", "count": 54},
        {"check": "exact_evidence_or_explained_missing", "status": "PASS", "count": 54},
        {
            "check": "transparent_exact_substring_repairs",
            "status": "PASS",
            "count": repair_counts["exact_substring_repair"],
        },
        {"check": "uncertain_rows_isolated_review_only", "status": "PASS", "count": len(normalized) - len(new_resolved)},
        {"check": "all_required_adjudications_completed", "status": "PASS", "count": len(consolidated)},
        {"check": "dev_labels_read", "status": "PASS", "count": 0},
        {"check": "test_labels_read", "status": "PASS", "count": 0},
        {"check": "teacher_api_calls", "status": "PASS", "count": 0},
        {"check": "gpu_or_training_runs", "status": "PASS", "count": 0},
    ]
    write_csv(args.out_dir / "tables" / "exp27n_adjudication_completion.csv", completion_rows)
    write_csv(args.out_dir / "tables" / "exp27n_adjudication_distribution.csv", distribution_rows)
    write_csv(args.out_dir / "tables" / "exp27n_source_disagreement_to_model_review.csv", source_rows)
    write_csv(args.out_dir / "tables" / "exp27n_adjudication_qc.csv", qc_rows)

    decision = {
        "experiment": "exp27n_selective_model_adjudication_ingest",
        "status": "PASS",
        "returned_rows": len(normalized),
        "new_resolved_model_silver": len(new_resolved),
        "new_review_only": len(normalized) - len(new_resolved),
        "all_required_adjudications_completed": len(consolidated),
        "all_required_resolved_model_silver": len(all_resolved),
        "all_required_review_only": len(all_review_only),
        "evidence_substring_repairs": repair_counts["exact_substring_repair"],
        "model_review_is_silver": True,
        "human_external_validation": False,
        "dev_test_labels_read": False,
        "teacher_api_calls": 0,
        "gpu_required": False,
        "model_training_runs": 0,
        "proceed_to_361_downstream_dataset_construction": True,
        "proceed_to_training": False,
        "proceed_to_full_3326_expansion": False,
        "proceed_to_dev_test_relabeling": False,
        "next_step": "prepare_controlled_361_row_in_place_downstream_pilot_datasets",
    }
    write_json(args.out_dir / "decision" / "exp27n_selective_adjudication_ingest_decision.json", decision)
    report = "\n".join(
        [
            "# Exp27N Selective Model Adjudication Ingest",
            "",
            "The single GPT-5.6Pro session returned all 54 target-aware blind reviews.",
            "These annotations are model-reviewed silver, not human expert gold.",
            "",
            "## Completion and QC",
            "",
            f"- returned/schema-valid rows: {len(normalized)}/54",
            f"- direct exact evidence rows: {repair_counts['exact']}",
            f"- explained missing-evidence rows: {repair_counts['missing_evidence_explained']}",
            f"- exact-substring evidence repairs: {repair_counts['exact_substring_repair']}",
            f"- new resolved model-silver rows: {len(new_resolved)}",
            f"- new review-only rows: {len(normalized) - len(new_resolved)}",
            f"- all required adjudications completed: {len(consolidated)}/70",
            f"- all required resolved model-silver rows: {len(all_resolved)}",
            f"- all required review-only rows: {len(all_review_only)}",
            "- returned failure buckets: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(Counter(row["failure_bucket"] for row in normalized).items())
            ),
            "- returned final scores: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(Counter(row["final_score"] for row in normalized).items())
            ),
            "",
            "One returned evidence field was a faithful paraphrase rather than an exact",
            "source substring. It was replaced with an exact sentence from the same",
            "evaluator output. Its score, range, failure bucket, confidence, and training",
            "use were not changed.",
            "",
            "## Interpretation",
            "",
            "The source-disagreement table compares existing labels and teacher outputs",
            "to the model-review silver result only. It must not be presented as accuracy",
            "against human expert gold. Unclear or low-confidence rows remain review_only",
            "with zero downstream quality weight.",
            "",
            "## Decision",
            "",
            "Exp27N passes its completion and evidence-QC gates. The next permitted step",
            "is construction of the controlled 361-row in-place downstream pilot variants.",
            "This does not authorize model training yet, full-3326 expansion, or dev/test",
            "relabeling.",
            "",
        ]
    )
    write_text(args.out_dir / "reports" / "exp27n_selective_adjudication_ingest_report.md", report)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--exp27i-dir", type=Path, default=DEFAULT_EXP27I)
    parser.add_argument("--exp27m-dir", type=Path, default=DEFAULT_EXP27M)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(ingest(parse_args()), ensure_ascii=False, sort_keys=True))
