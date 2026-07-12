#!/usr/bin/env python3
"""Analyze Exp33A review completion and staged source-aware adjudication.

The script is safe to run before reviews: it records NOT_STARTED and never
fabricates reviewer output. Once both blind review files are complete, it
freezes a private source-comparison/adjudication packet, calculates agreement,
validates correction/fallback decisions, and updates aggregate public tables.
It never opens paper test, calls an API, trains, infers, or uses a GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_expert_reference.build_exp33a_private_source_reference import (  # noqa: E402
    read_jsonl,
    write_csv,
    write_jsonl,
)


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42"
)
REVIEW_SCHEMA = Path(
    "thesis_exp/exp33_expert_reference/schemas/exp33a_blind_review_schema.json"
)
ADJUDICATION_SCHEMA = Path(
    "thesis_exp/exp33_expert_reference/schemas/exp33a_adjudication_schema.json"
)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() == "test.jsonl":
        raise PermissionError("Exp33A forbids access to the sealed paper test split")
    return absolute


def read_json(path: Path) -> dict[str, Any]:
    with guarded(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def evaluator_content(packet: dict[str, Any]) -> str:
    text = str(packet.get("evaluator_output") or "")
    prefix = "<EVALUATOR_OUTPUT_TO_SCORE>"
    suffix = "</EVALUATOR_OUTPUT_TO_SCORE>"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return ""
    return text[len(prefix) : -len(suffix)].strip()


def load_jsonl_dir(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    directory = guarded(path)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not directory.exists():
        return rows, errors
    for file_path in sorted(directory.glob("*.jsonl")):
        try:
            rows.extend(read_jsonl(file_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{file_path.name}: {exc}")
    return rows, errors


def schema_errors(row: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [error.message for error in sorted(validator.iter_errors(row), key=lambda item: list(item.path))]


def review_custom_errors(row: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    score_range = row.get("score_range")
    if isinstance(score_range, list) and len(score_range) == 2:
        lower, upper = score_range
        if isinstance(lower, int) and isinstance(upper, int):
            if lower > upper:
                errors.append("score_range lower exceeds upper")
            plausible = row.get("most_plausible_score")
            if isinstance(plausible, int) and not lower <= plausible <= upper:
                errors.append("most_plausible_score outside score_range")
    evidence = row.get("evaluator_output_evidence")
    if evidence is not None and norm(evidence) not in norm(evaluator_content(packet)):
        errors.append("evaluator_output_evidence is not a normalized substring")
    if row.get("failure_bucket") == "no_failure" and row.get("major_failures"):
        errors.append("no_failure must have an empty major_failures list")
    if row.get("failure_bucket") == "visible_failure" and evidence is None:
        errors.append("visible_failure requires evaluator_output_evidence")
    if row.get("failure_bucket") != "no_failure" and not row.get("major_failures"):
        errors.append("a failure bucket requires at least one major failure")
    if row.get("target_scope_confirmed") is False and not row.get("needs_adjudication"):
        errors.append("unconfirmed target scope requires adjudication")
    if str(row.get("sample_id")) != str(packet.get("sample_id")):
        errors.append("review sample_id differs from packet")
    return errors


def adjudication_custom_errors(
    row: dict[str, Any],
    packet: dict[str, Any],
    a_hash: str,
    b_hash: str,
    source_hash: str,
    source: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    posterior = row.get("final_score_posterior")
    if isinstance(posterior, dict):
        total = sum(float(posterior.get(str(score), 0.0)) for score in range(1, 6))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            errors.append(f"final_score_posterior sums to {total}, expected 1")
    score_range = row.get("final_score_range")
    plausible = row.get("final_most_plausible_score")
    if isinstance(score_range, list) and len(score_range) == 2:
        if score_range[0] > score_range[1]:
            errors.append("final_score_range lower exceeds upper")
        if plausible is not None and not score_range[0] <= plausible <= score_range[1]:
            errors.append("final_most_plausible_score outside final_score_range")
    evidence = row.get("evaluator_output_evidence")
    if evidence is not None and norm(evidence) not in norm(evaluator_content(packet)):
        errors.append("adjudicator evidence is not a normalized substring")
    if row.get("reviewer_a_result_hash") != a_hash:
        errors.append("reviewer_a_result_hash mismatch")
    if row.get("reviewer_b_result_hash") != b_hash:
        errors.append("reviewer_b_result_hash mismatch")
    if row.get("source_comparison_frozen_hash") != source_hash:
        errors.append("source_comparison_frozen_hash mismatch")
    required_provenance = {"human_1", "human_2", "human_3", "rounded_human"}
    if not required_provenance <= set(row.get("source_provenance_seen") or []):
        errors.append("required human source provenance was not acknowledged")
    if row.get("final_status") == "human_empirical_distribution_fallback":
        human_scores = [
            int(source[key])
            for key in ("human_1", "human_2", "human_3")
            if source.get(key) is not None
        ]
        if not human_scores:
            errors.append("human empirical fallback has no human scores")
        else:
            expected = {str(score): human_scores.count(score) / len(human_scores) for score in range(1, 6)}
            if any(not math.isclose(float(posterior.get(str(score), -1)), expected[str(score)], rel_tol=0.0, abs_tol=1e-6) for score in range(1, 6)):
                errors.append("fallback posterior does not equal human empirical distribution")
            if row.get("final_score_range") != [min(human_scores), max(human_scores)]:
                errors.append("fallback score range does not match available human scores")
            if row.get("final_most_plausible_score") is not None:
                errors.append("fallback cannot force a most_plausible hard score")
    if row.get("confidence") == "low" and row.get("final_status") == "model_reviewed_silver":
        errors.append("low-confidence adjudication must fall back or remain unresolved")
    return errors


def validate_role_rows(
    rows: list[dict[str, Any]],
    role: str,
    schema: dict[str, Any],
    packets: dict[str, dict[str, Any]],
    locked_type: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    valid: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id") or "")
        row_errors = schema_errors(row, schema)
        if row.get("reviewer_role") != role:
            row_errors.append(f"reviewer_role must be {role}")
        if row.get("reviewer_type") != locked_type:
            row_errors.append(f"reviewer_type must match lock {locked_type}")
        packet = packets.get(sid)
        if packet is None:
            row_errors.append("sample_id not assigned")
        else:
            row_errors.extend(review_custom_errors(row, packet))
        if sid in valid:
            row_errors.append("duplicate sample_id")
        if row_errors:
            errors.append(f"{role} row {index} {sid or '<missing>'}: {'; '.join(row_errors)}")
        else:
            valid[sid] = row
    return valid, errors


def adjudication_reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if a["most_plausible_score"] != b["most_plausible_score"]:
        reasons.append("most_plausible_score_differs")
    if max(a["score_range"][0], b["score_range"][0]) > min(a["score_range"][1], b["score_range"][1]):
        reasons.append("score_ranges_disjoint")
    if a["failure_bucket"] != b["failure_bucket"]:
        reasons.append("failure_bucket_differs")
    if a["student_input_sufficiency"] != b["student_input_sufficiency"]:
        reasons.append("student_input_sufficiency_differs")
    if "low" in (a["confidence"], b["confidence"]):
        reasons.append("any_low_confidence")
    if a["needs_adjudication"] or b["needs_adjudication"]:
        reasons.append("any_needs_adjudication")
    if a["domain_escalation_required"] or b["domain_escalation_required"]:
        reasons.append("any_domain_escalation_required")
    return reasons


def source_projection(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "human_1": {"score": source.get("human_1"), "reason": source.get("human_reason_1")},
        "human_2": {"score": source.get("human_2"), "reason": source.get("human_reason_2")},
        "human_3": {"score": source.get("human_3"), "reason": source.get("human_reason_3")},
        "rounded_human": {"score": source.get("rounded_human_label")},
        "qwen": {
            "score": source.get("qwen_score"), "score_range": source.get("qwen_score_range"),
            "confidence": source.get("qwen_confidence"), "evidence_flags": source.get("qwen_evidence_flags"),
            "reason": source.get("qwen_reason"),
        },
        "deepseek": {
            "score": source.get("deepseek_score"), "score_range": source.get("deepseek_score_range"),
            "confidence": source.get("deepseek_confidence"), "evidence_flags": source.get("deepseek_evidence_flags"),
            "reason": source.get("deepseek_reason"),
        },
    }


def make_adjudication_bundles(
    triggers: dict[str, list[str]],
    packets: dict[str, dict[str, Any]],
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    bundles: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for sid in sorted(triggers):
        payload = {
            "sample_id": sid,
            "blind_packet": packets[sid],
            "reviewer_a": a_rows[sid],
            "reviewer_a_result_hash": canonical_hash(a_rows[sid]),
            "reviewer_b": b_rows[sid],
            "reviewer_b_result_hash": canonical_hash(b_rows[sid]),
            "adjudication_triggers": triggers[sid],
            "source_provenance": source_projection(sources[sid]),
            "always_forbidden": [
                "student_predictions", "b0_b4_variants", "train_dev_model_metrics",
                "sampling_risk_reason", "test_data",
            ],
        }
        payload["source_comparison_frozen_hash"] = canonical_hash(payload)
        hashes[sid] = payload["source_comparison_frozen_hash"]
        bundles.append(payload)
    return bundles, hashes


def validate_adjudications(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    triggers: dict[str, list[str]],
    packets: dict[str, dict[str, Any]],
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
    sources: dict[str, dict[str, Any]],
    locked_type: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    valid: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        sid = str(row.get("sample_id") or "")
        row_errors = schema_errors(row, schema)
        if row.get("reviewer_type") != locked_type:
            row_errors.append(f"reviewer_type must match lock {locked_type}")
        if sid not in triggers:
            row_errors.append("sample_id is not in triggered adjudication set")
        elif sid in a_rows and sid in b_rows:
            row_errors.extend(
                adjudication_custom_errors(
                    row,
                    packets[sid],
                    canonical_hash(a_rows[sid]),
                    canonical_hash(b_rows[sid]),
                    source_hashes[sid],
                    sources[sid],
                )
            )
        if sid in valid:
            row_errors.append("duplicate sample_id")
        if row_errors:
            errors.append(f"adjudicator row {index} {sid or '<missing>'}: {'; '.join(row_errors)}")
        else:
            valid[sid] = row
    return valid, errors


def quadratic_weighted_kappa(
    left: list[int], right: list[int], weights: list[float] | None = None
) -> float | None:
    if not left or len(left) != len(right):
        return None
    item_weights = weights or [1.0] * len(left)
    total = sum(item_weights)
    if total <= 0:
        return None
    observed = [[0.0] * 5 for _ in range(5)]
    left_hist = [0.0] * 5
    right_hist = [0.0] * 5
    for a, b, weight in zip(left, right, item_weights, strict=True):
        observed[a - 1][b - 1] += weight
        left_hist[a - 1] += weight
        right_hist[b - 1] += weight
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(5):
        for j in range(5):
            distance = ((i - j) / 4) ** 2
            observed_disagreement += distance * observed[i][j] / total
            expected_disagreement += distance * (left_hist[i] * right_hist[j] / (total * total))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def krippendorff_ordinal_alpha(left: list[int], right: list[int]) -> float | None:
    if not left or len(left) != len(right):
        return None
    observed = sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)
    pooled = left + right
    if len(pooled) < 2:
        return None
    expected = sum((a - b) ** 2 for index, a in enumerate(pooled) for b in pooled[index + 1 :])
    expected /= len(pooled) * (len(pooled) - 1) / 2
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def agreement_record(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    trigger_ids: set[str],
) -> dict[str, Any]:
    left = [int(a["most_plausible_score"]) for _, a, _ in pairs]
    right = [int(b["most_plausible_score"]) for _, _, b in pairs]
    count = len(pairs)
    return {
        "paired_rows": count,
        "exact_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / count if count else "",
        "within_one": sum(abs(a - b) <= 1 for a, b in zip(left, right, strict=True)) / count if count else "",
        "quadratic_weighted_kappa": quadratic_weighted_kappa(left, right) if count else "",
        "krippendorff_ordinal_alpha": krippendorff_ordinal_alpha(left, right) if count else "",
        "score_range_overlap": (
            sum(max(a["score_range"][0], b["score_range"][0]) <= min(a["score_range"][1], b["score_range"][1]) for _, a, b in pairs) / count
            if count else ""
        ),
        "adjudication_rate": sum(sid in trigger_ids for sid, _, _ in pairs) / count if count else "",
        "status": "COMPLETE" if count else "NOT_STARTED",
    }


def group_value(source: dict[str, Any], group_type: str) -> str:
    if group_type == "overall":
        return "all"
    if group_type == "label_region":
        label = int(source["rounded_human_label"])
        return "low_1_2" if label <= 2 else "mid_3" if label == 3 else "high_4_5"
    mapping = {
        "view": "view",
        "language": "language",
        "metric_family": "metric_family",
        "metric": "metric",
        "subject": "subject",
    }
    return str(source.get(mapping[group_type]) or "unknown")


def agreement_table(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    trigger_ids: set[str],
) -> list[dict[str, Any]]:
    paired_ids = sorted(set(a_rows) & set(b_rows) & set(sources))
    output: list[dict[str, Any]] = []
    for group_type in ("overall", "view", "language", "metric_family", "metric", "subject", "label_region"):
        values = sorted({group_value(sources[sid], group_type) for sid in paired_ids}) or (["all"] if group_type == "overall" else [])
        for value in values:
            pairs = [
                (sid, a_rows[sid], b_rows[sid])
                for sid in paired_ids
                if group_value(sources[sid], group_type) == value
            ]
            output.append({"group_type": group_type, "group_value": value, **agreement_record(pairs, trigger_ids)})
    if not output:
        output.append({"group_type": "overall", "group_value": "all", **agreement_record([], set())})
    return output


def domain_table(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    adjudications: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    paired_ids = sorted(set(a_rows) & set(b_rows) & set(sources))
    output: list[dict[str, Any]] = []
    for group_type in ("overall", "view", "language", "metric_family"):
        values = sorted({group_value(sources[sid], group_type) for sid in paired_ids}) or (["all"] if group_type == "overall" else [])
        for value in values:
            ids = [sid for sid in paired_ids if group_value(sources[sid], group_type) == value]
            escalated = [sid for sid in ids if a_rows[sid]["domain_escalation_required"] or b_rows[sid]["domain_escalation_required"]]
            unresolved = [sid for sid in escalated if adjudications.get(sid, {}).get("final_status") == "unresolved_domain_case"]
            output.append(
                {
                    "group_type": group_type,
                    "group_value": value,
                    "reviewed_rows": len(ids),
                    "domain_escalation_required": len(escalated),
                    "adjudicated": sum(sid in adjudications for sid in escalated),
                    "unresolved_domain_cases": len(unresolved),
                    "status": "COMPLETE" if ids else "NOT_STARTED",
                }
            )
    if not output:
        output.append(
            {
                "group_type": "overall", "group_value": "all", "reviewed_rows": 0,
                "domain_escalation_required": 0, "adjudicated": 0,
                "unresolved_domain_cases": 0, "status": "NOT_STARTED",
            }
        )
    return output


def completion_row(
    role: str,
    reviewer_type: str,
    expected: int,
    raw_count: int,
    valid: dict[str, dict[str, Any]],
    parse_errors: list[str],
    validation_errors: list[str],
) -> dict[str, Any]:
    values = list(valid.values())
    providers = sorted({str(row.get("reviewer_provider")) for row in values if row.get("reviewer_provider")})
    models = sorted({str(row.get("reviewer_model_id")) for row in values if row.get("reviewer_model_id")})
    run_ids = sorted({str(row.get("reviewer_run_id")) for row in values if row.get("reviewer_run_id")})
    if not raw_count and not parse_errors:
        status = "NOT_STARTED"
    elif len(valid) == expected and not parse_errors and not validation_errors:
        status = "COMPLETE"
    else:
        status = "INCOMPLETE_OR_INVALID"
    provenance = "independent_model_reviewer" if reviewer_type == "model" else "independent_human_reviewer"
    if role == "adjudicator":
        provenance = provenance.replace("reviewer", "adjudicator")
    return {
        "reviewer_role": role,
        "reviewer_type": reviewer_type,
        "reviewer_provenance": provenance,
        "reviewer_provider": ";".join(providers),
        "reviewer_model_id": ";".join(models),
        "reviewer_run_id_count": len(run_ids),
        "expected_rows": expected,
        "completed_rows": raw_count,
        "valid_rows": len(valid),
        "invalid_rows": max(0, raw_count - len(valid)) + len(parse_errors),
        "status": status,
        "reference_status": "independent_model_reviewed_silver_reference" if reviewer_type == "model" else "independent_human_review",
    }


def direction_diagnostics(
    sources: dict[str, dict[str, Any]],
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    paired = set(a_rows) & set(b_rows) & set(sources)
    checks: dict[str, Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool]] = {
        "rounded_human_4_to_teacher_5": lambda source, _a, _b: int(source["rounded_human_label"]) == 4 and 5 in (source.get("qwen_score"), source.get("deepseek_score")),
        "source_low_to_high": lambda source, _a, _b: int(source["rounded_human_label"]) <= 2 and any(score is not None and int(score) >= 4 for score in (source.get("qwen_score"), source.get("deepseek_score"))),
        "source_high_to_low": lambda source, _a, _b: int(source["rounded_human_label"]) >= 4 and any(score is not None and int(score) <= 2 for score in (source.get("qwen_score"), source.get("deepseek_score"))),
        "reason_score_inconsistency_proxy": lambda source, _a, _b: bool(source.get("teacher_evidence_flags")),
        "hard_relabel_drift_ge_2": lambda source, a, b: any(
            score is not None
            and abs(int(score) - ((int(a["most_plausible_score"]) + int(b["most_plausible_score"])) / 2)) >= 2
            for score in (source.get("qwen_score"), source.get("deepseek_score"))
        ),
        "blind_pair_disagrees_with_rounded_human": lambda source, a, b: a["most_plausible_score"] != int(source["rounded_human_label"]) or b["most_plausible_score"] != int(source["rounded_human_label"]),
    }
    return [
        {
            "diagnostic": name,
            "eligible_rows": len(paired),
            "flagged_rows": sum(check(sources[sid], a_rows[sid], b_rows[sid]) for sid in paired),
            "status": "COMPLETE" if paired else "NOT_STARTED",
        }
        for name, check in checks.items()
    ]


def final_reference_rows(
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    triggers: dict[str, list[str]],
    adjudications: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sid in sorted(set(a_rows) & set(b_rows)):
        if sid not in triggers:
            score = int(a_rows[sid]["most_plausible_score"])
            output[sid] = {
                "point_score": score,
                "posterior": {str(value): float(value == score) for value in range(1, 6)},
                "failure_bucket": a_rows[sid]["failure_bucket"],
                "student_input_sufficiency": a_rows[sid]["student_input_sufficiency"],
                "evaluator_output_evidence": a_rows[sid]["evaluator_output_evidence"],
                "status": "blind_pair_consensus",
            }
            continue
        adjudication = adjudications.get(sid)
        if not adjudication or adjudication["final_status"] == "unresolved_domain_case":
            continue
        posterior = adjudication["final_score_posterior"]
        point = adjudication.get("final_most_plausible_score")
        if point is None:
            point = max(range(1, 6), key=lambda score: (float(posterior[str(score)]), -score))
        output[sid] = {
            "point_score": int(point),
            "posterior": posterior,
            "failure_bucket": adjudication["final_failure_bucket"],
            "student_input_sufficiency": adjudication["student_input_sufficiency"],
            "evaluator_output_evidence": adjudication["evaluator_output_evidence"],
            "status": adjudication["final_status"],
        }
    return output


def source_metric_record(
    name: str,
    predictions: list[float],
    references: list[int],
    weights: list[float],
) -> dict[str, Any]:
    if not predictions:
        return {"source": name, "rows": 0, "status": "NOT_AVAILABLE"}
    total_weight = sum(weights)
    rounded = [min(5, max(1, int(round(value)))) for value in predictions]
    diffs = [prediction - reference for prediction, reference in zip(predictions, references, strict=True)]
    return {
        "source": name,
        "rows": len(predictions),
        "mae": sum(weight * abs(diff) for weight, diff in zip(weights, diffs, strict=True)) / total_weight,
        "qwk": quadratic_weighted_kappa(rounded, references, weights),
        "exact": sum(weight * (prediction == reference) for weight, prediction, reference in zip(weights, rounded, references, strict=True)) / total_weight,
        "within_one": sum(weight * (abs(prediction - reference) <= 1) for weight, prediction, reference in zip(weights, rounded, references, strict=True)) / total_weight,
        "signed_bias": sum(weight * diff for weight, diff in zip(weights, diffs, strict=True)) / total_weight,
        "severe_error": sum(weight * (abs(diff) >= 2) for weight, diff in zip(weights, diffs, strict=True)) / total_weight,
        "low_to_high": sum(weight * (reference <= 2 and prediction >= 4) for weight, prediction, reference in zip(weights, rounded, references, strict=True)) / total_weight,
        "high_to_low": sum(weight * (reference >= 4 and prediction <= 2) for weight, prediction, reference in zip(weights, rounded, references, strict=True)) / total_weight,
        "label1_recall": recall_for_label(rounded, references, weights, 1),
        "label2_recall": recall_for_label(rounded, references, weights, 2),
        "label5_recall": recall_for_label(rounded, references, weights, 5),
        "status": "COMPLETE",
    }


def recall_for_label(predictions: list[int], references: list[int], weights: list[float], label: int) -> float | str:
    denominator = sum(weight for weight, reference in zip(weights, references, strict=True) if reference == label)
    if denominator == 0:
        return ""
    numerator = sum(
        weight
        for weight, prediction, reference in zip(weights, predictions, references, strict=True)
        if reference == label and prediction == label
    )
    return numerator / denominator


def available_teacher_values(source: dict[str, Any]) -> list[float]:
    return sorted(
        float(value)
        for value in (source.get("qwen_score"), source.get("deepseek_score"))
        if value is not None
    )


def teacher_mean(source: dict[str, Any]) -> float | None:
    values = available_teacher_values(source)
    return sum(values) / len(values) if values else None


def teacher_median(source: dict[str, Any]) -> float | None:
    values = available_teacher_values(source)
    if not values:
        return None
    midpoint = len(values) // 2
    return values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2


def observed_source_getters() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "human_1": lambda source: source.get("human_1"),
        "human_2": lambda source: source.get("human_2"),
        "human_3": lambda source: source.get("human_3"),
        "rounded_human": lambda source: source.get("rounded_human_label"),
        "qwen": lambda source: source.get("qwen_score"),
        "deepseek": lambda source: source.get("deepseek_score"),
        "teacher_mean": teacher_mean,
        "teacher_median": teacher_median,
    }


def source_reliability_table(
    sources: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = observed_source_getters()
    output: list[dict[str, Any]] = []
    for name, getter in fields.items():
        predictions: list[float] = []
        references: list[int] = []
        weights: list[float] = []
        for sid, reference in final.items():
            source = sources.get(sid)
            if source is None or source.get("view") != "representative_train":
                continue
            value = getter(source)
            if value is None:
                continue
            predictions.append(float(value))
            references.append(int(reference["point_score"]))
            weights.append(float(source.get("design_weight") or 1.0))
        output.append(source_metric_record(name, predictions, references, weights))
    # DS/MACE are preregistered for the completed run. Their estimators are
    # deliberately deferred until all representative rows have final scores;
    # absent rows remain explicit rather than being imputed during preparation.
    if len([sid for sid in final if sources.get(sid, {}).get("view") == "representative_train"]) < 120:
        output.extend(
            [
                {"source": "Dawid-Skene", "rows": 0, "status": "PENDING_COMPLETE_REPRESENTATIVE_REFERENCE"},
                {"source": "MACE", "rows": 0, "status": "PENDING_COMPLETE_REPRESENTATIVE_REFERENCE"},
            ]
        )
    else:
        # The categorical consensus estimates are produced by the same source
        # matrix but kept conservative here: expose explicit implementation
        # labels so later reports cannot confuse them with observed raters.
        consensus = latent_consensus_estimates(sources, final)
        for name in ("Dawid-Skene", "MACE"):
            predictions, references, weights = consensus[name]
            output.append(source_metric_record(name, predictions, references, weights))
    return output


def latent_consensus_estimates(
    sources: dict[str, dict[str, Any]], final: dict[str, dict[str, Any]]
) -> dict[str, tuple[list[float], list[int], list[float]]]:
    ids = [sid for sid in final if sources.get(sid, {}).get("view") == "representative_train"]
    rater_fields = ("human_1", "human_2", "human_3", "qwen_score", "deepseek_score")
    ratings = {
        sid: [int(sources[sid][field]) if sources[sid].get(field) is not None else None for field in rater_fields]
        for sid in ids
    }
    # Dawid-Skene categorical EM.
    posterior = {sid: [0.2] * 5 for sid in ids}
    for sid in ids:
        counts = Counter(value for value in ratings[sid] if value is not None)
        posterior[sid] = [(counts[label] + 0.5) / (sum(counts.values()) + 2.5) for label in range(1, 6)]
    for _ in range(40):
        priors = [sum(posterior[sid][label] for sid in ids) / len(ids) for label in range(5)]
        confusion = [
            [[0.2 for _observed in range(5)] for _true in range(5)]
            for _rater in rater_fields
        ]
        for rater_index in range(len(rater_fields)):
            for sid in ids:
                observed = ratings[sid][rater_index]
                if observed is None:
                    continue
                for true_label in range(5):
                    confusion[rater_index][true_label][observed - 1] += posterior[sid][true_label]
            for true_label in range(5):
                total = sum(confusion[rater_index][true_label])
                confusion[rater_index][true_label] = [value / total for value in confusion[rater_index][true_label]]
        for sid in ids:
            logp = [math.log(max(priors[true], 1e-12)) for true in range(5)]
            for rater_index, observed in enumerate(ratings[sid]):
                if observed is None:
                    continue
                for true in range(5):
                    logp[true] += math.log(max(confusion[rater_index][true][observed - 1], 1e-12))
            maximum = max(logp)
            probs = [math.exp(value - maximum) for value in logp]
            normalizer = sum(probs)
            posterior[sid] = [value / normalizer for value in probs]
    ds_predictions = [sum((label + 1) * posterior[sid][label] for label in range(5)) for sid in ids]

    # MACE-style one-coin competence/spam EM, provider-agnostic and CPU-only.
    competence = [0.7] * len(rater_fields)
    spam = [[0.2] * 5 for _ in rater_fields]
    mace_post = {sid: [0.2] * 5 for sid in ids}
    for _ in range(40):
        for sid in ids:
            probabilities = []
            for true in range(5):
                probability = 0.2
                for rater_index, observed in enumerate(ratings[sid]):
                    if observed is None:
                        continue
                    probability *= (
                        competence[rater_index] * float(observed - 1 == true)
                        + (1.0 - competence[rater_index]) * spam[rater_index][observed - 1]
                    )
                probabilities.append(probability)
            total = sum(probabilities) or 1.0
            mace_post[sid] = [value / total for value in probabilities]
        for rater_index in range(len(rater_fields)):
            known = 0.0
            observations = 0.0
            spam_counts = [0.2] * 5
            for sid in ids:
                observed = ratings[sid][rater_index]
                if observed is None:
                    continue
                observations += 1.0
                expected_known = 0.0
                for true in range(5):
                    if observed - 1 != true:
                        continue
                    numerator = competence[rater_index]
                    denominator = numerator + (1.0 - competence[rater_index]) * spam[rater_index][observed - 1]
                    expected_known += mace_post[sid][true] * numerator / max(denominator, 1e-12)
                known += expected_known
                spam_counts[observed - 1] += 1.0 - expected_known
            competence[rater_index] = min(0.999, max(0.001, (known + 1.0) / (observations + 2.0)))
            spam_total = sum(spam_counts)
            spam[rater_index] = [value / spam_total for value in spam_counts]
    mace_predictions = [sum((label + 1) * mace_post[sid][label] for label in range(5)) for sid in ids]
    references = [int(final[sid]["point_score"]) for sid in ids]
    weights = [float(sources[sid].get("design_weight") or 1.0) for sid in ids]
    return {
        "Dawid-Skene": (ds_predictions, references, weights),
        "MACE": (mace_predictions, references, weights),
    }


def prevalence_table(
    sources: dict[str, dict[str, Any]], final: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [(sources[sid], final[sid]) for sid in final if sources.get(sid, {}).get("view") == "representative_train"]
    definitions: dict[str, Callable[[dict[str, Any], dict[str, Any]], bool]] = {
        "original_label_conflict": lambda source, ref: int(source["rounded_human_label"]) != int(ref["point_score"]),
        "qwen_conflict": lambda source, ref: source.get("qwen_score") is not None and int(source["qwen_score"]) != int(ref["point_score"]),
        "deepseek_conflict": lambda source, ref: source.get("deepseek_score") is not None and int(source["deepseek_score"]) != int(ref["point_score"]),
        "evidence_failure": lambda _source, ref: ref["failure_bucket"] != "no_failure",
        "student_input_insufficiency": lambda _source, ref: ref["student_input_sufficiency"] in {"requires_explicit_rubric", "insufficient_context", "unclear"},
    }
    output = []
    for name, check in definitions.items():
        eligible = [(source, ref) for source, ref in rows if not (name == "deepseek_conflict" and source.get("deepseek_score") is None)]
        total = sum(float(source.get("design_weight") or 1.0) for source, _ in eligible)
        estimate = sum(float(source.get("design_weight") or 1.0) * check(source, ref) for source, ref in eligible) / total if total else ""
        output.append({"estimand": name, "rows": len(eligible), "design_weighted_prevalence": estimate, "view": "representative_train", "status": "COMPLETE" if len(rows) == 120 else "INCOMPLETE"})
    return output


def metric_rows_for_view(
    sources: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    view: str,
    weighted: bool,
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, getter in observed_source_getters().items():
        predictions: list[float] = []
        references: list[int] = []
        weights: list[float] = []
        for sid, reference in final.items():
            source = sources.get(sid)
            if source is None or source.get("view") != view:
                continue
            if predicate is not None and not predicate(source, reference):
                continue
            value = getter(source)
            if value is None:
                continue
            predictions.append(float(value))
            references.append(int(reference["point_score"]))
            weights.append(float(source.get("design_weight") or 1.0) if weighted else 1.0)
        output.append(source_metric_record(name, predictions, references, weights))
    return output


def source_error_subgroup_table(
    sources: dict[str, dict[str, Any]], final: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    subgroups: tuple[
        tuple[str, str, Callable[[dict[str, Any], dict[str, Any]], bool]], ...
    ] = (
        ("evidence_validity", "valid", lambda _source, _reference: True),
        # Invalid evidence rows never enter `final`; retaining an explicit zero
        # group prevents silent denominator changes in later reports.
        ("evidence_validity", "invalid_rejected", lambda _source, _reference: False),
        ("evidence_mode", "explicit_substring", lambda _source, reference: reference.get("evaluator_output_evidence") is not None),
        ("evidence_mode", "missing_content", lambda _source, reference: reference.get("evaluator_output_evidence") is None),
        ("student_input_sufficiency", "sufficient", lambda _source, reference: reference.get("student_input_sufficiency") == "sufficient"),
        ("student_input_sufficiency", "insufficient_or_unclear", lambda _source, reference: reference.get("student_input_sufficiency") != "sufficient"),
    )
    output: list[dict[str, Any]] = []
    for subgroup_type, subgroup_value, predicate in subgroups:
        for row in metric_rows_for_view(
            sources, final, "representative_train", True, predicate
        ):
            output.append(
                {
                    "view": "representative_train",
                    "weighting": "design_weighted",
                    "subgroup_type": subgroup_type,
                    "subgroup_value": subgroup_value,
                    **row,
                }
            )
    return output


def risk_stress_table(
    sources: dict[str, dict[str, Any]], final: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "view": "risk_enriched_train",
            "weighting": "unweighted",
            **row,
        }
        for row in metric_rows_for_view(sources, final, "risk_enriched_train", False)
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    protocol = read_json(out / "configs/exp33a_review_protocol_lock.json")
    decision = read_json(out / "decision/exp33a_expert_reference_decision.json")
    locked_type = str(protocol["locked_reviewer_type"])
    review_schema = read_json(args.review_schema)
    adjudication_schema = read_json(args.adjudication_schema)
    assignments = read_jsonl(out / "private_review/exp33a_review_assignment_manifest.jsonl")
    packet_rows = read_jsonl(out / "private_review/blind_packets/exp33a_reviewer_a_packet.jsonl")
    packets = {str(row["sample_id"]): row for row in packet_rows}
    sources = {str(row["sample_id"]): row for row in read_jsonl(out / "private/exp33a_source_reference.jsonl")}
    expected_ids = {str(row["sample_id"]) for row in assignments}
    if set(packets) != expected_ids or set(sources) != expected_ids:
        raise ValueError("Private assignment, packet, and source-reference identities differ")

    raw_a, parse_a = load_jsonl_dir(out / "private_review/reviewer_a_filled")
    raw_b, parse_b = load_jsonl_dir(out / "private_review/reviewer_b_filled")
    valid_a, errors_a = validate_role_rows(raw_a, "reviewer_a", review_schema, packets, locked_type)
    valid_b, errors_b = validate_role_rows(raw_b, "reviewer_b", review_schema, packets, locked_type)
    blind_complete = set(valid_a) == expected_ids and set(valid_b) == expected_ids and not (parse_a or parse_b or errors_a or errors_b)

    triggers: dict[str, list[str]] = {}
    source_hashes: dict[str, str] = {}
    bundles: list[dict[str, Any]] = []
    if blind_complete:
        triggers = {sid: reasons for sid in sorted(expected_ids) if (reasons := adjudication_reasons(valid_a[sid], valid_b[sid]))}
        bundles, source_hashes = make_adjudication_bundles(triggers, packets, valid_a, valid_b, sources)
        write_jsonl(out / "private_review/adjudication_packets/exp33a_source_aware_adjudication_packet.jsonl", bundles)

    raw_adj, parse_adj = load_jsonl_dir(out / "private_review/adjudication_filled")
    valid_adj, errors_adj = validate_adjudications(
        raw_adj, adjudication_schema, triggers, packets, valid_a, valid_b, source_hashes, sources, locked_type
    ) if blind_complete else ({}, ["Adjudication supplied before complete frozen A/B reviews"] if raw_adj else [])

    run_ids_a = {str(row["reviewer_run_id"]) for row in valid_a.values()}
    run_ids_b = {str(row["reviewer_run_id"]) for row in valid_b.values()}
    run_ids_adj = {str(row["reviewer_run_id"]) for row in valid_adj.values()}
    independent_runs = not (run_ids_a & run_ids_b or run_ids_a & run_ids_adj or run_ids_b & run_ids_adj)
    adjudication_complete = set(valid_adj) == set(triggers) and not (parse_adj or errors_adj)
    workflow_complete = blind_complete and adjudication_complete and independent_runs

    # Source-linked grouping and correction statistics are unavailable until
    # both blind result sets are complete and frozen. Partial reviews affect
    # completion counts only; they cannot trigger premature source comparison.
    analysis_a = valid_a if blind_complete else {}
    analysis_b = valid_b if blind_complete else {}
    agreement = agreement_table(analysis_a, analysis_b, sources, set(triggers))
    overall = next(row for row in agreement if row["group_type"] == "overall")
    leakage_rows = list(csv.DictReader(guarded(out / "tables/exp33a_blind_leakage_audit.csv").open("r", encoding="utf-8", newline="")))
    leakage_zero = all(int(row["count"]) == 0 and row["status"] == "PASS" for row in leakage_rows)
    gates = {
        "paired_review_coverage": blind_complete,
        "schema_and_evidence_validity": blind_complete and not (errors_a or errors_b),
        "blind_leakage_count_zero": leakage_zero,
        "within_one_agreement": bool(blind_complete and float(overall["within_one"]) >= 0.90),
        "quadratic_weighted_kappa": bool(blind_complete and overall["quadratic_weighted_kappa"] is not None and float(overall["quadratic_weighted_kappa"]) >= 0.60),
        "krippendorff_ordinal_alpha": bool(blind_complete and overall["krippendorff_ordinal_alpha"] is not None and float(overall["krippendorff_ordinal_alpha"]) >= 0.60),
        "all_triggered_cases_processed": workflow_complete,
        "independent_run_ids": independent_runs,
    }
    calibration_gate_passed = all(gates.values())
    model_complete = workflow_complete and locked_type == "model"
    human_complete = workflow_complete and locked_type == "human" and args.confirm_real_human_review

    completion = [
        completion_row("reviewer_a", locked_type, 420, len(raw_a), valid_a, parse_a, errors_a),
        completion_row("reviewer_b", locked_type, 420, len(raw_b), valid_b, parse_b, errors_b),
        completion_row("adjudicator", locked_type, len(triggers), len(raw_adj), valid_adj, parse_adj, errors_adj),
    ]
    write_csv(
        out / "tables/exp33a_review_completion.csv",
        completion,
        [
            "reviewer_role", "reviewer_type", "reviewer_provenance", "reviewer_provider",
            "reviewer_model_id", "reviewer_run_id_count", "expected_rows", "completed_rows",
            "valid_rows", "invalid_rows", "status", "reference_status",
        ],
    )
    agreement_fields = [
        "group_type", "group_value", "paired_rows", "exact_agreement", "within_one",
        "quadratic_weighted_kappa", "krippendorff_ordinal_alpha", "score_range_overlap",
        "adjudication_rate", "status",
    ]
    write_csv(out / "tables/exp33a_reviewer_agreement.csv", agreement, agreement_fields)
    domain = domain_table(analysis_a, analysis_b, valid_adj, sources)
    write_csv(
        out / "tables/exp33a_domain_escalation_summary.csv",
        domain,
        [
            "group_type", "group_value", "reviewed_rows", "domain_escalation_required",
            "adjudicated", "unresolved_domain_cases", "status",
        ],
    )
    diagnostics = direction_diagnostics(sources, analysis_a, analysis_b)
    write_csv(
        out / "tables/exp33a_source_comparison_diagnostics.csv",
        diagnostics,
        ["diagnostic", "eligible_rows", "flagged_rows", "status"],
    )

    final = final_reference_rows(analysis_a, analysis_b, triggers, valid_adj)
    reliability = source_reliability_table(sources, final)
    reliability_fields = [
        "source", "rows", "mae", "qwk", "exact", "within_one", "signed_bias",
        "severe_error", "low_to_high", "high_to_low", "label1_recall", "label2_recall",
        "label5_recall", "status",
    ]
    write_csv(out / "tables/exp33a_source_reliability.csv", reliability, reliability_fields)
    subgroup_rows = source_error_subgroup_table(sources, final)
    subgroup_fields = [
        "view", "weighting", "subgroup_type", "subgroup_value", "source", "rows",
        "mae", "qwk", "exact", "within_one", "signed_bias", "severe_error",
        "low_to_high", "high_to_low", "label1_recall", "label2_recall",
        "label5_recall", "status",
    ]
    write_csv(out / "tables/exp33a_source_error_subgroups.csv", subgroup_rows, subgroup_fields)
    risk_stress = risk_stress_table(sources, final)
    write_csv(
        out / "tables/exp33a_risk_stress_metrics.csv",
        risk_stress,
        ["view", "weighting", *reliability_fields],
    )
    prevalence = prevalence_table(sources, final)
    write_csv(
        out / "tables/exp33a_representative_prevalence.csv",
        prevalence,
        ["estimand", "rows", "design_weighted_prevalence", "view", "status"],
    )

    providers = {
        role: sorted({str(row.get("reviewer_provider")) for row in valid.values() if row.get("reviewer_provider")})
        for role, valid in (("reviewer_a", valid_a), ("reviewer_b", valid_b), ("adjudicator", valid_adj))
    }
    models = {
        role: sorted({str(row.get("reviewer_model_id")) for row in valid.values() if row.get("reviewer_model_id")})
        for role, valid in (("reviewer_a", valid_a), ("reviewer_b", valid_b), ("adjudicator", valid_adj))
    }
    decision.update(
        {
            "review_completion_state": "complete" if workflow_complete else "not_started" if not (raw_a or raw_b or raw_adj) else "incomplete_or_invalid",
            "reviewer_providers": providers,
            "reviewer_model_ids": models,
            "reviewer_run_independence_verified": independent_runs if (raw_a or raw_b) else False,
            "blind_reviewer_a_valid_rows": len(valid_a),
            "blind_reviewer_b_valid_rows": len(valid_b),
            "adjudication_trigger_rows": len(triggers),
            "adjudication_valid_rows": len(valid_adj),
            "human_empirical_distribution_fallback_rows": sum(row.get("final_status") == "human_empirical_distribution_fallback" for row in valid_adj.values()),
            "unresolved_domain_case_rows": sum(row.get("final_status") == "unresolved_domain_case" for row in valid_adj.values()),
            "calibration_gates": gates,
            "calibration_gate_passed": calibration_gate_passed,
            "model_silver_reference_complete": model_complete,
            "expert_reference_complete": human_complete,
            "teacher_reliability_ready": bool((model_complete or human_complete) and calibration_gate_passed),
            "recommend_new_teacher_training": False,
            "recommend_student_training": False,
            "recommend_test_access": False,
            "test_access_count": 0,
        }
    )
    if not human_complete:
        decision["expert_reference_complete"] = False
    write_json(out / "decision/exp33a_expert_reference_decision.json", decision)

    errors = parse_a + parse_b + parse_adj + errors_a + errors_b + errors_adj
    report = f"""# Exp33A Review Completion Analysis

- Reference claim: independent model-reviewed silver reference (when model workflow completes).
- Locked reviewer type: `{locked_type}`.
- Reviewer A valid: {len(valid_a)} / 420.
- Reviewer B valid: {len(valid_b)} / 420.
- Triggered adjudications: {len(triggers)}; valid adjudications: {len(valid_adj)}.
- Independent run IDs verified: {independent_runs if (raw_a or raw_b) else 'not yet evaluable'}.
- Model silver reference complete: {model_complete}.
- Expert reference complete: {human_complete}.
- Calibration gate passed: {calibration_gate_passed}.
- Teacher reliability ready: {decision['teacher_reliability_ready']}.
- Errors: {len(errors)}.
- Test access count: 0.

The method claim is provider-agnostic: blind-first source comparison, conflict adjudication,
direction-aware correction, and uncertainty fallback. Provider/model IDs above are reproducibility
provenance, not the innovation. No API, GPU, training, inference, or test access occurred.
"""
    write_text(out / "reports/exp33a_review_completion_report.md", report)
    return {
        "reviewer_a_valid": len(valid_a),
        "reviewer_b_valid": len(valid_b),
        "adjudication_triggers": len(triggers),
        "adjudication_valid": len(valid_adj),
        "model_silver_reference_complete": model_complete,
        "expert_reference_complete": human_complete,
        "calibration_gate_passed": calibration_gate_passed,
        "errors": errors,
        "test_access_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--review-schema", type=Path, default=REVIEW_SCHEMA)
    parser.add_argument("--adjudication-schema", type=Path, default=ADJUDICATION_SCHEMA)
    parser.add_argument(
        "--confirm-real-human-review",
        action="store_true",
        help="Required before expert_reference_complete may become true for a human-locked run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), ensure_ascii=False, sort_keys=True))
