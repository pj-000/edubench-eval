#!/usr/bin/env python3
"""Prepare Exp33B direction-aware train-label aggregation artifacts.

Exp33B is CPU-only data aggregation and audit. It reads the locked paper train
split, Exp33A representative-train model-reviewed silver review outputs, and
train-only teacher annotations resolved through the Exp33A/Exp28 locks. It
does not read dev/test, train a model, call APIs, run student inference, or use
GPU. Full 2,654-row supervision is produced only when preregistered
representative-train quality gates pass, and only under a gitignored private
directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_expert_reference.build_exp33a_private_source_reference import (  # noqa: E402
    DEFAULT_EXP28E_DECISION,
    DEFAULT_TEACHER_SUMMARY_DIR,
    annotation_projection,
    display_path,
    resolve_teacher_inputs,
    sha256_file,
)


LABELS = (1, 2, 3, 4, 5)
SOURCE_NAMES = ("human_1", "human_2", "human_3", "qwen", "deepseek")
BASELINE_ORDER = (
    "rounded_human",
    "human_median",
    "human_mean",
    "qwen",
    "deepseek",
    "teacher_mean",
    "teacher_median",
    "Dawid-Skene",
    "MACE",
    "equal_weight_fusion",
    "DRGA",
)
MATURE_BASELINES = tuple(name for name in BASELINE_ORDER if name != "DRGA")

DEFAULT_CONFIG = Path(
    "thesis_exp/exp33_direction_aware_aggregation/configs/exp33b_drga_preregistration_config.json"
)
DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_EXP33A_OUT = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_direction_aware_aggregation/outputs/exp33b_direction_aware_aggregation_seed42"
)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() in {"dev.jsonl", "test.jsonl"}:
        raise PermissionError(f"Exp33B forbids access to sealed/nontrain split: {absolute}")
    return absolute


def read_json(path: Path) -> dict[str, Any]:
    with guarded(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with guarded(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_fraction(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:14], 16) / float(16**14 - 1)


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Row has no sample identifier")
    return str(value)


def as_int_label(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        label = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return min(5, max(1, label))


def one_hot(label: int) -> list[float]:
    return [1.0 if value == label else 0.0 for value in LABELS]


def normalize(probs: Iterable[float]) -> list[float]:
    values = [max(0.0, float(value)) for value in probs]
    total = sum(values)
    if total <= 0:
        return [0.2] * 5
    return [value / total for value in values]


def expected_score(probs: list[float]) -> float:
    return sum(label * probs[label - 1] for label in LABELS)


def hard_label(probs: list[float]) -> int:
    return max(LABELS, key=lambda label: (probs[label - 1], -label))


def entropy(probs: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probs)


def normalized_entropy(probs: list[float]) -> float:
    return entropy(probs) / math.log(5.0)


def numeric_confidence(value: Any) -> float:
    if value is None or value == "":
        return 0.65
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    mapping = {"low": 0.35, "medium": 0.65, "high": 0.9}
    return mapping.get(str(value).strip().casefold(), 0.65)


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def human_scores(source: dict[str, Any]) -> list[int]:
    values = [as_int_label(source.get(field)) for field in ("human_1", "human_2", "human_3")]
    return [value for value in values if value is not None]


def teacher_scores(source: dict[str, Any]) -> list[int]:
    values = [as_int_label(source.get("qwen_score")), as_int_label(source.get("deepseek_score"))]
    return [value for value in values if value is not None]


def human_empirical_distribution(source: dict[str, Any]) -> list[float]:
    values = human_scores(source)
    if not values:
        rounded = as_int_label(source.get("rounded_human_label")) or 3
        return one_hot(rounded)
    counts = Counter(values)
    return [counts[label] / len(values) for label in LABELS]


def human_empirical_hard(source: dict[str, Any]) -> int:
    dist = human_empirical_distribution(source)
    median = safe_median([float(value) for value in human_scores(source)])
    if median is None:
        return hard_label(dist)
    return max(LABELS, key=lambda label: (dist[label - 1], -abs(label - median), -label))


def score_direction(original: int, teacher: int | None) -> str | None:
    if teacher is None:
        return None
    if original <= 2 and teacher >= 4:
        return "low_to_high"
    if original >= 4 and teacher <= 2:
        return "high_to_low"
    if teacher > original:
        return "up"
    if teacher < original:
        return "down"
    return "same"


def source_observation(source: dict[str, Any], name: str) -> int | None:
    if name in {"human_1", "human_2", "human_3"}:
        return as_int_label(source.get(name))
    if name == "qwen":
        return as_int_label(source.get("qwen_score"))
    if name == "deepseek":
        return as_int_label(source.get("deepseek_score"))
    raise KeyError(name)


def source_confidence(source: dict[str, Any], name: str) -> float:
    if name == "qwen":
        return numeric_confidence(source.get("qwen_confidence"))
    if name == "deepseek":
        return numeric_confidence(source.get("deepseek_confidence"))
    return 1.0


def source_evidence_flags(source: dict[str, Any], name: str) -> list[str]:
    value = source.get(f"{name}_evidence_flags")
    return [str(item) for item in value] if isinstance(value, list) else []


def source_score_cap(source: dict[str, Any], name: str) -> int | None:
    if name not in {"qwen", "deepseek"}:
        return None
    return as_int_label(source.get(f"{name}_score_cap"))


def teacher_projection(row: dict[str, Any] | None) -> dict[str, Any]:
    projected = annotation_projection(row)
    return {
        "score": projected["score"],
        "score_range": projected["score_range"],
        "confidence": projected["confidence"],
        "evidence_flags": projected["evidence_flags"],
        "major_failures": projected["major_failures"],
        "score_cap": projected["score_cap"],
        "reason_present": bool(projected["reason"]),
        "rubric_assessment_items": len(projected["rubric_assessment"] or []),
    }


def reviewer_trigger_reasons(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(a["most_plausible_score"]) != int(b["most_plausible_score"]):
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


def final_point_from_adjudication(row: dict[str, Any], posterior: list[float]) -> int:
    point = row.get("final_most_plausible_score")
    if point is None:
        return hard_label(posterior)
    label = as_int_label(point)
    if label is None:
        raise ValueError(f"Invalid adjudicated final score: {point}")
    return label


def load_selected_train_views(exp33a_out: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(exp33a_out / "private/exp33a_selected_sample_manifest.jsonl")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("source_split")) != "train":
            continue
        selected[str(row["sample_id"])] = dict(row)
    return selected


def load_train_sources(
    split_dir: Path,
    selected_train: dict[str, dict[str, Any]],
    teacher_resolved: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    train_rows = read_jsonl(split_dir / "train.jsonl")
    if len(train_rows) != 2654:
        raise ValueError(f"Locked train split size mismatch: {len(train_rows)}")
    primary = {str(row["sample_id"]): row for row in teacher_resolved["primary"]["rows"]}
    secondary = {str(row["sample_id"]): row for row in teacher_resolved["secondary"]["rows"]}
    if set(primary) != {sample_id(row) for row in train_rows}:
        raise ValueError("Primary teacher coverage is not exactly the locked train split")

    output: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        sid = sample_id(row)
        original = int(row["label_5"])
        qwen = teacher_projection(primary.get(sid))
        deepseek = teacher_projection(secondary.get(sid))
        selected = selected_train.get(sid, {})
        source = {
            "sample_id": sid,
            "source_split": "train",
            "view": selected.get("view", "full_train"),
            "question_key": str(row["question_key"]),
            "language": row.get("language"),
            "metric_family": row.get("metric_group"),
            "metric": row.get("metric_canonical"),
            "subject": row.get("subject_canonical"),
            "human_1": row.get("human_1_5"),
            "human_2": row.get("human_2_5"),
            "human_3": row.get("human_3_5"),
            "human_mean": row.get("human_mean_5"),
            "rounded_human_label": original,
            "qwen_score": qwen["score"],
            "qwen_score_range": qwen["score_range"],
            "qwen_confidence": qwen["confidence"],
            "qwen_evidence_flags": qwen["evidence_flags"],
            "qwen_major_failures": qwen["major_failures"],
            "qwen_score_cap": qwen["score_cap"],
            "qwen_reason_present": qwen["reason_present"],
            "qwen_rubric_assessment_items": qwen["rubric_assessment_items"],
            "deepseek_score": deepseek["score"],
            "deepseek_score_range": deepseek["score_range"],
            "deepseek_confidence": deepseek["confidence"],
            "deepseek_evidence_flags": deepseek["evidence_flags"],
            "deepseek_major_failures": deepseek["major_failures"],
            "deepseek_score_cap": deepseek["score_cap"],
            "deepseek_reason_present": deepseek["reason_present"],
            "deepseek_rubric_assessment_items": deepseek["rubric_assessment_items"],
            "teacher_direction": {
                "qwen": score_direction(original, qwen["score"]),
                "deepseek": score_direction(original, deepseek["score"]),
            },
            "teacher_evidence_flags": sorted(set(qwen["evidence_flags"] + deepseek["evidence_flags"])),
            "campaign_transition_type": (
                f"qwen:{score_direction(original, qwen['score']) or 'missing'}|"
                f"deepseek:{score_direction(original, deepseek['score']) or 'missing'}"
            ),
            "design_weight": selected.get("design_weight"),
            "inclusion_probability": selected.get("inclusion_probability"),
            "sampling_risk_reason": selected.get("sampling_risk_reason") or [],
        }
        output[sid] = source
    return output


def load_final_silver_reference(
    exp33a_out: Path,
    expected_ids: set[str],
    *,
    require_complete: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reviewer_dir = repo_path(exp33a_out / "private_review/reviewer_filled")
    adjudication_dir = repo_path(exp33a_out / "private_review/adjudication_filled")
    raw_a = [
        row
        for path in sorted(reviewer_dir.glob("exp33a_reviewer_a*_results.jsonl"))
        for row in read_jsonl(path)
    ]
    raw_b = [
        row
        for path in sorted(reviewer_dir.glob("exp33a_reviewer_b*_results.jsonl"))
        for row in read_jsonl(path)
    ]
    a_rows = {str(row["sample_id"]): row for row in raw_a if str(row.get("sample_id")) in expected_ids}
    b_rows = {str(row["sample_id"]): row for row in raw_b if str(row.get("sample_id")) in expected_ids}
    missing_a = sorted(expected_ids - set(a_rows))
    missing_b = sorted(expected_ids - set(b_rows))
    if missing_a or missing_b:
        status = {
            "status": "PENDING",
            "missing_reviewer_a": len(missing_a),
            "missing_reviewer_b": len(missing_b),
            "expected_rows": len(expected_ids),
        }
        if require_complete:
            raise ValueError(f"Exp33A silver review rows incomplete: {status}")
        return {}, status

    triggers = {
        sid: reasons
        for sid in sorted(expected_ids)
        if (reasons := reviewer_trigger_reasons(a_rows[sid], b_rows[sid]))
    }
    raw_adj = [
        row
        for path in sorted(adjudication_dir.glob("exp33a_adjudicator*_results.jsonl"))
        for row in read_jsonl(path)
    ]
    adj = {str(row["sample_id"]): row for row in raw_adj if str(row.get("sample_id")) in expected_ids}
    missing_adj = sorted(set(triggers) - set(adj))
    if missing_adj:
        status = {
            "status": "PENDING",
            "missing_adjudications": len(missing_adj),
            "triggered_rows": len(triggers),
            "expected_rows": len(expected_ids),
        }
        if require_complete:
            raise ValueError(f"Exp33A adjudication rows incomplete: {status}")
        return {}, status

    final: dict[str, dict[str, Any]] = {}
    for sid in sorted(expected_ids):
        if sid not in triggers:
            score = int(a_rows[sid]["most_plausible_score"])
            posterior = one_hot(score)
            final[sid] = {
                "sample_id": sid,
                "point_score": score,
                "posterior": posterior,
                "status": "blind_pair_consensus",
                "triggered": False,
                "adjudication_triggers": [],
            }
            continue
        adjudication = adj[sid]
        if adjudication.get("final_status") == "unresolved_domain_case":
            if require_complete:
                raise ValueError(f"Unresolved domain case cannot calibrate Exp33B: {sid}")
            continue
        posterior_map = adjudication["final_score_posterior"]
        posterior = normalize([float(posterior_map.get(str(label), 0.0)) for label in LABELS])
        final[sid] = {
            "sample_id": sid,
            "point_score": final_point_from_adjudication(adjudication, posterior),
            "posterior": posterior,
            "status": adjudication["final_status"],
            "triggered": True,
            "adjudication_triggers": triggers[sid],
        }
    return final, {
        "status": "COMPLETE" if len(final) == len(expected_ids) else "INCOMPLETE",
        "expected_rows": len(expected_ids),
        "final_rows": len(final),
        "triggered_rows": len(triggers),
        "adjudicated_rows": len(adj),
    }


def make_calibration_records(
    final: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sid, reference in sorted(final.items()):
        source = sources.get(sid)
        if source is None or source.get("view") != "representative_train":
            raise ValueError(f"Final representative row missing train source: {sid}")
        record = {
            "sample_id": sid,
            "language": str(source.get("language") or "unknown"),
            "reference_point": int(reference["point_score"]),
            "reference_posterior": list(reference["posterior"]),
            "reference_status": reference["status"],
            "reference_triggered": bool(reference["triggered"]),
            "design_weight": float(source.get("design_weight") or 1.0),
            "source": source,
        }
        records.append(record)
    if len(records) != 120:
        raise ValueError(f"Representative calibration set must have 120 rows, got {len(records)}")
    return records


def assign_folds(records: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[f"{record['reference_point']}|{record['language']}"].append(record)
    assignment: dict[str, int] = {}
    for key, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: stable_fraction(seed, f"fold|{key}", str(row["sample_id"])),
        )
        for index, row in enumerate(ordered):
            assignment[str(row["sample_id"])] = index % folds
    return assignment


def fit_supervised_reliability(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    smoothing = float(config["source_confusion_smoothing"])
    class_counts = [smoothing] * 5
    for record in records:
        weight = float(record["design_weight"])
        for index, prob in enumerate(record["reference_posterior"]):
            class_counts[index] += weight * float(prob)
    class_prior = normalize(class_counts)

    source_models: dict[str, dict[str, Any]] = {}
    for name in SOURCE_NAMES:
        counts = [[smoothing for _observed in LABELS] for _true in LABELS]
        obs_weight = 0.0
        exact = 0.0
        within_one = 0.0
        severe = 0.0
        signed_bias = 0.0
        for record in records:
            source = record["source"]
            observed = source_observation(source, name)
            if observed is None:
                continue
            row_weight = float(record["design_weight"])
            obs_weight += row_weight
            for true_label in LABELS:
                posterior_weight = row_weight * float(record["reference_posterior"][true_label - 1])
                counts[true_label - 1][observed - 1] += posterior_weight
                exact += posterior_weight * float(observed == true_label)
                within_one += posterior_weight * float(abs(observed - true_label) <= 1)
                severe += posterior_weight * float(abs(observed - true_label) >= 2)
                signed_bias += posterior_weight * (observed - true_label)
        confusion = [normalize(row) for row in counts]
        if obs_weight <= 0:
            reliability = 0.2
            exact_rate = within_rate = 0.0
            severe_rate = 1.0
            bias = 0.0
        else:
            exact_rate = exact / obs_weight
            within_rate = within_one / obs_weight
            severe_rate = severe / obs_weight
            bias = signed_bias / obs_weight
            reliability = max(0.0, min(1.0, (exact_rate - 0.2) / 0.8))
        floor = float(config["source_weight_floor"])
        cap = float(config["source_weight_cap"])
        power = float(config["source_weight_power"])
        source_weight = floor + (cap - floor) * (reliability**power)
        source_models[name] = {
            "rows": sum(1 for record in records if source_observation(record["source"], name) is not None),
            "weighted_observations": obs_weight,
            "confusion": confusion,
            "exact_soft": exact_rate,
            "within_one_soft": within_rate,
            "severe_soft": severe_rate,
            "signed_bias_soft": bias,
            "reliability": reliability,
            "source_weight": source_weight,
        }
    return {
        "class_prior": class_prior,
        "source_models": source_models,
        "training_rows": len(records),
        "training_weight": sum(float(record["design_weight"]) for record in records),
    }


def strong_low_to_high_evidence(source: dict[str, Any], label: int) -> bool:
    teacher_high = 0
    for name in ("qwen", "deepseek"):
        observed = source_observation(source, name)
        if observed is None or observed < label:
            continue
        confidence = source_confidence(source, name)
        cap = source_score_cap(source, name)
        if confidence >= 0.8 and (cap is None or cap >= label):
            teacher_high += 1
    return teacher_high >= 2


def strong_high_to_low_evidence(source: dict[str, Any], label: int) -> bool:
    for name in ("qwen", "deepseek"):
        observed = source_observation(source, name)
        cap = source_score_cap(source, name)
        flags = set(source_evidence_flags(source, name))
        if observed is not None and observed <= label and cap is not None and cap <= label:
            if flags & {"major_failure_flagged", "score_cap_present", "rubric_item_not_met"}:
                return True
    return False


def direction_flag_payload(source: dict[str, Any], posterior: list[float], label: int) -> dict[str, Any]:
    rounded = as_int_label(source.get("rounded_human_label")) or human_empirical_hard(source)
    return {
        "human4_to_label5": bool(rounded == 4 and label == 5),
        "low_to_high": bool(rounded <= 2 and label >= 4),
        "high_to_low": bool(rounded >= 4 and label <= 2),
        "score_cap_exceeded": any(
            (cap := source_score_cap(source, name)) is not None and label > cap
            for name in ("qwen", "deepseek")
        ),
        "teacher_low_to_high_signal": any(
            (as_int_label(source.get("rounded_human_label")) or 3) <= 2
            and (source_observation(source, name) or 0) >= 4
            for name in ("qwen", "deepseek")
        ),
        "teacher_high_to_low_signal": any(
            (as_int_label(source.get("rounded_human_label")) or 3) >= 4
            and (source_observation(source, name) or 6) <= 2
            for name in ("qwen", "deepseek")
        ),
        "posterior_entropy": normalized_entropy(posterior),
    }


def predict_drga(source: dict[str, Any], model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    empirical = human_empirical_distribution(source)
    empirical_smoothing = float(config["human_prior_smoothing"])
    empirical_smoothed = normalize([value + empirical_smoothing for value in empirical])
    prior = model["class_prior"]
    logp = [
        float(config["human_empirical_prior_weight"]) * math.log(max(empirical_smoothed[index], 1e-12))
        + float(config["global_prior_weight"]) * math.log(max(prior[index], 1e-12))
        for index in range(5)
    ]

    observed_sources = 0
    for name in SOURCE_NAMES:
        observed = source_observation(source, name)
        if observed is None:
            continue
        observed_sources += 1
        source_model = model["source_models"][name]
        weight = float(source_model["source_weight"])
        if name in {"qwen", "deepseek"}:
            confidence_factor = float(config["confidence_weight_floor"]) + (
                1.0 - float(config["confidence_weight_floor"])
            ) * source_confidence(source, name)
            evidence_bonus = float(config["teacher_evidence_bonus"]) if source_evidence_flags(source, name) else 0.0
            weight *= confidence_factor + evidence_bonus
        for label in LABELS:
            logp[label - 1] += weight * math.log(
                max(float(source_model["confusion"][label - 1][observed - 1]), 1e-12)
            )

    penalties = config["direction_penalties"]
    rounded = as_int_label(source.get("rounded_human_label")) or human_empirical_hard(source)
    for label in LABELS:
        penalty = 0.0
        if rounded == 4 and label == 5:
            human_fives = sum(1 for value in human_scores(source) if value == 5)
            teachers_five = sum(1 for value in teacher_scores(source) if value == 5)
            if human_fives < 2 and teachers_five < 2:
                penalty += float(penalties["unsupported_human4_to_label5"])
        if rounded <= 2 and label >= 4 and not strong_low_to_high_evidence(source, label):
            penalty += float(penalties["dangerous_low_to_high"])
        if rounded >= 4 and label <= 2 and not strong_high_to_low_evidence(source, label):
            penalty += float(penalties["unsupported_high_to_low"])
        if rounded >= 4 and label <= rounded - 2 and not strong_high_to_low_evidence(source, label):
            penalty += float(penalties["over_lowering_high_human_per_point"]) * float(rounded - label)
        for name in ("qwen", "deepseek"):
            cap = source_score_cap(source, name)
            if cap is not None and label > cap:
                penalty += float(penalties["score_cap_exceeded_per_point"]) * float(label - cap)
        logp[label - 1] -= penalty

    maximum = max(logp)
    posterior = normalize(math.exp(value - maximum) for value in logp)
    max_prob = max(posterior)
    entropy_value = normalized_entropy(posterior)
    fallback_config = config["fallback"]
    fallback = (
        entropy_value > float(fallback_config["entropy_threshold"])
        or max_prob < float(fallback_config["max_probability_threshold"])
        or observed_sources < int(fallback_config["min_observed_sources"])
    )
    if fallback:
        posterior = human_empirical_distribution(source)
        label = human_empirical_hard(source)
        status = "human_empirical_distribution_fallback"
    else:
        label = hard_label(posterior)
        status = "drga_model_reviewed_silver_aggregation"
    return {
        "method": "DRGA",
        "posterior": posterior,
        "score_prediction": expected_score(posterior),
        "hard_label": label,
        "status": status,
        "fallback": fallback,
        "uncertainty_entropy": normalized_entropy(posterior),
        "direction_flags": direction_flag_payload(source, posterior, label),
        "observed_sources": observed_sources,
    }


def scalar_prediction(method: str, value: float | int | None) -> dict[str, Any] | None:
    if value is None:
        return None
    score = min(5.0, max(1.0, float(value)))
    label = min(5, max(1, int(round(score))))
    return {
        "method": method,
        "posterior": one_hot(label),
        "score_prediction": score,
        "hard_label": label,
        "status": "COMPLETE",
        "fallback": False,
        "uncertainty_entropy": 0.0,
        "direction_flags": {},
    }


def equal_weight_fusion_prediction(source: dict[str, Any]) -> dict[str, Any] | None:
    labels = [
        source_observation(source, name)
        for name in SOURCE_NAMES
        if source_observation(source, name) is not None
    ]
    if not labels:
        return None
    counts = Counter(labels)
    posterior = [counts[label] / len(labels) for label in LABELS]
    return {
        "method": "equal_weight_fusion",
        "posterior": posterior,
        "score_prediction": expected_score(posterior),
        "hard_label": hard_label(posterior),
        "status": "COMPLETE",
        "fallback": False,
        "uncertainty_entropy": normalized_entropy(posterior),
        "direction_flags": {},
    }


def fixed_baseline_prediction(method: str, source: dict[str, Any]) -> dict[str, Any] | None:
    humans = [float(value) for value in human_scores(source)]
    teachers = [float(value) for value in teacher_scores(source)]
    if method == "rounded_human":
        return scalar_prediction(method, source.get("rounded_human_label"))
    if method == "human_mean":
        return scalar_prediction(method, safe_mean(humans))
    if method == "human_median":
        return scalar_prediction(method, safe_median(humans))
    if method == "qwen":
        return scalar_prediction(method, source.get("qwen_score"))
    if method == "deepseek":
        return scalar_prediction(method, source.get("deepseek_score"))
    if method == "teacher_mean":
        return scalar_prediction(method, safe_mean(teachers))
    if method == "teacher_median":
        return scalar_prediction(method, safe_median(teachers))
    if method == "equal_weight_fusion":
        return equal_weight_fusion_prediction(source)
    raise KeyError(method)


def fit_ds_model(records: list[dict[str, Any]], iterations: int = 45) -> dict[str, Any]:
    ids = [str(record["sample_id"]) for record in records]
    ratings = {
        str(record["sample_id"]): [source_observation(record["source"], name) for name in SOURCE_NAMES]
        for record in records
    }
    posterior: dict[str, list[float]] = {}
    for sid in ids:
        counts = Counter(value for value in ratings[sid] if value is not None)
        posterior[sid] = normalize((counts[label] + 0.5) for label in LABELS)
    confusion = [[[0.2 for _observed in LABELS] for _true in LABELS] for _source in SOURCE_NAMES]
    priors = [0.2] * 5
    if not ids:
        return {"priors": priors, "confusion": confusion}
    for _ in range(iterations):
        priors = normalize(sum(posterior[sid][index] for sid in ids) for index in range(5))
        confusion = [[[0.2 for _observed in LABELS] for _true in LABELS] for _source in SOURCE_NAMES]
        for source_index in range(len(SOURCE_NAMES)):
            for sid in ids:
                observed = ratings[sid][source_index]
                if observed is None:
                    continue
                for true_index in range(5):
                    confusion[source_index][true_index][observed - 1] += posterior[sid][true_index]
            for true_index in range(5):
                confusion[source_index][true_index] = normalize(confusion[source_index][true_index])
        for sid in ids:
            logp = [math.log(max(priors[index], 1e-12)) for index in range(5)]
            for source_index, observed in enumerate(ratings[sid]):
                if observed is None:
                    continue
                for true_index in range(5):
                    logp[true_index] += math.log(max(confusion[source_index][true_index][observed - 1], 1e-12))
            maximum = max(logp)
            posterior[sid] = normalize(math.exp(value - maximum) for value in logp)
    return {"priors": priors, "confusion": confusion}


def predict_ds(source: dict[str, Any], model: dict[str, Any]) -> dict[str, Any] | None:
    ratings = [source_observation(source, name) for name in SOURCE_NAMES]
    if not any(value is not None for value in ratings):
        return None
    logp = [math.log(max(float(model["priors"][index]), 1e-12)) for index in range(5)]
    for source_index, observed in enumerate(ratings):
        if observed is None:
            continue
        for true_index in range(5):
            logp[true_index] += math.log(
                max(float(model["confusion"][source_index][true_index][observed - 1]), 1e-12)
            )
    maximum = max(logp)
    posterior = normalize(math.exp(value - maximum) for value in logp)
    return {
        "method": "Dawid-Skene",
        "posterior": posterior,
        "score_prediction": expected_score(posterior),
        "hard_label": hard_label(posterior),
        "status": "COMPLETE",
        "fallback": False,
        "uncertainty_entropy": normalized_entropy(posterior),
        "direction_flags": {},
    }


def fit_mace_model(records: list[dict[str, Any]], iterations: int = 45) -> dict[str, Any]:
    ids = [str(record["sample_id"]) for record in records]
    ratings = {
        str(record["sample_id"]): [source_observation(record["source"], name) for name in SOURCE_NAMES]
        for record in records
    }
    competence = [0.7] * len(SOURCE_NAMES)
    spam = [[0.2] * 5 for _ in SOURCE_NAMES]
    posterior = {sid: [0.2] * 5 for sid in ids}
    for _ in range(iterations):
        for sid in ids:
            probabilities = []
            for true_index in range(5):
                probability = 0.2
                for source_index, observed in enumerate(ratings[sid]):
                    if observed is None:
                        continue
                    probability *= (
                        competence[source_index] * float(observed - 1 == true_index)
                        + (1.0 - competence[source_index]) * spam[source_index][observed - 1]
                    )
                probabilities.append(probability)
            posterior[sid] = normalize(probabilities)
        for source_index in range(len(SOURCE_NAMES)):
            known = 0.0
            observations = 0.0
            spam_counts = [0.2] * 5
            for sid in ids:
                observed = ratings[sid][source_index]
                if observed is None:
                    continue
                observations += 1.0
                expected_known = 0.0
                for true_index in range(5):
                    if observed - 1 != true_index:
                        continue
                    numerator = competence[source_index]
                    denominator = numerator + (1.0 - competence[source_index]) * spam[source_index][observed - 1]
                    expected_known += posterior[sid][true_index] * numerator / max(denominator, 1e-12)
                known += expected_known
                spam_counts[observed - 1] += 1.0 - expected_known
            competence[source_index] = min(0.999, max(0.001, (known + 1.0) / (observations + 2.0)))
            spam[source_index] = normalize(spam_counts)
    return {"competence": competence, "spam": spam}


def predict_mace(source: dict[str, Any], model: dict[str, Any]) -> dict[str, Any] | None:
    ratings = [source_observation(source, name) for name in SOURCE_NAMES]
    if not any(value is not None for value in ratings):
        return None
    probabilities = []
    for true_index in range(5):
        probability = 0.2
        for source_index, observed in enumerate(ratings):
            if observed is None:
                continue
            probability *= (
                model["competence"][source_index] * float(observed - 1 == true_index)
                + (1.0 - model["competence"][source_index]) * model["spam"][source_index][observed - 1]
            )
        probabilities.append(probability)
    posterior = normalize(probabilities)
    return {
        "method": "MACE",
        "posterior": posterior,
        "score_prediction": expected_score(posterior),
        "hard_label": hard_label(posterior),
        "status": "COMPLETE",
        "fallback": False,
        "uncertainty_entropy": normalized_entropy(posterior),
        "direction_flags": {},
    }


def crossfit_predictions(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    folds = int(config["folds"])
    assignment = assign_folds(records, folds, int(config["seed"]))
    for record in records:
        record["fold"] = assignment[str(record["sample_id"])]
    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    drga_fold_models: dict[str, Any] = {}
    for fold in range(folds):
        train_records = [record for record in records if int(record["fold"]) != fold]
        holdout_records = [record for record in records if int(record["fold"]) == fold]
        drga_model = fit_supervised_reliability(train_records, config)
        ds_model = fit_ds_model(train_records)
        mace_model = fit_mace_model(train_records)
        drga_fold_models[str(fold)] = {
            "training_rows": drga_model["training_rows"],
            "class_prior": drga_model["class_prior"],
            "source_weights": {
                name: drga_model["source_models"][name]["source_weight"] for name in SOURCE_NAMES
            },
        }
        for record in holdout_records:
            fold_rows.append(
                {
                    "fold": fold,
                    "language": record["language"],
                    "reference_point": record["reference_point"],
                    "rows": 1,
                    "design_weight": record["design_weight"],
                }
            )
            source = record["source"]
            method_predictions: dict[str, dict[str, Any] | None] = {
                method: fixed_baseline_prediction(method, source)
                for method in (
                    "rounded_human",
                    "human_median",
                    "human_mean",
                    "qwen",
                    "deepseek",
                    "teacher_mean",
                    "teacher_median",
                    "equal_weight_fusion",
                )
            }
            method_predictions["Dawid-Skene"] = predict_ds(source, ds_model)
            method_predictions["MACE"] = predict_mace(source, mace_model)
            method_predictions["DRGA"] = predict_drga(source, drga_model, config)
            for method in BASELINE_ORDER:
                prediction = method_predictions.get(method)
                if prediction is None:
                    continue
                row = {
                    "sample_id": record["sample_id"],
                    "fold": fold,
                    "method": method,
                    "reference_point": record["reference_point"],
                    "reference_posterior": record["reference_posterior"],
                    "design_weight": record["design_weight"],
                    **prediction,
                }
                predictions.append(row)
    return predictions, fold_rows, drga_fold_models


def quadratic_weighted_kappa(predictions: list[int], references: list[int], weights: list[float]) -> float:
    if not predictions:
        return float("nan")
    n = 5
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    pred_hist = [0.0] * n
    ref_hist = [0.0] * n
    total = 0.0
    for pred, ref, weight in zip(predictions, references, weights, strict=True):
        observed[ref - 1][pred - 1] += weight
        ref_hist[ref - 1] += weight
        pred_hist[pred - 1] += weight
        total += weight
    if total <= 0:
        return float("nan")
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / ((n - 1) ** 2)
            expected = ref_hist[i] * pred_hist[j] / total
            numerator += weight * observed[i][j]
            denominator += weight * expected
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - numerator / denominator


def recall_for_label(predictions: list[int], references: list[int], weights: list[float], label: int) -> float | str:
    denominator = sum(weight for reference, weight in zip(references, weights, strict=True) if reference == label)
    if denominator <= 0:
        return ""
    numerator = sum(
        weight
        for pred, reference, weight in zip(predictions, references, weights, strict=True)
        if reference == label and pred == label
    )
    return numerator / denominator


def calibration_metrics(rows: list[dict[str, Any]], bins: int) -> tuple[float, float, float]:
    total_weight = sum(float(row["weight"]) for row in rows)
    if total_weight <= 0:
        return float("nan"), float("nan"), float("nan")
    brier = 0.0
    log_loss = 0.0
    bin_stats = [{"weight": 0.0, "confidence": 0.0, "correct": 0.0} for _ in range(bins)]
    for row in rows:
        posterior = row["posterior"]
        ref = int(row["reference"])
        weight = float(row["weight"])
        brier += weight * sum(
            (posterior[label - 1] - float(label == ref)) ** 2 for label in LABELS
        )
        log_loss += weight * (-math.log(max(posterior[ref - 1], 1e-12)))
        confidence = max(posterior)
        correct = float(int(row["hard_label"]) == ref)
        bin_index = min(bins - 1, int(confidence * bins))
        bin_stats[bin_index]["weight"] += weight
        bin_stats[bin_index]["confidence"] += weight * confidence
        bin_stats[bin_index]["correct"] += weight * correct
    ece = 0.0
    for item in bin_stats:
        if item["weight"] <= 0:
            continue
        avg_conf = item["confidence"] / item["weight"]
        avg_acc = item["correct"] / item["weight"]
        ece += item["weight"] / total_weight * abs(avg_conf - avg_acc)
    return brier / total_weight, log_loss / total_weight, ece


def metric_record(method: str, predictions: list[dict[str, Any]], weighted: bool, config: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in predictions if row["method"] == method]
    if not rows:
        return {"method": method, "rows": 0, "status": "NOT_AVAILABLE"}
    weights = [float(row["design_weight"]) if weighted else 1.0 for row in rows]
    references = [int(row["reference_point"]) for row in rows]
    hard = [int(row["hard_label"]) for row in rows]
    scores = [float(row["score_prediction"]) for row in rows]
    total_weight = sum(weights)
    diffs = [score - reference for score, reference in zip(scores, references, strict=True)]
    cal_rows = [
        {
            "posterior": list(row["posterior"]),
            "reference": int(row["reference_point"]),
            "hard_label": int(row["hard_label"]),
            "weight": weight,
        }
        for row, weight in zip(rows, weights, strict=True)
    ]
    brier, log_loss, ece = calibration_metrics(cal_rows, int(config["metrics"]["ece_bins"]))
    return {
        "method": method,
        "rows": len(rows),
        "weighting": "design_weighted" if weighted else "unweighted",
        "mae": sum(weight * abs(diff) for weight, diff in zip(weights, diffs, strict=True)) / total_weight,
        "qwk": quadratic_weighted_kappa(hard, references, weights),
        "exact": sum(weight * float(pred == ref) for weight, pred, ref in zip(weights, hard, references, strict=True)) / total_weight,
        "within_one": sum(weight * float(abs(pred - ref) <= 1) for weight, pred, ref in zip(weights, hard, references, strict=True)) / total_weight,
        "signed_bias": sum(weight * diff for weight, diff in zip(weights, diffs, strict=True)) / total_weight,
        "severe_error": sum(weight * float(abs(pred - ref) >= int(config["metrics"]["severe_error_distance"])) for weight, pred, ref in zip(weights, hard, references, strict=True)) / total_weight,
        "low_to_high": sum(weight * float(ref <= 2 and pred >= 4) for weight, pred, ref in zip(weights, hard, references, strict=True)) / total_weight,
        "high_to_low": sum(weight * float(ref >= 4 and pred <= 2) for weight, pred, ref in zip(weights, hard, references, strict=True)) / total_weight,
        "label1_recall": recall_for_label(hard, references, weights, 1),
        "label2_recall": recall_for_label(hard, references, weights, 2),
        "label5_recall": recall_for_label(hard, references, weights, 5),
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "fallback_rate": sum(weight * float(row.get("fallback", False)) for weight, row in zip(weights, rows, strict=True)) / total_weight,
        "status": "COMPLETE",
    }


def metrics_table(predictions: list[dict[str, Any]], weighted: bool, config: dict[str, Any]) -> list[dict[str, Any]]:
    return [metric_record(method, predictions, weighted, config) for method in BASELINE_ORDER]


def source_reliability_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in SOURCE_NAMES:
        source_model = model["source_models"][name]
        rows.append(
            {
                "source": name,
                "rows": source_model["rows"],
                "weighted_observations": source_model["weighted_observations"],
                "exact_soft": source_model["exact_soft"],
                "within_one_soft": source_model["within_one_soft"],
                "severe_soft": source_model["severe_soft"],
                "signed_bias_soft": source_model["signed_bias_soft"],
                "reliability": source_model["reliability"],
                "source_weight": source_model["source_weight"],
            }
        )
    return rows


def fold_balance_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["fold"]), str(record["language"]), int(record["reference_point"]))
        grouped.setdefault(
            key,
            {
                "fold": key[0],
                "language": key[1],
                "reference_point": key[2],
                "rows": 0,
                "design_weight": 0.0,
            },
        )
        grouped[key]["rows"] += 1
        grouped[key]["design_weight"] += float(record["design_weight"])
    return [grouped[key] for key in sorted(grouped)]


def direction_summary_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drga = [row for row in predictions if row["method"] == "DRGA"]
    fields = (
        "human4_to_label5",
        "low_to_high",
        "high_to_low",
        "score_cap_exceeded",
        "teacher_low_to_high_signal",
        "teacher_high_to_low_signal",
    )
    output = []
    total_weight = sum(float(row["design_weight"]) for row in drga)
    for field in fields:
        value = sum(
            float(row["design_weight"]) * float(bool(row.get("direction_flags", {}).get(field)))
            for row in drga
        )
        output.append(
            {
                "flag": field,
                "rows": sum(bool(row.get("direction_flags", {}).get(field)) for row in drga),
                "design_weighted_rate": value / total_weight if total_weight else "",
                "status": "COMPLETE" if drga else "NOT_AVAILABLE",
            }
        )
    output.append(
        {
            "flag": "fallback",
            "rows": sum(bool(row.get("fallback")) for row in drga),
            "design_weighted_rate": (
                sum(float(row["design_weight"]) * float(bool(row.get("fallback"))) for row in drga) / total_weight
                if total_weight
                else ""
            ),
            "status": "COMPLETE" if drga else "NOT_AVAILABLE",
        }
    )
    return output


def quality_gates(metrics: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {str(row["method"]): row for row in metrics}
    rounded = by_method["rounded_human"]
    drga = by_method["DRGA"]
    eps = 1e-12
    mature_complete = [
        row
        for row in metrics
        if row.get("method") in MATURE_BASELINES and row.get("status") == "COMPLETE" and row.get("mae") != ""
    ]
    best_mature = min(mature_complete, key=lambda row: (float(row["mae"]), -float(row["qwk"]))) if mature_complete else None
    posterior_valid = all(
        len(row["posterior"]) == 5
        and all(value >= -1e-12 for value in row["posterior"])
        and abs(sum(row["posterior"]) - 1.0) <= 1e-8
        for row in predictions
    )
    fallback_recomputable = True
    for row in predictions:
        if row["method"] != "DRGA" or not row.get("fallback"):
            continue
        empirical = human_empirical_distribution(row["source"]) if "source" in row else None
        if empirical is not None and any(abs(a - b) > 1e-8 for a, b in zip(empirical, row["posterior"], strict=True)):
            fallback_recomputable = False
            break
    gates = {
        "drga_not_worse_than_rounded_human_mae": float(drga["mae"]) <= float(rounded["mae"]) + eps,
        "drga_not_worse_than_rounded_human_qwk": float(drga["qwk"]) >= float(rounded["qwk"]) - eps,
        "drga_not_worse_low_to_high": float(drga["low_to_high"]) <= float(rounded["low_to_high"]) + eps,
        "drga_not_worse_high_to_low": float(drga["high_to_low"]) <= float(rounded["high_to_low"]) + eps,
        "posterior_valid": posterior_valid,
        "fallback_recomputable": fallback_recomputable,
        "no_dev_or_test_access": True,
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "rounded_human": {
            "mae": rounded["mae"],
            "qwk": rounded["qwk"],
            "low_to_high": rounded["low_to_high"],
            "high_to_low": rounded["high_to_low"],
        },
        "drga": {
            "mae": drga["mae"],
            "qwk": drga["qwk"],
            "low_to_high": drga["low_to_high"],
            "high_to_low": drga["high_to_low"],
        },
        "best_mature_baseline": best_mature,
        "drga_vs_best_mature": (
            {
                "best_mature_method": best_mature["method"],
                "mae_delta_drga_minus_best": float(drga["mae"]) - float(best_mature["mae"]),
                "qwk_delta_drga_minus_best": float(drga["qwk"]) - float(best_mature["qwk"]),
            }
            if best_mature
            else None
        ),
    }


def private_prediction_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in predictions:
        posterior = list(row["posterior"])
        output = {
            "sample_id": row["sample_id"],
            "fold": row["fold"],
            "method": row["method"],
            "reference_point": row["reference_point"],
            "design_weight": row["design_weight"],
            "score_prediction": row["score_prediction"],
            "hard_label": row["hard_label"],
            "fallback": row.get("fallback", False),
            "uncertainty_entropy": row.get("uncertainty_entropy", ""),
            "status": row.get("status", ""),
            "p1": posterior[0],
            "p2": posterior[1],
            "p3": posterior[2],
            "p4": posterior[3],
            "p5": posterior[4],
            "direction_flags": json.dumps(row.get("direction_flags", {}), sort_keys=True),
        }
        rows.append(output)
    return rows


def source_provenance(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "human_scores_present": len(human_scores(source)),
        "rounded_human_label": source.get("rounded_human_label"),
        "qwen": {
            "score": source.get("qwen_score"),
            "confidence": source.get("qwen_confidence"),
            "evidence_flags": source.get("qwen_evidence_flags") or [],
            "score_cap": source.get("qwen_score_cap"),
            "reason_present": bool(source.get("qwen_reason_present")),
            "rubric_assessment_items": source.get("qwen_rubric_assessment_items"),
        },
        "deepseek": {
            "score": source.get("deepseek_score"),
            "confidence": source.get("deepseek_confidence"),
            "evidence_flags": source.get("deepseek_evidence_flags") or [],
            "score_cap": source.get("deepseek_score_cap"),
            "reason_present": bool(source.get("deepseek_reason_present")),
            "rubric_assessment_items": source.get("deepseek_rubric_assessment_items"),
        },
        "teacher_direction": source.get("teacher_direction"),
        "campaign_transition_type": source.get("campaign_transition_type"),
    }


def build_full_train_supervision(
    sources: dict[str, dict[str, Any]],
    model: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    minimum = float(config["supervision"]["min_sample_weight"])
    power = float(config["supervision"]["uncertainty_weight_power"])
    for sid, source in sorted(sources.items()):
        prediction = predict_drga(source, model, config)
        posterior = prediction["posterior"]
        uncertainty = normalized_entropy(posterior)
        sample_weight = minimum + (1.0 - minimum) * ((1.0 - uncertainty) ** power)
        flags = prediction["direction_flags"]
        rows.append(
            {
                "sample_id": sid,
                "hard_label": prediction["hard_label"],
                "fallback_label": human_empirical_hard(source) if prediction["fallback"] else "",
                "sample_weight": sample_weight,
                "uncertainty_entropy": uncertainty,
                "aggregation_status": prediction["status"],
                "p1": posterior[0],
                "p2": posterior[1],
                "p3": posterior[2],
                "p4": posterior[3],
                "p5": posterior[4],
                "source_provenance": json.dumps(source_provenance(source), ensure_ascii=False, sort_keys=True),
                "direction_flags": json.dumps(flags, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def public_supervision_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return [{"status": "NOT_GENERATED_GATE_FAILED", "rows": 0}]
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        status = str(row["aggregation_status"])
        label = str(row["hard_label"])
        key = (status, label)
        grouped.setdefault(
            key,
            {
                "aggregation_status": status,
                "hard_label": label,
                "rows": 0,
                "avg_sample_weight": 0.0,
                "avg_uncertainty_entropy": 0.0,
            },
        )
        grouped[key]["rows"] += 1
        grouped[key]["avg_sample_weight"] += float(row["sample_weight"])
        grouped[key]["avg_uncertainty_entropy"] += float(row["uncertainty_entropy"])
    output = []
    for key in sorted(grouped):
        item = grouped[key]
        count = item["rows"]
        item["avg_sample_weight"] /= count
        item["avg_uncertainty_entropy"] /= count
        output.append(item)
    return output


def risk_stress_metrics(
    risk_ids: set[str],
    sources: dict[str, dict[str, Any]],
    exp33a_out: Path,
    final_model: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    risk_final, risk_status = load_final_silver_reference(exp33a_out, risk_ids, require_complete=False)
    if risk_status.get("status") != "COMPLETE" or not risk_final:
        return [
            {
                "view": "risk_enriched_train",
                "weighting": "unweighted",
                "method": method,
                "rows": 0,
                "status": "PENDING_RISK_SILVER",
            }
            for method in BASELINE_ORDER
        ], risk_status
    records = []
    for sid, reference in sorted(risk_final.items()):
        source = sources[sid]
        records.append(
            {
                "sample_id": sid,
                "reference_point": int(reference["point_score"]),
                "reference_posterior": reference["posterior"],
                "design_weight": 1.0,
                "source": source,
            }
        )
    predictions = []
    ds_model = fit_ds_model(records)
    mace_model = fit_mace_model(records)
    for record in records:
        source = record["source"]
        method_predictions: dict[str, dict[str, Any] | None] = {
            method: fixed_baseline_prediction(method, source)
            for method in (
                "rounded_human",
                "human_median",
                "human_mean",
                "qwen",
                "deepseek",
                "teacher_mean",
                "teacher_median",
                "equal_weight_fusion",
            )
        }
        method_predictions["Dawid-Skene"] = predict_ds(source, ds_model)
        method_predictions["MACE"] = predict_mace(source, mace_model)
        method_predictions["DRGA"] = predict_drga(source, final_model, config)
        for method, prediction in method_predictions.items():
            if prediction is None:
                continue
            predictions.append(
                {
                    "sample_id": record["sample_id"],
                    "method": method,
                    "reference_point": record["reference_point"],
                    "design_weight": 1.0,
                    **prediction,
                }
            )
    rows = metrics_table(predictions, False, config)
    for row in rows:
        row["view"] = "risk_enriched_train"
        row["weighting"] = "unweighted"
    return rows, risk_status


def hash_public_and_private(out_dir: Path, private_paths: list[Path]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"private": {}, "public": {}}
    for path in private_paths:
        absolute = repo_path(path)
        if absolute.is_file():
            artifacts["private"][display_path(absolute)] = sha256_file(absolute)
    for folder in ("tables", "reports", "decision", "hashes", "configs"):
        for path in sorted((repo_path(out_dir) / folder).glob("**/*")):
            if not path.is_file():
                continue
            if path.name == "exp33b_artifact_hashes.json":
                continue
            if path.name.startswith("exp33b_validation_"):
                continue
            if path.name in {"exp33b_validation_report.md", "exp33b_validation_checks.csv"}:
                continue
            else:
                artifacts["public"][display_path(path)] = sha256_file(path)
    return artifacts


def write_report(
    out_dir: Path,
    metrics: list[dict[str, Any]],
    gate_payload: dict[str, Any],
    risk_status: dict[str, Any],
    supervision_rows: int,
) -> None:
    by_method = {row["method"]: row for row in metrics}
    drga = by_method["DRGA"]
    rounded = by_method["rounded_human"]
    best = gate_payload.get("best_mature_baseline") or {}
    report = f"""# Exp33B Direction-Aware Rubric-Grounded Aggregation

- Reference source: Exp33A `representative_train` model-reviewed silver posterior.
- Calibration rows: 120; cross-fitting: 5 folds stratified by silver point label and language.
- Paper protocol preserved: train/dev/test remain the locked 2654/664/2218 triple-key-isolated split.
- Dev/test access: none.
- Training/API/GPU/student inference: none.
- Risk view stress test: {risk_status.get("status", "PENDING")}; risk view is not used as prevalence.
- Full train supervision generated: {supervision_rows} rows.

## DRGA formula

`p(y|x) proportional H_x(y)^alpha * pi(y)^beta * product_s C_s[y, obs_s]^w_s * D_y(x)`

`H_x` is the human three-score empirical distribution, `pi` is the fold training
silver prior, `C_s` is the ordinal source confusion matrix fitted from
model-reviewed silver posterior, and `D_y` applies preregistered direction and
score-cap penalties. High-entropy or low-confidence rows fall back exactly to
the human empirical distribution.

## Cross-fit headline

| method | rows | MAE | QWK | low_to_high | high_to_low | fallback_rate |
|---|---:|---:|---:|---:|---:|---:|
| rounded_human | {rounded.get("rows")} | {rounded.get("mae")} | {rounded.get("qwk")} | {rounded.get("low_to_high")} | {rounded.get("high_to_low")} | {rounded.get("fallback_rate")} |
| DRGA | {drga.get("rows")} | {drga.get("mae")} | {drga.get("qwk")} | {drga.get("low_to_high")} | {drga.get("high_to_low")} | {drga.get("fallback_rate")} |

Best mature aggregation baseline by MAE: `{best.get("method", "NA")}` with MAE
`{best.get("mae", "NA")}` and QWK `{best.get("qwk", "NA")}`. DRGA is not
declared superior unless the table supports that claim.

## Gate decision

- Passed: {gate_payload["passed"]}.
- Gates: `{json.dumps(gate_payload["gates"], sort_keys=True)}`.

This is a model-reviewed silver aggregation experiment, not human expert gold.
"""
    write_text(out_dir / "reports/exp33b_direction_aware_aggregation_report.md", report)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    out_dir = args.out_dir
    repo_path(out_dir).mkdir(parents=True, exist_ok=True)
    for folder in ("configs", "tables", "reports", "decision", "hashes", "private"):
        repo_path(out_dir / folder).mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "configs/exp33b_drga_preregistration_config.json", config)

    teacher_manifest, teacher_resolved = resolve_teacher_inputs(args.teacher_summary_dir, args.exp28e_decision)
    selected_train = load_selected_train_views(args.exp33a_out_dir)
    representative_ids = {sid for sid, row in selected_train.items() if row.get("view") == "representative_train"}
    risk_ids = {sid for sid, row in selected_train.items() if row.get("view") == "risk_enriched_train"}
    if len(representative_ids) != 120:
        raise ValueError(f"Expected 120 representative train IDs, got {len(representative_ids)}")

    sources = load_train_sources(args.split_dir, selected_train, teacher_resolved)
    final, final_status = load_final_silver_reference(args.exp33a_out_dir, representative_ids, require_complete=True)
    records = make_calibration_records(final, sources)
    predictions, _fold_rows, fold_models = crossfit_predictions(records, config)

    # Keep source in private prediction rows for fallback validation, but never
    # write raw source text/reasons into public artifacts.
    by_source = {record["sample_id"]: record["source"] for record in records}
    for row in predictions:
        row["source"] = by_source[row["sample_id"]]

    crossfit_metrics = metrics_table(predictions, True, config)
    gate_payload = quality_gates(crossfit_metrics, predictions)
    final_model = fit_supervised_reliability(records, config)
    risk_rows, risk_status = risk_stress_metrics(risk_ids, sources, args.exp33a_out_dir, final_model, config)

    supervision: list[dict[str, Any]] = []
    supervision_path = out_dir / "private/exp33b_train_supervision.csv"
    if gate_payload["passed"] and not args.no_private_supervision:
        supervision = build_full_train_supervision(sources, final_model, config)
        write_csv(
            supervision_path,
            supervision,
            [
                "sample_id",
                "hard_label",
                "fallback_label",
                "sample_weight",
                "uncertainty_entropy",
                "aggregation_status",
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "source_provenance",
                "direction_flags",
            ],
        )
    elif repo_path(supervision_path).exists():
        # A failed gate must not leave a stale train-supervision artifact from
        # an earlier exploratory run that happened to pass.
        repo_path(supervision_path).unlink()

    private_predictions_path = out_dir / "private/exp33b_representative_crossfit_predictions.csv"
    write_csv(
        private_predictions_path,
        private_prediction_rows(predictions),
        [
            "sample_id",
            "fold",
            "method",
            "reference_point",
            "design_weight",
            "score_prediction",
            "hard_label",
            "fallback",
            "uncertainty_entropy",
            "status",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "direction_flags",
        ],
    )

    write_csv(
        out_dir / "tables/exp33b_crossfit_metrics.csv",
        crossfit_metrics,
        [
            "method",
            "rows",
            "weighting",
            "mae",
            "qwk",
            "exact",
            "within_one",
            "signed_bias",
            "severe_error",
            "low_to_high",
            "high_to_low",
            "label1_recall",
            "label2_recall",
            "label5_recall",
            "brier",
            "log_loss",
            "ece",
            "fallback_rate",
            "status",
        ],
    )
    write_csv(
        out_dir / "tables/exp33b_source_reliability.csv",
        source_reliability_rows(final_model),
        [
            "source",
            "rows",
            "weighted_observations",
            "exact_soft",
            "within_one_soft",
            "severe_soft",
            "signed_bias_soft",
            "reliability",
            "source_weight",
        ],
    )
    write_csv(
        out_dir / "tables/exp33b_fold_balance.csv",
        fold_balance_rows(records),
        ["fold", "language", "reference_point", "rows", "design_weight"],
    )
    write_csv(
        out_dir / "tables/exp33b_direction_flags.csv",
        direction_summary_rows(predictions),
        ["flag", "rows", "design_weighted_rate", "status"],
    )
    write_csv(
        out_dir / "tables/exp33b_risk_stress_metrics.csv",
        risk_rows,
        [
            "view",
            "weighting",
            "method",
            "rows",
            "mae",
            "qwk",
            "exact",
            "within_one",
            "signed_bias",
            "severe_error",
            "low_to_high",
            "high_to_low",
            "label1_recall",
            "label2_recall",
            "label5_recall",
            "brier",
            "log_loss",
            "ece",
            "fallback_rate",
            "status",
        ],
    )
    write_csv(
        out_dir / "tables/exp33b_train_supervision_public_summary.csv",
        public_supervision_summary(supervision),
        [
            "aggregation_status",
            "hard_label",
            "rows",
            "avg_sample_weight",
            "avg_uncertainty_entropy",
            "status",
        ],
    )

    input_hashes = {
        "train_split": {
            "path": display_path(repo_path(args.split_dir / "train.jsonl")),
            "sha256": sha256_file(args.split_dir / "train.jsonl"),
        },
        "exp33a_reviewer_a": {
            "path": display_path(repo_path(args.exp33a_out_dir / "private_review/reviewer_filled/exp33a_reviewer_a_results.jsonl")),
            "sha256": sha256_file(args.exp33a_out_dir / "private_review/reviewer_filled/exp33a_reviewer_a_results.jsonl"),
        },
        "exp33a_reviewer_b": {
            "path": display_path(repo_path(args.exp33a_out_dir / "private_review/reviewer_filled/exp33a_reviewer_b_results.jsonl")),
            "sha256": sha256_file(args.exp33a_out_dir / "private_review/reviewer_filled/exp33a_reviewer_b_results.jsonl"),
        },
        "exp33a_adjudicator": {
            "path": display_path(repo_path(args.exp33a_out_dir / "private_review/adjudication_filled/exp33a_adjudicator_results.jsonl")),
            "sha256": sha256_file(args.exp33a_out_dir / "private_review/adjudication_filled/exp33a_adjudicator_results.jsonl"),
        },
        "teachers": teacher_manifest,
    }
    write_json(out_dir / "hashes/exp33b_input_hashes.json", input_hashes)

    private_paths = [private_predictions_path]
    if supervision:
        private_paths.append(supervision_path)
    private_hashes = {
        display_path(repo_path(path)): sha256_file(path)
        for path in private_paths
        if repo_path(path).is_file()
    }

    decision = {
        "experiment": "Exp33B Direction-Aware Rubric-Grounded Aggregation",
        "method": "Direction-Aware Rubric-Grounded Aggregation",
        "method_short_name": "DRGA",
        "reference_status": "model-reviewed silver, not human expert gold",
        "paper_protocol": "train=2654/dev=664/test=2218; triple-key-disjoint, not question-key-excluded",
        "representative_silver_status": final_status,
        "risk_silver_status": risk_status,
        "risk_view_used_for_prevalence": False,
        "crossfit_folds": int(config["folds"]),
        "crossfit_stratification": config["fold_stratification"],
        "calibration_rows": len(records),
        "quality_gate_passed": gate_payload["passed"],
        "quality_gates": gate_payload["gates"],
        "quality_gate_details": gate_payload,
        "full_train_supervision_generated": bool(supervision),
        "full_train_supervision_rows": len(supervision),
        "future_train_rows_retained": 2654,
        "dev_rows_read": 0,
        "test_access_count": 0,
        "clean_dev_read": False,
        "api_called": False,
        "gpu_used": False,
        "training_run": False,
        "student_inference_run": False,
        "downstream_student_metric_tuning": False,
        "private_supervision_sha256": private_hashes.get(display_path(repo_path(supervision_path))),
        "private_crossfit_predictions_sha256": private_hashes.get(display_path(repo_path(private_predictions_path))),
        "fold_models": fold_models,
    }
    write_json(out_dir / "decision/exp33b_direction_aware_aggregation_decision.json", decision)
    write_report(out_dir, crossfit_metrics, gate_payload, risk_status, len(supervision))
    artifact_hashes = hash_public_and_private(out_dir, private_paths)
    write_json(out_dir / "hashes/exp33b_artifact_hashes.json", artifact_hashes)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--exp33a-out-dir", type=Path, default=DEFAULT_EXP33A_OUT)
    parser.add_argument("--teacher-summary-dir", type=Path, default=DEFAULT_TEACHER_SUMMARY_DIR)
    parser.add_argument("--exp28e-decision", type=Path, default=DEFAULT_EXP28E_DECISION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-private-supervision", action="store_true")
    return parser.parse_args()


def main() -> None:
    decision = prepare(parse_args())
    print(
        json.dumps(
            {
                "status": "PASS" if decision["quality_gate_passed"] else "GATE_FAILED",
                "full_train_supervision_generated": decision["full_train_supervision_generated"],
                "full_train_supervision_rows": decision["full_train_supervision_rows"],
                "dev_rows_read": decision["dev_rows_read"],
                "test_access_count": decision["test_access_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
