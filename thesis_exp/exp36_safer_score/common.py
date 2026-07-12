"""Shared constants and helpers for the locked Exp36A workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis_exp.src.edujudge.exp03.templates import make_prompt
from thesis_exp.src.edujudge.exp27p.common import prediction_metrics


ROOT = Path("thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42")
PRIVATE = ROOT / "private"
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
DEV_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl")
QWEN_PATH = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/"
    "private/qwen/p0_holistic_zero_shot/all_train.jsonl"
)
DEEPSEEK_PATH = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/"
    "private/deepseek/p0_holistic_zero_shot/secondary_route.jsonl"
)
LABELS = (1, 2, 3, 4, 5)
VARIANTS = (
    "v0_original_hard",
    "v0h_human_soft",
    "v1_qwen_hard",
    "v2_qwen_range_soft",
    "v3_naive_human_qwen",
    "v4_human_soft_logit_adjustment",
    "v5_safer_score",
    "v7_shuffled_teacher_control",
    "v6a_no_failure_aux",
    "v6b_no_student_uncertainty",
    "v6c_no_curriculum",
)
FAILURE_CLASSES = (
    "no_major_failure",
    "missing_content_or_key_point",
    "factual_or_rubric_mismatch",
    "insufficient_reasoning_or_evidence",
    "task_constraint_or_format_violation",
    "unclear_or_other",
)
FAILURE_MAP = {
    "no_major_failure": 0,
    "missing_key_point": 1,
    "missing_content": 1,
    "factual_or_rubric_mismatch": 2,
    "insufficient_evidence": 3,
    "reasoning_gap": 3,
    "task_constraint_violation": 4,
    "format_violation": 4,
    "unclear": 5,
}


def forbid_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Exp36A forbids test access: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    forbid_test_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_model_text(row: dict[str, Any]) -> str:
    """Locked paper input: question + answer + evaluation dimension only."""
    return make_prompt(
        {
            "question": row.get("question"),
            "answer": row.get("answer"),
            "metric_canonical": row.get("metric_canonical"),
        },
        "A2_question_answer_metric",
    )


def normalized_entropy(values: Iterable[float], classes: int = 5) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[arr > 0]
    return float(-(arr * np.log(arr)).sum() / math.log(classes)) if len(arr) else 0.0


def human_distribution(row: dict[str, Any]) -> list[float]:
    scores = [int(round(float(row[f"human_{index}"]))) for index in (1, 2, 3)]
    return [scores.count(label) / 3.0 for label in LABELS]


def one_hot(score: int) -> list[float]:
    return [1.0 if label == int(score) else 0.0 for label in LABELS]


def qwen_distribution(annotation: dict[str, Any]) -> tuple[list[float], str]:
    score = int(annotation["score"])
    value = annotation.get("plausible_score_range") or annotation.get("score_range")
    if isinstance(value, dict):
        lower, upper = int(value.get("lower", score)), int(value.get("upper", score))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = int(value[0]), int(value[1])
    else:
        return one_hot(score), "hard_score_no_range"
    lower, upper = max(1, lower), min(5, upper)
    mass = [math.exp(-abs(label - score)) if lower <= label <= upper else 0.0 for label in LABELS]
    total = sum(mass)
    return [value / total for value in mass], "plausible_score_range"


def coarse_failures(annotation: dict[str, Any]) -> set[int]:
    return {FAILURE_MAP.get(str(value), 5) for value in (annotation.get("major_failures") or ["unclear"])}


def failure_vector(annotation: dict[str, Any]) -> list[float]:
    classes = coarse_failures(annotation)
    return [1.0 if index in classes else 0.0 for index in range(len(FAILURE_CLASSES))]


def deepseek_failure_confirms(qwen: dict[str, Any], deepseek: dict[str, Any] | None) -> bool:
    return deepseek is not None and bool(coarse_failures(qwen) & coarse_failures(deepseek))


def evidence_gate(raw: dict[str, Any], expected_id: str) -> tuple[bool, list[str]]:
    annotation = raw.get("annotation") or {}
    reasons: list[str] = []
    if raw.get("schema_errors"):
        reasons.append("schema_errors")
    if str(raw.get("sample_id")) != expected_id or str(annotation.get("sample_id")) != expected_id:
        reasons.append("target_scope_invalid")
    if not str(annotation.get("reason") or "").strip():
        reasons.append("reason_missing")
    assessments = annotation.get("rubric_assessment") or []
    if not assessments:
        reasons.append("rubric_finding_missing")
    for item in assessments:
        verdict = str(item.get("verdict") or "")
        evidence = str(item.get("evidence") or "").strip()
        if not evidence and verdict != "not_applicable":
            reasons.append("evidence_invalid")
            break
    score = int(annotation.get("score") or 0)
    failures = set(annotation.get("major_failures") or [])
    if not 1 <= score <= 5:
        reasons.append("score_invalid")
    if failures == {"no_major_failure"} and score <= 2:
        reasons.append("score_reason_inconsistent")
    cap = annotation.get("score_cap")
    if cap is not None and score > int(cap):
        reasons.append("score_cap_inconsistent")
    text = " ".join(str(value).lower() for value in annotation.get("major_failures") or [])
    if "answer_key" in text or "reference_uncertain" in text:
        reasons.append("answer_key_uncertainty")
    return not reasons, sorted(set(reasons))


def direction_factor(human: int, qwen: int, deepseek: dict[str, Any] | None, qwen_ann: dict[str, Any]) -> tuple[float, str]:
    score_confirmed = deepseek is not None and abs(qwen - int(deepseek["score"])) <= 1
    failure_confirmed = deepseek_failure_confirms(qwen_ann, deepseek)
    if human <= 2 and qwen >= 4:
        return 0.0, "human_low_qwen_high_blocked"
    if human <= 2 and qwen == 3:
        return 0.5, "human_low_qwen_mid_limited"
    if human == 4 and qwen == 5:
        return 0.25, "human4_qwen5_limited"
    if human >= 4 and qwen <= 2:
        return (0.5, "extreme_confirmed") if score_confirmed and failure_confirmed else (0.0, "extreme_unconfirmed")
    if abs(qwen - human) >= 2:
        return (0.25, "large_gap_confirmed") if score_confirmed else (0.0, "large_gap_unconfirmed")
    return 1.0, "adjacent_or_equal"


def curriculum_rho(epoch: int) -> float:
    if epoch <= 3:
        return 0.0
    if epoch == 4:
        return 1.0 / 3.0
    if epoch == 5:
        return 2.0 / 3.0
    return 1.0


def dynamic_target(row: dict[str, Any], epoch: int, variant: str) -> list[float]:
    human = np.asarray(row["human_target_5"], dtype=float)
    teacher = np.asarray(row["teacher_target_5"], dtype=float)
    lam = float(row.get("teacher_lambda", 0.0))
    if variant == "v6b_no_student_uncertainty":
        lam = float(row.get("teacher_lambda_no_student_uncertainty", 0.0))
    rho = 1.0 if variant == "v6c_no_curriculum" else curriculum_rho(epoch)
    target = (1.0 - rho * lam) * human + rho * lam * teacher
    return target.tolist()


def seeded_shuffle(values: list[Any], seed: int) -> list[Any]:
    copied = list(values)
    random.Random(seed).shuffle(copied)
    return copied


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return prediction_metrics(rows)

