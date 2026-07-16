"""Shared protocol and deterministic scoring utilities for Exp48A."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "thesis_exp/exp48_eduq_tail"
OUT = MODULE / "outputs/exp48a_qualification_pilot"
TRAIN = ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
PRIVATE = OUT / "private"

ALLOWED_STATES = {"satisfied", "partial", "violated"}
VERIFIER_STATES = ALLOWED_STATES | {"unclear"}
TARGET_SCORES = (2, 3, 5)
SEED = 48

PUBLIC_DIRS = ("configs", "tables", "reports", "decision", "hashes", "prompts", "schemas")
PRIVATE_DIRS = (
    "private/source_packets", "private/generated_families", "private/verifier_packets",
    "private/verifier_a", "private/verifier_b", "private/adjudication",
    "private/final_silver", "logs_private",
)


def ensure_layout() -> None:
    ensure_output_layout(OUT)


def ensure_output_layout(out: Path) -> None:
    for name in PUBLIC_DIRS + PRIVATE_DIRS:
        (out / name).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    latin = re.findall(r"[a-z0-9]+", normalized)
    han = re.findall(r"[\u4e00-\u9fff]", normalized)
    return set(latin + han)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = re.sub(r"\s+", "", normalize_text(text))
    return {normalized[i:i+n] for i in range(max(0, len(normalized) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def criterion_map(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in family["criteria"]}


def apply_score_program(family: dict[str, Any], states: dict[str, str]) -> int | None:
    """Apply the locked, non-holistic EduQ-TAIL v1 score program."""
    if set(states.values()) - VERIFIER_STATES or "unclear" in states.values():
        return None
    program = family["score_program"]
    criteria = criterion_map(family)
    required = list(program["score5_required_essential_ids"])
    supporting = [key for key, value in criteria.items() if value["type"] == "supporting"]
    prohibited = list(program.get("prohibited_ids", []))
    major = list(program["score2_major_omission_ids"])

    if any(states.get(key) == "violated" for key in prohibited):
        return 2
    essential_nonfailed = sum(states.get(key) in {"satisfied", "partial"} for key in required)
    if any(states.get(key) == "violated" for key in major) and essential_nonfailed >= 1:
        return 2
    supporting_satisfied = sum(states.get(key) == "satisfied" for key in supporting)
    if (
        all(states.get(key) == "satisfied" for key in required)
        and supporting_satisfied >= int(program["score5_min_supporting_satisfied"])
    ):
        return 5
    return 3


def validate_family(family: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {"family_id", "synthetic_question_key", "metric", "language", "synthetic_question", "criteria", "score_program", "answers"}
    missing = sorted(required - set(family))
    if missing:
        return [f"missing_fields:{','.join(missing)}"]
    criteria = family.get("criteria", [])
    if not 4 <= len(criteria) <= 6:
        errors.append("criteria_count_not_4_to_6")
    ids = [str(item.get("id", "")) for item in criteria]
    if len(ids) != len(set(ids)) or any(not key for key in ids):
        errors.append("criterion_ids_invalid")
    if any(item.get("type") not in {"essential", "supporting", "prohibited"} for item in criteria):
        errors.append("criterion_type_invalid")
    if sum(item.get("type") == "essential" for item in criteria) < 2:
        errors.append("fewer_than_two_essential_criteria")
    program = family.get("score_program", {})
    if program.get("version") != "eduq_tail_v1":
        errors.append("score_program_version_invalid")
    referenced = set(program.get("score5_required_essential_ids", [])) | set(program.get("score2_major_omission_ids", [])) | set(program.get("prohibited_ids", []))
    if referenced - set(ids):
        errors.append("score_program_references_unknown_criterion")
    types = {str(item.get("id")): item.get("type") for item in criteria}
    if any(types.get(key) != "essential" for key in program.get("score5_required_essential_ids", [])):
        errors.append("score5_required_ids_must_be_essential")
    if any(types.get(key) != "essential" for key in program.get("score2_major_omission_ids", [])):
        errors.append("score2_major_omission_ids_must_be_essential")
    if any(types.get(key) != "prohibited" for key in program.get("prohibited_ids", [])):
        errors.append("prohibited_ids_must_reference_prohibited_criteria")
    supporting_count = sum(value == "supporting" for value in types.values())
    if not 0 <= int(program.get("score5_min_supporting_satisfied", -1)) <= supporting_count:
        errors.append("score5_supporting_threshold_invalid")
    answers = family.get("answers", [])
    answer_ids = [str(answer.get("answer_id", "")) for answer in answers]
    if len(answer_ids) != len(set(answer_ids)) or any(not value for value in answer_ids):
        errors.append("answer_ids_invalid")
    if sorted(answer.get("intended_score") for answer in answers) != list(TARGET_SCORES):
        errors.append("answers_must_cover_2_3_5_once")
    texts = [normalize_text(answer.get("text", "")) for answer in answers]
    if len(texts) != len(set(texts)) or any(not text for text in texts):
        errors.append("answer_text_duplicate_or_empty")
    lengths = [max(1, len(text)) for text in texts]
    if lengths and max(lengths) / min(lengths) > 1.5:
        errors.append(f"answer_length_ratio_above_1p5:ratio={max(lengths)/min(lengths):.3f}:lengths={lengths}")
    for answer in answers:
        states = answer.get("intended_criterion_states", {})
        if set(states) != set(ids) or set(states.values()) - ALLOWED_STATES:
            errors.append(f"invalid_intended_states:{answer.get('answer_id','unknown')}")
            continue
        if apply_score_program(family, states) != answer.get("intended_score"):
            errors.append(f"intended_states_score_mismatch:{answer.get('answer_id','unknown')}")
        text = normalize_text(answer.get("text", ""))
        if re.search(r"(?:target|intended|目标|预期)\s*(?:score|分数|得分)?\s*[:=]?\s*[235]\b", text):
            errors.append(f"target_score_leakage:{answer.get('answer_id','unknown')}")
    return sorted(set(errors))


def verifier_state_map(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for answer in row.get("answers", []):
        output[str(answer["anonymous_answer_id"])] = {
            str(item["criterion_id"]): str(item["status"]) for item in answer.get("criteria", [])
        }
    return output


def counts(rows: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in rows).items()))
