"""Shared, train-only helpers for Exp37A.

This module deliberately has no model-training or API code.  It also rejects
the paper-like dev/test split paths so an accidental evaluation-set read fails
early.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("thesis_exp/exp37_failure_evidence_qualification/outputs/exp37a_failure_evidence_qualification_seed42")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
REASON_PATHS = (
    Path("5-grades/5_human_1.jsonl"),
    Path("5-grades/5_human_2.jsonl"),
    Path("5-grades/5_human_3.jsonl"),
    Path("5-grades/5_merge_human_metric_en.jsonl"),
    Path("5-grades/5_merge_human_metric_zh.jsonl"),
)
QWEN_MANIFEST = Path(
    "thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/tables/"
    "exp36a_resolved_teacher_input_manifest.csv"
)
V5_PATH = Path(
    "thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/private/data/"
    "exp36a_v5_safer_score_train.jsonl"
)
V7_PATH = Path(
    "thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/private/data/"
    "exp36a_v7_shuffled_teacher_control_train.jsonl"
)
OOF_PATH = Path(
    "thesis_exp/exp36_safer_score/outputs/exp36a_safer_score_seed42/private/"
    "exp36a_oof_human_baseline_predictions.jsonl"
)
FAILURE_CLASSES = (
    "no_major_failure",
    "missing_content_or_key_point",
    "factual_or_rubric_mismatch",
    "insufficient_reasoning_or_evidence",
    "task_constraint_or_format_violation",
    "unclear_or_other",
)


def reject_eval_path(path: Path) -> None:
    """Reject explicit dev/test split files while allowing ordinary words."""
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    name = path.name.lower()
    if name in {"dev.jsonl", "test.jsonl", "dev.json", "test.json"}:
        raise ValueError(f"Exp37A is train-only and forbids: {path}")
    if "/dev/" in normalized or "/test/" in normalized:
        raise ValueError(f"Exp37A is train-only and forbids: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    reject_eval_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value = unicodedata.normalize("NFKC", value).replace("\\n", "\n")
    return " ".join(value.split()).strip()


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id") or row.get("record_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample/record id")
    return str(value)


def question_key(row: dict[str, Any]) -> str:
    return str(row.get("question_key") or row.get("question_id") or "")


def sha256_text(value: Any) -> str:
    return hashlib.sha256(norm(value).encode("utf-8")).hexdigest()


def packet_hash(packet: dict[str, Any]) -> str:
    payload = {key: value for key, value in packet.items() if key != "packet_hash" and not str(key).startswith("_")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, str]:
    return (norm(row.get("language")), norm(row.get("metric_group") or row.get("metric_canonical")))


def subject(row: dict[str, Any]) -> str:
    return norm(row.get("subject_canonical") or row.get("subject_raw"))


def largest_remainder_quota(counts: dict[Any, int], total: int) -> dict[Any, int]:
    if not counts or total <= 0:
        return {key: 0 for key in counts}
    total_available = sum(counts.values())
    raw = {key: total * value / total_available for key, value in counts.items()}
    quota = {key: min(counts[key], int(math.floor(value))) for key, value in raw.items()}
    remainder = total - sum(quota.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - math.floor(raw[key])), str(key)))
    for key in order:
        if remainder <= 0:
            break
        if quota[key] < counts[key]:
            quota[key] += 1
            remainder -= 1
    return quota


def stratified_sample(rows: list[dict[str, Any]], total: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(stratum(row), subject(row))].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    quotas = largest_remainder_quota({key: len(values) for key, values in groups.items()}, total)
    chosen = [row for key in sorted(groups, key=str) for row in groups[key][:quotas[key]]]
    return sorted(chosen, key=sample_id)


def load_reason_rows(paths: Iterable[Path] = REASON_PATHS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            row = dict(row)
            row["_source_file"] = path.name
            rows.append(row)
    return rows


def reason_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return only exact keys; no fuzzy or substring matching is performed."""
    question = norm(row.get("question"))
    answer = norm(row.get("answer") or row.get("response"))
    metric = norm(row.get("metric") or row.get("metric_canonical") or row.get("principle"))
    model = norm(row.get("generator_model") or row.get("answer_model") or row.get("model"))
    return [(question, answer, metric, model), (question, answer, metric, "")]


def train_reason_keys(row: dict[str, Any]) -> list[tuple[str, ...]]:
    question = norm(row.get("question"))
    answer = norm(row.get("answer"))
    metric = norm(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id"))
    model = norm(row.get("answer_model") or row.get("generator_model"))
    return [(question, answer, metric, model), (question, answer, metric, "")]


def strict_reason_recovery(train_rows: list[dict[str, Any]], reason_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    by_sample_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reason_rows:
        for key in reason_keys(row):
            by_key[key].append(row)
        direct_id = row.get("sample_id") or row.get("record_id") or row.get("id")
        if direct_id:
            by_sample_id[str(direct_id)].append(row)
    recovered: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for train_row in train_rows:
        sid = sample_id(train_row)
        candidates: list[dict[str, Any]] = []
        seen: set[int] = set()
        for candidate in by_sample_id.get(sid, []):
            candidates.append(candidate)
            seen.add(id(candidate))
        for key in train_reason_keys(train_row):
            for candidate in by_key.get(key, []):
                marker = id(candidate)
                if marker not in seen:
                    candidates.append(candidate)
                    seen.add(marker)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_source[str(candidate.get("_source_file") or "unknown")].append(candidate)
        label = int(round(float(train_row["label_5"])))
        usable: list[dict[str, Any]] = []
        ambiguous_sources = 0
        inconsistent_sources = 0
        for source_rows in by_source.values():
            if len(source_rows) != 1:
                if source_rows:
                    ambiguous_sources += 1
                continue
            candidate = source_rows[0]
            try:
                consistent = int(float(candidate["score"])) == label
            except (KeyError, TypeError, ValueError):
                consistent = False
            if consistent:
                usable.append(candidate)
            else:
                inconsistent_sources += 1
        status = "unique_match" if usable else ("ambiguous" if ambiguous_sources else "unavailable")
        summary.append({
            "sample_id": sid,
            "recovered": int(status == "unique_match"),
            "match_status": status,
            "candidate_count": len(candidates),
            "unique_match_count": len(usable),
            "ambiguous_count": ambiguous_sources,
            "unavailable_count": int(status == "unavailable"),
            "score_consistent": len(usable),
            "score_inconsistent": inconsistent_sources,
            "source_files": "|".join(sorted(by_source)),
        })
        for row in usable:
            recovered.append({
                "sample_id": sid,
                "source_file": row.get("_source_file"),
                "source_model": row.get("model") or row.get("generator_model"),
                "source_score": int(float(row["score"])),
                "human_reason": row.get("reason"),
                "principle": row.get("principle"),
            })
    return summary, recovered


def load_teacher_map(manifest: Path) -> tuple[dict[str, dict[str, Any]], str]:
    if not manifest.exists():
        return {}, "missing_manifest"
    rows = read_csv(manifest)
    qwen = next((row.get("resolved_path", "") for row in rows if row.get("provider") == "qwen"), "")
    path = Path(qwen)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}, f"missing_qwen:{path}"
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not isinstance(row.get("annotation"), dict):
            continue
        sid = str(row.get("sample_id") or row["annotation"].get("sample_id") or "")
        if sid:
            output[sid] = row["annotation"]
    try:
        display_path = str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        display_path = str(path)
    return output, display_path


def normalize_failure(value: str) -> str:
    aliases = {
        "missing_key_point": "missing_content_or_key_point",
        "missing_content": "missing_content_or_key_point",
        "insufficient_evidence": "insufficient_reasoning_or_evidence",
        "reasoning_gap": "insufficient_reasoning_or_evidence",
        "task_constraint_violation": "task_constraint_or_format_violation",
        "format_violation": "task_constraint_or_format_violation",
        "unclear": "unclear_or_other",
    }
    return aliases.get(str(value), str(value))


def normalized_substring(needle: str, haystack: str) -> bool:
    return norm(needle) in norm(haystack) if norm(needle) else False
