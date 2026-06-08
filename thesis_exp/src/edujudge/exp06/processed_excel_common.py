"""Shared helpers for Exp6-1 processed Excel source audit."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.src.edujudge.exp06 import EXP0_SPLIT_DIR, PROCESSED_EXCEL_AUDIT_DIR, REPO_ROOT, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.common import (
    OFFICIAL_METRIC_SET,
    base_language,
    label_5_from_score,
    normalize_metric,
    numeric_score,
    source_rel,
    stable_qa_hash,
    stable_question_hash,
    stable_triple_hash,
)
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_csv
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


PROCESSED_EXCEL_SOURCES = [
    "deepseek_output/processed_excel_data_1.jsonl",
    "deepseek_output/processed_excel_data_1_en.jsonl",
    "deepseek_output/processed_excel_data_1_zh.jsonl",
]

SOURCE_PRIORITY = {
    "deepseek_output/processed_excel_data_1.jsonl": 0,
    "deepseek_output/processed_excel_data_1_en.jsonl": 1,
    "deepseek_output/processed_excel_data_1_zh.jsonl": 2,
}


def output_path(name: str) -> Path:
    ensure_exp06_dirs()
    return PROCESSED_EXCEL_AUDIT_DIR / name


def source_paths() -> list[Path]:
    return [REPO_ROOT / source for source in PROCESSED_EXCEL_SOURCES]


def load_source_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def canonical_record_key(record: dict[str, Any]) -> str:
    payload = {
        "id": record.get("id"),
        "question": normalize_text(record.get("question")),
        "response": normalize_text(record.get("response")),
        "scores": [
            {
                "criterion": normalize_text(item.get("criterion")) if isinstance(item, dict) else "",
                "score": item.get("score") if isinstance(item, dict) else "",
                "reason": normalize_text(item.get("reason")) if isinstance(item, dict) else "",
            }
            for item in record.get("scores", [])
        ],
    }
    return sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def candidate_key(row: dict[str, Any]) -> str:
    return sha1_text(
        normalize_text(row.get("question")),
        normalize_text(row.get("answer")),
        normalize_text(row.get("metric_canonical")),
        row.get("target_label_5"),
        normalize_text(row.get("reason")),
    )


def expand_source(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = source_rel(path)
    for record_index, record in enumerate(load_source_records(path)):
        question = stringify(record.get("question"))
        answer = stringify(record.get("response"))
        language = base_language("", question, answer)
        scores = record.get("scores") or []
        for score_index, score_item in enumerate(scores):
            if not isinstance(score_item, dict):
                continue
            metric_raw = score_item.get("criterion", "")
            metric_canonical = normalize_metric(metric_raw)
            score_raw = score_item.get("score", "")
            target_label = label_5_from_score(path, score_raw, metric_raw)
            question_key = stable_question_hash(question)
            qa_key = stable_qa_hash(question, answer)
            triple_key = stable_triple_hash(question, answer, metric_canonical)
            row = {
                "candidate_id": sha1_text(source, record_index, score_index, metric_canonical, score_raw)[:16],
                "source_file": source,
                "source_record_index": record_index,
                "source_score_index": score_index,
                "source_record_id": record.get("id", ""),
                "level": record.get("level", ""),
                "model": record.get("model", ""),
                "question": question,
                "answer": answer,
                "metric_raw": stringify(metric_raw),
                "metric_canonical": metric_canonical,
                "score_raw": stringify(score_raw),
                "score_numeric": numeric_score(score_raw),
                "target_label_5": target_label,
                "reason": stringify(score_item.get("reason", "")),
                "language": language,
                "question_key": question_key,
                "qa_key": qa_key,
                "triple_key": triple_key,
                "candidate_key": "",
                "valid_metric": metric_canonical in OFFICIAL_METRIC_SET,
                "valid_label": target_label in {1, 2, 3, 4, 5},
                "valid_question_answer": bool(question and answer),
            }
            row["candidate_key"] = candidate_key(row)
            rows.append(row)
    return rows


def all_candidate_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in source_paths():
        out.extend(expand_source(path))
    return out


def load_split_indexes() -> dict[str, dict[str, set[str]]]:
    indexes: dict[str, dict[str, set[str]]] = {}
    for split in ["train", "dev", "test"]:
        rows = read_jsonl(EXP0_SPLIT_DIR / f"{split}.jsonl")
        indexes[split] = {
            "question_key": {stable_question_hash(row.get("question")) for row in rows},
            "qa_key": {stable_qa_hash(row.get("question"), row.get("answer")) for row in rows},
            "triple_key": {
                stable_triple_hash(row.get("question"), row.get("answer"), row.get("metric_canonical") or row.get("metric_raw"))
                for row in rows
            },
            "exp0_question_key": {stringify(row.get("question_key")) for row in rows if row.get("question_key")},
            "exp0_triple_key": {stringify(row.get("triple_key")) for row in rows if row.get("triple_key")},
        }
    return indexes


def records_by_source() -> dict[str, list[dict[str, Any]]]:
    return {source_rel(path): load_source_records(path) for path in source_paths()}


def rows_by_source(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[stringify(row.get("source_file"))].append(row)
    return dict(out)


def count_duplicates(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    counts = Counter(stringify(row.get(key)) for row in rows)
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    return duplicate_groups, duplicate_rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_processed_csv(name: str, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    write_csv(output_path(name), rows, fieldnames)


def summarize_low_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    low = [row for row in rows if stringify(row.get("target_label_5")) in {"1", "2"} or row.get("target_label_5") in {1, 2}]
    return {
        "total_low_score_candidates": len(low),
        "by_label": dict(Counter(stringify(row.get("target_label_5")) for row in low)),
        "by_metric": dict(Counter(stringify(row.get("metric_canonical")) for row in low)),
        "by_language": dict(Counter(stringify(row.get("language")) for row in low)),
        "by_source_file": dict(Counter(stringify(row.get("source_file")) for row in low)),
    }


def small_preview(value: Any, length: int = 200) -> str:
    return truncate_text(value, length)
