"""Leakage checks for Exp 0 splits."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import (
    OUTPUT_DIR,
    REPO_ROOT,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_exp_dirs,
    read_jsonl,
    write_csv,
    write_text,
)
from thesis_exp.src.edujudge.utils.text_norm import normalize_text, stringify, truncate_text


SPLIT_NAMES = ["train", "dev", "test"]
SPLIT_DIRS = ["paper_like_triple_seed42", "question_seed42"]
SYNTHETIC_FILES = ["sampled_merge_50_new.json", "sampled_merge_50_new_swift.json"]


def _load_split(split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: read_jsonl(split_dir / f"{split}.jsonl") for split in SPLIT_NAMES}


def _key_sets(assignments: dict[str, list[dict[str, Any]]], key: str) -> dict[str, set[str]]:
    return {split: {stringify(row.get(key, "")) for row in rows if row.get(key, "")} for split, rows in assignments.items()}


def _pairwise_overlap(sets: dict[str, set[str]]) -> list[tuple[str, str, set[str]]]:
    out = []
    for idx, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[idx + 1 :]:
            out.append((left, right, sets[left] & sets[right]))
    return out


def _qa_key(row: dict[str, Any]) -> str:
    return sha1_text(normalize_text(row.get("question")), normalize_text(row.get("answer")))


def _record_duplicates(assignments: dict[str, list[dict[str, Any]]], key: str) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    for split, rows in assignments.items():
        for row in rows:
            value = stringify(row.get(key, ""))
            if not value:
                continue
            counts[value] += 1
            examples.setdefault(value, {"split": split, "record_id": row.get("record_id"), "question": row.get("question")})
    return [
        {
            "key": key,
            "value": value,
            "count": count,
            "example_split": examples[value]["split"],
            "example_record_id": examples[value]["record_id"],
            "question_preview": truncate_text(examples[value]["question"], 180),
        }
        for value, count in counts.items()
        if count > 1
    ]


def _full_duplicate_rows(assignments: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    for split, rows in assignments.items():
        for row in rows:
            value = sha1_text(
                normalize_text(row.get("question")),
                normalize_text(row.get("answer")),
                normalize_text(row.get("metric_canonical") or row.get("metric_raw")),
                row.get("label_5"),
                row.get("human_1"),
                row.get("human_2"),
                row.get("human_3"),
            )
            counts[value] += 1
            examples.setdefault(value, {"split": split, "record_id": row.get("record_id"), "question": row.get("question")})
    return [
        {
            "key": "full_record_key",
            "value": value,
            "count": count,
            "example_split": examples[value]["split"],
            "example_record_id": examples[value]["record_id"],
            "question_preview": truncate_text(examples[value]["question"], 180),
        }
        for value, count in counts.items()
        if count > 1
    ]


def _extract_user_question_from_synthetic(record: dict[str, Any]) -> str:
    text = ""
    if isinstance(record.get("instruction"), str):
        text = record["instruction"]
    elif isinstance(record.get("messages"), list):
        for msg in record["messages"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = stringify(msg.get("content"))
                break
    if not text:
        text = stringify(record)

    # Try to recover the first dialogue user content embedded in the prompt.
    match = re.search(r"\[\{['\"]role['\"]:\s*['\"]user['\"],\s*['\"]content['\"]:\s*(?P<quote>['\"])(?P<body>.*?)(?P=quote)\}", text, re.DOTALL)
    if match:
        body = match.group("body")
        try:
            return ast.literal_eval(match.group("quote") + body + match.group("quote"))
        except (SyntaxError, ValueError):
            return body
    marker = "对话："
    if marker in text:
        return text.split(marker, 1)[1][:2000]
    return text[:2000]


def _synthetic_question_keys() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in SYNTHETIC_FILES:
        path = REPO_ROOT / name
        keys: set[str] = set()
        if not path.exists():
            out[name] = keys
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out[name] = keys
            continue
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = list(data.values())
        else:
            iterable = []
        for item in iterable:
            if isinstance(item, dict):
                question_text = _extract_user_question_from_synthetic(item)
            else:
                question_text = stringify(item)
            if question_text:
                keys.add(sha1_text(normalize_text(question_text)))
        out[name] = keys
    return out


def _add_overlap_rows(
    summary_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    split_name: str,
    check: str,
    key_type: str,
    overlaps: list[tuple[str, str, set[str]]],
    fail_condition: bool,
    warning_condition: bool = True,
) -> None:
    total_overlap = sum(len(values) for _, _, values in overlaps)
    severity = "PASS"
    if total_overlap and fail_condition:
        severity = "FAIL"
    elif total_overlap and warning_condition:
        severity = "WARNING"
    summary_rows.append(
        {
            "split_name": split_name,
            "check": check,
            "key_type": key_type,
            "overlap_count": total_overlap,
            "severity": severity,
        }
    )
    for left, right, values in overlaps:
        if values:
            detail_rows.append(
                {
                    "split_name": split_name,
                    "check": check,
                    "key_type": key_type,
                    "pair": f"{left}-{right}",
                    "overlap_count": len(values),
                    "severity": severity,
                    "example_keys": sorted(values)[:10],
                }
            )


def run_checks() -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_exp_dirs()
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    synthetic_keys = _synthetic_question_keys()

    for split_name in SPLIT_DIRS:
        split_dir = SPLITS_DIR / split_name
        if not split_dir.exists():
            summary_rows.append(
                {
                    "split_name": split_name,
                    "check": "split files exist",
                    "key_type": "file",
                    "overlap_count": "",
                    "severity": "FAIL",
                }
            )
            continue
        assignments = _load_split(split_dir)
        for key in ["triple_key", "question_key", "answer_key"]:
            fail = key == "triple_key" or (split_name == "question_seed42" and key == "question_key")
            warning = True
            if split_name == "paper_like_triple_seed42" and key == "question_key":
                warning = True
            _add_overlap_rows(
                summary_rows,
                detail_rows,
                split_name,
                f"{key} cross-split overlap",
                key,
                _pairwise_overlap(_key_sets(assignments, key)),
                fail_condition=fail,
                warning_condition=warning,
            )

        qa_sets = {
            split: {_qa_key(row) for row in rows if row.get("question") and row.get("answer")}
            for split, rows in assignments.items()
        }
        _add_overlap_rows(
            summary_rows,
            detail_rows,
            split_name,
            "normalized question+answer cross-split overlap",
            "qa_key",
            _pairwise_overlap(qa_sets),
            fail_condition=False,
            warning_condition=True,
        )

        duplicate_records = _record_duplicates(assignments, "record_id")
        duplicate_full_records = _full_duplicate_rows(assignments)
        for check_name, duplicates in [
            ("duplicate record_id across split files", duplicate_records),
            ("duplicate full scored item across split files", duplicate_full_records),
        ]:
            severity = "FAIL" if duplicates and "record_id" in check_name else "WARNING" if duplicates else "PASS"
            summary_rows.append(
                {
                    "split_name": split_name,
                    "check": check_name,
                    "key_type": "record_id" if "record_id" in check_name else "full_record_key",
                    "overlap_count": len(duplicates),
                    "severity": severity,
                }
            )
            for duplicate in duplicates[:50]:
                detail_rows.append(
                    {
                        "split_name": split_name,
                        "check": check_name,
                        "key_type": duplicate["key"],
                        "pair": "all",
                        "overlap_count": duplicate["count"],
                        "severity": severity,
                        "example_keys": [duplicate["value"], duplicate["question_preview"]],
                    }
                )

        heldout_question_keys = _key_sets(assignments, "question_key")["dev"] | _key_sets(assignments, "question_key")["test"]
        for source_name, keys in synthetic_keys.items():
            overlap = keys & heldout_question_keys
            severity = "WARNING" if overlap else "PASS"
            summary_rows.append(
                {
                    "split_name": split_name,
                    "check": "synthetic sampled question overlap with dev/test",
                    "key_type": source_name,
                    "overlap_count": len(overlap),
                    "severity": severity,
                }
            )
            if overlap:
                detail_rows.append(
                    {
                        "split_name": split_name,
                        "check": "synthetic sampled question overlap with dev/test",
                        "key_type": source_name,
                        "pair": "synthetic-dev/test",
                        "overlap_count": len(overlap),
                        "severity": severity,
                        "example_keys": sorted(overlap)[:10],
                    }
                )

    severities = {row["severity"] for row in summary_rows}
    if "FAIL" in severities:
        status = "FAIL"
    elif "WARNING" in severities:
        status = "WARNING"
    else:
        status = "PASS"
    return status, summary_rows, detail_rows


def write_report(status: str, summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_split[row["split_name"]].append(row)

    lines = [
        "# Leakage Report",
        "",
        f"Overall status: **{status}**",
        "",
        "Interpretation: triple-key overlap is a failure for both split strategies. Question-key overlap is a warning for `paper_like_triple_seed42` because that split only promises question-answer-metric isolation, but it is a failure for `question_seed42`.",
        "",
    ]
    for split_name in SPLIT_DIRS:
        split_rows = by_split.get(split_name, [])
        split_status = "FAIL" if any(row["severity"] == "FAIL" for row in split_rows) else "WARNING" if any(row["severity"] == "WARNING" for row in split_rows) else "PASS"
        lines.extend(
            [
                f"## {split_name}: {split_status}",
                "",
                "| check | key_type | overlap_count | severity |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in split_rows:
            lines.append(f"| {row['check']} | {row['key_type']} | {row['overlap_count']} | {row['severity']} |")
        lines.append("")

    if detail_rows:
        lines.extend(
            [
                "## Details",
                "",
                "Detailed examples are written to `tables/leakage_details.csv`.",
            ]
        )
    write_text(OUTPUT_DIR / "leakage_report.md", "\n".join(lines))


def main() -> None:
    status, summary_rows, detail_rows = run_checks()
    write_csv(
        TABLES_DIR / "leakage_summary.csv",
        summary_rows,
        ["split_name", "check", "key_type", "overlap_count", "severity"],
    )
    write_csv(
        TABLES_DIR / "leakage_details.csv",
        detail_rows,
        ["split_name", "check", "key_type", "pair", "overlap_count", "severity", "example_keys"],
    )
    write_report(status, summary_rows, detail_rows)
    print(f"Leakage check status: {status}")


if __name__ == "__main__":
    main()
