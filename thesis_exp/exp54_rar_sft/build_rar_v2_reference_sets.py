"""Build deterministic, train-only R1/R3 rationale reference sets for RAR-SFT V2.

This stage consumes the frozen RAR-0A exact-alignment inventory. It performs
only deterministic text normalization and high-precision explicit-score
redaction. It does not call a model, read dev/test, perform semantic filtering,
construct R2 donors, or train anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    text_sha256,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_ALIGNED = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar0_alignment/private/aligned_reason_candidates.jsonl"
)
DEFAULT_RAR0_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar0_alignment/protocol/rar0a_protocol_lock.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"

EXPECTED_TRAIN_ROWS = 2654
EXPECTED_REASON_ROWS = 1612
EXPECTED_NO_REASON_ROWS = 1042
EXPECTED_ALL_RATER_REFERENCES = 4836
EXPECTED_LABEL_CONSISTENT_REFERENCES = 3934
EXPECTED_DISSENTING_REFERENCES = 902
EXPECTED_CONSISTENT_REFERENCE_DISTRIBUTION = {2: 902, 3: 710}


@dataclass(frozen=True)
class ScorePattern:
    rule_id: str
    pattern: re.Pattern[str]


SCORE_PATTERNS = (
    ScorePattern(
        "en_score_assignment",
        re.compile(
            r"(?ix)"
            r"\b(?:final\s+)?(?:score|rating)\s*(?:is|of|[:=])\s*"
            r"(?:10|[1-9])(?:\s*/\s*(?:5|10))?\s*(?:points?|stars?)?\b"
        ),
    ),
    ScorePattern(
        "en_rater_verdict",
        re.compile(
            r"(?ix)"
            r"\b(?:i\s+)?(?:would\s+)?(?:give|gave|award|awarded|rate|rated)\s+"
            r"(?:this\s+(?:answer|response)\s+)?(?:it\s+)?(?:a\s+)?"
            r"(?:10|[1-9])(?:\s*/\s*(?:5|10))?\s*(?:points?|stars?)?\b"
        ),
    ),
    ScorePattern(
        "zh_rater_verdict",
        re.compile(
            r"(?:因此|所以|故|综上|综合评定)?\s*"
            r"(?:只能\s*)?"
            r"(?:给到|给予|给(?:出)?(?:了)?|评定为|评为|可评|可得|评|打)\s*"
            r"[一二三四五六七八九十0-9]+\s*(?:/\s*(?:5|10))?\s*分"
            r"(?:的评分)?"
        ),
    ),
    ScorePattern(
        "zh_terminal_score_verdict",
        re.compile(
            r"(?:因此|所以|故|综上)?\s*(?:最终\s*)?得分\s*"
            r"[一二三四五六七八九十0-9]+\s*(?:/\s*(?:5|10))?\s*分"
            r"\s*[。.!！]?$"
        ),
    ),
    ScorePattern(
        "zh_explicit_score_assignment",
        re.compile(
            r"(?:最终\s*)?(?:评分|得分)\s*(?:为|是|[:：=])\s*"
            r"[一二三四五六七八九十0-9]+\s*(?:/\s*(?:5|10))?\s*分?"
        ),
    ),
)


def normalize_reason(value: Any) -> str:
    """Apply only NFKC and whitespace normalization."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\\n", "\n")
    return " ".join(text.split()).strip()


def _non_overlapping_matches(text: str) -> list[tuple[int, int, str, str]]:
    candidates: list[tuple[int, int, str, str]] = []
    for spec in SCORE_PATTERNS:
        for match in spec.pattern.finditer(text):
            candidates.append((match.start(), match.end(), spec.rule_id, match.group(0)))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    accepted: list[tuple[int, int, str, str]] = []
    cursor = -1
    for candidate in candidates:
        start, end, _, _ = candidate
        if start < cursor:
            continue
        accepted.append(candidate)
        cursor = end
    return accepted


def redact_explicit_scores(text: Any) -> tuple[str, list[dict[str, Any]]]:
    """Remove high-precision rater-score reports without rewriting other semantics.

    Character spans are relative to the NFKC + whitespace-normalized text before
    redaction. This is the aligned source text consumed by this stage.
    """
    normalized = normalize_reason(text)
    matches = _non_overlapping_matches(normalized)
    if not matches:
        return normalized, []
    pieces: list[str] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    output_length = 0
    for start, end, rule_id, matched_text in matches:
        prefix = normalized[cursor:start]
        pieces.append(prefix)
        output_length += len(prefix)
        events.append(
            {
                "rule_id": rule_id,
                "original_start": start,
                "original_end": end,
                "original_text": matched_text,
                "replacement": "",
                "cleaned_insertion_offset": output_length,
            }
        )
        cursor = end
    pieces.append(normalized[cursor:])
    cleaned = normalize_reason("".join(pieces))
    cleaned = re.sub(r"\s+([,.;，。；])", r"\1", cleaned)
    cleaned = re.sub(r"([,，;；])\s*([。.!！])", r"\2", cleaned)
    cleaned = cleaned.strip(" ,，;；")
    return cleaned, events


def contains_explicit_score_report(text: Any) -> bool:
    return bool(_non_overlapping_matches(normalize_reason(text)))


def _reference(
    record_id: str,
    rater_id: str,
    reason: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    source_text = normalize_reason(reason.get("reason"))
    source_hash = str(reason.get("reason_sha256") or "")
    if not source_text:
        return None, []
    if source_hash and text_sha256(source_text) != source_hash:
        raise ValueError(f"{record_id}/{rater_id}: aligned reason hash mismatch")
    cleaned, redactions = redact_explicit_scores(source_text)
    if contains_explicit_score_report(cleaned):
        raise ValueError(f"{record_id}/{rater_id}: explicit score remains after redaction")
    reference = {
        "reference_id": f"{record_id}:{rater_id}",
        "rater_id": rater_id,
        "rater_score": int(reason["source_score"]),
        "aggregate_score_consistent": bool(reason["aggregate_score_consistent"]),
        "reason": cleaned,
        "source_reason_sha256": source_hash or text_sha256(source_text),
        "clean_reason_sha256": text_sha256(cleaned) if cleaned else "",
        "redaction_count": len(redactions),
    }
    audit_rows = [
        {
            "record_id": record_id,
            "rater_id": rater_id,
            "source_reason_sha256": reference["source_reason_sha256"],
            "clean_reason_sha256": reference["clean_reason_sha256"],
            **event,
        }
        for event in redactions
    ]
    return (reference if cleaned else None), audit_rows


def build_reference_sets(
    aligned_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    all_rater_rows: list[dict[str, Any]] = []
    consistent_rows: list[dict[str, Any]] = []
    redaction_audit: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(aligned_rows):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"row {index}: missing record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        label = int(row["score"])
        if label not in range(1, 6):
            raise ValueError(f"{record_id}: score outside 1-5")

        references: list[dict[str, Any]] = []
        human_reasons = row.get("human_reasons") or {}
        for rater_id in ("human_1", "human_2", "human_3"):
            if rater_id not in human_reasons:
                continue
            reference, events = _reference(record_id, rater_id, human_reasons[rater_id])
            redaction_audit.extend(events)
            if reference is not None:
                references.append(reference)
        references.sort(key=lambda ref: ref["rater_id"])
        label_consistent = [ref for ref in references if ref["aggregate_score_consistent"]]

        common = {
            "record_id": record_id,
            "question_key": str(row.get("question_key") or ""),
            "label_5": label,
            "metric_id": str(row.get("metric_id") or ""),
            "language": str(row.get("language") or ""),
        }
        inventory.append(
            {
                **common,
                "aligned_reference_count": len(references),
                "label_consistent_reference_count": len(label_consistent),
                "dissenting_reference_count": len(references) - len(label_consistent),
                "has_aligned_reason": bool(references),
            }
        )
        all_rater_rows.append(
            {
                **common,
                "rationale_mask": int(bool(references)),
                "reference_count": len(references),
                "references": references,
            }
        )
        consistent_rows.append(
            {
                **common,
                "rationale_mask": int(bool(label_consistent)),
                "reference_count": len(label_consistent),
                "references": label_consistent,
            }
        )
    return inventory, all_rater_rows, consistent_rows, redaction_audit


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _coverage_row(
    rows: list[dict[str, Any]],
    all_rater_by_id: dict[str, dict[str, Any]],
    consistent_by_id: dict[str, dict[str, Any]],
    stratum_type: str,
    **keys: Any,
) -> dict[str, Any]:
    all_active = sum(all_rater_by_id[row["record_id"]]["rationale_mask"] for row in rows)
    consistent_active = sum(consistent_by_id[row["record_id"]]["rationale_mask"] for row in rows)
    all_refs = sum(all_rater_by_id[row["record_id"]]["reference_count"] for row in rows)
    consistent_refs = sum(consistent_by_id[row["record_id"]]["reference_count"] for row in rows)
    return {
        "stratum_type": stratum_type,
        "label_5": keys.get("label_5", ""),
        "metric_id": keys.get("metric_id", ""),
        "language": keys.get("language", ""),
        "n_rows": len(rows),
        "all_rater_active_rows": all_active,
        "all_rater_active_rate": _rate(all_active, len(rows)),
        "all_rater_references": all_refs,
        "label_consistent_active_rows": consistent_active,
        "label_consistent_active_rate": _rate(consistent_active, len(rows)),
        "label_consistent_references": consistent_refs,
    }


def coverage_table(
    inventory: list[dict[str, Any]],
    all_rater_rows: list[dict[str, Any]],
    consistent_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_rater_by_id = {row["record_id"]: row for row in all_rater_rows}
    consistent_by_id = {row["record_id"]: row for row in consistent_rows}
    output = [
        _coverage_row(
            inventory, all_rater_by_id, consistent_by_id, "overall"
        )
    ]
    dimensions = [
        ("label", ("label_5",)),
        ("metric", ("metric_id",)),
        ("language", ("language",)),
        ("label_metric_language", ("label_5", "metric_id", "language")),
    ]
    for stratum_type, fields in dimensions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            grouped[tuple(row[field] for field in fields)].append(row)
        for values, rows in sorted(
            grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            output.append(
                _coverage_row(
                    rows,
                    all_rater_by_id,
                    consistent_by_id,
                    stratum_type,
                    **dict(zip(fields, values)),
                )
            )
    return output


def summarize(
    inventory: list[dict[str, Any]],
    all_rater_rows: list[dict[str, Any]],
    consistent_rows: list[dict[str, Any]],
    redaction_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    all_active = sum(row["rationale_mask"] for row in all_rater_rows)
    consistent_active = sum(row["rationale_mask"] for row in consistent_rows)
    all_refs = sum(row["reference_count"] for row in all_rater_rows)
    consistent_refs = sum(row["reference_count"] for row in consistent_rows)
    distribution = Counter(
        row["reference_count"] for row in consistent_rows if row["reference_count"]
    )
    remaining_leakage = [
        ref["reference_id"]
        for row in all_rater_rows
        for ref in row["references"]
        if contains_explicit_score_report(ref["reason"])
    ]
    duplicate_reference_ids = [
        reference_id
        for reference_id, count in Counter(
            ref["reference_id"] for row in all_rater_rows for ref in row["references"]
        ).items()
        if count != 1
    ]
    checks = {
        "train_rows_exact": len(inventory) == EXPECTED_TRAIN_ROWS,
        "row_ids_match": [row["record_id"] for row in inventory]
        == [row["record_id"] for row in all_rater_rows]
        == [row["record_id"] for row in consistent_rows],
        "all_rater_active_rows_exact": all_active == EXPECTED_REASON_ROWS,
        "label_consistent_active_rows_exact": consistent_active == EXPECTED_REASON_ROWS,
        "no_reason_rows_exact": len(inventory) - all_active == EXPECTED_NO_REASON_ROWS,
        "all_rater_reference_count_exact": all_refs == EXPECTED_ALL_RATER_REFERENCES,
        "label_consistent_reference_count_exact": consistent_refs
        == EXPECTED_LABEL_CONSISTENT_REFERENCES,
        "dissenting_reference_count_exact": all_refs - consistent_refs
        == EXPECTED_DISSENTING_REFERENCES,
        "consistent_reference_distribution_exact": dict(sorted(distribution.items()))
        == EXPECTED_CONSISTENT_REFERENCE_DISTRIBUTION,
        "all_active_reasons_nonempty": all(
            ref["reason"] for row in all_rater_rows for ref in row["references"]
        ),
        "all_reference_ids_unique": not duplicate_reference_ids,
        "explicit_score_leakage_zero": not remaining_leakage,
        "consistent_subset_of_all": all(
            {ref["reference_id"] for ref in consistent["references"]}
            <= {ref["reference_id"] for ref in all_rater["references"]}
            for all_rater, consistent in zip(all_rater_rows, consistent_rows)
        ),
    }
    return {
        "status": "REFERENCE_SETS_READY" if all(checks.values()) else "REFERENCE_SETS_NO_GO",
        "dev_accessed": False,
        "test_accessed": False,
        "api_used": False,
        "model_used": False,
        "training_used": False,
        "counts": {
            "train_rows": len(inventory),
            "reason_covered_rows": all_active,
            "no_reason_rows": len(inventory) - all_active,
            "all_rater_references": all_refs,
            "label_consistent_references": consistent_refs,
            "dissenting_references": all_refs - consistent_refs,
            "consistent_reference_distribution": dict(sorted(distribution.items())),
            "redaction_events": len(redaction_audit),
            "remaining_explicit_score_reports": len(remaining_leakage),
        },
        "checks": checks,
        "remaining_leakage_reference_ids": remaining_leakage,
        "duplicate_reference_ids": duplicate_reference_ids,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    checks = "\n".join(
        f"- `{name}`: `{'PASS' if passed else 'FAIL'}`"
        for name, passed in summary["checks"].items()
    )
    report = f"""# RAR-SFT V2 reference-set readiness report

## Boundary

- Status: `{summary['status']}`
- Train rows: `{counts['train_rows']}`
- Dev/test accessed: `false`
- API/model/GPU/training used: `false`
- Semantic filtering or rewriting: `false`

## Counts

- Reason-covered rows: `{counts['reason_covered_rows']}`
- Score-only fallback rows: `{counts['no_reason_rows']}`
- R1 all-rater references: `{counts['all_rater_references']}`
- R3 label-consistent references: `{counts['label_consistent_references']}`
- Dissenting references retained for R1: `{counts['dissenting_references']}`
- R3 reference-count distribution: `{json.dumps(counts['consistent_reference_distribution'], sort_keys=True)}`
- Explicit-score redaction events: `{counts['redaction_events']}`
- Remaining detected explicit score reports: `{counts['remaining_explicit_score_reports']}`

## Deterministic checks

{checks}

## Interpretation

This stage validates source alignment, deterministic cleaning, provenance, and reference-set
membership only. It does not claim semantic correctness or human validation. R2 donor matching,
training manifests, token-budget auditing, model snapshot locking, and all training remain locked.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=DEFAULT_ALIGNED)
    parser.add_argument("--rar0-lock", type=Path, default=DEFAULT_RAR0_LOCK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.aligned)
    if not args.aligned.exists():
        raise FileNotFoundError(args.aligned)
    if not args.rar0_lock.exists():
        raise FileNotFoundError(args.rar0_lock)
    aligned_rows = read_jsonl(args.aligned, protect_split=True)
    rar0_lock = json.loads(args.rar0_lock.read_text(encoding="utf-8"))
    if rar0_lock.get("dev_accessed") or rar0_lock.get("test_accessed"):
        raise PermissionError("RAR-0A source lock indicates evaluation-split access")
    if len(aligned_rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS} aligned train rows, found {len(aligned_rows)}"
        )

    inventory, all_rater_rows, consistent_rows, redaction_audit = build_reference_sets(
        aligned_rows
    )
    summary = summarize(inventory, all_rater_rows, consistent_rows, redaction_audit)

    out_dir: Path = args.out_dir
    data_dir = out_dir / "data"
    audit_dir = out_dir / "audit"
    protocol_dir = out_dir / "protocol"
    report_dir = out_dir / "reports"

    paths = {
        "aligned_reason_inventory": data_dir / "aligned_reason_inventory.jsonl",
        "all_rater_reference_sets": data_dir / "all_rater_reference_sets.jsonl",
        "label_consistent_reference_sets": data_dir
        / "label_consistent_reference_sets.jsonl",
        "explicit_score_redaction_audit": audit_dir
        / "explicit_score_redaction_audit.csv",
        "coverage": audit_dir / "coverage_by_label_metric_language.csv",
        "readiness_json": audit_dir / "reference_set_readiness_report.json",
        "readiness_md": report_dir / "reference_set_readiness_report.md",
    }
    write_jsonl(paths["aligned_reason_inventory"], inventory)
    write_jsonl(paths["all_rater_reference_sets"], all_rater_rows)
    write_jsonl(paths["label_consistent_reference_sets"], consistent_rows)
    write_csv(paths["explicit_score_redaction_audit"], redaction_audit)
    write_csv(
        paths["coverage"],
        coverage_table(inventory, all_rater_rows, consistent_rows),
    )
    write_json(paths["readiness_json"], summary)
    _write_report(paths["readiness_md"], summary)

    output_hashes = {
        name: file_sha256(path)
        for name, path in paths.items()
        if path.exists() and name not in {"readiness_json", "readiness_md"}
    }
    data_lock = {
        "experiment": "Exp54 RAR-SFT V2 deterministic reference sets",
        "status": summary["status"],
        "dev_accessed": False,
        "test_accessed": False,
        "input": {
            "aligned_path": str(args.aligned.relative_to(REPO_ROOT)),
            "aligned_sha256": file_sha256(args.aligned),
            "rar0_lock_path": str(args.rar0_lock.relative_to(REPO_ROOT)),
            "rar0_lock_sha256": file_sha256(args.rar0_lock),
            "rar0_source_hashes": rar0_lock.get("source_hashes"),
        },
        "output_hashes": output_hashes,
        "score_redaction_rule_ids": [spec.rule_id for spec in SCORE_PATTERNS],
    }
    write_json(protocol_dir / "reference_set_data_lock.json", data_lock)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "REFERENCE_SETS_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
