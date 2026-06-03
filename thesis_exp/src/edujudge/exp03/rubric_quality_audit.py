"""Quality audit for Exp3 metric-level rubric text."""

from __future__ import annotations

import argparse
import difflib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import EXP03_REPORTS_DIR, EXP03_TABLES_DIR, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.rubric_sources import audit_rubric_sources
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


SIMILARITY_WARNING_THRESHOLD = 0.90
SPECIAL_ZH_PAIR = frozenset({"Scenario Element Integration", "Instruction Following & Task Completion"})


def normalize_rubric_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def pair_similarity(left: str, right: str) -> float:
    return float(difflib.SequenceMatcher(None, normalize_rubric_text(left), normalize_rubric_text(right)).ratio())


def severity_for(metric_a: str, metric_b: str, similarity: float, exact_duplicate: bool) -> str:
    if exact_duplicate and metric_a != metric_b:
        return "ERROR"
    if similarity >= SIMILARITY_WARNING_THRESHOLD and metric_a != metric_b:
        return "WARNING"
    return "PASS"


def load_source_rows(input_path: Path) -> list[dict[str, str]]:
    if not input_path.exists():
        audit_rubric_sources()
    rows = read_csv(input_path)
    return [row for row in rows if str(row.get("rubric_text") or "").strip()]


def audit_rubric_quality(input_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    input_path = input_path or (EXP03_TABLES_DIR / "rubric_source_audit.csv")
    source_rows = load_source_rows(input_path)
    by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_language[str(row.get("language") or "unknown")].append(row)

    audit_rows: list[dict[str, Any]] = []
    for language, rows in sorted(by_language.items()):
        rows = sorted(rows, key=lambda row: str(row.get("metric_canonical") or ""))
        for idx, left in enumerate(rows):
            for right in rows[idx + 1 :]:
                metric_a = str(left.get("metric_canonical") or "")
                metric_b = str(right.get("metric_canonical") or "")
                text_a = normalize_rubric_text(left.get("rubric_text") or "")
                text_b = normalize_rubric_text(right.get("rubric_text") or "")
                exact_duplicate = bool(text_a and text_a == text_b and metric_a != metric_b)
                similarity = pair_similarity(text_a, text_b)
                severity = severity_for(metric_a, metric_b, similarity, exact_duplicate)
                notes = "ok"
                if exact_duplicate:
                    notes = "Different metrics have exactly identical rubric text."
                elif severity == "WARNING":
                    notes = f"Different metrics have highly similar rubric text >= {SIMILARITY_WARNING_THRESHOLD:.2f}."
                if language == "zh" and frozenset({metric_a, metric_b}) == SPECIAL_ZH_PAIR:
                    notes = "SPECIAL_CHECK zh Scenario Element Integration vs Instruction Following & Task Completion. " + notes
                audit_rows.append(
                    {
                        "language": language,
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "similarity": similarity,
                        "exact_duplicate": exact_duplicate,
                        "severity": severity,
                        "notes": notes,
                    }
                )

    write_csv(EXP03_TABLES_DIR / "rubric_quality_audit.csv", audit_rows)
    write_rubric_quality_report(audit_rows, input_path)
    return audit_rows


def overall_status(rows: list[dict[str, Any]]) -> str:
    severities = {str(row.get("severity")) for row in rows}
    if "ERROR" in severities:
        return "ERROR"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


def special_zh_status(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        metrics = frozenset({str(row.get("metric_a")), str(row.get("metric_b"))})
        if row.get("language") == "zh" and metrics == SPECIAL_ZH_PAIR:
            return str(row.get("severity"))
    return "MISSING"


def write_rubric_quality_report(rows: list[dict[str, Any]], input_path: Path) -> None:
    status = overall_status(rows)
    special_status = special_zh_status(rows)
    error_rows = [row for row in rows if row.get("severity") == "ERROR"]
    warning_rows = [row for row in rows if row.get("severity") == "WARNING"]
    lines = [
        "# Exp3 Rubric Quality Audit",
        "",
        f"Overall status: **{status}**",
        "",
        f"- Input: `{relpath(input_path)}`",
        f"- Pairwise comparisons: {len(rows)}",
        f"- ERROR pairs: {len(error_rows)}",
        f"- WARNING pairs: {len(warning_rows)}",
        f"- zh Scenario Element Integration vs Instruction Following & Task Completion: **{special_status}**",
        "",
        "Exact duplicates across different metrics are marked ERROR. Highly similar cross-metric",
        f"rubrics are marked WARNING at similarity >= {SIMILARITY_WARNING_THRESHOLD:.2f}.",
        "",
        f"CSV: `{relpath(EXP03_TABLES_DIR / 'rubric_quality_audit.csv')}`",
    ]
    if error_rows:
        lines.extend(["", "## ERROR Pairs", "", "| language | metric_a | metric_b | similarity | notes |", "| --- | --- | --- | ---: | --- |"])
        for row in error_rows[:20]:
            lines.append(
                f"| {row['language']} | {row['metric_a']} | {row['metric_b']} | "
                f"{float(row['similarity']):.6f} | {str(row['notes']).replace('|', '/')} |"
            )
    if warning_rows:
        lines.extend(["", "## WARNING Pairs", "", "| language | metric_a | metric_b | similarity | notes |", "| --- | --- | --- | ---: | --- |"])
        for row in warning_rows[:20]:
            lines.append(
                f"| {row['language']} | {row['metric_a']} | {row['metric_b']} | "
                f"{float(row['similarity']):.6f} | {str(row['notes']).replace('|', '/')} |"
            )
    write_text(EXP03_REPORTS_DIR / "rubric_quality_audit.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Exp3 rubric text quality.")
    parser.add_argument("--input", type=Path, default=EXP03_TABLES_DIR / "rubric_source_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_rubric_quality(args.input)
    print(f"Rubric quality status: {overall_status(rows)}")
    print(f"Special zh SEI vs IFTC status: {special_zh_status(rows)}")
    print(f"Output: {relpath(EXP03_TABLES_DIR / 'rubric_quality_audit.csv')}")


if __name__ == "__main__":
    main()
