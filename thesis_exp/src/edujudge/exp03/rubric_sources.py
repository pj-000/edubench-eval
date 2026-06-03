"""Audit and normalize rubric sources for Exp3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import EXP03_REPORTS_DIR, EXP03_TABLES_DIR, SPLIT_PATHS, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.templates import rubric_to_text
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


def load_split_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, path in SPLIT_PATHS.items():
        for row in read_jsonl(path):
            rows.append({"split": split, **row})
    return rows


def normalize_rubric(row: dict[str, Any]) -> str:
    return rubric_to_text(row.get("rubric")).strip()


def audit_rubric_sources(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    rows = rows or load_split_rows()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = stringify(row.get("metric_canonical")).strip()
        metric_group = stringify(row.get("metric_group")).strip()
        language = stringify(row.get("language")).strip() or "unknown"
        grouped[(metric, metric_group, language)].append(row)

    audit_rows: list[dict[str, Any]] = []
    missing_groups: list[str] = []
    for (metric, metric_group, language), group_rows in sorted(grouped.items()):
        rubric_counter = Counter(normalize_rubric(row) for row in group_rows if normalize_rubric(row))
        n_rows = len(group_rows)
        covered = sum(rubric_counter.values())
        coverage = covered / n_rows if n_rows else 0.0
        if not rubric_counter:
            missing_groups.append(f"{metric}/{language}")
            audit_rows.append(
                {
                    "metric_canonical": metric,
                    "metric_group": metric_group,
                    "language": language,
                    "rubric_source": "missing",
                    "rubric_text": "",
                    "n_rows": n_rows,
                    "coverage": coverage,
                    "is_sample_level": False,
                    "is_metric_level": False,
                    "notes": "FAIL: no rubric text was found for this metric/language group.",
                }
            )
            continue

        rubric_text, most_common_count = rubric_counter.most_common(1)[0]
        unique_count = len(rubric_counter)
        is_metric_level = unique_count == 1
        is_sample_level = unique_count > 1
        source = "split_row_field_metric_language_level" if is_metric_level else "split_row_field_variable_by_row"
        notes = (
            "Rubric is read from the split row field but is constant within this metric/language group; "
            "treat as metric-level description, not sample-specific human annotation."
            if is_metric_level
            else "Rubric text varies within the metric/language group; inspect before claiming sample-level provenance."
        )
        if most_common_count != covered:
            notes += f" Most common rubric covers {most_common_count}/{covered} covered rows."
        audit_rows.append(
            {
                "metric_canonical": metric,
                "metric_group": metric_group,
                "language": language,
                "rubric_source": source,
                "rubric_text": rubric_text,
                "n_rows": n_rows,
                "coverage": coverage,
                "is_sample_level": is_sample_level,
                "is_metric_level": is_metric_level,
                "notes": notes,
            }
        )

    write_csv(EXP03_TABLES_DIR / "rubric_source_audit.csv", audit_rows)
    write_rubric_audit_report(audit_rows, missing_groups)
    if missing_groups:
        raise RuntimeError(f"Missing rubric text for metric/language groups: {missing_groups}")
    return audit_rows


def write_rubric_audit_report(audit_rows: list[dict[str, Any]], missing_groups: list[str]) -> None:
    total_rows = sum(int(row["n_rows"]) for row in audit_rows)
    covered_rows = sum(int(row["n_rows"]) for row in audit_rows if float(row["coverage"]) == 1.0)
    metric_level_groups = sum(1 for row in audit_rows if bool(row["is_metric_level"]))
    sample_level_groups = sum(1 for row in audit_rows if bool(row["is_sample_level"]))
    status = "FAIL" if missing_groups else "PASS"
    lines = [
        "# Exp3 Rubric Source Audit",
        "",
        f"Overall status: **{status}**",
        "",
        f"- Source split rows covered: {covered_rows}/{total_rows}",
        f"- Metric/language groups audited: {len(audit_rows)}",
        f"- Metric-level groups: {metric_level_groups}",
        f"- Variable row-level groups: {sample_level_groups}",
        "",
        "Important provenance note: rubric is read from the split row field, but in this dataset it is",
        "constant within each metric/language group. Therefore Exp3 treats it as metric-level rubric",
        "description, not sample-specific human annotation.",
        "",
        f"CSV: `{relpath(EXP03_TABLES_DIR / 'rubric_source_audit.csv')}`",
    ]
    if missing_groups:
        lines.extend(["", "## Missing Groups", "", *[f"- {item}" for item in missing_groups]])
    write_text(EXP03_REPORTS_DIR / "rubric_source_audit.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Exp3 rubric sources.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = audit_rubric_sources()
    print(f"Rubric audit rows: {len(rows)}")
    print(f"Output: {relpath(EXP03_TABLES_DIR / 'rubric_source_audit.csv')}")


if __name__ == "__main__":
    main()
