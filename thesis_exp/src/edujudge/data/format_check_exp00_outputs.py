"""Formatting checks for Exp 0.1 generated config and reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from thesis_exp.src.edujudge.data.reference_contract import CONTRACT_PATH
from thesis_exp.src.edujudge.utils.io import OUTPUT_DIR, TABLES_DIR, ensure_exp_dirs, write_csv, write_text


MARKDOWN_FILES = [
    OUTPUT_DIR / "report.md",
    OUTPUT_DIR / "data_card.md",
    OUTPUT_DIR / "review_package.md",
    OUTPUT_DIR / "leakage_report.md",
    OUTPUT_DIR / "official_source_audit.md",
    OUTPUT_DIR / "subject_alignment_report.md",
    OUTPUT_DIR / "split_reference_check.md",
    OUTPUT_DIR / "sanity_check_exp00_reference.md",
]

REQUIRED_NONEMPTY_CSV_FILES = [
    TABLES_DIR / "leakage_summary.csv",
    TABLES_DIR / "sanity_check_results.csv",
    TABLES_DIR / "split_stats.csv",
    TABLES_DIR / "official_source_inventory.csv",
    TABLES_DIR / "subject_mapping.csv",
    TABLES_DIR / "metric_mapping.csv",
    TABLES_DIR / "scenario_mapping.csv",
]

REQUIRED_YAML_FIELDS = [
    ("official_edubench", "expected_scenarios"),
    ("official_edubench", "expected_metrics"),
    ("pdf_audit_corpus", "expected_total_scored_items"),
    ("pdf_audit_corpus", "expected_train_pool_rows"),
    ("pdf_audit_corpus", "expected_heldout_test_rows"),
    ("score_mapping", "ten_to_five"),
]

MAX_MARKDOWN_LINE_LENGTH = 240


def _get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def check_reference_contract() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    data = yaml.safe_load(text)
    rows.append(
        {
            "artifact": str(CONTRACT_PATH),
            "check": "yaml.safe_load",
            "status": "PASS" if isinstance(data, dict) else "FAIL",
            "observed": type(data).__name__,
            "expected": "dict",
            "notes": f"{line_count} physical lines",
        }
    )
    rows.append(
        {
            "artifact": str(CONTRACT_PATH),
            "check": "multi-line YAML",
            "status": "PASS" if line_count >= 20 else "FAIL",
            "observed": line_count,
            "expected": ">=20",
            "notes": "prevents one-line flow-style contract output",
        }
    )
    if not isinstance(data, dict):
        return rows
    for keys in REQUIRED_YAML_FIELDS:
        try:
            value = _get_nested(data, keys)
        except KeyError:
            rows.append(
                {
                    "artifact": str(CONTRACT_PATH),
                    "check": ".".join(keys),
                    "status": "FAIL",
                    "observed": "missing",
                    "expected": "present",
                    "notes": "",
                }
            )
            continue
        status = "PASS"
        expected = "present"
        if keys == ("official_edubench", "expected_scenarios"):
            status = "PASS" if isinstance(value, list) and len(value) == 9 else "FAIL"
            expected = "list length 9"
        elif keys == ("official_edubench", "expected_metrics"):
            metric_count = sum(len(items) for items in value.values()) if isinstance(value, dict) else 0
            status = "PASS" if metric_count == 12 else "FAIL"
            expected = "12 metrics across groups"
            value = metric_count
        elif keys[0] == "pdf_audit_corpus":
            status = "PASS" if isinstance(value, int) and value > 0 else "FAIL"
            expected = "positive integer"
        elif keys == ("score_mapping", "ten_to_five"):
            status = "PASS" if isinstance(value, dict) and len(value) == 5 else "FAIL"
            expected = "5 mapping buckets"
        rows.append(
            {
                "artifact": str(CONTRACT_PATH),
                "check": ".".join(keys),
                "status": status,
                "observed": value,
                "expected": expected,
                "notes": "",
            }
        )
    return rows


def _markdown_table_blocks(lines: list[str]) -> list[tuple[int, list[str]]]:
    blocks: list[tuple[int, list[str]]] = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].lstrip().startswith("|"):
            idx += 1
            continue
        start = idx + 1
        block: list[str] = []
        while idx < len(lines) and lines[idx].lstrip().startswith("|"):
            block.append(lines[idx])
            idx += 1
        blocks.append((start, block))
    return blocks


def _is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def check_markdown_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        heading_errors = 0
        for idx, line in enumerate(lines):
            if not line.startswith("#"):
                continue
            prev_ok = idx == 0 or lines[idx - 1].strip() == ""
            next_ok = idx + 1 == len(lines) or lines[idx + 1].strip() == ""
            if not (prev_ok and next_ok):
                heading_errors += 1
        table_errors = 0
        table_blocks = _markdown_table_blocks(lines)
        for _, block in table_blocks:
            if len(block) < 2 or not _is_separator(block[1]):
                table_errors += 1
        max_line = max((len(line) for line in lines), default=0)
        checks = [
            ("multi-line markdown", len(lines) >= 5, len(lines), ">=5"),
            ("heading separation", heading_errors == 0, heading_errors, 0),
            ("table block format", table_errors == 0, table_errors, 0),
            ("raw line length", max_line <= MAX_MARKDOWN_LINE_LENGTH, max_line, f"<={MAX_MARKDOWN_LINE_LENGTH}"),
        ]
        for check, ok, observed, expected in checks:
            rows.append(
                {
                    "artifact": str(path),
                    "check": check,
                    "status": "PASS" if ok else "FAIL",
                    "observed": observed,
                    "expected": expected,
                    "notes": f"{len(table_blocks)} table block(s)",
                }
            )
    return rows


def check_csv_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_csv_files = sorted(TABLES_DIR.glob("*.csv"))
    required = {path.resolve() for path in REQUIRED_NONEMPTY_CSV_FILES}
    existing = {path.resolve() for path in all_csv_files}
    for path in REQUIRED_NONEMPTY_CSV_FILES:
        if path.resolve() in existing:
            continue
        rows.append(
            {
                "artifact": str(path),
                "check": "required CSV exists",
                "status": "FAIL",
                "observed": "missing",
                "expected": "present",
                "notes": "required Exp0.1 audit CSV",
            }
        )
    for path in all_csv_files:
        physical_lines = sum(1 for _ in path.open("r", encoding="utf-8", newline=""))
        dataframe = pd.read_csv(path)
        require_data = path.resolve() in required
        rows.append(
            {
                "artifact": str(path),
                "check": "pandas.read_csv",
                "status": "PASS",
                "observed": f"{len(dataframe)} rows / {len(dataframe.columns)} columns",
                "expected": "readable CSV",
                "notes": f"{physical_lines} physical lines",
            }
        )
        rows.append(
            {
                "artifact": str(path),
                "check": "header row",
                "status": "PASS" if len(dataframe.columns) > 0 else "FAIL",
                "observed": len(dataframe.columns),
                "expected": ">0",
                "notes": "",
            }
        )
        rows.append(
            {
                "artifact": str(path),
                "check": "one physical line per record",
                "status": "PASS" if physical_lines == len(dataframe) + 1 else "FAIL",
                "observed": physical_lines,
                "expected": len(dataframe) + 1,
                "notes": "header plus one line per parsed data row",
            }
        )
        if require_data:
            rows.append(
                {
                    "artifact": str(path),
                    "check": "required data rows",
                    "status": "PASS" if len(dataframe) > 0 and physical_lines >= 2 else "FAIL",
                    "observed": len(dataframe),
                    "expected": ">0",
                    "notes": "required Exp0.1 audit CSV",
                }
            )
    return rows


def run_checks() -> tuple[str, list[dict[str, Any]]]:
    ensure_exp_dirs()
    rows = []
    rows.extend(check_reference_contract())
    rows.extend(check_markdown_files())
    rows.extend(check_csv_files())
    status = "FAIL" if any(row["status"] == "FAIL" for row in rows) else "PASS"
    return status, rows


def main() -> None:
    status, rows = run_checks()
    write_csv(
        TABLES_DIR / "format_check_results.csv",
        rows,
        ["artifact", "check", "status", "observed", "expected", "notes"],
    )
    lines = [
        "# Exp 0.1 Formatting Check",
        "",
        f"Overall status: **{status}**",
        "",
        "| artifact | check | status | observed | expected | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact']} | {row['check']} | {row['status']} | {row['observed']} | {row['expected']} | {row.get('notes', '')} |"
        )
    write_text(OUTPUT_DIR / "format_check_exp00_outputs.md", "\n".join(lines))
    print(f"Exp 0.1 formatting check status: {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
