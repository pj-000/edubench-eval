"""CPU-only setup checks for Exp2 CE baseline data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp02 import (
    EXPECTED_SPLIT_ROWS,
    EXP02_DATA_DIR,
    EXP02_OUTPUT_DIR,
    EXP02_TABLES_DIR,
    SPLIT_DIR,
    ensure_exp02_dirs,
)
from thesis_exp.src.edujudge.exp02.build_exp02_dataset import build_exp02_dataset
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def run_sanity_check() -> list[dict[str, Any]]:
    ensure_exp02_dirs()
    if not all((EXP02_DATA_DIR / f"{split}.jsonl").exists() for split in EXPECTED_SPLIT_ROWS):
        build_exp02_dataset()
    rows: list[dict[str, Any]] = []
    split_ids: dict[str, set[str]] = {}
    for split, expected in EXPECTED_SPLIT_ROWS.items():
        source = SPLIT_DIR / f"{split}.jsonl"
        output = EXP02_DATA_DIR / f"{split}.jsonl"
        add(rows, f"{split} source exists", "PASS" if source.exists() else "FAIL", relpath(source), "exists")
        add(rows, f"{split} output exists", "PASS" if output.exists() else "FAIL", relpath(output), "exists")
        data = read_jsonl(output) if output.exists() else []
        add(rows, f"{split} row count", "PASS" if len(data) == expected else "WARN", len(data), expected)
        labels = sorted({row.get("label_5") for row in data})
        add(rows, f"{split} labels in 1-5", "PASS" if labels == [1, 2, 3, 4, 5] else "FAIL", labels, [1, 2, 3, 4, 5])
        missing_text = sum(1 for row in data if not row.get("text"))
        add(rows, f"{split} nonempty text", "PASS" if missing_text == 0 else "FAIL", missing_text, 0)
        split_ids[split] = {str(row.get("id")) for row in data}
        counts = Counter(row.get("label_5") for row in data)
        add(rows, f"{split} label distribution", "PASS", dict(sorted(counts.items())), "recorded")
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        overlap = split_ids[left] & split_ids[right]
        add(rows, f"{left}/{right} record_id overlap", "PASS" if not overlap else "FAIL", len(overlap), 0)

    write_csv(EXP02_TABLES_DIR / "sanity_check_exp02_setup.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "WARN/FAIL"
    write_text(
        EXP02_OUTPUT_DIR / "sanity_check_exp02_setup.md",
        f"""# Exp2 Setup Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def main() -> None:
    rows = run_sanity_check()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp2 setup sanity statuses: {', '.join(statuses)}")
    print(f"Outputs: {relpath(EXP02_OUTPUT_DIR / 'sanity_check_exp02_setup.md')}")


if __name__ == "__main__":
    main()

