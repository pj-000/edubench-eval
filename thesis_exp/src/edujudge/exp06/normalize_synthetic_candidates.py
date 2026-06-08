"""Normalize existing synthetic candidates into an audit-only table."""

from __future__ import annotations

from typing import Any

from thesis_exp.src.edujudge.exp06 import EXP06_TABLES_DIR, NORMALIZED_FIELDS, ensure_exp06_dirs
from thesis_exp.src.edujudge.exp06.common import iter_candidate_paths, normalize_records_from_file, write_rows


def build_candidate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in iter_candidate_paths():
        rows.extend(normalize_records_from_file(path))
    return rows


def main() -> None:
    ensure_exp06_dirs()
    rows = build_candidate_rows()
    write_rows(EXP06_TABLES_DIR / "synthetic_candidate_rows.csv", rows, NORMALIZED_FIELDS)
    print(f"Wrote {len(rows)} rows to {EXP06_TABLES_DIR / 'synthetic_candidate_rows.csv'}")


if __name__ == "__main__":
    main()
