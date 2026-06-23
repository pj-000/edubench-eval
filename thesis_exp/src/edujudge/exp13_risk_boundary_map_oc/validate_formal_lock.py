"""Validate the Exp13 formal-lock contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp13_risk_boundary_map_oc import (
    EXP13_CONFIG_DIR,
    EXP13_REPORTS_DIR,
    EXP13_TABLES_DIR,
    ensure_exp13_dirs,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_csv, relpath, write_csv, write_text


LOCKED_RUN = "score_proj_l2h_lam0p20"
LOCKED_SEEDS = [42, 43, 44]
LOCK_PATH = EXP13_CONFIG_DIR / "exp13_formal_locked.yaml"
REPORT_PATH = EXP13_REPORTS_DIR / "exp13_formal_lock_report.md"
CHECKS_PATH = EXP13_TABLES_DIR / "exp13_formal_lock_checks.csv"


def _add(rows: list[dict[str, Any]], check_name: str, status: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if status else "FAIL", "details": details})


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _bool_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def validate(allow_run_override: bool = False) -> list[dict[str, Any]]:
    ensure_exp13_dirs()
    rows: list[dict[str, Any]] = []

    _add(rows, "formal lock YAML exists", LOCK_PATH.exists(), relpath(LOCK_PATH))
    if not LOCK_PATH.exists():
        write_csv(CHECKS_PATH, rows)
        return rows

    config = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8")) or {}
    locked_runs = list(config.get("locked_runs") or [])
    locked_seeds = [int(seed) for seed in config.get("locked_seeds") or []]
    selection_source = str(config.get("selection_source") or "")
    selection_path = _repo_path(selection_source)

    runs_ok = locked_runs == [LOCKED_RUN] or (allow_run_override and LOCKED_RUN in locked_runs)
    _add(rows, "locked run is score_proj_l2h_lam0p20", runs_ok, locked_runs)
    _add(rows, "locked seeds are 42/43/44", locked_seeds == LOCKED_SEEDS, locked_seeds)
    _add(rows, "selection source is configured", bool(selection_source), selection_source)
    _add(rows, "selection source CSV exists", selection_path.exists(), relpath(selection_path))

    source_rows: list[dict[str, str]] = []
    headers: list[str] = []
    if selection_path.exists():
        source_rows = read_csv(selection_path)
        headers = list(source_rows[0].keys()) if source_rows else []
    _add(rows, "selection source CSV has no test columns", not any(header.startswith("test_") for header in headers), headers)

    locked_source_rows = [row for row in source_rows if row.get("run_name") == LOCKED_RUN]
    _add(rows, "selection source contains locked run", bool(locked_source_rows), LOCKED_RUN)
    _add(
        rows,
        "locked run uses_test_for_selection is false",
        bool(locked_source_rows) and all(_bool_false(row.get("uses_test_for_selection", "")) for row in locked_source_rows),
        [row.get("uses_test_for_selection", "") for row in locked_source_rows],
    )

    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp13 Formal Lock Report",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "Formal config was locked from dev-only scout results.",
        "Test metrics are only for final held-out evaluation.",
        "No test labels are used for tuning.",
        "",
        "## Locked Contract",
        "",
        f"- Locked from commit: `{config.get('locked_from_commit', '')}`",
        f"- Locked mode: `{config.get('locked_from_mode', '')}`",
        f"- Selection source: `{selection_source}`",
        f"- Locked runs: `{', '.join(str(run) for run in locked_runs)}`",
        f"- Locked seeds: `{', '.join(str(seed) for seed in locked_seeds)}`",
        f"- Selection rule: `{config.get('selection_rule', '')}`",
        f"- Selection delta: `{config.get('selection_delta', '')}`",
        f"- Eval test: `{config.get('eval_test', '')}`",
        "",
        "## Checks",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_csv(CHECKS_PATH, rows)
    write_text(REPORT_PATH, "\n".join(lines))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Exp13 formal-lock metadata.")
    parser.add_argument(
        "--allow-run-override",
        action="store_true",
        help="Allow a user-edited lock file to contain extra runs, while still requiring the locked run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = validate(allow_run_override=args.allow_run_override)
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"Exp13 formal lock validation failed. See {relpath(REPORT_PATH)}")
    print("Exp13 formal lock validation PASS")


if __name__ == "__main__":
    main()
