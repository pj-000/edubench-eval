"""Sanity checks for available Exp3 run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp03 import EXP03_OUTPUT_DIR, EXP03_TABLES_DIR, TEMPLATE_NAMES, ensure_exp03_dirs, template_run_dir
from thesis_exp.src.edujudge.exp03.collect_exp03_results import collect_exp03_results
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_text


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def check_jsonl(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "missing"
    rows = read_jsonl(path)
    required = {"record_id", "label_5", "pred_label_5", "pred_score_expected", "prob_1", "prob_5"}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    return ("PASS" if rows and not missing else "FAIL", f"rows={len(rows)} missing={missing}")


def check_arrays(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "missing"
    arrays = np.load(path, allow_pickle=True)
    required = {"logits_dev", "logits_test", "probs_dev", "probs_test", "labels_dev", "labels_test", "record_ids_dev", "record_ids_test"}
    missing = sorted(required - set(arrays.files))
    if missing:
        return "FAIL", f"missing={missing}"
    logits_dev_shape = arrays["logits_dev"].shape
    logits_test_shape = arrays["logits_test"].shape
    classes = logits_dev_shape[1] if len(logits_dev_shape) > 1 else "NA"
    return "PASS", f"all required arrays present; dev={logits_dev_shape[0]} test={logits_test_shape[0]} classes={classes}"


def run_output_sanity(allow_pending: bool = True) -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    summary = collect_exp03_results()
    rows: list[dict[str, Any]] = []
    summary_by_template = {row["template_name"]: row for row in summary}
    for template_name in TEMPLATE_NAMES:
        status = str(summary_by_template.get(template_name, {}).get("status") or "pending")
        run_dir = template_run_dir(template_name)
        if status == "pending" and allow_pending:
            add(rows, f"{template_name} run status", "PASS", "pending", "pending allowed")
            continue
        add(rows, f"{template_name} run status", "PASS" if status in {"completed", "reused_exp02"} else "FAIL", status, "completed/reused_exp02")
        for split in ["dev", "test"]:
            pred_status, pred_detail = check_jsonl(run_dir / "predictions" / f"predictions_{split}.jsonl")
            add(rows, f"{template_name} predictions_{split}", pred_status, pred_detail, "readable prediction JSONL")
        arr_status, arr_detail = check_arrays(run_dir / "arrays" / "dev_test_arrays.npz")
        add(rows, f"{template_name} dev_test_arrays.npz", arr_status, arr_detail, "required array keys")
        for table in ["metrics_summary.csv", "per_bin_metrics.csv", "low_score_metrics.csv", "high_score_metrics.csv", "metric_level_metrics.csv", "scenario_level_metrics.csv"]:
            path = run_dir / "tables" / table
            add(rows, f"{template_name} {table}", "PASS" if path.exists() else "FAIL", relpath(path) if path.exists() else "missing", "exists")

    write_csv(EXP03_TABLES_DIR / "sanity_check_exp03_outputs.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP03_OUTPUT_DIR / "sanity_check_exp03_outputs.md",
        f"""# Exp3 Output Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check available Exp3 outputs.")
    parser.add_argument("--strict", action="store_true", help="Require all templates to be completed/reused.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_output_sanity(allow_pending=not args.strict)
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp3 output sanity statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP03_OUTPUT_DIR / 'sanity_check_exp03_outputs.md')}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
