"""Sanity checks for Exp4 objective outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp04 import (
    EXPECTED_SPLIT_ROWS,
    EXP04_OUTPUT_DIR,
    EXP04_TABLES_DIR,
    OBJECTIVE_IDS,
    ensure_exp04_dirs,
    run_dir,
)
from thesis_exp.src.edujudge.exp04.collect_exp04_results import collect_exp04_results
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = str(row.get(col, "")).replace("|", "\\|")
            if len(value) > 100:
                value = value[:97] + "..."
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_jsonl(path: Path, split: str) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "missing"
    rows = read_jsonl(path)
    required = {"record_id", "label_5", "human_mean_5", "pred_label_5", "pred_score_expected"}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    expected = EXPECTED_SPLIT_ROWS[split]
    status = "PASS" if len(rows) == expected and not missing else "FAIL"
    return status, f"rows={len(rows)} expected={expected} missing={missing}"


def check_arrays(path: Path, objective_id: str) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "missing"
    arrays = np.load(path, allow_pickle=True)
    required = {
        "logits_dev",
        "logits_test",
        "probs_dev",
        "probs_test",
        "labels_dev",
        "labels_test",
        "record_ids_dev",
        "record_ids_test",
    }
    missing = sorted(required - set(arrays.files))
    if missing:
        return "FAIL", f"missing={missing}"
    expected_outputs = 5 if objective_id == "O1_classification" else 1 if objective_id.startswith("O2_") else 4
    logits_shape = arrays["logits_test"].shape
    observed_outputs = logits_shape[1] if len(logits_shape) > 1 else "NA"
    output_ok = observed_outputs == expected_outputs
    status = "PASS" if output_ok else "FAIL"
    return status, f"test={logits_shape[0]} outputs={observed_outputs} expected_outputs={expected_outputs}"


def run_output_sanity(allow_pending: bool = True) -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    summary = collect_exp04_results()
    rows: list[dict[str, Any]] = []
    by_objective = {row["objective_id"]: row for row in summary}
    for objective_id in OBJECTIVE_IDS:
        status = str(by_objective.get(objective_id, {}).get("status") or "pending")
        path = run_dir(objective_id)
        if status == "pending" and allow_pending:
            add(rows, f"{objective_id} run status", "PASS", "pending", "pending allowed")
            continue
        add(
            rows,
            f"{objective_id} run status",
            "PASS" if status in {"completed", "reused_exp03_a4", "eval_only"} else "FAIL",
            status,
            "completed/reused/eval_only",
        )
        for split in ["dev", "test"]:
            pred_status, pred_detail = check_jsonl(path / "predictions" / f"predictions_{split}.jsonl", split)
            add(rows, f"{objective_id} predictions_{split}", pred_status, pred_detail, "readable prediction JSONL")
        arr_status, arr_detail = check_arrays(path / "arrays" / "dev_test_arrays.npz", objective_id)
        add(rows, f"{objective_id} dev_test_arrays.npz", arr_status, arr_detail, "required array keys")
        for table in [
            "metrics_summary.csv",
            "per_bin_metrics.csv",
            "low_score_metrics.csv",
            "high_score_metrics.csv",
            "metric_level_metrics.csv",
            "scenario_level_metrics.csv",
            "score_distribution.csv",
        ]:
            table_path = path / "tables" / table
            add(rows, f"{objective_id} {table}", "PASS" if table_path.exists() else "FAIL", relpath(table_path) if table_path.exists() else "missing", "exists")
        metadata_path = path / "run_metadata.json"
        if metadata_path.exists():
            json.loads(metadata_path.read_text(encoding="utf-8"))
        add(rows, f"{objective_id} run_metadata.json", "PASS" if metadata_path.exists() else "FAIL", relpath(metadata_path) if metadata_path.exists() else "missing", "valid JSON")

    write_csv(EXP04_TABLES_DIR / "sanity_check_exp04_outputs.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP04_OUTPUT_DIR / "sanity_check_exp04_outputs.md",
        f"""# Exp4 Output Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp4 objective outputs.")
    parser.add_argument("--strict", action="store_true", help="Require all O1-O3 outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_output_sanity(allow_pending=not args.strict)
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp4 output sanity statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP04_OUTPUT_DIR / 'sanity_check_exp04_outputs.md')}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
