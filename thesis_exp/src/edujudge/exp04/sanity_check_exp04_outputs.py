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
    checkpoint_dir,
    ensure_exp04_dirs,
    run_dir,
)
from thesis_exp.src.edujudge.exp04.collect_exp04_results import collect_exp04_results
from thesis_exp.src.edujudge.utils.io import read_csv, read_jsonl, relpath, write_csv, write_text


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


def checkpoint_selection_markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "objective_id",
        "selection_metric",
        "expected_best_epoch",
        "actual_best_epoch",
        "expected_best_value",
        "actual_best_value",
        "status",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.10f}"
    except (TypeError, ValueError):
        return ""


def best_history_row(path: Path, metric: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        try:
            candidates.append((float(row[metric]), row))
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        return {}
    return min(candidates, key=lambda item: item[0])[1]


def check_checkpoint_selection(status_by_objective: dict[str, str], allow_pending: bool) -> list[dict[str, Any]]:
    checks = [
        ("O2_regression_smoothl1", "MAE_expected", "dev_MAE_expected"),
        ("O3_ordinal", "MAE_label", "dev_MAE_label"),
    ]
    rows: list[dict[str, Any]] = []
    for objective_id, history_metric, metadata_metric in checks:
        if status_by_objective.get(objective_id) == "pending" and allow_pending:
            rows.append(
                {
                    "objective_id": objective_id,
                    "selection_metric": metadata_metric,
                    "expected_best_epoch": "pending",
                    "actual_best_epoch": "pending",
                    "expected_best_value": "",
                    "actual_best_value": "",
                    "status": "PASS",
                }
            )
            continue
        expected = best_history_row(run_dir(objective_id) / "tables" / "dev_metrics_history.csv", history_metric)
        checkpoint_metadata = load_json(checkpoint_dir(objective_id) / "best" / "exp04_head_metadata.json")
        run_metadata = load_json(run_dir(objective_id) / "run_metadata.json")
        metadata = checkpoint_metadata or run_metadata
        expected_epoch = expected.get("epoch")
        expected_value = expected.get(history_metric)
        actual_epoch = metadata.get("best_epoch")
        actual_value = metadata.get("best_selection_metric_value")
        actual_metric = metadata.get("best_selection_metric_name")
        try:
            epoch_ok = int(float(actual_epoch)) == int(float(expected_epoch))
        except (TypeError, ValueError):
            epoch_ok = False
        try:
            value_ok = abs(float(actual_value) - float(expected_value)) <= 1e-8
        except (TypeError, ValueError):
            value_ok = False
        metric_ok = actual_metric == metadata_metric
        rows.append(
            {
                "objective_id": objective_id,
                "selection_metric": metadata_metric,
                "expected_best_epoch": expected_epoch or "",
                "actual_best_epoch": actual_epoch or "",
                "expected_best_value": format_float(expected_value),
                "actual_best_value": format_float(actual_value),
                "status": "PASS" if epoch_ok and value_ok and metric_ok else "FAIL",
            }
        )
    return rows


def run_output_sanity(allow_pending: bool = True) -> list[dict[str, Any]]:
    ensure_exp04_dirs()
    summary = collect_exp04_results()
    rows: list[dict[str, Any]] = []
    by_objective = {row["objective_id"]: row for row in summary}
    status_by_objective = {objective_id: str(row.get("status") or "pending") for objective_id, row in by_objective.items()}
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

    selection_rows = check_checkpoint_selection(status_by_objective, allow_pending)
    for row in selection_rows:
        add(
            rows,
            f"{row['objective_id']} checkpoint selection",
            row["status"],
            f"epoch={row['actual_best_epoch']} value={row['actual_best_value']}",
            f"epoch={row['expected_best_epoch']} value={row['expected_best_value']}",
            row["selection_metric"],
        )

    write_csv(EXP04_TABLES_DIR / "sanity_check_exp04_outputs.csv", rows)
    write_csv(
        EXP04_TABLES_DIR / "checkpoint_selection_sanity.csv",
        selection_rows,
        fieldnames=[
            "objective_id",
            "selection_metric",
            "expected_best_epoch",
            "actual_best_epoch",
            "expected_best_value",
            "actual_best_value",
            "status",
        ],
    )
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP04_OUTPUT_DIR / "sanity_check_exp04_outputs.md",
        f"""# Exp4 Output Sanity Check

Overall status: **{overall}**

## Checkpoint Selection

{checkpoint_selection_markdown_table(selection_rows)}

## Output Files

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
