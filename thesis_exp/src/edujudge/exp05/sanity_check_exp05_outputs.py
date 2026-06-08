"""Sanity checks for Exp5 L1 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp05 import EXP05_OUTPUT_DIR, EXP05_TABLES_DIR, L1_RUN_ID, ensure_exp05_dirs, l1_run_dir
from thesis_exp.src.edujudge.exp05.collect_exp05_results import collect_exp05_results
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


def check_jsonl(path: Path, max_rows: int | None) -> tuple[str, str]:
    if not path.exists():
        return "FAIL", "missing"
    rows = read_jsonl(path)
    required = {
        "record_id",
        "label_5",
        "human_mean_5",
        "pred_label",
        "pred_label_5",
        "pred_score_expected",
        "prob_gt_1",
        "prob_gt_2",
        "prob_gt_3",
        "prob_gt_4",
        "logit_gt_1",
        "logit_gt_2",
        "logit_gt_3",
        "logit_gt_4",
    }
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    row_ok = len(rows) > 0 and (max_rows is None or len(rows) <= max_rows)
    return ("PASS" if row_ok and not missing else "FAIL", f"rows={len(rows)} missing={missing}")


def check_arrays(path: Path, max_rows: int | None) -> tuple[str, str]:
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
    outputs_ok = arrays["logits_test"].shape[1] == 4 if "logits_test" in arrays.files else False
    rows_ok = True
    if max_rows is not None and "logits_test" in arrays.files:
        rows_ok = arrays["logits_test"].shape[0] <= max_rows
    status = "PASS" if not missing and outputs_ok and rows_ok else "FAIL"
    return status, f"test_shape={arrays['logits_test'].shape if 'logits_test' in arrays.files else 'NA'} missing={missing}"


def check_checkpoint_selection(rows: list[dict[str, Any]], run_path: Path, allow_pending: bool) -> None:
    history_path = run_path / "tables" / "dev_metrics_history.csv"
    metadata_path = run_path / "run_metadata.json"
    if not history_path.exists() and allow_pending:
        add(rows, "L1 checkpoint selection", "PASS", "pending", "pending allowed")
        return
    if not history_path.exists() or not metadata_path.exists():
        add(rows, "L1 checkpoint selection", "FAIL", "missing history/metadata", "exists")
        return
    history = read_csv(history_path)
    best = min(history, key=lambda row: float(row["MAE_label"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value_ok = abs(float(best["MAE_label"]) - float(metadata.get("best_selection_metric_value"))) <= 1e-8
    metric_ok = metadata.get("best_selection_metric_name") == "dev_MAE_label"
    add(
        rows,
        "L1 checkpoint selection",
        "PASS" if value_ok and metric_ok else "FAIL",
        f"epoch={metadata.get('best_epoch')} value={metadata.get('best_selection_metric_value')}",
        f"dev_MAE_label min={best['MAE_label']}",
    )


def run_output_sanity(allow_pending: bool = True, smoke: bool = False) -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    if not smoke:
        collect_exp05_results()
    rows: list[dict[str, Any]] = []
    run_path = l1_run_dir(smoke=smoke)
    max_rows = 8 if smoke else None
    status = "completed" if (run_path / "tables" / "metrics_summary.csv").exists() else "pending"
    if status == "pending" and allow_pending:
        add(rows, f"{L1_RUN_ID} run status", "PASS", "pending", "pending allowed")
    else:
        add(rows, f"{L1_RUN_ID} run status", "PASS" if status == "completed" else "FAIL", status, "completed")
        for split in ["dev", "test"]:
            pred_status, pred_detail = check_jsonl(run_path / "predictions" / f"predictions_{split}.jsonl", max_rows)
            add(rows, f"L1 predictions_{split}", pred_status, pred_detail, "readable JSONL")
        arr_status, arr_detail = check_arrays(run_path / "arrays" / "dev_test_arrays.npz", max_rows)
        add(rows, "L1 dev_test_arrays.npz", arr_status, arr_detail, "required array keys, logits dim=4")
        for table in [
            "metrics_summary.csv",
            "per_bin_metrics.csv",
            "low_score_metrics.csv",
            "high_score_metrics.csv",
            "metric_level_metrics.csv",
            "scenario_level_metrics.csv",
        ]:
            path = run_path / "tables" / table
            add(rows, f"L1 {table}", "PASS" if path.exists() else "FAIL", relpath(path) if path.exists() else "missing", "exists")
        check_checkpoint_selection(rows, run_path, allow_pending=allow_pending or smoke)
    suffix = "_smoke" if smoke else ""
    write_csv(EXP05_TABLES_DIR / f"sanity_check_exp05_outputs{suffix}.csv", rows)
    if not smoke:
        write_csv(EXP05_TABLES_DIR / "sanity_check_exp05_outputs.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    out_path = EXP05_OUTPUT_DIR / ("sanity_check_exp05_outputs_smoke.md" if smoke else "sanity_check_exp05_outputs.md")
    write_text(
        out_path,
        f"""# Exp5 Output Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp5 L1 outputs.")
    parser.add_argument("--strict", action="store_true", help="Require L1 outputs.")
    parser.add_argument("--smoke", action="store_true", help="Check smoke-test output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_output_sanity(allow_pending=not args.strict, smoke=args.smoke)
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp5 output sanity statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / ('sanity_check_exp05_outputs_smoke.md' if args.smoke else 'sanity_check_exp05_outputs.md'))}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
