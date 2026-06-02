"""Postprocess entry point for completed Exp2 CE baseline runs.

This module intentionally does only lightweight file/shape checks for now. It is
the stable hook to extend later with plots and paper tables after a formal run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp02 import EXP02_OUTPUT_DIR
from thesis_exp.src.edujudge.utils.io import relpath, write_text


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def add(rows: list[dict[str, Any]], artifact: str, path: Path, status: str, detail: str) -> None:
    rows.append({"artifact": artifact, "path": relpath(path), "status": status, "detail": detail})


def find_predictions_test(output_dir: Path) -> Path:
    candidates = [
        output_dir / "predictions" / "predictions_test.jsonl",
        output_dir / "predictions_test.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["artifact", "path", "status", "detail"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def postprocess(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tables_dir = output_dir / "tables"
    arrays_dir = output_dir / "arrays"
    required_csvs = [
        ("metrics_summary.csv", tables_dir / "metrics_summary.csv"),
        ("per_bin_metrics.csv", tables_dir / "per_bin_metrics.csv"),
        ("low_score_metrics.csv", tables_dir / "low_score_metrics.csv"),
    ]

    for name, path in required_csvs:
        if not path.exists():
            add(rows, name, path, "FAIL", "missing")
            continue
        csv_rows = read_csv_rows(path)
        add(rows, name, path, "PASS", f"rows={len(csv_rows)} columns={len(csv_rows[0]) if csv_rows else 0}")

    predictions_path = find_predictions_test(output_dir)
    if predictions_path.exists():
        prediction_rows = read_jsonl_rows(predictions_path)
        required_prediction_fields = {
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "prob_5",
            "logit_1",
            "logit_2",
            "logit_3",
            "logit_4",
            "logit_5",
            "pred_score_expected",
            "abs_error_label",
            "signed_error_label",
            "abs_error_expected",
            "signed_error_expected",
        }
        missing = sorted(required_prediction_fields - set(prediction_rows[0])) if prediction_rows else []
        status = "PASS" if prediction_rows and not missing else "FAIL"
        detail = f"rows={len(prediction_rows)} missing_fields={missing}"
        add(rows, "predictions_test.jsonl", predictions_path, status, detail)
    else:
        add(rows, "predictions_test.jsonl", predictions_path, "FAIL", "missing")

    arrays_path = arrays_dir / "exp02_dev_test_arrays.npz"
    if arrays_path.exists():
        arrays = np.load(arrays_path, allow_pickle=True)
        expected_keys = {
            "logits_dev",
            "logits_test",
            "probs_dev",
            "probs_test",
            "labels_dev",
            "labels_test",
            "record_ids_dev",
            "record_ids_test",
        }
        missing = sorted(expected_keys - set(arrays.files))
        shape_summary = {key: list(arrays[key].shape) for key in arrays.files}
        status = "PASS" if not missing else "FAIL"
        add(rows, "exp02_dev_test_arrays.npz", arrays_path, status, f"missing_keys={missing} shapes={shape_summary}")
    else:
        add(rows, "exp02_dev_test_arrays.npz", arrays_path, "FAIL", "missing")

    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        output_dir / "postprocess_check.md",
        f"""# Exp2 Postprocess Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check completed Exp2 CE baseline outputs.")
    parser.add_argument("--output_dir", type=Path, default=EXP02_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = postprocess(args.output_dir)
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp2 postprocess statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(args.output_dir / 'postprocess_check.md')}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
