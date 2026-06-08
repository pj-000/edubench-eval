"""Sanity checks for Exp5 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp05 import (
    EXPECTED_SPLIT_ROWS,
    EXP05_OUTPUT_DIR,
    EXP05_TABLES_DIR,
    L1_RUN_ID,
    L2_RUN_CONFIGS,
    ensure_exp05_dirs,
    l1_run_dir,
    l2_run_dir,
)
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


def check_jsonl(path: Path, expected_rows: int | None, max_rows: int | None) -> tuple[str, str]:
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
    row_ok = len(rows) > 0
    if expected_rows is not None:
        row_ok = row_ok and len(rows) == expected_rows
    if max_rows is not None:
        row_ok = row_ok and len(rows) <= max_rows
    return ("PASS" if row_ok and not missing else "FAIL", f"rows={len(rows)} missing={missing}")


def check_arrays(path: Path, expected_rows: int | None, max_rows: int | None) -> tuple[str, str]:
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
    if expected_rows is not None and "logits_test" in arrays.files:
        rows_ok = arrays["logits_test"].shape[0] == expected_rows
    if max_rows is not None and "logits_test" in arrays.files:
        rows_ok = rows_ok and arrays["logits_test"].shape[0] <= max_rows
    status = "PASS" if not missing and outputs_ok and rows_ok else "FAIL"
    return status, f"test_shape={arrays['logits_test'].shape if 'logits_test' in arrays.files else 'NA'} missing={missing}"


def check_checkpoint_selection(rows: list[dict[str, Any]], run_id: str, run_path: Path, allow_pending: bool) -> None:
    history_path = run_path / "tables" / "dev_metrics_history.csv"
    metadata_path = run_path / "run_metadata.json"
    if not history_path.exists() and allow_pending:
        add(rows, f"{run_id} checkpoint selection", "PASS", "pending", "pending allowed")
        return
    if not history_path.exists() or not metadata_path.exists():
        add(rows, f"{run_id} checkpoint selection", "FAIL", "missing history/metadata", "exists")
        return
    history = read_csv(history_path)
    best = min(history, key=lambda row: float(row["MAE_label"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value_ok = abs(float(best["MAE_label"]) - float(metadata.get("best_selection_metric_value"))) <= 1e-8
    metric_ok = metadata.get("best_selection_metric_name") == "dev_MAE_label"
    add(
        rows,
        f"{run_id} checkpoint selection",
        "PASS" if value_ok and metric_ok else "FAIL",
        f"epoch={metadata.get('best_epoch')} value={metadata.get('best_selection_metric_value')}",
        f"dev_MAE_label min={best['MAE_label']}",
    )


def expected_rows_for(split: str, smoke: bool) -> int | None:
    return None if smoke else EXPECTED_SPLIT_ROWS[split]


def check_run_outputs(
    rows: list[dict[str, Any]],
    run_id: str,
    run_path: Path,
    allow_pending: bool,
    smoke: bool,
    require_loss_debug: bool,
) -> None:
    max_rows = 8 if smoke else None
    status = "completed" if (run_path / "tables" / "metrics_summary.csv").exists() else "pending"
    if status == "pending" and allow_pending:
        add(rows, f"{run_id} run status", "PASS", "pending", "pending allowed")
        return
    add(rows, f"{run_id} run status", "PASS" if status == "completed" else "FAIL", status, "completed")
    for split in ["dev", "test"]:
        pred_status, pred_detail = check_jsonl(
            run_path / "predictions" / f"predictions_{split}.jsonl",
            expected_rows=expected_rows_for(split, smoke),
            max_rows=max_rows,
        )
        add(rows, f"{run_id} predictions_{split}", pred_status, pred_detail, "readable JSONL")
    arr_status, arr_detail = check_arrays(
        run_path / "arrays" / "dev_test_arrays.npz",
        expected_rows=expected_rows_for("test", smoke),
        max_rows=max_rows,
    )
    add(rows, f"{run_id} dev_test_arrays.npz", arr_status, arr_detail, "required array keys, logits dim=4")
    for table in [
        "metrics_summary.csv",
        "per_bin_metrics.csv",
        "low_score_metrics.csv",
        "high_score_metrics.csv",
        "metric_level_metrics.csv",
        "scenario_level_metrics.csv",
        "score_distribution.csv",
    ]:
        path = run_path / "tables" / table
        add(rows, f"{run_id} {table}", "PASS" if path.exists() else "FAIL", relpath(path) if path.exists() else "missing", "exists")
    if require_loss_debug:
        debug_path = run_path / "logs" / "loss_debug_history.csv"
        add(rows, f"{run_id} loss_debug_history.csv", "PASS" if debug_path.exists() else "FAIL", relpath(debug_path) if debug_path.exists() else "missing", "exists")
        summary_path = run_path / "run_summary.md"
        summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        config_ok = all(fragment in summary for fragment in ["lambda_low", "margin", "use_class_weights: `false`"])
        add(rows, f"{run_id} config recorded in run_summary", "PASS" if config_ok else "FAIL", relpath(summary_path) if summary_path.exists() else "missing", "lambda/margin/no class weights")
    metadata_path = run_path / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metric_ok = metadata.get("best_selection_metric_name") in {"dev_MAE_label", None}
    if require_loss_debug and status == "completed":
        metric_ok = metadata.get("best_selection_metric_name") == "dev_MAE_label"
    add(rows, f"{run_id} selection metric", "PASS" if metric_ok else "FAIL", metadata.get("best_selection_metric_name"), "dev_MAE_label")
    if require_loss_debug and status == "completed":
        add(
            rows,
            f"{run_id} test after best checkpoint",
            "PASS" if metadata.get("test_evaluated_after_best_checkpoint") is True else "FAIL",
            metadata.get("test_evaluated_after_best_checkpoint"),
            "true",
        )
    check_checkpoint_selection(rows, run_id, run_path, allow_pending=allow_pending or smoke)


def check_no_tracked_weights(rows: list[dict[str, Any]]) -> None:
    import subprocess

    from thesis_exp.src.edujudge.utils.io import REPO_ROOT

    result = subprocess.run(
        ["git", "ls-files", "thesis_exp/artifacts", "*.safetensors", "*.pt", "*.pth", "*.bin", "*/hf_cache/*"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    add(rows, "no checkpoint/weights tracked", "PASS" if result.returncode == 0 and not tracked else "FAIL", tracked, "[]")


def run_output_sanity(allow_pending: bool = True, smoke: bool = False, l2_only: bool = False) -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    if not smoke:
        collect_exp05_results()
    rows: list[dict[str, Any]] = []
    if not l2_only:
        check_run_outputs(rows, L1_RUN_ID, l1_run_dir(smoke=smoke), allow_pending, smoke, require_loss_debug=False)
    for run_id in L2_RUN_CONFIGS:
        check_run_outputs(rows, run_id, l2_run_dir(run_id, smoke=smoke), allow_pending, smoke, require_loss_debug=True)
    if not smoke:
        summary = read_csv(EXP05_TABLES_DIR / "loss_ablation_summary.csv") if (EXP05_TABLES_DIR / "loss_ablation_summary.csv").exists() else []
        present = {row.get("loss_id"): row.get("status") for row in summary}
        add(rows, "L0/L1 metrics present in summary", "PASS" if "L0_exp04_o3_ordinal" in present and L1_RUN_ID in present else "FAIL", present, "L0 and L1 present")
    check_no_tracked_weights(rows)
    suffix = "_l2_smoke" if smoke and l2_only else "_smoke" if smoke else ""
    write_csv(EXP05_TABLES_DIR / f"sanity_check_exp05_outputs{suffix}.csv", rows)
    if not smoke:
        write_csv(EXP05_TABLES_DIR / "sanity_check_exp05_outputs.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    if smoke and l2_only:
        out_path = EXP05_OUTPUT_DIR / "sanity_check_exp05_outputs_l2_smoke.md"
    else:
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
    parser = argparse.ArgumentParser(description="Check Exp5 low-score loss outputs.")
    parser.add_argument("--strict", action="store_true", help="Require configured outputs.")
    parser.add_argument("--smoke", action="store_true", help="Check smoke-test output path.")
    parser.add_argument("--l2-only", action="store_true", help="Check only L2a/L2b outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_output_sanity(allow_pending=not args.strict, smoke=args.smoke, l2_only=args.l2_only)
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp5 output sanity statuses: {', '.join(statuses)}")
    if args.smoke and args.l2_only:
        out_path = EXP05_OUTPUT_DIR / "sanity_check_exp05_outputs_l2_smoke.md"
    else:
        out_path = EXP05_OUTPUT_DIR / ("sanity_check_exp05_outputs_smoke.md" if args.smoke else "sanity_check_exp05_outputs.md")
    print(f"Output: {relpath(out_path)}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
