"""Collect available Exp3 input-ablation outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import (
    EXP03_ARRAYS_DIR,
    EXP03_PREDICTIONS_DIR,
    EXP03_TABLES_DIR,
    TEMPLATE_NAMES,
    ensure_exp03_dirs,
    template_run_dir,
)
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import build_exp03_datasets
from thesis_exp.src.edujudge.exp03.templates import get_template_spec
from thesis_exp.src.edujudge.exp03.train_input_ablation import normalize_run_files, reuse_exp02_outputs
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv


DEFAULT_MODEL_NAME = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_row(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("split") == split), {})


def metadata_status(run_dir: Path, template_name: str) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = str(metadata.get("status") or "")
        if status:
            return status
    if (run_dir / "tables" / "metrics_summary.csv").exists():
        return "completed"
    if template_name == "A2_question_answer_metric":
        return "reused_exp02"
    return "pending"


def weighted_length_summary(template_name: str) -> dict[str, Any]:
    path = EXP03_TABLES_DIR / "template_length_stats.csv"
    if not path.exists():
        build_exp03_datasets()
    rows = [row for row in read_csv(path) if row.get("template_name") == template_name]
    if not rows:
        return {"mean_token_length": None, "truncation_rate": None}
    total_n = sum(int(float(row.get("n") or 0)) for row in rows)
    if not total_n:
        return {"mean_token_length": None, "truncation_rate": None}
    mean_tokens = sum((float(row.get("mean_token_length") or 0) * int(float(row.get("n") or 0))) for row in rows) / total_n
    trunc_rate = sum((float(row.get("truncation_rate") or 0) * int(float(row.get("n") or 0))) for row in rows) / total_n
    return {"mean_token_length": mean_tokens, "truncation_rate": trunc_rate}


def copy_available_artifacts(template_name: str, run_dir: Path) -> None:
    for split in ["dev", "test"]:
        src = run_dir / "predictions" / f"predictions_{split}.jsonl"
        if src.exists():
            dst = EXP03_PREDICTIONS_DIR / f"{template_name}_predictions_{split}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    arrays_src = run_dir / "arrays" / "dev_test_arrays.npz"
    if arrays_src.exists():
        arrays_dst = EXP03_ARRAYS_DIR / f"{template_name}_dev_test_arrays.npz"
        arrays_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arrays_src, arrays_dst)


def collect_table(template_name: str, status: str, run_dir: Path, table_name: str) -> list[dict[str, Any]]:
    path = run_dir / "tables" / table_name
    if not path.exists():
        return []
    spec = get_template_spec(template_name)
    rows = []
    for row in read_csv(path):
        rows.append(
            {
                "template_name": template_name,
                "ablation_id": spec.ablation_id,
                "status": status,
                **row,
            }
        )
    return rows


def summarize_template(template_name: str, run_dir: Path) -> dict[str, Any]:
    spec = get_template_spec(template_name)
    if template_name == "A2_question_answer_metric" and not (run_dir / "tables" / "metrics_summary.csv").exists():
        reuse_exp02_outputs(run_dir, DEFAULT_MODEL_NAME)
    if (run_dir / "tables" / "metrics_summary.csv").exists():
        normalize_run_files(run_dir, template_name, DEFAULT_MODEL_NAME, metadata_status(run_dir, template_name))

    status = metadata_status(run_dir, template_name)
    length = weighted_length_summary(template_name)
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    if not metrics_path.exists():
        return {
            "template_name": template_name,
            "ablation_id": spec.ablation_id,
            "status": status if status != "reused_exp02" else "pending",
            "model_name": "",
            "input_fields": ", ".join(spec.input_fields),
            "test_accuracy": None,
            "test_MAE_label": None,
            "test_MAE_expected": None,
            "test_signed_bias": None,
            "test_macro_f1": None,
            "test_qwk": None,
            "test_kendall_tau": None,
            "test_acc_at_1": None,
            "test_acc_at_2": None,
            "test_acc_at_5": None,
            "test_low_to_high_rate": None,
            "test_low_signed_bias": None,
            "test_high_to_mid_or_low_rate": None,
            "mean_token_length": length["mean_token_length"],
            "truncation_rate": length["truncation_rate"],
            "run_dir": relpath(run_dir),
        }

    copy_available_artifacts(template_name, run_dir)
    metrics = read_csv(metrics_path)
    test = first_row(metrics, "test")
    return {
        "template_name": template_name,
        "ablation_id": spec.ablation_id,
        "status": status,
        "model_name": DEFAULT_MODEL_NAME,
        "input_fields": ", ".join(spec.input_fields),
        "test_accuracy": float_or_none(test.get("Accuracy") or test.get("Exact Match")),
        "test_MAE_label": float_or_none(test.get("MAE_label")),
        "test_MAE_expected": float_or_none(test.get("MAE_expected")),
        "test_signed_bias": float_or_none(test.get("Signed Bias label")),
        "test_macro_f1": float_or_none(test.get("Macro-F1")),
        "test_qwk": float_or_none(test.get("Quadratic Weighted Kappa")),
        "test_kendall_tau": float_or_none(test.get("Kendall tau")),
        "test_acc_at_1": float_or_none(test.get("Acc@1")),
        "test_acc_at_2": float_or_none(test.get("Acc@2")),
        "test_acc_at_5": float_or_none(test.get("Acc@5")),
        "test_low_to_high_rate": float_or_none(test.get("low_to_high_rate")),
        "test_low_signed_bias": float_or_none(test.get("low_signed_bias")),
        "test_high_to_mid_or_low_rate": float_or_none(test.get("high_to_mid_or_low_rate")),
        "mean_token_length": length["mean_token_length"],
        "truncation_rate": length["truncation_rate"],
        "run_dir": relpath(run_dir),
    }


def collect_exp03_results() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    if not (EXP03_TABLES_DIR / "template_length_stats.csv").exists():
        build_exp03_datasets()
    summary_rows = []
    low_rows: list[dict[str, Any]] = []
    per_bin_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for template_name in TEMPLATE_NAMES:
        run_dir = template_run_dir(template_name)
        summary = summarize_template(template_name, run_dir)
        summary_rows.append(summary)
        status = str(summary["status"])
        if status in {"completed", "reused_exp02"}:
            low_rows.extend(collect_table(template_name, status, run_dir, "low_score_metrics.csv"))
            per_bin_rows.extend(collect_table(template_name, status, run_dir, "per_bin_metrics.csv"))
            metric_rows.extend(collect_table(template_name, status, run_dir, "metric_level_metrics.csv"))
            scenario_rows.extend(collect_table(template_name, status, run_dir, "scenario_level_metrics.csv"))

    write_csv(EXP03_TABLES_DIR / "input_ablation_summary.csv", summary_rows)
    write_csv(EXP03_TABLES_DIR / "input_ablation_low_score.csv", low_rows)
    write_csv(EXP03_TABLES_DIR / "input_ablation_per_bin.csv", per_bin_rows)
    write_csv(EXP03_TABLES_DIR / "input_ablation_metric_level.csv", metric_rows)
    write_csv(EXP03_TABLES_DIR / "input_ablation_scenario_level.csv", scenario_rows)
    token_rows = read_csv(EXP03_TABLES_DIR / "template_length_stats.csv")
    write_csv(EXP03_TABLES_DIR / "input_ablation_token_length.csv", token_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp3 input-ablation outputs.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = collect_exp03_results()
    statuses = {row["template_name"]: row["status"] for row in rows}
    print(f"Exp3 collected templates: {statuses}")
    print(f"Output: {relpath(EXP03_TABLES_DIR / 'input_ablation_summary.csv')}")


if __name__ == "__main__":
    main()
