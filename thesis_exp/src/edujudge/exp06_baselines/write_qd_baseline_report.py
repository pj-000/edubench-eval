"""Write review artifacts for Exp6 question-disjoint baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_baselines import (
    A4_QD_DATASET_DIR,
    EXP06_QD_CHECKPOINTS_DIR,
    EXP06_QD_REPORTS_DIR,
    EXP06_QD_RUNS_DIR,
    EXP06_QD_TABLES_DIR,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    ensure_exp06_qd_dirs,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, md_table, read_csv, relpath, write_csv, write_text
from thesis_exp.src.edujudge.utils.text_norm import stringify


RUN_IDS = [QD_B0_RUN_ID, QD_B1_RUN_ID]
METRIC_FIELDS = [
    "run_id",
    "status",
    "split",
    "n",
    "MAE_label",
    "MAE_expected",
    "Exact Match",
    "Macro-F1",
    "Quadratic Weighted Kappa",
    "low_to_high_rate",
    "epoch",
    "global_step",
    "checkpoint_state_dict_mb",
]


def safe_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def file_size_mb(path: Path) -> str:
    if not path.exists():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.1f}"


def run_status(run_id: str) -> str:
    metadata = safe_read_json(EXP06_QD_RUNS_DIR / run_id / "run_metadata.json")
    if metadata.get("status"):
        return stringify(metadata["status"])
    metrics_path = EXP06_QD_RUNS_DIR / run_id / "tables" / "metrics_summary.csv"
    log_path = EXP06_QD_REPORTS_DIR.parent / "logs" / f"{run_id}.log"
    if metrics_path.exists():
        return "completed_no_metadata"
    if log_path.exists():
        return "running_or_failed"
    return "pending"


def collect_run_rows(run_id: str) -> list[dict[str, Any]]:
    status = run_status(run_id)
    metrics_path = EXP06_QD_RUNS_DIR / run_id / "tables" / "metrics_summary.csv"
    checkpoint_path = EXP06_QD_CHECKPOINTS_DIR / run_id / "best" / "state_dict.pt"
    rows = safe_read_csv(metrics_path)
    if not rows:
        return [
            {
                "run_id": run_id,
                "status": status,
                "split": "",
                "checkpoint_state_dict_mb": file_size_mb(checkpoint_path),
            }
        ]
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {field: row.get(field, "") for field in METRIC_FIELDS if field not in {"run_id", "status", "checkpoint_state_dict_mb"}}
        item["run_id"] = run_id
        item["status"] = status
        item["checkpoint_state_dict_mb"] = file_size_mb(checkpoint_path)
        out.append(item)
    return out


def as_float(value: Any) -> float | None:
    try:
        return float(stringify(value))
    except ValueError:
        return None


def test_metric(rows: list[dict[str, Any]], run_id: str, metric: str) -> float | None:
    for row in rows:
        if row.get("run_id") == run_id and row.get("split") == "test":
            return as_float(row.get(metric))
    return None


def comparison_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ["MAE_label", "MAE_expected", "Exact Match", "Macro-F1", "Quadratic Weighted Kappa", "low_to_high_rate"]:
        b0 = test_metric(metric_rows, QD_B0_RUN_ID, metric)
        b1 = test_metric(metric_rows, QD_B1_RUN_ID, metric)
        rows.append(
            {
                "metric": metric,
                "QD-B0_test": "" if b0 is None else f"{b0:.6g}",
                "QD-B1_test": "" if b1 is None else f"{b1:.6g}",
                "B1_minus_B0": "" if b0 is None or b1 is None else f"{b1 - b0:.6g}",
            }
        )
    return rows


def tracked_checkpoint_files() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--", str(EXP06_QD_CHECKPOINTS_DIR.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [f"GIT_ERROR:{exc}"]
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def write_report() -> dict[str, Any]:
    ensure_exp06_qd_dirs()
    dataset_rows = safe_read_csv(EXP06_QD_TABLES_DIR / "qd_a4_dataset_sanity.csv")
    metric_rows: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        metric_rows.extend(collect_run_rows(run_id))
    compare = comparison_rows(metric_rows)
    tracked_ckpts = tracked_checkpoint_files()
    write_csv(EXP06_QD_TABLES_DIR / "qd_baseline_metrics_summary.csv", metric_rows, METRIC_FIELDS)
    write_csv(EXP06_QD_TABLES_DIR / "qd_baseline_test_comparison.csv", compare)

    complete_runs = {
        row.get("run_id")
        for row in metric_rows
        if row.get("status") in {"completed", "completed_no_metadata"} and row.get("split") == "test"
    }
    overall = "PASS" if set(RUN_IDS) <= complete_runs and not tracked_ckpts else "PENDING"
    if tracked_ckpts:
        overall = "FAIL"
    lines = [
        "# Exp6 Question-disjoint Baseline Review",
        "",
        f"Overall status: **{overall}**",
        "",
        "Scope: QD-B0 and QD-B1 rerun the human-only baselines on `question_seed42` so Exp6",
        "synthetic-generation results have a question-disjoint comparison point.",
        "",
        "## Dataset",
        "",
        md_table(dataset_rows, ["split", "rows", "expected_rows", "status", "path"], max_rows=10),
        "",
        f"Dataset card: `{relpath(A4_QD_DATASET_DIR / 'dataset_card.md')}`",
        "",
        "## Runs",
        "",
        md_table(metric_rows, METRIC_FIELDS, max_rows=12),
        "",
        "## Test Comparison",
        "",
        md_table(compare, ["metric", "QD-B0_test", "QD-B1_test", "B1_minus_B0"], max_rows=10),
        "",
        "## Artifact Guardrails",
        "",
        f"- Checkpoint directory: `{relpath(EXP06_QD_CHECKPOINTS_DIR)}`",
        f"- Tracked checkpoint files: **{len(tracked_ckpts)}**",
    ]
    if tracked_ckpts:
        lines.extend(["", "Tracked checkpoint paths:", ""])
        lines.extend(f"- `{path}`" for path in tracked_ckpts[:20])
    write_text(EXP06_QD_REPORTS_DIR / "qd_baseline_review.md", "\n".join(lines))
    return {"overall": overall, "metric_rows": len(metric_rows), "tracked_checkpoint_files": len(tracked_ckpts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = write_report()
    print(
        "Wrote Exp6 QD baseline report: "
        f"overall={result['overall']} rows={result['metric_rows']} "
        f"tracked_checkpoint_files={result['tracked_checkpoint_files']}"
    )


if __name__ == "__main__":
    main()
