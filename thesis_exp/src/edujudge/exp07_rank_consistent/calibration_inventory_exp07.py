"""Inventory Exp7/QD calibration inputs without modifying raw artifacts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_CHECKPOINTS_DIR,
    EXP07_OUTPUT_DIR,
    EXP07_REPORTS_DIR,
    EXP07_RUNS_DIR,
    EXP07_TABLES_DIR,
    EXP07_RUN_ID,
    QD_B0_RUN_ID,
    QD_B1_RUN_ID,
    QD_BASELINE_RUNS_DIR,
    ensure_exp07_dirs,
)
from thesis_exp.src.edujudge.utils.io import THESIS_DIR, relpath, write_csv, write_text


INVENTORY_FIELDS = [
    "run_id",
    "dev_logits_available",
    "test_logits_available",
    "dev_probs_available",
    "test_probs_available",
    "dev_labels_available",
    "test_labels_available",
    "dev_record_ids_available",
    "test_record_ids_available",
    "dev_predictions_available",
    "test_predictions_available",
    "arrays_available",
    "checkpoint_available",
    "calibration_ready",
    "blocking_reason",
]

CORE_KEYS = [
    "dev_logits_available",
    "test_logits_available",
    "dev_probs_available",
    "test_probs_available",
    "dev_labels_available",
    "test_labels_available",
    "dev_record_ids_available",
    "test_record_ids_available",
    "dev_predictions_available",
    "test_predictions_available",
    "arrays_available",
]


@dataclass(frozen=True)
class RunInventoryConfig:
    run_id: str
    run_dir: Path
    checkpoint_path: Path


RUNS = [
    RunInventoryConfig(
        run_id=QD_B0_RUN_ID,
        run_dir=QD_BASELINE_RUNS_DIR / QD_B0_RUN_ID,
        checkpoint_path=THESIS_DIR
        / "artifacts"
        / "exp06_question_disjoint_baselines"
        / "checkpoints"
        / QD_B0_RUN_ID
        / "best"
        / "state_dict.pt",
    ),
    RunInventoryConfig(
        run_id=QD_B1_RUN_ID,
        run_dir=QD_BASELINE_RUNS_DIR / QD_B1_RUN_ID,
        checkpoint_path=THESIS_DIR
        / "artifacts"
        / "exp06_question_disjoint_baselines"
        / "checkpoints"
        / QD_B1_RUN_ID
        / "best"
        / "state_dict.pt",
    ),
    RunInventoryConfig(
        run_id=EXP07_RUN_ID,
        run_dir=EXP07_RUNS_DIR / EXP07_RUN_ID,
        checkpoint_path=EXP07_CHECKPOINTS_DIR / EXP07_RUN_ID / "best" / "state_dict.pt",
    ),
]


def required_paths(run: RunInventoryConfig) -> dict[str, Path]:
    arrays_dir = run.run_dir / "arrays"
    return {
        "dev_logits_available": arrays_dir / "logits_dev.npy",
        "test_logits_available": arrays_dir / "logits_test.npy",
        "dev_probs_available": arrays_dir / "probs_dev.npy",
        "test_probs_available": arrays_dir / "probs_test.npy",
        "dev_labels_available": arrays_dir / "labels_dev.npy",
        "test_labels_available": arrays_dir / "labels_test.npy",
        "dev_record_ids_available": arrays_dir / "record_ids_dev.txt",
        "test_record_ids_available": arrays_dir / "record_ids_test.txt",
        "dev_predictions_available": run.run_dir / "predictions" / "predictions_dev.jsonl",
        "test_predictions_available": run.run_dir / "predictions" / "predictions_test.jsonl",
        "arrays_available": arrays_dir / "dev_test_arrays.npz",
        "checkpoint_available": run.checkpoint_path,
    }


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(THESIS_DIR.parent))


def _remote_existing_paths(
    paths: list[Path],
    *,
    host: str | None,
    port: int,
    repo: str,
) -> tuple[set[str], str]:
    if not host:
        return set(), "not_checked"
    rels = sorted({_repo_relative(path) for path in paths})
    quoted_paths = " ".join(shlex.quote(path) for path in rels)
    command = (
        f"cd {shlex.quote(repo)} && "
        f"for path in {quoted_paths}; do "
        'if [ -e "$path" ]; then printf "%s\\n" "$path"; fi; '
        "done"
    )
    result = subprocess.run(
        ["ssh", "-p", str(port), host, command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return set(), f"server_check_failed: {detail}"
    return set(result.stdout.splitlines()), "checked"


def _availability(path: Path, remote_paths: set[str]) -> str:
    if path.exists():
        return "yes_local"
    if _repo_relative(path) in remote_paths:
        return "yes_server"
    return "no"


def _ready_status(row: dict[str, str]) -> str:
    values = [row[key] for key in CORE_KEYS]
    if all(value == "yes_local" for value in values):
        return "yes_local"
    if all(value in {"yes_local", "yes_server"} for value in values):
        return "yes_server"
    return "no"


def _blocking_reason(row: dict[str, str]) -> str:
    if row["calibration_ready"] == "yes_local":
        return "ready in local workspace"
    if row["calibration_ready"] == "yes_server":
        return "ready on server; sync arrays/predictions locally or run calibration on server"
    missing = [key.replace("_available", "") for key in CORE_KEYS if row[key] == "no"]
    if row["checkpoint_available"] in {"yes_local", "yes_server"}:
        return "missing " + ", ".join(missing) + "; export logits/probs/predictions eval-only from checkpoint"
    return "missing " + ", ".join(missing) + "; no checkpoint found for eval-only export"


def build_inventory(*, server_host: str | None, server_port: int, server_repo: str) -> tuple[list[dict[str, str]], str]:
    all_paths: list[Path] = []
    for run in RUNS:
        all_paths.extend(required_paths(run).values())
    remote_paths, server_status = _remote_existing_paths(
        all_paths,
        host=server_host,
        port=server_port,
        repo=server_repo,
    )
    rows: list[dict[str, str]] = []
    for run in RUNS:
        row: dict[str, str] = {"run_id": run.run_id}
        for key, path in required_paths(run).items():
            row[key] = _availability(path, remote_paths)
        row["calibration_ready"] = _ready_status(row)
        row["blocking_reason"] = _blocking_reason(row)
        rows.append(row)
    return rows, server_status


def _recommendation_lines(status_by_run: dict[str, dict[str, str]], server_status: str) -> list[str]:
    qdr1 = status_by_run[EXP07_RUN_ID]
    baselines = [status_by_run[QD_B0_RUN_ID], status_by_run[QD_B1_RUN_ID]]
    lines = [
        "- Do not start Exp7-B yet; the next step is risk-aware ordinal calibration feasibility/export planning.",
    ]
    if qdr1["calibration_ready"] in {"yes_local", "yes_server"}:
        location = "local" if qdr1["calibration_ready"] == "yes_local" else "server"
        lines.append(
            f"- QD-R1 calibration is possible from the available {location} dev/test logits and probabilities, "
            "but the raw QD-R1 scorer overestimates low scores."
        )
    else:
        lines.append("- QD-R1 calibration is blocked until its dev/test logits and probabilities are available.")
    if all(row["calibration_ready"] == "yes_local" for row in baselines):
        lines.append("- QD-B0/QD-B1 calibration inputs are ready in the local workspace.")
    elif all(row["calibration_ready"] in {"yes_local", "yes_server"} for row in baselines):
        lines.append(
            "- QD-B0/QD-B1 calibration inputs are available via the checked server inventory; "
            "sync them locally or run the calibration workflow on the server."
        )
    elif all(row["checkpoint_available"] in {"yes_local", "yes_server"} for row in baselines):
        lines.append(
            "- QD-B0/QD-B1 logits/probs are missing, but checkpoints are available; "
            "export logits/probs/predictions eval-only before calibration."
        )
    elif server_status == "not_checked":
        lines.append(
            "- QD-B0/QD-B1 local calibration inputs are missing in this workspace; "
            "run with server inventory enabled or sync/export the baseline logits before calibration."
        )
    else:
        lines.append("- QD-B0/QD-B1 calibration is blocked until logits/probs/predictions or checkpoints are available.")
    return lines


def write_report(rows: list[dict[str, str]], server_status: str) -> None:
    status_by_run = {row["run_id"]: row for row in rows}
    lines = [
        "# Exp7 Calibration Feasibility",
        "",
        "Scope: inventory only. No model training, API calls, synthetic generation, calibration fitting, or raw prediction/array edits were performed.",
        "",
        f"Server check: `{server_status}`.",
        "",
        "## Inventory",
        "",
        "| run_id | dev logits | test logits | dev probs | test probs | dev predictions | test predictions | arrays | checkpoint | calibration ready |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {dev_logits_available} | {test_logits_available} | {dev_probs_available} | "
            "{test_probs_available} | {dev_predictions_available} | {test_predictions_available} | "
            "{arrays_available} | {checkpoint_available} | {calibration_ready} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Feasibility Notes",
            "",
        ]
    )
    for run_id in [QD_B0_RUN_ID, QD_B1_RUN_ID, EXP07_RUN_ID]:
        row = status_by_run[run_id]
        lines.append(f"- {run_id}: `{row['calibration_ready']}`; {row['blocking_reason']}.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    lines.extend(_recommendation_lines(status_by_run, server_status))
    write_text(EXP07_REPORTS_DIR / "exp07_calibration_feasibility.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory calibration logits/probs availability for Exp7.")
    parser.add_argument("--server_host", default=None, help="Optional SSH host, for example user@host.")
    parser.add_argument("--server_port", type=int, default=22)
    parser.add_argument("--server_repo", default="/home/jpang/edubench-eval-exp2")
    args = parser.parse_args()
    ensure_exp07_dirs()
    rows, server_status = build_inventory(
        server_host=args.server_host,
        server_port=args.server_port,
        server_repo=args.server_repo,
    )
    write_csv(EXP07_TABLES_DIR / "exp07_calibration_logit_inventory.csv", rows, INVENTORY_FIELDS)
    write_report(rows, server_status)
    print(f"Wrote calibration inventory: {relpath(EXP07_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
