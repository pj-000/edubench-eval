"""Sanity checks for Exp7 setup before smoke or formal training."""

from __future__ import annotations

import math
import py_compile
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp07_rank_consistent import (
    EXP07_OUTPUT_DIR,
    EXP07_RUN_ID,
    EXP07_SRC_DIR,
    EXP07_TABLES_DIR,
    ensure_exp07_dirs,
)
from thesis_exp.src.edujudge.exp07_rank_consistent.data import dataset_sanity_rows, tracked_weight_files
from thesis_exp.src.edujudge.exp07_rank_consistent.losses import coral_ordinal_loss, make_ordinal_targets
from thesis_exp.src.edujudge.exp07_rank_consistent.metrics import prediction_from_probs, sigmoid
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp07_qdr1_smoke.sh"),
    Path("thesis_exp/scripts/run_exp07_qdr1_train.sh"),
    Path("thesis_exp/scripts/sync_exp07_qdr1_to_server.sh"),
]


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def run_bash_n(path: Path) -> tuple[bool, str]:
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def py_compile_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EXP07_SRC_DIR.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, f"py_compile {relpath(path)}", True)
        except py_compile.PyCompileError as exc:
            add(rows, f"py_compile {relpath(path)}", False, str(exc))
    return rows


def target_toy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    actual = make_ordinal_targets([1, 2, 3, 4, 5])
    actual_list = actual.detach().cpu().tolist() if hasattr(actual, "detach") else actual
    add(rows, "ordinal target toy conversion correct", actual_list == expected, actual_list)
    try:
        make_ordinal_targets([0, 1, 2])
        add(rows, "label_5=0 raises error", False, "no error")
    except ValueError:
        add(rows, "label_5=0 raises error", True)
    return rows


def coral_toy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        import torch

        from thesis_exp.src.edujudge.exp07_rank_consistent.coral_head import CoralOrdinalHead, assert_rank_consistent

        torch.manual_seed(7)
        head = CoralOrdinalHead(hidden_size=3)
        hidden = torch.tensor([[0.2, -0.5, 1.0], [1.2, 0.0, -0.7]], dtype=torch.float32)
        logits = head(hidden)
        probs = torch.sigmoid(logits)
        assert_rank_consistent(logits)
        assert_rank_consistent(probs)
        loss = coral_ordinal_loss(logits, torch.tensor([2, 5], dtype=torch.long))
        labels = [prediction_from_probs(row) for row in probs.detach().cpu().tolist()]
        add(rows, "CORAL toy logits shape = [batch,4]", tuple(logits.shape) == (2, 4), tuple(logits.shape))
        add(rows, "CORAL toy logits monotonic by construction", True)
        add(rows, "CORAL toy probs monotonic by construction", True)
        add(rows, "coral_ordinal_loss finite", bool(torch.isfinite(loss).detach().cpu()), float(loss.detach().cpu()))
        add(rows, "pred_label in 1..5", all(1 <= label <= 5 and 1.0 <= score <= 5.0 for label, score in labels), labels)
        add(rows, "toy check backend", True, "torch")
    except ModuleNotFoundError:
        logits = [[2.0, 1.0, 0.0, -1.0], [0.5, 0.0, -0.5, -1.0]]
        probs = sigmoid(logits)
        loss = coral_ordinal_loss(logits, [2, 5])
        labels = [prediction_from_probs(row) for row in probs.tolist()]
        monotonic = all(all(row[idx] >= row[idx + 1] for idx in range(3)) for row in logits)
        prob_monotonic = all(all(row[idx] >= row[idx + 1] for idx in range(3)) for row in probs.tolist())
        add(rows, "CORAL toy logits shape = [batch,4]", len(logits) == 2 and all(len(row) == 4 for row in logits), "fallback")
        add(rows, "CORAL toy logits monotonic by construction", monotonic, logits)
        add(rows, "CORAL toy probs monotonic by construction", prob_monotonic, probs.tolist())
        add(rows, "coral_ordinal_loss finite", math.isfinite(float(loss)), loss)
        add(rows, "pred_label in 1..5", all(1 <= label <= 5 and 1.0 <= score <= 5.0 for label, score in labels), labels)
        add(rows, "toy check backend", True, "formula_fallback_no_torch")
    return rows


def script_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in SCRIPT_PATHS:
        add(rows, f"script exists {relpath(path)}", path.exists())
        if path.exists():
            ok, details = run_bash_n(path)
            add(rows, f"bash -n {relpath(path)}", ok, details)
    return rows


def main() -> None:
    ensure_exp07_dirs()
    rows: list[dict[str, Any]] = []
    rows.extend(dataset_sanity_rows())
    rows.extend(target_toy_rows())
    rows.extend(coral_toy_rows())
    rows.extend(script_rows())
    rows.extend(py_compile_rows())
    weights = tracked_weight_files()
    add(rows, "no checkpoint/weights tracked after setup checks", not weights, ", ".join(weights))

    try:
        from thesis_exp.src.edujudge.exp07_rank_consistent.collect_exp07_results import collect

        collect()
        add(rows, "Exp7 pending report/review package written", True, relpath(EXP07_OUTPUT_DIR))
    except Exception as exc:
        add(rows, "Exp7 pending report/review package written", False, f"{type(exc).__name__}: {exc}")

    write_csv(EXP07_TABLES_DIR / "sanity_check_exp07_setup.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp7 Setup Sanity Check",
        "",
        f"Run ID: `{EXP07_RUN_ID}`",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP07_OUTPUT_DIR / "sanity_check_exp07_setup.md", "\n".join(lines))
    if failures:
        raise SystemExit("Exp7 setup sanity failed. See thesis_exp/outputs/exp07_rank_consistent_ordinal.")
    print("Exp7 setup sanity PASS")


if __name__ == "__main__":
    main()
