"""Sanity checks for QD-PR2 anchored pairwise fine-tuning scaffold."""

from __future__ import annotations

import math
import py_compile
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import (
    QD_B1_CHECKPOINT_DIR,
    QDPR2_DATASET_DIR,
    QDPR2_OUTPUT_DIR,
    QDPR2_TABLES_DIR,
    ensure_exp09_dirs,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import read_split, tracked_weight_files
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.losses import (
    anchor_bce_with_logits,
    monotonic_regularization,
    total_anchored_pairwise_training_loss,
)
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.qdpr2_setup import prepare_qdpr2_setup
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp09_qdpr2_smoke.sh"),
    Path("thesis_exp/scripts/run_exp09_qdpr2_train.sh"),
    Path("thesis_exp/scripts/sync_exp09_qdpr2_to_server.sh"),
]
SOURCE_PATHS = sorted(Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal").glob("*.py"))
TOY_LOSS_FIELDS = [
    "case",
    "L_total",
    "L_point",
    "L_pair",
    "L_anchor",
    "L_mono",
    "mean_score_gap",
    "low_high_pair_loss",
    "mono_pair_violation_rate",
]


def add(rows: list[dict[str, Any]], check_name: str, status: str, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": status, "details": details})


def checkpoint_state() -> tuple[str, str]:
    state_dict = QD_B1_CHECKPOINT_DIR / "state_dict.pt"
    metadata = QD_B1_CHECKPOINT_DIR / "exp05_head_metadata.json"
    tokenizer = QD_B1_CHECKPOINT_DIR / "tokenizer.json"
    if state_dict.exists() and metadata.exists() and tokenizer.exists():
        return "PASS", relpath(QD_B1_CHECKPOINT_DIR)
    return "BLOCKED", f"BLOCKED_MISSING_QDB1_CHECKPOINT: {relpath(QD_B1_CHECKPOINT_DIR)}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_human_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source_type") or "").strip().lower()
    generator = str(row.get("generator_model") or "").strip()
    if source == "synthetic":
        return False
    if source == "human":
        return True
    return generator == ""


def toy_loss_rows() -> list[dict[str, Any]]:
    win_logits = np.array([[3.0, 1.5, 0.2, -1.2], [4.0, 2.0, 1.0, -0.5]], dtype=np.float64)
    lose_logits = np.array([[1.5, 0.2, -1.0, -2.5], [2.0, 0.6, -0.4, -2.0]], dtype=np.float64)
    ref_win_logits = win_logits + np.array([[0.1, -0.1, 0.0, 0.0], [0.0, 0.1, -0.1, 0.0]])
    ref_lose_logits = lose_logits + np.array([[0.0, 0.0, 0.1, -0.1], [-0.1, 0.0, 0.0, 0.1]])
    labels_win = np.array([4, 5])
    labels_lose = np.array([2, 3])
    label_gap = labels_win - labels_lose
    low_high = np.array([1, 0])
    class_weights = np.array([0.0, 1.5, 1.5, 0.7, 0.65, 0.65])
    total, debug = total_anchored_pairwise_training_loss(
        win_logits,
        lose_logits,
        ref_win_logits,
        ref_lose_logits,
        labels_win,
        labels_lose,
        label_gap,
        low_high,
        class_weights,
        lambda_pair=0.05,
        lambda_anchor=0.5,
        lambda_mono=0.1,
    )
    row = {"case": "anchored_pairwise_toy", "L_total": total, **debug}
    return [{field: row.get(field, "") for field in TOY_LOSS_FIELDS}]


def run_checks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_exp09_dirs()
    setup = prepare_qdpr2_setup(write_shards=False)
    rows: list[dict[str, Any]] = []
    checkpoint_status, checkpoint_details = checkpoint_state()
    add(rows, "QD-B1 checkpoint exists", checkpoint_status, checkpoint_details)
    for split in ["train", "dev", "test"]:
        split_rows = read_split(QDPR2_DATASET_DIR, split)
        add(rows, f"{split} rows are human only", "PASS" if all(is_human_row(row) for row in split_rows) else "FAIL")
        add(rows, f"{split} has no synthetic rows", "PASS" if not any(row.get("source_type") == "synthetic" for row in split_rows) else "FAIL")
    pairs = list(setup["train_pairs"]) + list(setup["dev_pairs"])
    add(rows, "pair win_label > lose_label", "PASS" if all(int(pair["win_label_5"]) > int(pair["lose_label_5"]) for pair in pairs) else "FAIL")
    add(rows, "high-comparability pair rates reported", "PASS" if setup["comparability_rows"] else "FAIL")
    add(rows, "pair weights finite", "PASS" if all(finite(pair.get("pair_weight")) for pair in pairs) else "FAIL")
    add(rows, "pair margins finite", "PASS" if all(finite(pair.get("pair_margin")) for pair in pairs) else "FAIL")
    add(rows, "anchor targets available", checkpoint_status, "on-the-fly QD-B1 reference logits from train pairs")
    toy_rows = toy_loss_rows()
    toy = toy_rows[0]
    for key in ["L_point", "L_pair", "L_anchor", "L_mono", "L_total"]:
        add(rows, f"{key} finite", "PASS" if finite(toy.get(key)) else "FAIL", toy.get(key))
    mono_good, _ = monotonic_regularization(np.array([[2.0, 1.0, 0.0, -1.0]], dtype=np.float64))
    mono_bad, _ = monotonic_regularization(np.array([[0.0, 1.0, 0.5, 2.0]], dtype=np.float64))
    add(
        rows,
        "toy monotonic regularizer positive only on violation",
        "PASS" if float(mono_good) == 0.0 and float(mono_bad) > 0.0 else "FAIL",
        f"non_violation={mono_good}; violation={mono_bad}",
    )
    py_failures = []
    for path in SOURCE_PATHS:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            py_failures.append(f"{relpath(path)}: {exc}")
    add(rows, "py_compile pass", "PASS" if not py_failures else "FAIL", "; ".join(py_failures))
    bash_failures = []
    for path in SCRIPT_PATHS:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            bash_failures.append(f"{relpath(path)}: {(result.stderr or result.stdout).strip()}")
    add(rows, "bash -n pass", "PASS" if not bash_failures else "FAIL", "; ".join(bash_failures))
    weights = tracked_weight_files()
    add(rows, "no checkpoint/weights tracked", "PASS" if not weights else "FAIL", ", ".join(weights))
    return rows, toy_rows


def write_sanity(rows: list[dict[str, Any]], toy_rows: list[dict[str, Any]]) -> str:
    write_csv(QDPR2_TABLES_DIR / "qdpr2_setup_sanity.csv", rows)
    write_csv(QDPR2_TABLES_DIR / "qdpr2_toy_loss_scales.csv", toy_rows, fieldnames=TOY_LOSS_FIELDS)
    failures = [row for row in rows if row["status"] == "FAIL"]
    blocked = [row for row in rows if row["status"] == "BLOCKED"]
    if failures:
        status = "FAIL"
    elif blocked:
        status = "BLOCKED_MISSING_QDB1_CHECKPOINT"
    else:
        status = "PASS"
    lines = [
        "# QD-PR2 Setup Sanity",
        "",
        f"Status: `{status}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(QDPR2_OUTPUT_DIR / "sanity_check_qdpr2_setup.md", "\n".join(lines))
    return status


def main() -> None:
    rows, toy_rows = run_checks()
    status = write_sanity(rows, toy_rows)
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        raise SystemExit(f"QD-PR2 setup sanity FAIL. See {relpath(QDPR2_OUTPUT_DIR / 'sanity_check_qdpr2_setup.md')}")
    print(f"QD-PR2 setup sanity {status}")


if __name__ == "__main__":
    main()
