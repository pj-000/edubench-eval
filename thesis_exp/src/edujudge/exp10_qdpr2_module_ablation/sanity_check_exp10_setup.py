"""Sanity checks for Exp10 QD-PR2 module ablation scaffold."""

from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import QD_B1_CHECKPOINT_DIR, QDPR2_DATASET_DIR
from thesis_exp.src.edujudge.exp09_pairwise_ordinal.data import read_split, tracked_weight_files
from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    ABLATION_LAMBDAS,
    ABLATION_ORDER,
    EXP10_CONFIG_DIR,
    EXP10_OUTPUT_DIR,
    EXP10_TABLES_DIR,
    ensure_exp10_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp10_qdpr2_module_ablation.sh"),
    Path("thesis_exp/scripts/sync_exp10_qdpr2_module_ablation_to_server.sh"),
]
SOURCE_PATHS = [
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/losses.py"),
    Path("thesis_exp/src/edujudge/exp09_pairwise_ordinal/train_qdpr2_anchored_pairwise.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/__init__.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/collect_exp10_results.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/sanity_check_exp10_setup.py"),
    Path("thesis_exp/src/edujudge/exp10_qdpr2_module_ablation/readability_check_exp10.py"),
]


def add(rows: list[dict[str, Any]], check_name: str, status: str, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": status, "details": details})


def checkpoint_status() -> tuple[str, str]:
    required = ["state_dict.pt", "exp05_head_metadata.json", "tokenizer.json"]
    missing = [name for name in required if not (QD_B1_CHECKPOINT_DIR / name).exists()]
    if missing:
        return "BLOCKED", f"BLOCKED_MISSING_QDB1_CHECKPOINT: {relpath(QD_B1_CHECKPOINT_DIR)} missing {missing}"
    return "PASS", relpath(QD_B1_CHECKPOINT_DIR)


def config_rows() -> list[dict[str, Any]]:
    rows = []
    for name in ABLATION_ORDER:
        path = EXP10_CONFIG_DIR / f"exp10_{name}.yaml"
        if not path.exists():
            rows.append({"ablation_name": name, "config_path": relpath(path), "status": "FAIL", "details": "missing"})
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        expected = ABLATION_LAMBDAS[name]
        mismatches = [
            f"{key}: expected {value}, got {data.get(key)}"
            for key, value in expected.items()
            if float(data.get(key, -9999)) != float(value)
        ]
        status = "PASS" if not mismatches else "FAIL"
        rows.append(
            {
                "ablation_name": name,
                "config_path": relpath(path),
                "status": status,
                "lambda_point": data.get("lambda_point"),
                "lambda_pair": data.get("lambda_pair"),
                "lambda_anchor": data.get("lambda_anchor"),
                "lambda_mono": data.get("lambda_mono"),
                "output_dir": data.get("output_dir"),
                "checkpoint_output_dir": data.get("checkpoint_output_dir"),
                "details": "; ".join(mismatches),
            }
        )
    return rows


def run_checks() -> list[dict[str, Any]]:
    ensure_exp10_dirs()
    rows: list[dict[str, Any]] = []
    ckpt_status, ckpt_details = checkpoint_status()
    add(rows, "QD-B1 checkpoint exists", ckpt_status, ckpt_details)
    for split, expected in [("train", 3326), ("dev", 1107), ("test", 1103)]:
        split_rows = read_split(QDPR2_DATASET_DIR, split)
        questions = {str(row.get("question") or row.get("question_id") or row.get("question_key")) for row in split_rows}
        add(rows, f"{split} split row count", "PASS" if len(split_rows) == expected else "FAIL", len(split_rows))
        add(rows, f"{split} split has question identities", "PASS" if len(questions) > 0 else "FAIL", len(questions))
        add(rows, f"{split} split has no synthetic rows", "PASS" if not any(str(row.get("source_type")).lower() == "synthetic" for row in split_rows) else "FAIL")
    config_checks = config_rows()
    write_csv(EXP10_TABLES_DIR / "exp10_config_sanity.csv", config_checks)
    add(rows, "all ablation configs exist and lambdas match", "PASS" if all(row["status"] == "PASS" for row in config_checks) else "FAIL")
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
    return rows


def write_sanity(rows: list[dict[str, Any]]) -> str:
    write_csv(EXP10_TABLES_DIR / "exp10_setup_sanity.csv", rows)
    failures = [row for row in rows if row["status"] == "FAIL"]
    blocked = [row for row in rows if row["status"] == "BLOCKED"]
    if failures:
        status = "FAIL"
    elif blocked:
        status = "BLOCKED_MISSING_QDB1_CHECKPOINT"
    else:
        status = "PASS"
    lines = [
        "# Exp10 QD-PR2 Module Ablation Sanity",
        "",
        f"Status: `{status}`",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP10_OUTPUT_DIR / "sanity_check_exp10_setup.md", "\n".join(lines))
    return status


def main() -> None:
    rows = run_checks()
    status = write_sanity(rows)
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(f"Exp10 sanity FAIL. See {relpath(EXP10_OUTPUT_DIR / 'sanity_check_exp10_setup.md')}")
    print(f"Exp10 sanity {status}")


if __name__ == "__main__":
    main()
