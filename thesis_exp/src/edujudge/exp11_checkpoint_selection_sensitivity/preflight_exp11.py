"""Preflight checks for Exp11 checkpoint-selection sensitivity."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import QD_B1_CHECKPOINT_DIR, QDPR2_DATASET_DIR
from thesis_exp.src.edujudge.exp11_checkpoint_selection_sensitivity import (
    DEFAULT_GAMMA,
    DEFAULT_MAE_GUARD_DELTA,
    DEFAULT_MONO_BETA,
    EXP11_LOCAL_RUNS_DIR,
    EXP11_REPORTS_DIR,
    EXP11_TABLES_DIR,
    SELECTION_RULES,
    ensure_exp11_dirs,
    seed_checkpoint_dir,
    seed_run_dir,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_csv, write_text


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _split_path(split: str) -> Path:
    return QDPR2_DATASET_DIR / f"{split}.jsonl"


def _checkpoint_exists(path: Path) -> bool:
    return all((path / name).exists() for name in ["state_dict.pt", "exp05_head_metadata.json", "tokenizer.json"])


def _assignment(seeds: list[str], gpus: list[str]) -> dict[str, str]:
    return {seed: gpus[idx % len(gpus)] for idx, seed in enumerate(seeds)} if gpus else {seed: "" for seed in seeds}


def build_preflight_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = yaml.safe_load(args.config_path.read_text(encoding="utf-8")) if args.config_path.exists() else {}
    seeds = args.seeds.split()
    gpus = args.gpu_list.replace(",", " ").split()
    assignment = _assignment(seeds, gpus)
    formal_caps = {
        name: os.environ.get(name, "")
        for name in ["MAX_TRAIN_SAMPLES", "MAX_EVAL_SAMPLES", "MAX_TRAIN_PAIRS", "MAX_DEV_PAIRS"]
    }
    rows = [
        {"item": "repo_root", "value": REPO_ROOT, "status": "PASS", "details": ""},
        {"item": "current_git_commit", "value": _git_commit(), "status": "PASS", "details": ""},
        {"item": "output_dir", "value": "thesis_exp/outputs/exp11_checkpoint_selection_sensitivity", "status": "PASS", "details": ""},
        {"item": "local_run_root", "value": relpath(EXP11_LOCAL_RUNS_DIR), "status": "PASS", "details": ""},
        {
            "item": "local_checkpoint_dir_ignored",
            "value": relpath(EXP11_LOCAL_RUNS_DIR),
            "status": "PASS" if "thesis_exp/runs" in relpath(EXP11_LOCAL_RUNS_DIR) else "FAIL",
            "details": "thesis_exp/.gitignore ignores runs/",
        },
        {"item": "train_data_path", "value": relpath(_split_path("train")), "status": "PASS" if _split_path("train").exists() else "FAIL", "details": ""},
        {"item": "dev_data_path", "value": relpath(_split_path("dev")), "status": "PASS" if _split_path("dev").exists() else "FAIL", "details": ""},
        {"item": "test_data_path", "value": relpath(_split_path("test")), "status": "PASS" if _split_path("test").exists() else "FAIL", "details": ""},
        {
            "item": "QD_B1_checkpoint_path",
            "value": relpath(args.qd_b1_checkpoint_dir),
            "status": "PASS" if _checkpoint_exists(args.qd_b1_checkpoint_dir) else "BLOCKED",
            "details": "requires state_dict.pt, exp05_head_metadata.json, tokenizer.json",
        },
        {"item": "model_name", "value": args.model_name_or_path, "status": "PASS", "details": ""},
        {"item": "seed_list", "value": args.seeds, "status": "PASS" if seeds else "FAIL", "details": ""},
        {"item": "epochs", "value": args.epochs, "status": "PASS", "details": ""},
        {"item": "lambda_point", "value": config.get("lambda_point", 1.0), "status": "PASS", "details": ""},
        {"item": "lambda_pair", "value": config.get("lambda_pair", 0.05), "status": "PASS", "details": ""},
        {"item": "lambda_anchor", "value": config.get("lambda_anchor", 0.5), "status": "PASS", "details": ""},
        {"item": "lambda_mono", "value": config.get("lambda_mono", 0.1), "status": "PASS", "details": ""},
        {"item": "pair_dataset_size_train", "value": config.get("pair_dataset_size_train", 10000), "status": "PASS", "details": ""},
        {"item": "pair_dataset_size_dev", "value": config.get("pair_dataset_size_dev", 3000), "status": "PASS", "details": ""},
        {"item": "max_pairs_per_record", "value": config.get("max_pairs_per_record", 80), "status": "PASS", "details": ""},
        {"item": "max_pairs_per_low_record", "value": config.get("max_pairs_per_low_record", 100), "status": "PASS", "details": ""},
        {"item": "selection_rules", "value": ", ".join(SELECTION_RULES), "status": "PASS", "details": ""},
        {"item": "gamma", "value": args.gamma, "status": "PASS", "details": ""},
        {"item": "delta", "value": args.delta, "status": "PASS", "details": ""},
        {"item": "beta", "value": args.beta, "status": "PASS", "details": ""},
        {"item": "gpu_assignment", "value": assignment, "status": "PASS" if assignment else "FAIL", "details": ""},
        {
            "item": "formal_run_has_no_sample_caps",
            "value": formal_caps,
            "status": "PASS" if not any(formal_caps.values()) else "FAIL",
            "details": "Formal Exp11 must not use sample or pair caps.",
        },
    ]
    for seed in seeds:
        rows.append({"item": f"seed_{seed}_run_dir", "value": relpath(seed_run_dir(seed)), "status": "PASS", "details": ""})
        rows.append(
            {
                "item": f"seed_{seed}_checkpoint_dir",
                "value": relpath(seed_checkpoint_dir(seed)),
                "status": "PASS",
                "details": "local ignored checkpoint directory",
            }
        )
    return rows


def write_preflight(rows: list[dict[str, Any]]) -> str:
    ensure_exp11_dirs()
    write_csv(EXP11_TABLES_DIR / "exp11_preflight.csv", rows)
    status = "PASS"
    if any(row["status"] == "FAIL" for row in rows):
        status = "FAIL"
    elif any(row["status"] == "BLOCKED" for row in rows):
        status = "BLOCKED"
    lines = [
        "# Exp11 Checkpoint Selection Sensitivity Preflight",
        "",
        f"Status: `{status}`",
        "",
        "| item | status | value | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {row['item']} | {row['status']} | `{row['value']}` | {row.get('details', '')} |" for row in rows)
    write_text(EXP11_REPORTS_DIR / "exp11_preflight_report.md", "\n".join(lines))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp11 preflight checks.")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gpu_list", default="6")
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--qd_b1_checkpoint_dir", type=Path, default=QD_B1_CHECKPOINT_DIR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--delta", type=float, default=DEFAULT_MAE_GUARD_DELTA)
    parser.add_argument("--beta", type=float, default=DEFAULT_MONO_BETA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = write_preflight(build_preflight_rows(args))
    print(f"Exp11 preflight {status}")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
