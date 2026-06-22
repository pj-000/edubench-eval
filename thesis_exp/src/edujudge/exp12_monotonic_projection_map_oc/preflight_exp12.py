"""Preflight checks for Exp12 monotonic projection / MAP-OC."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import QD_B1_CHECKPOINT_DIR, QDPR2_DATASET_DIR
from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc import (
    DEFAULT_PROJECTION_METHOD,
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    DEFAULT_SOFT_RISK_GAMMA,
    EXP11_LOCAL_RUNS_DIR,
    EXP11_TABLES_DIR,
    EXP12_LOCAL_RUNS_DIR,
    EXP12_REPORTS_DIR,
    EXP12_TABLES_DIR,
    ensure_exp12_dirs,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_csv, relpath, write_csv, write_text


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _split_path(split: str) -> Path:
    return QDPR2_DATASET_DIR / f"{split}.jsonl"


def _checkpoint_exists(path: Path) -> bool:
    return all((path / name).exists() for name in ["state_dict.pt", "exp05_head_metadata.json", "tokenizer.json"])


def _assignment(seeds: list[str], gpus: list[str]) -> dict[str, str]:
    return {seed: gpus[idx % len(gpus)] for idx, seed in enumerate(seeds)} if gpus else {seed: "" for seed in seeds}


def _exp11_selected_info_exists() -> bool:
    path = EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv"
    if not path.exists():
        return False
    return any(row.get("selection_rule") == DEFAULT_SELECTION_RULE for row in read_csv(path))


def _exp11_checkpoint_available(seed: str) -> bool:
    selected = EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv"
    if not selected.exists():
        return False
    for row in read_csv(selected):
        if str(row.get("seed")) == str(seed) and row.get("selection_rule") == DEFAULT_SELECTION_RULE:
            epoch = int(float(row.get("selected_epoch", 0)))
            ckpt = EXP11_LOCAL_RUNS_DIR / f"seed_{seed}" / "checkpoints" / f"epoch_{epoch:02d}" / "state_dict.pt"
            return ckpt.exists()
    return False


def build_preflight_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = yaml.safe_load(args.config_path.read_text(encoding="utf-8")) if args.config_path.exists() else {}
    seeds = args.seeds.split()
    gpus = args.gpu_list.replace(",", " ").split()
    formal_caps = {
        name: os.environ.get(name, "")
        for name in ["MAX_TRAIN_SAMPLES", "MAX_EVAL_SAMPLES", "MAX_TRAIN_PAIRS", "MAX_DEV_PAIRS"]
    }
    rows = [
        {"item": "repo_root", "value": REPO_ROOT, "status": "PASS", "details": ""},
        {"item": "current_git_commit", "value": _git_commit(), "status": "PASS", "details": ""},
        {"item": "output_dir", "value": "thesis_exp/outputs/exp12_monotonic_projection_map_oc", "status": "PASS", "details": ""},
        {"item": "local_run_root", "value": relpath(EXP12_LOCAL_RUNS_DIR), "status": "PASS", "details": ""},
        {
            "item": "local_checkpoint_dir_ignored",
            "value": relpath(EXP12_LOCAL_RUNS_DIR),
            "status": "PASS" if "thesis_exp/runs" in relpath(EXP12_LOCAL_RUNS_DIR) else "FAIL",
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
        {
            "item": "Exp11_outputs_exist",
            "value": relpath(EXP11_TABLES_DIR),
            "status": "PASS" if EXP11_TABLES_DIR.exists() else "BLOCKED",
            "details": "",
        },
        {
            "item": "Exp11_selected_checkpoint_info_exists",
            "value": DEFAULT_SELECTION_RULE,
            "status": "PASS" if _exp11_selected_info_exists() else "BLOCKED",
            "details": "Exp12A needs Exp11 selected epoch metadata.",
        },
        {"item": "model_name", "value": args.model_name_or_path, "status": "PASS", "details": ""},
        {"item": "seed_list", "value": args.seeds, "status": "PASS" if seeds else "FAIL", "details": ""},
        {"item": "epochs", "value": args.epochs, "status": "PASS", "details": ""},
        {"item": "projection_method", "value": config.get("projection_method", DEFAULT_PROJECTION_METHOD), "status": "PASS", "details": ""},
        {"item": "projection_in_decode", "value": config.get("projection_in_decode", False), "status": "PASS", "details": ""},
        {"item": "projection_in_pair_score", "value": config.get("projection_in_pair_score", False), "status": "PASS", "details": ""},
        {"item": "projection_in_point_loss", "value": config.get("projection_in_point_loss", False), "status": "PASS", "details": ""},
        {"item": "projection_in_anchor", "value": config.get("projection_in_anchor", False), "status": "PASS", "details": ""},
        {"item": "selection_rule", "value": config.get("selection_rule", DEFAULT_SELECTION_RULE), "status": "PASS", "details": ""},
        {"item": "selection_delta", "value": config.get("selection_delta", DEFAULT_SELECTION_DELTA), "status": "PASS", "details": ""},
        {"item": "gamma", "value": args.gamma, "status": "PASS", "details": ""},
        {"item": "lambda_point", "value": config.get("lambda_point", 1.0), "status": "PASS", "details": ""},
        {"item": "lambda_pair", "value": config.get("lambda_pair", 0.05), "status": "PASS", "details": ""},
        {"item": "lambda_anchor", "value": config.get("lambda_anchor", 0.5), "status": "PASS", "details": ""},
        {"item": "lambda_mono", "value": config.get("lambda_mono", 0.1), "status": "PASS", "details": ""},
        {"item": "eta_proj", "value": config.get("eta_proj", 0.1), "status": "PASS", "details": ""},
        {"item": "gpu_assignment", "value": _assignment(seeds, gpus), "status": "PASS" if gpus else "FAIL", "details": ""},
        {
            "item": "formal_run_has_no_sample_caps",
            "value": formal_caps,
            "status": "PASS" if not any(formal_caps.values()) else "FAIL",
            "details": "Formal Exp12 must not use sample or pair caps.",
        },
    ]
    for seed in seeds:
        rows.append(
            {
                "item": f"seed_{seed}_Exp11_checkpoint_or_prediction_cache",
                "value": _exp11_checkpoint_available(seed),
                "status": "PASS" if _exp11_checkpoint_available(seed) else "BLOCKED",
                "details": "Exp12A can regenerate raw probabilities from Exp11 local checkpoint on server.",
            }
        )
    return rows


def write_preflight(rows: list[dict[str, Any]]) -> str:
    ensure_exp12_dirs()
    write_csv(EXP12_TABLES_DIR / "exp12_preflight.csv", rows)
    status = "PASS"
    if any(row["status"] == "FAIL" for row in rows):
        status = "FAIL"
    elif any(row["status"] == "BLOCKED" for row in rows):
        status = "BLOCKED"
    lines = [
        "# Exp12 Monotonic Projection / MAP-OC Preflight",
        "",
        f"Status: `{status}`",
        "",
        "| item | status | value | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {row['item']} | {row['status']} | `{row['value']}` | {row.get('details', '')} |" for row in rows)
    write_text(EXP12_REPORTS_DIR / "exp12_preflight_report.md", "\n".join(lines))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp12 preflight checks.")
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gpu_list", default="6 7")
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--qd_b1_checkpoint_dir", type=Path, default=QD_B1_CHECKPOINT_DIR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_SOFT_RISK_GAMMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = write_preflight(build_preflight_rows(args))
    print(f"Exp12 preflight {status}")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
