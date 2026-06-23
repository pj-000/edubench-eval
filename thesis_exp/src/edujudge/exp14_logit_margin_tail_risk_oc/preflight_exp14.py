"""Preflight checks for Exp14 logit-margin tail-risk scout."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp09_pairwise_ordinal import QD_B1_CHECKPOINT_DIR, QDPR2_DATASET_DIR
from thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc import (
    CONFIG_BY_RUN,
    DEFAULT_SELECTION_DELTA,
    DEFAULT_SELECTION_RULE,
    EXP14_LOCAL_RUNS_DIR,
    EXP14_OUTPUT_DIR,
    EXP14_REPORTS_DIR,
    EXP14_TABLES_DIR,
    ensure_exp14_dirs,
)
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, relpath, write_csv, write_text


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _split_path(split: str) -> Path:
    return QDPR2_DATASET_DIR / f"{split}.jsonl"


def _checkpoint_exists(path: Path) -> bool:
    return all((path / name).exists() for name in ["state_dict.pt", "exp05_head_metadata.json", "tokenizer.json"])


def _config_rows(run_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = [
        "use_l2h_risk_loss",
        "use_l2h_logit_margin_loss",
        "lambda_l2h_logit_margin",
        "logit_margin_risk_threshold_t",
        "logit_margin_loss_type",
        "logit_margin_tail_fraction",
        "selection_rule",
        "selection_decode_mode",
        "projection_in_decode",
        "projection_in_pair_score",
    ]
    for run_name in run_names:
        path = CONFIG_BY_RUN.get(run_name)
        config = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.exists() else {}
        missing = [name for name in required if name not in config]
        rows.append(
            {
                "item": f"config_{run_name}",
                "value": relpath(path) if path else "",
                "status": "PASS" if path and path.exists() and not missing else "FAIL",
                "details": f"missing={missing}" if missing else "",
            }
        )
        if config:
            rows.extend(
                [
                    {
                        "item": f"{run_name}_exp13_probability_hinge_disabled",
                        "value": config.get("use_l2h_risk_loss"),
                        "status": "PASS" if config.get("use_l2h_risk_loss") is False else "FAIL",
                        "details": "Exp14 must not reuse Exp13 probability-space q3 hinge.",
                    },
                    {
                        "item": f"{run_name}_logit_margin_enabled",
                        "value": config.get("use_l2h_logit_margin_loss"),
                        "status": "PASS" if config.get("use_l2h_logit_margin_loss") is True else "FAIL",
                        "details": "",
                    },
                    {
                        "item": f"{run_name}_logit_margin_threshold_t",
                        "value": config.get("logit_margin_risk_threshold_t"),
                        "status": "PASS" if int(config.get("logit_margin_risk_threshold_t", 0)) == 3 else "FAIL",
                        "details": "threshold tensor index is 2 in the loss implementation",
                    },
                    {
                        "item": f"{run_name}_selection_rule",
                        "value": config.get("selection_rule"),
                        "status": "PASS" if config.get("selection_rule") == DEFAULT_SELECTION_RULE else "FAIL",
                        "details": "",
                    },
                    {
                        "item": f"{run_name}_selection_decode_mode",
                        "value": config.get("selection_decode_mode"),
                        "status": "PASS" if config.get("selection_decode_mode") == "projected" else "FAIL",
                        "details": "",
                    },
                ]
            )
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = args.seeds.split()
    gpus = args.gpu_list.replace(",", " ").split()
    run_names = args.runs.split()
    caps = {name: os.environ.get(name, "") for name in ["MAX_TRAIN_SAMPLES", "MAX_EVAL_SAMPLES", "MAX_TRAIN_PAIRS", "MAX_DEV_PAIRS"]}
    eval_test_allowed = args.eval_test == "0" or os.environ.get("ALLOW_EXP14_TEST", "0") == "1"
    rows = [
        {"item": "repo_root", "value": REPO_ROOT, "status": "PASS", "details": ""},
        {"item": "current_git_commit", "value": _git_commit(), "status": "PASS", "details": ""},
        {"item": "mode", "value": args.mode, "status": "PASS" if args.mode == "scout" else "FAIL", "details": "Exp14 currently implements scout-only by default."},
        {"item": "eval_test", "value": args.eval_test, "status": "PASS" if args.eval_test == "0" else "FAIL", "details": "Scout must not evaluate test."},
        {"item": "allow_exp14_test_guard", "value": os.environ.get("ALLOW_EXP14_TEST", "0"), "status": "PASS" if eval_test_allowed else "FAIL", "details": "EVAL_TEST=1 requires ALLOW_EXP14_TEST=1."},
        {"item": "output_dir", "value": relpath(EXP14_OUTPUT_DIR), "status": "PASS", "details": ""},
        {"item": "local_run_root", "value": relpath(EXP14_LOCAL_RUNS_DIR), "status": "PASS", "details": ""},
        {
            "item": "local_checkpoint_dir_ignored",
            "value": relpath(EXP14_LOCAL_RUNS_DIR),
            "status": "PASS" if "thesis_exp/runs" in relpath(EXP14_LOCAL_RUNS_DIR) else "FAIL",
            "details": "thesis_exp/.gitignore ignores runs/",
        },
        {"item": "train_data_path", "value": relpath(_split_path("train")), "status": "PASS" if _split_path("train").exists() else "FAIL", "details": ""},
        {"item": "dev_data_path", "value": relpath(_split_path("dev")), "status": "PASS" if _split_path("dev").exists() else "FAIL", "details": ""},
        {"item": "test_data_path", "value": relpath(_split_path("test")), "status": "PASS" if _split_path("test").exists() else "FAIL", "details": "present but not used in scout"},
        {
            "item": "QD_B1_checkpoint_path",
            "value": relpath(args.qd_b1_checkpoint_dir),
            "status": "PASS" if _checkpoint_exists(args.qd_b1_checkpoint_dir) else "BLOCKED",
            "details": "requires state_dict.pt, exp05_head_metadata.json, tokenizer.json",
        },
        {"item": "seed_list", "value": args.seeds, "status": "PASS" if seeds else "FAIL", "details": ""},
        {"item": "gpu_list", "value": args.gpu_list, "status": "PASS" if gpus else "FAIL", "details": ""},
        {"item": "run_list", "value": args.runs, "status": "PASS" if run_names else "FAIL", "details": ""},
        {"item": "epochs", "value": args.epochs, "status": "PASS", "details": ""},
        {"item": "selection_delta", "value": args.selection_delta, "status": "PASS" if float(args.selection_delta) == DEFAULT_SELECTION_DELTA else "WARN", "details": ""},
        {
            "item": "scout_has_no_sample_caps",
            "value": caps,
            "status": "PASS" if not any(caps.values()) else "FAIL",
            "details": "Exp14 scout runs should not use sample or pair caps.",
        },
    ]
    rows.extend(_config_rows(run_names))
    return rows


def write_preflight(rows: list[dict[str, Any]]) -> str:
    ensure_exp14_dirs()
    write_csv(EXP14_TABLES_DIR / "exp14_preflight.csv", rows)
    status = "PASS"
    if any(row["status"] == "FAIL" for row in rows):
        status = "FAIL"
    elif any(row["status"] == "BLOCKED" for row in rows):
        status = "BLOCKED"
    lines = [
        "# Exp14 Logit-Margin Tail-Risk OC Preflight",
        "",
        f"Status: `{status}`",
        "",
        "| item | status | value | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {row['item']} | {row['status']} | `{row['value']}` | {row.get('details', '')} |" for row in rows)
    write_text(EXP14_REPORTS_DIR / "exp14_preflight_report.md", "\n".join(lines))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp14 preflight checks.")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gpu_list", default="6 7")
    parser.add_argument("--runs", default=" ".join(CONFIG_BY_RUN))
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--mode", default="scout")
    parser.add_argument("--eval_test", default="0")
    parser.add_argument("--selection_delta", default="0.005")
    parser.add_argument("--qd_b1_checkpoint_dir", type=Path, default=QD_B1_CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = write_preflight(build_rows(args))
    print(f"Exp14 preflight {status}")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
