"""Run Exp12A decode-only projection evals from selected Exp11 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc import (
    CONFIG_BY_RUN,
    DEFAULT_SELECTION_RULE,
    EXP11_LOCAL_RUNS_DIR,
    EXP11_TABLES_DIR,
    exp12a_eval_dir,
    ensure_exp12_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv


def _selected_rows(rule: str) -> dict[str, dict[str, str]]:
    path = EXP11_TABLES_DIR / "exp11_selection_rule_summary.csv"
    if not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        if row.get("selection_rule") == rule:
            out[str(int(float(row.get("seed", 0))))] = row
    return out


def _checkpoint_for(seed: str, epoch: int) -> Path:
    return EXP11_LOCAL_RUNS_DIR / f"seed_{seed}" / "checkpoints" / f"epoch_{epoch:02d}"


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_exp12_dirs()
    selected = _selected_rows(args.selection_rule)
    rows: list[dict[str, Any]] = []
    seeds = args.seeds.split()
    gpus = args.gpu_list.replace(",", " ").split() or [""]
    for idx, seed in enumerate(seeds):
        selection = selected.get(seed)
        if not selection:
            rows.append({"seed": seed, "status": "SKIPPED", "reason": "missing Exp11 selected epoch", "output_dir": ""})
            continue
        epoch = int(float(selection.get("selected_epoch", 0)))
        checkpoint_dir = _checkpoint_for(seed, epoch)
        output_dir = exp12a_eval_dir(seed, epoch)
        if args.skip_completed and (output_dir / "run_metadata.json").exists():
            rows.append({"seed": seed, "status": "SKIPPED_COMPLETED", "reason": "", "output_dir": relpath(output_dir)})
            continue
        if args.reset_run_dir and output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)
        if not (checkpoint_dir / "state_dict.pt").exists():
            rows.append(
                {
                    "seed": seed,
                    "selected_epoch": epoch,
                    "status": "BLOCKED",
                    "reason": f"missing checkpoint {relpath(checkpoint_dir)}",
                    "output_dir": relpath(output_dir),
                }
            )
            continue
        env = os.environ.copy()
        gpu = gpus[idx % len(gpus)]
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            "-m",
            "thesis_exp.src.edujudge.exp09_pairwise_ordinal.train_qdpr2_anchored_pairwise",
            "--config_path",
            str(args.config_path),
            "--model_name_or_path",
            args.model_name_or_path,
            "--qd_b1_checkpoint_dir",
            str(args.qd_b1_checkpoint_dir),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--output_dir",
            str(output_dir),
            "--checkpoint_output_dir",
            str(output_dir / "checkpoints_unused"),
            "--eval_only",
            "--seed",
            seed,
            "--per_device_eval_batch_size",
            str(args.per_device_eval_batch_size),
            "--bf16",
            args.bf16,
            "--no_progress_bar",
        ]
        if args.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        result = subprocess.run(cmd, env=env)
        rows.append(
            {
                "seed": seed,
                "selected_epoch": epoch,
                "selected_global_step": selection.get("selected_global_step", ""),
                "checkpoint_dir": relpath(checkpoint_dir),
                "output_dir": relpath(output_dir),
                "gpu": gpu,
                "status": "COMPLETED" if result.returncode == 0 else "FAILED",
                "reason": "",
            }
        )
        if result.returncode != 0:
            break
    write_csv(Path("thesis_exp/outputs/exp12_monotonic_projection_map_oc/tables/exp12a_eval_manifest.csv"), rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp12A decode projection evals.")
    parser.add_argument("--config_path", type=Path, default=CONFIG_BY_RUN["decode_projection_only"])
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gpu_list", default="6 7")
    parser.add_argument("--selection_rule", default=DEFAULT_SELECTION_RULE)
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument(
        "--qd_b1_checkpoint_dir",
        type=Path,
        default=Path("thesis_exp/artifacts/exp06_question_disjoint_baselines/checkpoints/QD-B1_human_only_L1_weighted_ordinal/best"),
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--bf16", default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--reset_run_dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    rows = run(parse_args())
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    if any(row.get("status") == "FAILED" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
