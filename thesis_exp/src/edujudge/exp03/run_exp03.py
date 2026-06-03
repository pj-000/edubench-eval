"""Orchestrate Exp3 setup, smoke tests, and postprocessing."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import (
    CORE_TRAIN_TEMPLATES,
    EXP03_ARTIFACTS_DIR,
    EXP03_OUTPUT_DIR,
    EXP03_SMOKE_DIR,
    TEMPLATE_NAMES,
    ensure_exp03_dirs,
)
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import build_exp03_datasets
from thesis_exp.src.edujudge.exp03.postprocess_exp03_results import postprocess_exp03_results
from thesis_exp.src.edujudge.exp03.sanity_check_exp03_setup import run_sanity_check
from thesis_exp.src.edujudge.exp03.write_exp03_report import write_exp03_report
from thesis_exp.src.edujudge.utils.io import relpath, write_text


DEFAULT_MODEL_NAME = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        # Minimal fallback for simple scalar YAML used by these configs.
        data: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line or ":" not in line or line.startswith(" "):
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
        return data


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    for section in ["train", "smoke_test", "model", "data", "output"]:
        value = nested(config, section, key, default=None)
        if value is not None:
            return value
    return default


def bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_smoke_report(results: list[dict[str, Any]]) -> None:
    overall = "PASS" if all(row["returncode"] == 0 for row in results) else "FAIL"
    lines = [
        "# Exp3 Smoke Test Report",
        "",
        f"Overall status: **{overall}**",
        "",
        "| template | status | output_dir | log |",
        "| --- | --- | --- | --- |",
    ]
    for row in results:
        status = "PASS" if row["returncode"] == 0 else "FAIL"
        lines.append(f"| {row['template_name']} | {status} | `{row['output_dir']}` | `{row['log_path']}` |")
    write_text(EXP03_SMOKE_DIR / "smoke_test_report.md", "\n".join(lines))


def run_smoke(config: dict[str, Any], templates: list[str]) -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    build_exp03_datasets()
    run_sanity_check()
    model_name = str(config_value(config, "model_name_or_path", DEFAULT_MODEL_NAME))
    train_cfg = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
    smoke_cfg = config.get("smoke_test", {}) if isinstance(config.get("smoke_test"), dict) else {}

    def cfg(name: str, default: Any) -> Any:
        return smoke_cfg.get(name, train_cfg.get(name, config_value(config, name, default)))

    results = []
    for template_name in templates:
        output_dir = EXP03_SMOKE_DIR / template_name
        checkpoint_dir = EXP03_ARTIFACTS_DIR / "smoke_test" / template_name
        log_path = EXP03_SMOKE_DIR / f"{template_name}.log"
        cmd = [
            sys.executable,
            "-m",
            "thesis_exp.src.edujudge.exp03.train_input_ablation",
            "--template_name",
            template_name,
            "--model_name_or_path",
            model_name,
            "--output_dir",
            str(output_dir),
            "--checkpoint_output_dir",
            str(checkpoint_dir),
            "--max_length",
            str(cfg("max_length", 2048)),
            "--num_train_epochs",
            str(cfg("num_train_epochs", 0.01)),
            "--learning_rate",
            str(cfg("learning_rate", "2e-5")),
            "--weight_decay",
            str(cfg("weight_decay", "0.01")),
            "--warmup_ratio",
            str(cfg("warmup_ratio", "0.05")),
            "--per_device_train_batch_size",
            str(cfg("per_device_train_batch_size", 1)),
            "--per_device_eval_batch_size",
            str(cfg("per_device_eval_batch_size", 1)),
            "--gradient_accumulation_steps",
            str(cfg("gradient_accumulation_steps", 1)),
            "--max_train_samples",
            str(cfg("max_train_samples", 8)),
            "--max_eval_samples",
            str(cfg("max_eval_samples", 8)),
            "--bf16",
            str(cfg("bf16", "auto")),
            "--log_steps",
            str(cfg("log_steps", 1)),
            "--trust_remote_code",
            "--local_files_only",
            "--no_progress_bar",
        ]
        if bool_flag(cfg("gradient_checkpointing", True)):
            cmd.append("--gradient_checkpointing")
        if bool_flag(cfg("fp16", False)):
            cmd.append("--fp16")
        env = {**os.environ, "FORMAL_RUN": "0", "REQUIRE_CUDA": os.environ.get("REQUIRE_CUDA", "0")}
        output_dir.mkdir(parents=True, exist_ok=True)
        EXP03_SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT, check=False, env=env)
        results.append(
            {
                "template_name": template_name,
                "returncode": result.returncode,
                "output_dir": relpath(output_dir),
                "log_path": relpath(log_path),
            }
        )
    write_smoke_report(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp3 setup/smoke/postprocess stages.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--templates", nargs="+", default=CORE_TRAIN_TEMPLATES, choices=TEMPLATE_NAMES)
    parser.add_argument("--mode", choices=["build", "sanity", "smoke", "postprocess", "report", "all"], default="all")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "build":
        build_exp03_datasets()
    elif args.mode == "sanity":
        run_sanity_check()
    elif args.mode == "smoke":
        results = run_smoke(config, args.templates)
        if any(row["returncode"] != 0 for row in results):
            raise SystemExit(1)
    elif args.mode == "postprocess":
        postprocess_exp03_results(strict=args.strict)
    elif args.mode == "report":
        write_exp03_report()
    elif args.mode == "all":
        build_exp03_datasets()
        run_sanity_check()
        postprocess_exp03_results(strict=args.strict)
    print(f"Exp3 mode={args.mode} finished. Output: {relpath(EXP03_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
