"""Build the public, no-GPU Exp54 actual failure-bank generation plan."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    SEEDS,
    TRAIN_PATH,
    load_train_rows,
    sha256_file,
)


TRAINING_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
)
OUTPUT_PATH = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "actual_failure_bank/generation_plan.json"
)
CONFIG_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "actual_failure_bank_generation.json"
)
SOURCE_PATHS = (
    "thesis_exp/exp54_rar_sft/actual_failure_bank.py",
    "thesis_exp/exp54_rar_sft/build_actual_failure_bank_plan.py",
    "thesis_exp/exp54_rar_sft/run_actual_failure_bank_vllm.py",
    "thesis_exp/exp54_rar_sft/collect_actual_failure_bank.py",
    "thesis_exp/exp54_rar_sft/inference_contract.py",
    "thesis_exp/exp54_rar_sft/training_contract.py",
    "thesis_exp/exp54_rar_sft/run_dev_inference_vllm.py",
)


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_binding(seed: int) -> dict[str, Any]:
    checkpoint = (
        TRAINING_ROOT
        / f"seed{seed}"
        / "r3"
        / "checkpoint-logical-epoch-3"
    )
    adapter = checkpoint / "adapter"
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": "R3",
        "seed": seed,
        "logical_epoch_number": 3,
        "global_optimizer_step": 996,
        "test_accessed": False,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"{state_path}: checkpoint differs at {key}")
    return {
        "seed": seed,
        "checkpoint_relative_path": checkpoint.relative_to(
            TRAINING_ROOT
        ).as_posix(),
        "adapter_config": binding(adapter / "adapter_config.json"),
        "adapter_model": binding(adapter / "adapter_model.safetensors"),
        "trainer_state": binding(state_path),
    }


def build() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("status") != "TRAIN_ONLY_GREEDY_DIAGNOSTIC_PLANNED"
        or config.get("split") != "train"
        or config.get("expected_total_outputs") != 7962
        or config.get("preference_pair_construction_allowed") is not False
        or config.get("gpu_execution_allowed") is not False
    ):
        raise ValueError("actual failure-bank configuration differs")
    rows = load_train_rows()
    label_counts = Counter(int(row["label_5"]) for row in rows)
    metric_counts = Counter(str(row["metric_id"]) for row in rows)
    language_counts = Counter(str(row["language"]) for row in rows)
    if len(metric_counts) != 12 or set(language_counts) != {"en", "zh"}:
        raise ValueError("train coverage differs")
    checkpoints = [checkpoint_binding(seed) for seed in SEEDS]
    return {
        "schema_version": "exp54-actual-failure-bank-plan-v1",
        "status": "ACTUAL_FAILURE_BANK_PLAN_READY_NO_GPU_EXECUTION",
        "scientific_role": (
            "Neutral train-only inventory of actual R3-SFT failures; "
            "independent of the later actual/synthetic/hybrid pair decision."
        ),
        "configuration": binding(CONFIG_PATH),
        "train": {
            **binding(TRAIN_PATH),
            "rows": len(rows),
            "label_counts": {
                str(label): int(label_counts[label])
                for label in sorted(label_counts)
            },
            "metric_counts": {
                metric: int(metric_counts[metric])
                for metric in sorted(metric_counts)
            },
            "language_counts": {
                language: int(language_counts[language])
                for language in sorted(language_counts)
            },
        },
        "checkpoints": checkpoints,
        "generation": config["generation"],
        "expected_outputs_per_seed": len(rows),
        "expected_total_outputs": len(rows) * len(SEEDS),
        "sources": {
            path: binding(REPO_ROOT / path) for path in SOURCE_PATHS
        },
        "execution": {
            "gpu_used": False,
            "gpu_execution_allowed": False,
            "gpu_execution_requires_user_confirmation": True,
            "preference_pair_construction_started": False,
            "stochastic_rollout_started": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
    }


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    plan = build()
    write(OUTPUT_PATH, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
