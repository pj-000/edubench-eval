"""Audit the three train-only mechanism-control GPU smoke results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_smoke_plan_v1.json"
)
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_results(
    *,
    plan: dict[str, Any],
    result_by_arm: dict[str, dict[str, Any]],
    result_path_by_arm: dict[str, Path],
) -> dict[str, Any]:
    if tuple(plan.get("arms") or ()) != ARMS:
        raise ValueError("smoke plan arm vector differs")
    summaries = {}
    adapter_hashes = set()
    for arm in ARMS:
        result = result_by_arm.get(arm)
        if not isinstance(result, dict):
            raise ValueError(f"{arm}: smoke result is missing")
        training = result.get("training")
        if (
            result.get("status")
            != "MECHANISM_CONTROL_TRAIN_ONLY_SMOKE_COMPLETE"
            or result.get("arm") != arm
            or int(result.get("seed", -1)) != 42
            or int(training.get("optimizer_steps", -1)) != 1
            or result.get("dev_accessed") is not False
            or result.get("test_accessed") is not False
            or result.get("full_training_started") is not False
        ):
            raise ValueError(f"{arm}: smoke execution contract differs")
        expected_rows = int(plan["source_rows"][arm])
        if int(result.get("source_row_count", -1)) != expected_rows:
            raise ValueError(f"{arm}: smoke source-row count differs")
        losses = training.get("losses")
        if not isinstance(losses, list) or not losses:
            raise ValueError(f"{arm}: smoke losses are missing")
        numeric = [
            *losses,
            training.get("gradient_norm_before_clip"),
            training.get("parameter_update_l2"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError(f"{arm}: smoke numeric result is non-finite")
        if (
            float(training["gradient_norm_before_clip"]) <= 0
            or float(training["parameter_update_l2"]) <= 0
        ):
            raise ValueError(f"{arm}: smoke gradient/update is not positive")
        if (
            int(training.get("peak_allocated_bytes", 0)) <= 0
            or int(training.get("peak_reserved_bytes", 0))
            < int(training["peak_allocated_bytes"])
        ):
            raise ValueError(f"{arm}: smoke GPU-memory accounting differs")
        assignment = plan["gpu_assignments"][arm]
        identity = training.get("device_identity")
        if (
            identity.get("uuid") != assignment["uuid"]
            or identity.get("name") != assignment["name"]
        ):
            raise ValueError(f"{arm}: smoke CUDA identity differs")
        adapter_hash = str(training.get("adapter_model_sha256") or "")
        if len(adapter_hash) != 64 or adapter_hash in adapter_hashes:
            raise ValueError(f"{arm}: smoke adapter hash is invalid or reused")
        adapter_hashes.add(adapter_hash)
        result_path = result_path_by_arm[arm]
        adapter_path = result_path.parent / "adapter/adapter_model.safetensors"
        if (
            not adapter_path.is_file()
            or adapter_path.is_symlink()
            or sha256_file(adapter_path) != adapter_hash
        ):
            raise ValueError(f"{arm}: saved smoke adapter differs")
        summaries[arm] = {
            "source_rows": expected_rows,
            "micro_batches": int(training["micro_batches"]),
            "optimizer_steps": 1,
            "loss_sum": sum(float(value) for value in losses),
            "gradient_norm_before_clip": float(
                training["gradient_norm_before_clip"]
            ),
            "parameter_update_l2": float(training["parameter_update_l2"]),
            "peak_allocated_bytes": int(training["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(training["peak_reserved_bytes"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "adapter_model_sha256": adapter_hash,
            "result_sha256": sha256_file(result_path),
        }
    return {
        "schema_version": "exp54-mechanism-control-smoke-audit-v1",
        "status": "MECHANISM_CONTROL_SMOKE_PASS_FULL_TRAINING_NOT_STARTED",
        "arms": summaries,
        "all_losses_finite": True,
        "all_gradient_norms_positive": True,
        "all_parameter_updates_positive": True,
        "all_adapter_hashes_unique": True,
        "gpu_smoke_complete": True,
        "full_training_allowed": False,
        "full_training_started": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_root = args.artifact_root / "mechanism_control_smoke"
    output_path = smoke_root / "audit_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    reject_eval_path(args.plan)
    reject_eval_path(smoke_root)
    plan = read_json(args.plan)
    result_paths = {
        arm: smoke_root / arm.lower() / "result.json" for arm in ARMS
    }
    for path in result_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = audit_results(
        plan=plan,
        result_by_arm={
            arm: read_json(path) for arm, path in result_paths.items()
        },
        result_path_by_arm=result_paths,
    )
    _atomic_json(output_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
