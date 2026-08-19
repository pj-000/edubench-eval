"""Audit completed train-only SORC-DPO correctness-smoke results."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from safetensors import safe_open

from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK,
    DEFAULT_SMOKE_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    SMOKE_ROOT,
    _canonical_sha256,
    read_json,
)


RUNNER = Path(__file__).resolve().parent / "train_sorc_dpo_smoke.py"
AUTHORIZATION_ROOT = SMOKE_ROOT / "private/authorizations"
RUN_ROOT = SMOKE_ROOT / "runs"
OUTPUT = SMOKE_ROOT / "correctness_smoke_partial_report.json"
ARMS = {
    "P1_FIELD_DPO": 32,
    "P2_SORC_SCORE": 32,
    "P1_SYN_SEED42": 32,
}
ALLOWED_OUTPUT_FILES = {
    "adapter/README.md",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "result.json",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_regular(path: Path, *, label: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise PermissionError(f"{label} cannot be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{label} must be a regular file")


def _audit_authorization(
    authorization: dict[str, Any],
    *,
    arm: str,
    authorization_path: Path,
    run_dir: Path,
) -> None:
    if authorization.get("status") != "SORC_DPO_GPU_SMOKE_AUTHORIZED":
        raise PermissionError(f"{arm}: authorization status differs")
    exact = {
        "arm": arm,
        "seed": 42,
        "optimizer_steps": 1,
        "physical_micro_batch_pairs": 1,
        "gpu_smoke_allowed": True,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "smoke_lock_sha256": sha256_file(DEFAULT_SMOKE_LOCK),
        "smoke_plan_sha256": sha256_file(DEFAULT_SMOKE_PLAN),
        "training_config_sha256": sha256_file(DEFAULT_TRAINING_CONFIG),
        "checkpoint_lock_sha256": sha256_file(DEFAULT_CHECKPOINT_LOCK),
        "base_training_configuration_sha256": sha256_file(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        "preference_training_frozen_lock_sha256": sha256_file(
            DEFAULT_FROZEN_TRAINING_LOCK
        ),
        "smoke_implementation_candidate_lock_sha256": sha256_file(
            DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
        ),
        "runner_sha256": sha256_file(RUNNER),
        "cuda_device_name": "NVIDIA RTX A6000",
        "output_dir": str(run_dir.resolve(strict=True)),
    }
    base = read_json(DEFAULT_BASE_TRAINING_CONFIGURATION)
    exact["base_model_snapshot_identity_sha256"] = _canonical_sha256(
        base["model"]
    )
    for field, expected in exact.items():
        if authorization.get(field) != expected:
            raise ValueError(f"{arm}: authorization {field} differs")
    if (
        not isinstance(authorization.get("cuda_device_uuid"), str)
        or not str(authorization["cuda_device_uuid"]).startswith("GPU-")
    ):
        raise ValueError(f"{arm}: authorization UUID differs")
    if int(authorization["minimum_free_memory_bytes_before_load"]) < (
        40 * 1024**3
    ):
        raise ValueError(f"{arm}: free-memory floor differs")
    _require_regular(authorization_path, label=f"{arm} authorization")


def _audit_adapter(run_dir: Path, *, arm: str) -> dict[str, Any]:
    discovered = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"{arm}: output symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            discovered.add(relative)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"{arm}: irregular output: {relative}")
    if discovered != ALLOWED_OUTPUT_FILES:
        raise ValueError(f"{arm}: output file set differs")
    config_path = run_dir / "adapter/adapter_config.json"
    model_path = run_dir / "adapter/adapter_model.safetensors"
    config = read_json(config_path)
    base = read_json(DEFAULT_BASE_TRAINING_CONFIGURATION)
    preference = read_json(DEFAULT_TRAINING_CONFIG)
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": int(preference["model_and_adapter"]["lora_rank"]),
        "lora_alpha": int(
            preference["model_and_adapter"]["lora_alpha"]
        ),
        "lora_dropout": float(
            preference["model_and_adapter"]["lora_dropout"]
        ),
        "bias": "none",
        "modules_to_save": None,
        "inference_mode": True,
        "base_model_name_or_path": str(base["model"]["local_path"]),
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"{arm}: adapter config {field} differs")
    expected_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if set(config.get("target_modules") or []) != expected_targets:
        raise ValueError(f"{arm}: adapter target modules differ")
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != 504:
            raise ValueError(f"{arm}: adapter tensor count differs")
        parameter_count = 0
        dtypes = set()
        for key in keys:
            if not (
                key.endswith(".lora_A.weight")
                or key.endswith(".lora_B.weight")
            ):
                raise ValueError(f"{arm}: non-LoRA tensor saved: {key}")
            tensor = handle.get_tensor(key)
            parameter_count += int(tensor.numel())
            dtypes.add(str(tensor.dtype))
    if parameter_count != 33030144 or dtypes != {"torch.float32"}:
        raise ValueError(f"{arm}: saved LoRA specification differs")
    return {
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_model_sha256": sha256_file(model_path),
        "adapter_tensor_count": len(keys),
        "adapter_parameter_count": parameter_count,
        "adapter_dtypes": sorted(dtypes),
    }


def _audit_result(
    result: dict[str, Any],
    authorization: dict[str, Any],
    adapter: dict[str, Any],
    *,
    arm: str,
    pair_count: int,
) -> None:
    exact = {
        "schema_version": "exp54-sorc-dpo-gpu-smoke-result-v1",
        "status": "SORC_DPO_GPU_SMOKE_STEP_COMPLETE_REQUIRES_RESULT_AUDIT",
        "arm": arm,
        "seed": 42,
        "pair_count": pair_count,
        "optimizer_steps": 1,
        "dev_accessed": False,
        "test_accessed": False,
        "output_adapter_config_sha256": adapter[
            "adapter_config_sha256"
        ],
        "output_adapter_model_sha256": adapter[
            "adapter_model_sha256"
        ],
    }
    for field, expected in exact.items():
        if result.get(field) != expected:
            raise ValueError(f"{arm}: result {field} differs")
    for field in ("loss", "gradient_norm_before_clip"):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{arm}: result {field} is invalid")
    identity = result.get("cuda_device_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{arm}: CUDA identity is missing")
    for field in ("name", "uuid"):
        if identity.get(field) != authorization.get(
            f"cuda_device_{field}"
        ):
            raise ValueError(f"{arm}: CUDA {field} differs")
    free_before = int(identity["free_memory_bytes_before_load"])
    total = int(identity["total_memory_bytes"])
    peak = int(result["peak_gpu_memory_bytes"])
    if (
        free_before
        < int(authorization["minimum_free_memory_bytes_before_load"])
        or peak <= 0
        or peak >= total
    ):
        raise ValueError(f"{arm}: CUDA memory accounting differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = {}
    for arm, pair_count in ARMS.items():
        authorization_path = (
            AUTHORIZATION_ROOT / f"{arm.lower()}.json"
        )
        run_dir = RUN_ROOT / f"correctness-smoke-v1-{arm.lower()}"
        authorization = read_json(authorization_path)
        _audit_authorization(
            authorization,
            arm=arm,
            authorization_path=authorization_path,
            run_dir=run_dir,
        )
        adapter = _audit_adapter(run_dir, arm=arm)
        result_path = run_dir / "result.json"
        _require_regular(result_path, label=f"{arm} result")
        result = read_json(result_path)
        _audit_result(
            result,
            authorization,
            adapter,
            arm=arm,
            pair_count=pair_count,
        )
        arms[arm] = {
            "pair_count": pair_count,
            "optimizer_steps": 1,
            "loss": float(result["loss"]),
            "gradient_norm_before_clip": float(
                result["gradient_norm_before_clip"]
            ),
            "peak_gpu_memory_bytes": int(
                result["peak_gpu_memory_bytes"]
            ),
            "authorization_sha256": sha256_file(authorization_path),
            "result_sha256": sha256_file(result_path),
            **adapter,
        }
    report = {
        "schema_version": "exp54-sorc-dpo-correctness-smoke-partial-v1",
        "status": "NON_P3_CORRECTNESS_SMOKE_PASS_P3_PENDING_QUALIFICATION",
        "arms": arms,
        "non_p3_arms_passed": True,
        "p3_executed": False,
        "capacity_probe_executed": False,
        "source_hashes": {
            "result_auditor": sha256_file(Path(__file__)),
            "smoke_runner": sha256_file(RUNNER),
            "smoke_implementation_candidate_lock": sha256_file(
                DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
            ),
        },
        "formal_preference_training_allowed": False,
        "evaluator_called": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
