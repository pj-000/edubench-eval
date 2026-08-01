"""Build one exact, train-only SORC-DPO correctness-smoke authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    ARMS,
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK,
    DEFAULT_QUALIFICATION_CANDIDATE_LOCK,
    DEFAULT_SMOKE_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES,
    SMOKE_ROOT,
    _canonical_sha256,
    read_json,
    verify_p3_qualification_lock,
)


AUTHORIZATION_ROOT = SMOKE_ROOT / "private/authorizations"
RUN_ROOT = SMOKE_ROOT / "runs"
RUNNER = (
    Path(__file__).resolve().parent / "train_sorc_dpo_smoke.py"
)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_authorization(
    *,
    arm: str,
    cuda_device_uuid: str,
    rationale_qualification_lock: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not cuda_device_uuid.startswith("GPU-"):
        raise ValueError("CUDA device UUID must use the GPU- form")
    base_configuration = read_json(DEFAULT_BASE_TRAINING_CONFIGURATION)
    output_dir = (
        RUN_ROOT / f"correctness-smoke-v1-{arm.lower()}"
    ).resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    authorization = {
        "schema_version": "exp54-sorc-dpo-gpu-smoke-authorization-v1",
        "status": "SORC_DPO_GPU_SMOKE_AUTHORIZED",
        "gpu_smoke_allowed": True,
        "formal_preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "arm": arm,
        "seed": 42,
        "optimizer_steps": 1,
        "physical_micro_batch_pairs": 1,
        "smoke_lock_sha256": sha256_file(DEFAULT_SMOKE_LOCK),
        "smoke_plan_sha256": sha256_file(DEFAULT_SMOKE_PLAN),
        "training_config_sha256": sha256_file(DEFAULT_TRAINING_CONFIG),
        "checkpoint_lock_sha256": sha256_file(DEFAULT_CHECKPOINT_LOCK),
        "base_training_configuration_sha256": sha256_file(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        "base_model_snapshot_identity_sha256": _canonical_sha256(
            base_configuration["model"]
        ),
        "preference_training_frozen_lock_sha256": sha256_file(
            DEFAULT_FROZEN_TRAINING_LOCK
        ),
        "smoke_implementation_candidate_lock_sha256": sha256_file(
            DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
        ),
        "runner_sha256": sha256_file(RUNNER),
        "cuda_device_name": "NVIDIA RTX A6000",
        "cuda_device_uuid": cuda_device_uuid,
        "minimum_free_memory_bytes_before_load": (
            MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES
        ),
        "output_dir": str(output_dir),
    }
    if arm == "P3_JOINT_SORC":
        if rationale_qualification_lock is None:
            raise PermissionError(
                "P3 authorization requires final rationale qualification"
            )
        qualification_path = rationale_qualification_lock.resolve(strict=True)
        verify_p3_qualification_lock(
            qualification_path=qualification_path,
            qualification_candidate_lock_path=(
                DEFAULT_QUALIFICATION_CANDIDATE_LOCK
            ),
        )
        authorization.update(
            {
                "rationale_qualification_lock_path": str(
                    qualification_path
                ),
                "rationale_qualification_lock_sha256": sha256_file(
                    qualification_path
                ),
                "rationale_qualification_candidate_lock_sha256": sha256_file(
                    DEFAULT_QUALIFICATION_CANDIDATE_LOCK
                ),
            }
        )
    authorization_path = (
        AUTHORIZATION_ROOT / f"{arm.lower()}.json"
    )
    write_json_exclusive(authorization_path, authorization)
    return authorization_path, authorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cuda-device-uuid", required=True)
    parser.add_argument("--rationale-qualification-lock", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, authorization = build_authorization(
        arm=args.arm,
        cuda_device_uuid=args.cuda_device_uuid,
        rationale_qualification_lock=args.rationale_qualification_lock,
    )
    print(
        json.dumps(
            {
                "status": "SORC_DPO_GPU_SMOKE_AUTHORIZATION_WRITTEN",
                "arm": args.arm,
                "authorization_path": str(path),
                "authorization_sha256": sha256_file(path),
                "cuda_device_name": authorization["cuda_device_name"],
                "cuda_device_uuid": authorization["cuda_device_uuid"],
                "physical_micro_batch_pairs": 1,
                "optimizer_steps": 1,
                "formal_preference_training_allowed": False,
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
