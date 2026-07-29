"""Run the operator-approved LR-only train smoke on an idle A6000.

This wrapper intentionally skips the retired external root authorization
mechanism.  It keeps all frozen data/model/checkpoint checks, verifies the
single permitted LR override, and never reads dev or test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_CHECKPOINT_LOCK,
    DEFAULT_SMOKE_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    execute_one_smoke_step,
    load_and_validate_smoke_rows,
    read_json,
)


ARMS = ("P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC")
OVERLAY_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_lr5e6_followup_v1.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/smoke"
)
MINIMUM_FREE_BYTES = 40 * 1024**3


def verify_overlay() -> dict[str, Any]:
    overlay = read_json(OVERLAY_PATH)
    diagnostic_path = REPO_ROOT / str(
        overlay["trigger"]["diagnostic_report_path"]
    )
    diagnostic = read_json(diagnostic_path)
    if (
        overlay.get("status") != "TRAIN_ONLY_SMOKE_THEN_SEED42_SCOUT"
        or overlay["only_override"]
        != {
            "field": "optimization.learning_rate",
            "from": 5e-7,
            "to": 5e-6,
        }
        or sha256_file(DEFAULT_TRAINING_CONFIG)
        != str(overlay["parent_protocol"]["training_config_sha256"])
        or sha256_file(diagnostic_path)
        != str(overlay["trigger"]["diagnostic_report_sha256"])
        or diagnostic.get("status")
        != overlay["trigger"]["required_diagnostic_status"]
        or diagnostic["lr_only_followup_trigger"][
            "lr_only_followup_trigger_satisfied"
        ]
        is not True
        or overlay["boundaries"]["formal_lr5e7_checkpoints_preserved"]
        is not True
        or overlay["boundaries"]["test_allowed"] is not False
        or overlay["boundaries"]["dev_accessed_during_train_only_stages"]
        is not False
    ):
        raise PermissionError("LR-only exploratory overlay differs")
    return overlay


def run(*, arm: str, cuda_device_uuid: str) -> dict[str, Any]:
    overlay = verify_overlay()
    rows, _smoke_lock, _smoke_plan = load_and_validate_smoke_rows(
        arm=arm,
        smoke_lock_path=DEFAULT_SMOKE_LOCK,
        smoke_plan_path=DEFAULT_SMOKE_PLAN,
    )
    output_dir = OUTPUT_ROOT / arm.lower()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    authorization = {
        "cuda_device_name": "NVIDIA RTX A6000",
        "cuda_device_uuid": cuda_device_uuid,
        "minimum_free_memory_bytes_before_load": MINIMUM_FREE_BYTES,
        "physical_micro_batch_pairs": 1,
        "output_dir": str(output_dir),
    }
    result = execute_one_smoke_step(
        arm=arm,
        rows=rows,
        authorization=authorization,
        training_config_path=DEFAULT_TRAINING_CONFIG,
        checkpoint_lock_path=DEFAULT_CHECKPOINT_LOCK,
        base_training_configuration_path=(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        learning_rate_override=float(overlay["only_override"]["to"]),
        protocol_overlay_path=OVERLAY_PATH,
    )
    if (
        float(result["learning_rate"]) != 5e-6
        or result.get("dev_accessed") is not False
        or result.get("test_accessed") is not False
    ):
        raise ValueError("LR-only smoke result differs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cuda-device-uuid", required=True)
    args = parser.parse_args()
    result = run(arm=args.arm, cuda_device_uuid=args.cuda_device_uuid)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"],
                "learning_rate": result["learning_rate"],
                "loss": result["loss"],
                "gradient_norm_before_clip": result[
                    "gradient_norm_before_clip"
                ],
                "output_adapter_model_sha256": result[
                    "output_adapter_model_sha256"
                ],
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
