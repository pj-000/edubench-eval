"""Train one fixed multi-seed LR-only SORC-DPO arm on an idle A6000."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.run_user_sorc_dpo_lr_followup_smoke import (
    ARMS,
    OUTPUT_ROOT as SMOKE_OUTPUT_ROOT,
    OVERLAY_PATH as SMOKE_OVERLAY_PATH,
    verify_overlay,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_formal import execute
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import read_json


OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/train"
)
MULTISEED_OVERLAY_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_lr5e6_multiseed_execution_v1.json"
)


def verify_multiseed_overlay() -> dict[str, Any]:
    smoke_overlay = verify_overlay()
    overlay = read_json(MULTISEED_OVERLAY_PATH)
    if (
        overlay.get("status")
        != "TRAIN_ONLY_MULTI_SEED_EXECUTION_ALLOWED_NO_DEV"
        or overlay["parent_lr_followup_overlay"]["sha256"]
        != sha256_file(SMOKE_OVERLAY_PATH)
        or overlay["operator_decision"]["seeds"] != [42, 43, 44]
        or overlay["operator_decision"]["arms"] != list(ARMS)
        or overlay["operator_decision"]["run_all_three_seeds_before_any_new_dev"]
        is not True
        or overlay["scientific_variables"]["only_change_from_formal_null"]
        != "learning_rate_5e-7_to_5e-6"
        or overlay["execution_boundaries"]["existing_lr5e7_runs_preserved"]
        is not True
        or overlay["execution_boundaries"]["dev_allowed_during_training"]
        is not False
        or overlay["execution_boundaries"]["test_allowed"] is not False
    ):
        raise PermissionError("LR-only multi-seed execution overlay differs")
    return {
        "smoke": smoke_overlay,
        "multiseed": overlay,
    }


def verify_all_smokes() -> dict[str, dict[str, Any]]:
    output = {}
    adapter_hashes = set()
    for arm in ARMS:
        result_path = SMOKE_OUTPUT_ROOT / arm.lower() / "result.json"
        adapter_path = (
            SMOKE_OUTPUT_ROOT
            / arm.lower()
            / "adapter/adapter_model.safetensors"
        )
        result = read_json(result_path)
        if (
            result.get("status")
            != "SORC_DPO_GPU_SMOKE_STEP_COMPLETE_REQUIRES_RESULT_AUDIT"
            or result.get("arm") != arm
            or int(result.get("seed", -1)) != 42
            or int(result.get("optimizer_steps", -1)) != 1
            or float(result.get("learning_rate", -1.0)) != 5e-6
            or result.get("protocol_overlay_sha256")
            != sha256_file(SMOKE_OVERLAY_PATH)
            or result.get("dev_accessed") is not False
            or result.get("test_accessed") is not False
            or not adapter_path.is_file()
            or adapter_path.is_symlink()
            or sha256_file(adapter_path)
            != result.get("output_adapter_model_sha256")
        ):
            raise ValueError(f"{arm}: LR-only smoke result differs")
        adapter_hash = str(result["output_adapter_model_sha256"])
        if adapter_hash in adapter_hashes:
            raise ValueError("LR-only smoke adapters are not unique")
        adapter_hashes.add(adapter_hash)
        output[arm] = {
            "result_sha256": sha256_file(result_path),
            "adapter_model_sha256": adapter_hash,
            "loss": float(result["loss"]),
            "gradient_norm_before_clip": float(
                result["gradient_norm_before_clip"]
            ),
        }
    return output


def run(*, arm: str, seed: int, cuda_device_uuid: str) -> dict[str, Any]:
    overlays = verify_multiseed_overlay()
    smoke_bindings = verify_all_smokes()
    if seed not in (42, 43, 44):
        raise ValueError("LR-only follow-up seed differs")
    output_dir = OUTPUT_ROOT / arm.lower() / f"seed_{seed}"
    result = execute(
        arm=arm,
        seed=seed,
        cuda_device_uuid=cuda_device_uuid,
        output_dir=output_dir,
        expected_output_root=OUTPUT_ROOT,
        learning_rate_override=float(
            overlays["smoke"]["only_override"]["to"]
        ),
        protocol_overlay_path=MULTISEED_OVERLAY_PATH,
        result_schema_version="exp54-sorc-dpo-lr5e6-multiseed-result-v1",
    )
    result["smoke_bindings"] = smoke_bindings
    result_path = output_dir / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--cuda-device-uuid", required=True)
    args = parser.parse_args()
    result = run(
        arm=args.arm,
        seed=args.seed,
        cuda_device_uuid=args.cuda_device_uuid,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"],
                "seed": result["seed"],
                "learning_rate": result["learning_rate"],
                "optimizer_steps": result["optimizer_steps"],
                "elapsed_seconds": result["elapsed_seconds"],
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
