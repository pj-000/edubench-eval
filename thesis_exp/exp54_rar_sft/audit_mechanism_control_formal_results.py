"""Audit the nine train-only Exp54 mechanism-control checkpoints.

The public outputs contain only aggregate statistics and cryptographic hashes.
No train row, token sequence, prediction, rationale, dev row, or test row is
read or emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
SEEDS = (42, 43, 44)
EXPECTED_STEPS = {
    "R3_TOKENAVG": 996,
    "P1_FULLSEQ": 27,
    "P1_SYN_LR5E6": 27,
}
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_formal_plan_v1.json"
)
DEFAULT_FORMAL_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "mechanism_control_formal"
)
DEFAULT_AUDIT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "mechanism_control_formal_audit"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_regular(path: Path, *, label: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if metadata.st_size < 1:
        raise ValueError(f"{label} is empty")


def _require_finite_vector(
    values: Any,
    *,
    length: int,
    positive: bool,
    label: str,
) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{label} length differs")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{label} contains a non-finite value")
    if positive and not all(value > 0 for value in converted):
        raise ValueError(f"{label} contains a non-positive value")
    return converted


def _audit_safetensors(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    from safetensors import safe_open

    _require_regular(path, label=label)
    tensor_count = 0
    parameter_count = 0
    dtypes: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or any(
            "lora_A" not in key and "lora_B" not in key for key in keys
        ):
            raise ValueError(f"{label} contains a non-LoRA tensor")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.numel() < 1:
                raise ValueError(f"{label} contains an empty tensor")
            tensor_count += 1
            parameter_count += int(tensor.numel())
            dtypes.add(str(tensor.dtype))
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "dtypes": sorted(dtypes),
    }


def _result_directory(formal_root: Path, arm: str, seed: int) -> Path:
    return formal_root / arm.lower() / f"seed_{seed}"


def audit(
    *,
    formal_root: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _read_object(plan_path)
    if (
        plan.get("status")
        != "USER_APPROVED_TRAIN_ONLY_FULL_MECHANISM_CONTROLS"
        or plan.get("arms") != list(ARMS)
        or plan.get("seeds") != list(SEEDS)
        or plan.get("run_count") != 9
        or plan.get("dev_allowed") is not False
        or plan.get("test_allowed") is not False
        or plan.get("execution_device_override", {}).get(
            "must_be_disclosed_as_execution_hardware_difference"
        )
        is not True
    ):
        raise ValueError("formal mechanism-control plan differs")
    if formal_root.resolve(strict=True) != Path(
        plan["server_paths"]["output_root"]
    ).resolve(strict=True):
        raise ValueError("formal result root differs from the execution plan")
    failures = list(formal_root.glob("**/failure.json"))
    if failures:
        raise ValueError("formal result root contains a failure artifact")

    run_reports: dict[str, Any] = {}
    result_hashes: dict[str, str] = {}
    adapter_hashes: dict[str, str] = {}
    all_adapter_hashes: list[str] = []
    all_tensor_specs: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            run_key = f"{arm}:seed{seed}"
            directory = _result_directory(formal_root, arm, seed)
            result_path = directory / "result.json"
            progress_path = directory / "progress.json"
            for path, label in (
                (result_path, f"{run_key} result"),
                (progress_path, f"{run_key} progress"),
            ):
                _require_regular(path, label=label)
            result = _read_object(result_path)
            progress = _read_object(progress_path)
            expected_steps = EXPECTED_STEPS[arm]
            if (
                result.get("status")
                != "MECHANISM_CONTROL_FORMAL_TRAINING_COMPLETE"
                or result.get("arm") != arm
                or int(result.get("seed", -1)) != seed
                or int(result.get("optimizer_steps", -1)) != expected_steps
                or result.get("dev_accessed") is not False
                or result.get("test_accessed") is not False
                or progress.get("status") != "COMPLETE"
                or int(progress.get("optimizer_step", -1)) != expected_steps
                or int(progress.get("optimizer_steps", -1)) != expected_steps
                or float(progress.get("completion_percent", -1)) != 100.0
                or progress.get("dev_accessed") is not False
                or progress.get("test_accessed") is not False
            ):
                raise ValueError(f"{run_key}: completion contract differs")
            losses = _require_finite_vector(
                result.get("losses"),
                length=expected_steps,
                positive=True,
                label=f"{run_key} losses",
            )
            gradients = _require_finite_vector(
                result.get("gradient_norms_before_clip"),
                length=expected_steps,
                positive=True,
                label=f"{run_key} gradients",
            )
            if (
                not math.isfinite(float(result.get("elapsed_seconds", -1)))
                or float(result["elapsed_seconds"]) <= 0
                or int(result.get("peak_allocated_bytes", 0)) <= 0
                or int(result.get("peak_reserved_bytes", 0)) <= 0
            ):
                raise ValueError(f"{run_key}: runtime statistics differ")
            assignment = plan["gpu_assignments_by_arm_and_seed"][arm][
                str(seed)
            ]
            identity = result.get("device_identity", {})
            if (
                identity.get("uuid") != assignment["uuid"]
                or identity.get("name") != assignment["name"]
            ):
                raise ValueError(f"{run_key}: GPU identity differs")

            if arm == "R3_TOKENAVG":
                adapter_dir = (
                    directory / "checkpoint-logical-epoch-3/adapter"
                )
                if (
                    int(result.get("event_count", -1)) != 7_962
                    or int(result.get("logical_epochs", -1)) != 3
                    or result.get("loss_reduction")
                    != "token_average_active_scaled_by_2"
                    or float(result.get("learning_rate", -1)) != 1e-4
                ):
                    raise ValueError(f"{run_key}: R3 contract differs")
                for epoch in (1, 2, 3):
                    checkpoint = (
                        directory / f"checkpoint-logical-epoch-{epoch}"
                    )
                    state_path = checkpoint / "trainer_state.json"
                    for filename in (
                        "optimizer.pt",
                        "scheduler.pt",
                        "rng_state.pt",
                    ):
                        _require_regular(
                            checkpoint / filename,
                            label=f"{run_key} epoch {epoch} {filename}",
                        )
                    state = _read_object(state_path)
                    if (
                        state.get("status")
                        != "EXP54_FORMAL_CHECKPOINT_UNEVALUATED"
                        or state.get("arm") != arm
                        or int(state.get("seed", -1)) != seed
                        or int(state.get("logical_epoch_number", -1))
                        != epoch
                        or int(state.get("global_optimizer_step", -1))
                        != 332 * epoch
                        or state.get("dev_accessed") is not False
                        or state.get("test_accessed") is not False
                    ):
                        raise ValueError(
                            f"{run_key}: epoch {epoch} state differs"
                        )
            else:
                adapter_dir = directory / "adapter"
                expected_mode = (
                    "fullseq" if arm == "P1_FULLSEQ" else "field"
                )
                if (
                    int(result.get("pair_count", -1)) != 838
                    or int(result.get("preference_epochs", -1)) != 1
                    or float(result.get("learning_rate", -1)) != 5e-6
                    or float(result.get("beta", -1)) != 0.1
                    or result.get("log_probability_mode") != expected_mode
                    or int(result.get("physical_micro_batch_pairs", -1)) != 4
                    or int(result.get("gradient_accumulation_pairs", -1))
                    != 32
                    or int(result.get("final_group_pairs", -1)) != 6
                ):
                    raise ValueError(
                        f"{run_key}: preference contract differs"
                    )
                if result["output_adapter_model_sha256"] in set(
                    result["cold_start_adapter_hashes"].values()
                ):
                    raise ValueError(
                        f"{run_key}: output equals the cold-start adapter"
                    )

            config_path = adapter_dir / "adapter_config.json"
            model_path = adapter_dir / "adapter_model.safetensors"
            _require_regular(config_path, label=f"{run_key} adapter config")
            model_spec = _audit_safetensors(
                model_path,
                label=f"{run_key} adapter model",
            )
            if (
                sha256_file(config_path)
                != result["output_adapter_config_sha256"]
                or sha256_file(model_path)
                != result["output_adapter_model_sha256"]
            ):
                raise ValueError(f"{run_key}: output adapter hash differs")
            adapter_hash = sha256_file(model_path)
            all_adapter_hashes.append(adapter_hash)
            all_tensor_specs.append(model_spec)
            result_hashes[run_key] = sha256_file(result_path)
            adapter_hashes[run_key] = adapter_hash
            run_reports[run_key] = {
                "optimizer_steps": expected_steps,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_min": min(losses),
                "loss_max": max(losses),
                "gradient_norm_min": min(gradients),
                "gradient_norm_max": max(gradients),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "peak_allocated_bytes": int(
                    result["peak_allocated_bytes"]
                ),
                "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
                "adapter_sha256": adapter_hash,
                "adapter_tensor_spec": model_spec,
                "device_name": identity["name"],
                "device_uuid": identity["uuid"],
            }

    if len(set(all_adapter_hashes)) != 9:
        raise ValueError("the nine output adapter hashes are not unique")
    canonical_spec = all_tensor_specs[0]
    if any(spec != canonical_spec for spec in all_tensor_specs[1:]):
        raise ValueError("adapter tensor specifications differ across runs")

    report = {
        "schema_version": "exp54-mechanism-control-formal-audit-v1",
        "status": (
            "MECHANISM_CONTROL_FORMAL_RESULT_AUDIT_PASS_"
            "EVALUATION_NOT_STARTED"
        ),
        "formal_run_count": 9,
        "completed_run_count": 9,
        "failed_run_count": 0,
        "total_optimizer_steps": sum(EXPECTED_STEPS.values()) * 3,
        "all_losses_finite_and_positive": True,
        "all_gradient_norms_finite_and_positive": True,
        "all_progress_files_complete": True,
        "all_adapter_hashes_unique": True,
        "all_adapter_tensor_specs_equal": True,
        "all_adapter_files_content_validated": True,
        "r3_epoch_checkpoint_inventory_complete": True,
        "r3_execution_hardware_difference_disclosed": True,
        "run_reports": run_reports,
        "dev_accessed": False,
        "test_accessed": False,
        "evaluation_started": False,
    }
    lock = {
        "schema_version": "exp54-mechanism-control-formal-lock-v1",
        "status": "MECHANISM_CONTROL_FORMAL_CHECKPOINTS_FROZEN",
        "formal_plan_sha256": sha256_file(plan_path),
        "result_hashes": result_hashes,
        "adapter_hashes": adapter_hashes,
        "checkpoint_count": 9,
        "training_complete": True,
        "checkpoint_frozen": True,
        "dev_accessed": False,
        "test_accessed": False,
        "evaluation_started": False,
    }
    return report, lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=DEFAULT_FORMAL_ROOT,
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "audit_report.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "checkpoint_lock.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.report, args.lock):
        if path.exists():
            raise FileExistsError(path)
    report, lock = audit(
        formal_root=args.formal_root,
        plan_path=args.plan,
    )
    _write_object(args.report, report)
    lock["audit_report_sha256"] = sha256_file(args.report)
    _write_object(args.lock, lock)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
