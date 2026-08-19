"""Audit all formal train-only SORC-DPO runs and LoRA adapter artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.train_sorc_dpo_formal import (
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_TRAINING_CONFIG,
    OUTPUT_ROOT,
)


EXPECTED_RUNS = {
    ("P1_FIELD_DPO", 42),
    ("P1_FIELD_DPO", 43),
    ("P1_FIELD_DPO", 44),
    ("P2_SORC_SCORE", 42),
    ("P2_SORC_SCORE", 43),
    ("P2_SORC_SCORE", 44),
    ("P3_JOINT_SORC", 42),
    ("P3_JOINT_SORC", 43),
    ("P3_JOINT_SORC", 44),
    ("P1_SYN_SEED42", 42),
}
DEFAULT_REPORT = OUTPUT_ROOT / "formal_training_audit_report.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_directory(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / arm.lower() / f"seed_{seed}"


def _adapter_spec(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        if not keys:
            raise ValueError(f"{path}: empty adapter")
        tensors = {key: handle.get_tensor(key) for key in keys}
    if any("lora_A" not in key and "lora_B" not in key for key in keys):
        raise ValueError(f"{path}: non-LoRA tensor found")
    if any(
        forbidden in key
        for key in keys
        for forbidden in ("base_layer", "reference", "optimizer")
    ):
        raise ValueError(f"{path}: forbidden tensor namespace found")
    dtypes = sorted({str(tensor.dtype) for tensor in tensors.values()})
    shapes = {key: list(tensor.shape) for key, tensor in tensors.items()}
    total_parameters = sum(tensor.numel() for tensor in tensors.values())
    shape_payload = json.dumps(
        shapes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "tensor_count": len(keys),
        "total_parameters": total_parameters,
        "dtypes": dtypes,
        "key_and_shape_sha256": hashlib.sha256(shape_payload).hexdigest(),
    }


def audit() -> dict[str, Any]:
    config = _read_json(DEFAULT_TRAINING_CONFIG)
    frozen = _read_json(DEFAULT_FROZEN_TRAINING_LOCK)
    if (
        frozen.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or sha256_file(DEFAULT_TRAINING_CONFIG)
        != str(frozen["source_hashes"]["training_config"])
    ):
        raise ValueError("frozen training configuration differs")

    discovered = set()
    run_reports: dict[str, Any] = {}
    common_adapter_spec: dict[str, Any] | None = None
    adapter_hashes = set()
    for arm, seed in sorted(EXPECTED_RUNS):
        run_dir = _run_directory(arm, seed)
        progress_path = run_dir / "progress.json"
        result_path = run_dir / "result.json"
        adapter_config_path = run_dir / "adapter/adapter_config.json"
        adapter_model_path = run_dir / "adapter/adapter_model.safetensors"
        for path in (
            progress_path,
            result_path,
            adapter_config_path,
            adapter_model_path,
        ):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing or invalid formal artifact: {path}")
        progress = _read_json(progress_path)
        result = _read_json(result_path)
        adapter_config = _read_json(adapter_config_path)
        exact = {
            "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
            "arm": arm,
            "seed": seed,
            "optimizer_steps": 27,
            "preference_epochs": 1,
            "physical_micro_batch_pairs": 4,
            "dev_accessed": False,
            "test_accessed": False,
        }
        for field, expected in exact.items():
            if result.get(field) != expected:
                raise ValueError(f"{arm}/{seed}: result {field} differs")
        if (
            progress.get("status") != "COMPLETE"
            or progress.get("optimizer_step") != 27
            or progress.get("optimizer_steps") != 27
            or progress.get("overall_percent") != 100.0
            or progress.get("dev_accessed") is not False
            or progress.get("test_accessed") is not False
        ):
            raise ValueError(f"{arm}/{seed}: progress is not complete")
        losses = list(result.get("losses") or [])
        gradients = list(result.get("gradient_norms_before_clip") or [])
        if (
            len(losses) != 27
            or len(gradients) != 27
            or not all(math.isfinite(float(value)) for value in losses)
            or not all(math.isfinite(float(value)) for value in gradients)
            or not all(float(value) > 0 for value in gradients)
        ):
            raise ValueError(f"{arm}/{seed}: loss or gradient trace differs")
        if (
            sha256_file(adapter_config_path)
            != result["output_adapter_config_sha256"]
            or sha256_file(adapter_model_path)
            != result["output_adapter_model_sha256"]
            or result["training_config_sha256"]
            != sha256_file(DEFAULT_TRAINING_CONFIG)
            or result["frozen_training_lock_sha256"]
            != sha256_file(DEFAULT_FROZEN_TRAINING_LOCK)
        ):
            raise ValueError(f"{arm}/{seed}: result hash binding differs")
        if (
            adapter_config.get("peft_type") != "LORA"
            or int(adapter_config.get("r")) != 16
            or int(adapter_config.get("lora_alpha")) != 32
            or float(adapter_config.get("lora_dropout")) != 0.05
        ):
            raise ValueError(f"{arm}/{seed}: LoRA configuration differs")
        spec = _adapter_spec(adapter_model_path)
        if (
            spec["tensor_count"] != 504
            or spec["total_parameters"] != 33030144
            or spec["dtypes"] != ["torch.float32"]
        ):
            raise ValueError(f"{arm}/{seed}: LoRA tensor specification differs")
        if common_adapter_spec is None:
            common_adapter_spec = spec
        elif spec != common_adapter_spec:
            raise ValueError(f"{arm}/{seed}: LoRA shape specification differs")
        adapter_sha256 = sha256_file(adapter_model_path)
        if adapter_sha256 in adapter_hashes:
            raise ValueError(f"{arm}/{seed}: duplicate trained adapter bytes")
        adapter_hashes.add(adapter_sha256)
        discovered.add((arm, seed))
        run_reports[f"{arm}/seed_{seed}"] = {
            "status": "PASS",
            "optimizer_steps": 27,
            "loss_trace_finite": True,
            "gradient_trace_finite_positive": True,
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
            "adapter_model_sha256": adapter_sha256,
            "result_sha256": sha256_file(result_path),
        }
    if discovered != EXPECTED_RUNS:
        raise ValueError("formal run inventory differs")
    assert common_adapter_spec is not None
    return {
        "schema_version": "exp54-sorc-dpo-formal-training-audit-v1",
        "status": "SORC_DPO_FORMAL_TRAINING_AUDIT_PASS",
        "run_count": len(run_reports),
        "all_runs_complete": True,
        "all_runs_have_27_optimizer_steps": True,
        "all_loss_traces_finite": True,
        "all_gradient_traces_finite_positive": True,
        "all_adapter_hashes_unique": True,
        "common_adapter_spec": common_adapter_spec,
        "runs": run_reports,
        "source_hashes": {
            "training_config": sha256_file(DEFAULT_TRAINING_CONFIG),
            "frozen_training_lock": sha256_file(
                DEFAULT_FROZEN_TRAINING_LOCK
            ),
            "auditor": sha256_file(Path(__file__)),
        },
        "dev_accessed": False,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit()
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "all_runs_complete": report["all_runs_complete"],
                "adapter_spec": report["common_adapter_spec"],
                "report_sha256": sha256_file(args.output),
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
