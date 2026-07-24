"""Audit the complete Exp54 training configuration without starting training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    reject_eval_path,
    write_json,
)
from thesis_exp.exp54_rar_sft.freeze_materialized_manifests import (
    collect_private_hashes,
)
from thesis_exp.exp54_rar_sft.inference_contract import GENERATION_KWARGS
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    ARMS,
    SEEDS,
    TRAINING_SOURCE_PATHS,
    validate_training_configuration,
    verify_model_snapshot,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
DEFAULT_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"


def _read_object(path: Path) -> dict[str, Any]:
    reject_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_frozen_inputs(
    frozen_lock_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = _read_object(frozen_lock_path)
    if (
        frozen.get("status")
        != "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
        or frozen.get("manifest_frozen") is not True
        or frozen.get("training_configuration_review_allowed") is not True
    ):
        raise PermissionError("materialized manifests are not frozen for config review")
    if (
        frozen.get("smoke_training_allowed")
        or frozen.get("formal_training_allowed")
        or frozen.get("dev_accessed")
        or frozen.get("test_accessed")
        or frozen.get("training_used")
    ):
        raise PermissionError("frozen manifest lock crosses its authorization boundary")
    actual_private = collect_private_hashes(output_dir)
    if actual_private != frozen["private_artifact_hashes"]:
        raise ValueError("private data differs from frozen manifest lock")
    return frozen


def observe_runtime() -> dict[str, Any]:
    import torch

    packages = (
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "safetensors",
        "tokenizers",
        "numpy",
        "sentencepiece",
        "llamafactory",
    )
    driver_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    gpu_inventory = []
    for line in driver_output:
        name, driver, memory = (part.strip() for part in line.split(",", 2))
        gpu_inventory.append(
            {"name": name, "driver": driver, "memory_mib": int(memory)}
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        **{name: importlib.metadata.version(name) for name in packages},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gpu_inventory": gpu_inventory,
    }


def validate_runtime(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    names = (
        "python",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "safetensors",
        "tokenizers",
        "numpy",
        "sentencepiece",
        "llamafactory",
        "cuda_runtime",
        "cudnn",
    )
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in names
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"runtime differs from candidate: {mismatches}")
    if not observed["cuda_available"] or not observed["bf16_supported"]:
        raise ValueError("audited runtime lacks CUDA/bfloat16 support")
    matching_gpus = [
        gpu
        for gpu in observed["gpu_inventory"]
        if gpu["name"] == expected["required_gpu_name"]
        and gpu["driver"] == expected["driver"]
    ]
    if len(matching_gpus) < 4:
        raise ValueError("fewer than four audited RTX A6000 GPUs are available")
    return {
        "all_versions_match": True,
        "cuda_and_bf16_available": True,
        "matching_rtx_a6000_count": len(matching_gpus),
        "single_visible_gpu_required_at_training": (
            expected["visible_gpu_count"] == 1
        ),
    }


def audit_model_structure(config: dict[str, Any]) -> dict[str, Any]:
    from safetensors import safe_open

    model_path = Path(config["model"]["local_path"])
    model_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    weight_index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    expected_model_values = {
        "architectures": [config["model"]["architecture"]],
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
        "bos_token_id": config["model"]["pad_token_id"],
        "eos_token_id": config["model"]["eos_token_id"],
    }
    for key, expected in expected_model_values.items():
        if model_config.get(key) != expected:
            raise ValueError(f"model config differs at {key}")
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 398:
        raise ValueError("model weight index tensor count differs")
    expected_shards = {
        name
        for name in config["model"]["files"]
        if name.endswith(".safetensors")
    }
    if set(weight_map.values()) != expected_shards:
        raise ValueError("model weight index references unexpected shards")
    occurrences = {
        target: sum(
            f".{target}.weight" in tensor_name for tensor_name in weight_map
        )
        for target in config["lora"]["target_modules"]
    }
    expected_occurrences = config["lora"]["expected_target_occurrences_each"]
    if set(occurrences.values()) != {expected_occurrences}:
        raise ValueError("LoRA target occurrence counts differ")
    if weight_index.get("metadata", {}).get("total_size") != 8_045_591_552:
        raise ValueError("model weight-index total size differs")
    shapes: dict[str, tuple[int, ...]] = {}
    for shard_name in sorted(expected_shards):
        with safe_open(
            model_path / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            for tensor_name in shard.keys():
                shapes[tensor_name] = tuple(
                    shard.get_slice(tensor_name).get_shape()
                )
    if set(shapes) != set(weight_map):
        raise ValueError("safetensor headers differ from weight index")
    base_parameters = sum(math.prod(shape) for shape in shapes.values())
    if base_parameters != config["lora"]["expected_base_parameters"]:
        raise ValueError("base parameter count differs")
    lora_trainable_parameters = 0
    for tensor_name, shape in shapes.items():
        if any(
            tensor_name.endswith(f".{target}.weight")
            for target in config["lora"]["target_modules"]
        ):
            if len(shape) != 2:
                raise ValueError("LoRA target is not a matrix")
            lora_trainable_parameters += config["lora"]["rank"] * sum(shape)
    if (
        lora_trainable_parameters
        != config["lora"]["expected_trainable_parameters"]
    ):
        raise ValueError("derived LoRA trainable-parameter count differs")
    return {
        "architecture": model_config["architectures"][0],
        "weight_tensor_count": len(weight_map),
        "weight_index_total_size": weight_index["metadata"]["total_size"],
        "lora_target_occurrences": occurrences,
        "base_parameters": base_parameters,
        "lora_trainable_parameters": lora_trainable_parameters,
        "all_lora_targets_present_in_every_layer": True,
    }


def main() -> None:
    args = parse_args()
    config = _read_object(args.config)
    step_plan = validate_training_configuration(config)
    frozen = validate_frozen_inputs(args.frozen_lock, args.output_dir)
    model_hashes = verify_model_snapshot(config)
    model_structure = audit_model_structure(config)
    observed_runtime = observe_runtime()
    runtime_checks = validate_runtime(observed_runtime, config["runtime"])
    source_hashes = {
        name: file_sha256(path) for name, path in TRAINING_SOURCE_PATHS.items()
    }
    audit_source_hashes = {
        "configuration_auditor": file_sha256(Path(__file__)),
        "manifest_freezer": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/freeze_materialized_manifests.py"
        ),
    }
    generation_expected = config["generation"]
    if GENERATION_KWARGS != {
        "do_sample": generation_expected["do_sample"],
        "num_beams": generation_expected["num_beams"],
        "max_new_tokens": generation_expected["max_new_tokens"],
        "use_cache": True,
    }:
        raise ValueError("generation implementation differs from config")

    report = {
        "status": "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED",
        "scope": (
            "frozen train-only manifests, full base snapshot, runtime, LoRA, "
            "two-block loss, optimizer/scheduler, precision, deterministic "
            "single-GPU execution, checkpointing, generation, and parser"
        ),
        "manifest_freeze": {
            "status": frozen["status"],
            "frozen_lock_sha256": file_sha256(args.frozen_lock),
            "review_gate": frozen["review_gate"],
            "private_artifacts_rehashed_equal": True,
        },
        "configuration_sha256": file_sha256(args.config),
        "model_snapshot": {
            "model_id": config["model"]["model_id"],
            "immutable_revision": config["model"]["immutable_revision"],
            "file_hashes": model_hashes,
            "all_file_sizes_and_hashes_match": True,
            **model_structure,
        },
        "runtime_observed": observed_runtime,
        "runtime_checks": runtime_checks,
        "lora": config["lora"],
        "data_and_step_plan": {
            **config["data"],
            **step_plan,
            "physical_one_pass_equals_three_logical_epochs": True,
            "logical_epoch_remainder_is_flushed_not_carried": True,
        },
        "loss": config["loss"],
        "optimization": config["optimization"],
        "precision_and_memory": config["precision_and_memory"],
        "determinism": config["determinism"],
        "checkpointing": config["checkpointing"],
        "generation": config["generation"],
        "parser": config["parser"],
        "training_source_hashes": source_hashes,
        "audit_source_hashes": audit_source_hashes,
        "authorization_gate": {
            "external_reviewed_lock_required_by_entrypoint": True,
            "model_weights_loaded_before_authorization": False,
            "current_authorization_lock_exists": False,
            "smoke_training_allowed": False,
            "formal_training_allowed": False,
        },
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    report_path = (
        args.output_dir / "audit/training_configuration_candidate_report.json"
    )
    lock_path = (
        args.output_dir
        / "protocol/training_configuration_candidate_lock.json"
    )
    write_json(report_path, report)
    write_json(
        lock_path,
        {
            "status": report["status"],
            "manifest_frozen_lock_sha256": report["manifest_freeze"][
                "frozen_lock_sha256"
            ],
            "configuration_sha256": report["configuration_sha256"],
            "model_snapshot": report["model_snapshot"],
            "runtime_observed": report["runtime_observed"],
            "runtime_checks": report["runtime_checks"],
            "data_and_step_plan": report["data_and_step_plan"],
            "training_source_hashes": source_hashes,
            "audit_source_hashes": audit_source_hashes,
            "candidate_report_sha256": file_sha256(report_path),
            "external_reviewed_lock_required_by_entrypoint": True,
            "smoke_training_allowed": False,
            "formal_training_allowed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "training_used": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=DEFAULT_FROZEN_LOCK,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
