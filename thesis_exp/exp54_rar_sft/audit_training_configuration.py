"""Audit the complete Exp54 training configuration without starting training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
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
    require_adapter_free_base_model,
    validate_model_directory_listing,
    validate_training_configuration,
    verify_model_snapshot,
)
from thesis_exp.exp54_rar_sft.authorization_guard import (
    RUNTIME_SOURCE_PATHS,
    TRUSTED_AUTHORIZATION_DIGEST_PATH,
    closure_sha256,
    runtime_source_closure,
    sha256_file,
    verify_external_authorization,
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


def validate_tensor_shard_mapping(
    *,
    weight_map: dict[str, str],
    tensor_names_by_shard: dict[str, set[str]],
    expected_shards: set[str],
) -> dict[str, Any]:
    """Prove the index mapping equals each physical shard header."""
    if set(tensor_names_by_shard) != expected_shards:
        raise ValueError("physical safetensor shard set differs")
    indexed_shards = set(weight_map.values())
    if indexed_shards != expected_shards:
        raise ValueError("weight index references unexpected shards")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for names in tensor_names_by_shard.values():
        duplicates.update(seen & names)
        seen.update(names)
    if duplicates:
        raise ValueError("tensor names are duplicated across shards")
    unindexed = seen - set(weight_map)
    missing_indexed = set(weight_map) - seen
    if unindexed:
        raise ValueError("physical shards contain unindexed tensors")
    if missing_indexed:
        raise ValueError("weight index contains physically missing tensors")
    for shard_name, actual_names in tensor_names_by_shard.items():
        indexed_names = {
            tensor_name
            for tensor_name, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        }
        if actual_names != indexed_names:
            raise ValueError(
                f"per-tensor shard mapping differs for {shard_name}"
            )
    return {
        "per_tensor_shard_mapping_match": True,
        "duplicate_tensor_names_across_shards": 0,
        "unindexed_tensor_count": 0,
        "missing_indexed_tensor_count": 0,
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
    tensor_names_by_shard: dict[str, set[str]] = {}
    for shard_name in sorted(expected_shards):
        with safe_open(
            model_path / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            shard_names = set(shard.keys())
            tensor_names_by_shard[shard_name] = shard_names
            for tensor_name in shard_names:
                if tensor_name in shapes:
                    raise ValueError(
                        "tensor names are duplicated across safetensor shards"
                    )
                shapes[tensor_name] = tuple(
                    shard.get_slice(tensor_name).get_shape()
                )
    shard_mapping = validate_tensor_shard_mapping(
        weight_map=weight_map,
        tensor_names_by_shard=tensor_names_by_shard,
        expected_shards=expected_shards,
    )
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
        **shard_mapping,
        "all_lora_targets_present_in_every_layer": True,
    }


def audit_generation_cap(
    *,
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    lengths_by_arm: dict[str, list[int]] = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            path = (
                output_dir
                / "data"
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            rows = read_jsonl(path, protect_split=True)
            if len(rows) != 7_962:
                raise ValueError("generation-cap manifest row count differs")
            lengths_by_arm[arm].extend(
                len(row["target_token_ids"])
                + len(row["assistant_suffix_token_ids"])
                for row in rows
            )
    maximum: dict[str, int] = {}
    p95: dict[str, int] = {}
    exceeding: dict[str, int] = {}
    for arm, lengths in lengths_by_arm.items():
        ordered = sorted(lengths)
        maximum[arm] = ordered[-1]
        p95[arm] = ordered[math.ceil(0.95 * len(ordered)) - 1]
        exceeding[arm] = sum(length > max_new_tokens for length in lengths)
    covers = not any(exceeding.values())
    if not covers:
        raise ValueError("generation cap does not cover frozen train targets")
    return {
        "max_new_tokens": max_new_tokens,
        "max_materialized_assistant_tokens_by_arm": maximum,
        "p95_materialized_assistant_tokens_by_arm": p95,
        "events_exceeding_max_new_tokens_by_arm": exceeding,
        "generation_cap_covers_all_frozen_train_targets": True,
    }


def audit_authorization_authenticity(
    *,
    config_path: Path,
    frozen_lock_path: Path,
    source_closure: dict[str, str],
) -> dict[str, Any]:
    """Exercise trusted and untrusted digest paths without model loading."""
    source_digest = closure_sha256(source_closure)
    with tempfile.TemporaryDirectory(prefix="exp54-auth-probe-") as directory:
        root = Path(directory)
        trusted_digest_path = root / "trusted-digest"
        try:
            configuration_lock_path = root / "configuration-lock.json"
            configuration_lock = {
                "status": (
                    "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED"
                ),
                "configuration_sha256": sha256_file(config_path),
                "manifest_frozen_lock_sha256": sha256_file(frozen_lock_path),
                "candidate_report_sha256": "a" * 64,
                "runtime_source_closure": source_closure,
                "runtime_source_closure_sha256": source_digest,
                "smoke_training_allowed": False,
                "formal_training_allowed": False,
                "dev_accessed": False,
                "test_accessed": False,
                "training_used": False,
            }
            configuration_lock_path.write_text(
                json.dumps(configuration_lock, sort_keys=True),
                encoding="utf-8",
            )
            authorization_path = root / "authorization.json"
            authorization = {
                "schema_version": "exp54-external-authorization-v1",
                "authorization_mode": "formal",
                "reviewed_training_configuration_commit": "b" * 40,
                "review_verdict": "TRAINING_CONFIGURATION_PASS",
                "configuration_sha256": sha256_file(config_path),
                "training_configuration_lock_sha256": sha256_file(
                    configuration_lock_path
                ),
                "candidate_report_sha256": "a" * 64,
                "frozen_manifest_lock_sha256": sha256_file(
                    frozen_lock_path
                ),
                "runtime_source_closure_sha256": source_digest,
                "allowed_arms": list(ARMS),
                "allowed_seeds": list(SEEDS),
                "sealed_data_state": {
                    "dev_accessed": False,
                    "test_accessed": False,
                    "training_used": False,
                },
            }
            authorization_path.write_text(
                json.dumps(authorization, sort_keys=True),
                encoding="utf-8",
            )
            trusted_digest_path.write_text("0" * 64 + "\n", encoding="ascii")
            trusted_digest_path.chmod(0o400)
            try:
                verify_external_authorization(
                    authorization_lock_path=authorization_path,
                    config_path=config_path,
                    frozen_lock_path=frozen_lock_path,
                    training_config_lock_path=configuration_lock_path,
                    arm="R3",
                    seed=42,
                    trusted_digest_path=trusted_digest_path,
                    require_root_owned_digest=False,
                )
            except PermissionError:
                forged_rejected = True
            else:
                raise AssertionError(
                    "self-consistent locks bypassed the external digest"
                )
            trusted_digest_path.chmod(0o600)
            trusted_digest_path.write_text(
                sha256_file(authorization_path) + "\n",
                encoding="ascii",
            )
            trusted_digest_path.chmod(0o400)
            accepted = verify_external_authorization(
                authorization_lock_path=authorization_path,
                config_path=config_path,
                frozen_lock_path=frozen_lock_path,
                training_config_lock_path=configuration_lock_path,
                arm="R3",
                seed=42,
                trusted_digest_path=trusted_digest_path,
                require_root_owned_digest=False,
            )
            if accepted != authorization:
                raise AssertionError("trusted authorization payload changed")
        finally:
            trusted_digest_path.unlink(missing_ok=True)
    return {
        "authorization_authenticity_verified": True,
        "forged_lock_pair_rejected": forged_rejected,
        "trusted_external_digest_acceptance_verified": True,
        "untrusted_arbitrary_lock_paths_rejected": True,
    }


def main() -> None:
    args = parse_args()
    config = _read_object(args.config)
    step_plan = validate_training_configuration(config)
    frozen = validate_frozen_inputs(args.frozen_lock, args.output_dir)
    model_hashes = verify_model_snapshot(config)
    model_directory = validate_model_directory_listing(
        Path(config["model"]["local_path"]),
        expected_regular_files=config["model"]["directory_regular_files"],
        expected_listing_sha256=config["model"][
            "directory_listing_sha256"
        ],
    )
    model_structure = audit_model_structure(config)
    observed_runtime = observe_runtime()
    runtime_checks = validate_runtime(observed_runtime, config["runtime"])
    source_hashes = runtime_source_closure()
    source_closure_sha256 = closure_sha256(source_hashes)
    authorization_probes = audit_authorization_authenticity(
        config_path=args.config,
        frozen_lock_path=args.frozen_lock,
        source_closure=source_hashes,
    )
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
    generation_cap = audit_generation_cap(
        output_dir=args.output_dir,
        max_new_tokens=generation_expected["max_new_tokens"],
    )

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
            **model_directory,
            "historical_adapter_artifacts_absent": True,
            "pre_lora_base_model_adapter_free": True,
            "pre_lora_runtime_guard": require_adapter_free_base_model.__name__,
            "pre_lora_peft_config_absent": True,
            "pre_lora_adapter_parameter_count": 0,
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
        "generation_train_target_coverage": generation_cap,
        "parser": config["parser"],
        "runtime_source_closure": source_hashes,
        "runtime_source_closure_sha256": source_closure_sha256,
        "runtime_source_closure_complete": True,
        "audit_source_hashes": audit_source_hashes,
        "authorization_gate": {
            "external_reviewed_lock_required_by_entrypoint": True,
            "authorization_authentication_method": (
                "root-owned repository-external exact SHA-256 file"
            ),
            "trusted_authorization_digest_path": str(
                TRUSTED_AUTHORIZATION_DIGEST_PATH
            ),
            "trusted_digest_root_owned_required": True,
            "trusted_digest_group_or_other_writable_allowed": False,
            "authorization_signature_required": False,
            "external_digest_required": True,
            **authorization_probes,
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
            "runtime_source_closure": source_hashes,
            "runtime_source_closure_sha256": source_closure_sha256,
            "runtime_source_closure_complete": True,
            "audit_source_hashes": audit_source_hashes,
            "generation_train_target_coverage": generation_cap,
            "authorization_authentication": report["authorization_gate"],
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
