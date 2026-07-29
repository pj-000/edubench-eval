"""Validate or execute one explicitly authorized train-only SORC-DPO smoke step.

The default and currently usable mode is ``--validate-only``. GPU/model access
requires ``--execute`` plus a separately reviewed authorization file that
binds the exact runner, frozen package, arm, seed, output directory, and
one-step budget. No such authorization is produced by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path
from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
    field_dpo_per_pair,
    field_mean_logps,
    weighted_objective,
)


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
SMOKE_ROOT = RAR_ROOT / "preference_smoke"
DEFAULT_SMOKE_LOCK = SMOKE_ROOT / "smoke_package_lock.json"
DEFAULT_SMOKE_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/sorc_dpo_smoke_plan_v1.json"
)
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_training_candidate_v1.json"
)
DEFAULT_CHECKPOINT_LOCK = (
    RAR_ROOT / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
)
DEFAULT_BASE_TRAINING_CONFIGURATION = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json"
)
DEFAULT_FROZEN_TRAINING_LOCK = (
    RAR_ROOT
    / "preference_training_candidate/preference_training_frozen_lock.json"
)
DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK = (
    SMOKE_ROOT / "smoke_implementation_candidate_lock.json"
)
DEFAULT_QUALIFICATION_CANDIDATE_LOCK = (
    RAR_ROOT
    / "preference_pairs/rationale_qualification/candidate_lock.json"
)
DEFAULT_AUTHORIZATION = SMOKE_ROOT / "gpu_smoke_execution_authorization.json"
DEFAULT_FORMAL_RUN_ROOT = RAR_ROOT / "formal_runs"
FINAL_QUALIFICATION_SCHEMA_VERSION = (
    "exp54-sorc-rationale-qualification-v1-final-lock"
)
FINAL_QUALIFICATION_STATUS = (
    "SORC_RATIONALE_QUALIFICATION_COMPLETE_P3_ALLOWED"
)
MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES = 40 * 1024**3
ARMS = (
    "P1_FIELD_DPO",
    "P2_SORC_SCORE",
    "P3_JOINT_SORC",
    "P1_SYN_SEED42",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_regular_non_symlink(path: Path, *, label: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise PermissionError(f"{label} cannot be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{label} must be a regular file")


def _private_subset_path(
    arm: str,
    *,
    private_subset_dir: Path,
) -> Path:
    return private_subset_dir / f"{arm.lower()}.jsonl"


def load_and_validate_smoke_rows(
    *,
    arm: str,
    smoke_lock_path: Path,
    smoke_plan_path: Path,
    private_subset_dir: Path = SMOKE_ROOT / "private",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown smoke arm: {arm}")
    for path in (smoke_lock_path, smoke_plan_path):
        reject_eval_path(path)
        _require_regular_non_symlink(path, label=path.name)
    smoke_lock = read_json(smoke_lock_path)
    smoke_plan = read_json(smoke_plan_path)
    if (
        smoke_lock.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_NOT_FROZEN_EXECUTION_FORBIDDEN"
        or smoke_lock.get("gpu_smoke_allowed") is not False
        or smoke_plan["execution_contract"]["gpu_smoke_allowed"] is not False
    ):
        raise PermissionError("smoke package is not fail-closed")
    if sha256_file(smoke_plan_path) != str(
        smoke_lock["source_hashes"]["smoke_plan"]
    ):
        raise ValueError("smoke plan hash differs")
    subset_path = _private_subset_path(
        arm,
        private_subset_dir=private_subset_dir,
    )
    reject_eval_path(subset_path)
    _require_regular_non_symlink(subset_path, label=f"{arm} subset")
    if sha256_file(subset_path) != str(
        smoke_lock["private_subset_hashes"][arm]
    ):
        raise ValueError(f"{arm}: private smoke subset hash differs")
    rows = read_jsonl(subset_path)
    arm_plan = smoke_plan["arms"][arm]
    if len(rows) != int(arm_plan["pair_count"]):
        raise ValueError(f"{arm}: smoke pair count differs")
    if int(arm_plan["optimizer_steps"]) != 1:
        raise ValueError(f"{arm}: smoke must contain exactly one optimizer step")
    if int(arm_plan["accumulation_group_pair_count"]) != len(rows):
        raise ValueError(f"{arm}: accumulation-group count differs")
    if any(int(row["cutoff_len"]) != 2048 for row in rows):
        raise ValueError(f"{arm}: fixed padding cutoff differs")
    collator = SORCDPOPairCollator(pad_token_id=151643, cutoff_len=2048)
    for row in rows:
        collator([row])
    return rows, smoke_lock, smoke_plan


def verify_gpu_execution_authorization(
    *,
    path: Path,
    arm: str,
    smoke_lock_path: Path,
    smoke_plan_path: Path,
    training_config_path: Path,
    checkpoint_lock_path: Path,
    base_training_configuration_path: Path,
    frozen_training_lock_path: Path,
    implementation_candidate_lock_path: Path,
    qualification_candidate_lock_path: Path = (
        DEFAULT_QUALIFICATION_CANDIDATE_LOCK
    ),
) -> dict[str, Any]:
    """Verify an external exact authorization before any torch/CUDA/model use."""
    reject_eval_path(path)
    _require_regular_non_symlink(path, label="GPU smoke authorization")
    bound_files = {
        "smoke lock": smoke_lock_path,
        "smoke plan": smoke_plan_path,
        "preference training configuration": training_config_path,
        "checkpoint selection lock": checkpoint_lock_path,
        "base training configuration": base_training_configuration_path,
        "frozen preference-training lock": frozen_training_lock_path,
        "smoke implementation candidate lock": (
            implementation_candidate_lock_path
        ),
    }
    for label, bound_path in bound_files.items():
        reject_eval_path(bound_path)
        _require_regular_non_symlink(bound_path, label=label)
    authorization = read_json(path)
    if authorization.get("status") != "SORC_DPO_GPU_SMOKE_AUTHORIZED":
        raise PermissionError("GPU smoke authorization status is invalid")
    if authorization.get("gpu_smoke_allowed") is not True:
        raise PermissionError("GPU smoke is not authorized")
    if authorization.get("formal_preference_training_allowed") is not False:
        raise PermissionError("authorization improperly enables formal training")
    if authorization.get("dev_accessed") is not False or authorization.get(
        "test_accessed"
    ) is not False:
        raise PermissionError("authorization evaluation boundary differs")
    base_configuration = read_json(base_training_configuration_path)
    if not isinstance(base_configuration.get("model"), dict):
        raise ValueError("base training configuration lacks model identity")
    frozen_training = read_json(frozen_training_lock_path)
    implementation = read_json(implementation_candidate_lock_path)
    if frozen_training.get("status") != (
        "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
    ):
        raise PermissionError("frozen preference-training lock differs")
    if implementation.get("status") != (
        "SORC_DPO_SMOKE_IMPLEMENTATION_CANDIDATE_GPU_EXECUTION_FORBIDDEN"
    ):
        raise PermissionError("smoke implementation lock differs")
    exact = {
        "arm": arm,
        "seed": 42,
        "optimizer_steps": 1,
        "physical_micro_batch_pairs": 1,
        "smoke_lock_sha256": sha256_file(smoke_lock_path),
        "smoke_plan_sha256": sha256_file(smoke_plan_path),
        "training_config_sha256": sha256_file(training_config_path),
        "checkpoint_lock_sha256": sha256_file(checkpoint_lock_path),
        "base_training_configuration_sha256": sha256_file(
            base_training_configuration_path
        ),
        "base_model_snapshot_identity_sha256": _canonical_sha256(
            base_configuration["model"]
        ),
        "preference_training_frozen_lock_sha256": sha256_file(
            frozen_training_lock_path
        ),
        "smoke_implementation_candidate_lock_sha256": sha256_file(
            implementation_candidate_lock_path
        ),
        "runner_sha256": sha256_file(Path(__file__)),
        "cuda_device_name": "NVIDIA RTX A6000",
    }
    for field, expected in exact.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"GPU smoke authorization {field} differs")
    output_dir = Path(str(authorization.get("output_dir") or ""))
    if not output_dir.is_absolute():
        raise PermissionError("authorized output directory must be absolute")
    normalized_output = output_dir.resolve(strict=False)
    allowed_parent = (SMOKE_ROOT / "runs").resolve(strict=False)
    if normalized_output.parent != allowed_parent:
        raise PermissionError("authorized output directory is outside smoke root")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cuda_device_uuid = authorization.get("cuda_device_uuid")
    if (
        not isinstance(cuda_device_uuid, str)
        or not cuda_device_uuid.startswith("GPU-")
        or len(cuda_device_uuid) < 8
    ):
        raise PermissionError("authorized CUDA device UUID is invalid")
    minimum_free = authorization.get("minimum_free_memory_bytes_before_load")
    if (
        isinstance(minimum_free, bool)
        or not isinstance(minimum_free, int)
        or minimum_free < MINIMUM_AUTHORIZED_FREE_MEMORY_BYTES
    ):
        raise PermissionError("authorized free-memory floor is too small")
    if arm == "P3_JOINT_SORC":
        _require_regular_non_symlink(
            qualification_candidate_lock_path,
            label="rationale qualification candidate lock",
        )
        if authorization.get(
            "rationale_qualification_candidate_lock_sha256"
        ) != sha256_file(qualification_candidate_lock_path):
            raise PermissionError(
                "rationale qualification candidate binding differs"
            )
        qualification_path = Path(
            str(authorization.get("rationale_qualification_lock_path") or "")
        )
        expected_hash = str(
            authorization.get("rationale_qualification_lock_sha256") or ""
        )
        if not qualification_path.is_absolute() or len(expected_hash) != 64:
            raise PermissionError("P3 lacks rationale qualification binding")
        _require_regular_non_symlink(
            qualification_path,
            label="rationale qualification lock",
        )
        if sha256_file(qualification_path) != expected_hash:
            raise PermissionError("rationale qualification lock differs")
        verify_p3_qualification_lock(
            qualification_path=qualification_path,
            qualification_candidate_lock_path=(
                qualification_candidate_lock_path
            ),
        )
    return authorization


def verify_p3_qualification_lock(
    *,
    qualification_path: Path,
    qualification_candidate_lock_path: Path,
) -> dict[str, Any]:
    """Accept only the final reviewed qualification contract for P3."""
    qualification = read_json(qualification_path)
    candidate = read_json(qualification_candidate_lock_path)
    exact = {
        "schema_version": FINAL_QUALIFICATION_SCHEMA_VERSION,
        "status": FINAL_QUALIFICATION_STATUS,
        "rationale_blind_qualification_completed": True,
        "p3_preference_training_allowed": True,
        "evaluator_family_qualification_completed": True,
        "evaluator_family_count": 2,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for field, expected in exact.items():
        if qualification.get(field) != expected:
            raise PermissionError(
                f"rationale qualification {field} differs"
            )
    if "p3_training_allowed" in qualification:
        raise PermissionError("deprecated P3 qualification alias is forbidden")
    if qualification.get("source_hashes") != candidate.get("source_hashes"):
        raise PermissionError("rationale qualification source hashes differ")
    family_results = qualification.get("evaluator_family_results")
    if not isinstance(family_results, dict) or len(family_results) != 2:
        raise PermissionError(
            "rationale qualification must contain two evaluator families"
        )
    for family, result in family_results.items():
        if not isinstance(family, str) or not family:
            raise PermissionError("evaluator-family identity is invalid")
        if (
            not isinstance(result, dict)
            or result.get("qualification_completed") is not True
        ):
            raise PermissionError(
                f"evaluator-family qualification incomplete: {family}"
            )
        counts = [
            result.get("r3_wins"),
            result.get("r2_wins"),
            result.get("ties"),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise PermissionError(
                f"evaluator-family counts are invalid: {family}"
            )
        if int(result["r3_wins"]) <= int(result["r2_wins"]):
            raise PermissionError(
                f"evaluator-family does not qualify P3: {family}"
            )
    return qualification


def _seed42_checkpoint(
    checkpoint_lock_path: Path,
    *,
    formal_runs_root: Path = DEFAULT_FORMAL_RUN_ROOT,
) -> tuple[Path, dict[str, str]]:
    _require_regular_non_symlink(
        checkpoint_lock_path,
        label="checkpoint selection lock",
    )
    lock = read_json(checkpoint_lock_path)
    if (
        lock.get("status") != "SFT_DEV_CHECKPOINT_SELECTION_FROZEN"
        or lock.get("test_accessed") is not False
    ):
        raise ValueError("SFT checkpoint selection lock differs")
    selected = [
        value
        for value in lock["checkpoint_bindings"]
        if str(value["arm"]).upper() == "R3" and int(value["seed"]) == 42
    ]
    if len(selected) != 1 or int(selected[0]["selected_epoch"]) != 3:
        raise ValueError("seed42 R3 epoch-3 checkpoint binding differs")
    item = selected[0]
    relative = Path(str(item["checkpoint_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("checkpoint path escapes formal-runs root")
    resolved_root = formal_runs_root.resolve(strict=True)
    adapter_dir = (resolved_root / relative / "adapter").resolve(strict=True)
    if not adapter_dir.is_relative_to(resolved_root):
        raise PermissionError("checkpoint path escapes formal-runs root")
    hashes = {
        name: str(metadata["sha256"])
        for name, metadata in item["adapter_files"].items()
    }
    for name, expected in hashes.items():
        path = adapter_dir / name
        _require_regular_non_symlink(path, label=f"R3 {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"R3 checkpoint differs: {name}")
    return adapter_dir, hashes


def verify_base_model_snapshot(
    base_training_configuration_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Recheck the frozen base-model directory before importing torch."""
    _require_regular_non_symlink(
        base_training_configuration_path,
        label="base training configuration",
    )
    configuration = read_json(base_training_configuration_path)
    model = configuration.get("model")
    if not isinstance(model, dict):
        raise ValueError("base training configuration lacks model identity")
    base_path = Path(str(model.get("local_path") or ""))
    root_metadata = base_path.lstat()
    if stat.S_ISLNK(root_metadata.st_mode):
        raise PermissionError("base model root cannot be a symlink")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PermissionError("base model root must be a directory")
    expected_listing = model.get("directory_regular_files")
    if not isinstance(expected_listing, dict):
        raise ValueError("base model directory listing is missing")
    actual_listing: dict[str, int] = {}
    for path in sorted(base_path.rglob("*")):
        relative = path.relative_to(base_path).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"base model descendant symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode):
            actual_listing[relative] = int(metadata.st_size)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(
                f"base model descendant is not regular: {relative}"
            )
    if actual_listing != expected_listing:
        raise ValueError("base model directory listing differs")
    listing_payload = json.dumps(
        sorted(actual_listing.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(listing_payload).hexdigest() != str(
        model.get("directory_listing_sha256")
    ):
        raise ValueError("base model directory-listing hash differs")
    expected_files = model.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("base model file lock is missing")
    for name, expected in expected_files.items():
        path = base_path / str(name)
        _require_regular_non_symlink(path, label=f"base model {name}")
        if (
            path.stat().st_size != int(expected["size"])
            or sha256_file(path) != str(expected["sha256"])
        ):
            raise ValueError(f"base model file differs: {name}")
    return base_path, configuration


def verify_authorized_cuda_device(
    *,
    torch_module: Any,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    """Verify the single visible CUDA device before model loading."""
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("GPU smoke requires exactly one visible CUDA device")
    name = str(cuda.get_device_name(0))
    properties = cuda.get_device_properties(0)
    property_uuid = getattr(properties, "uuid", None)
    if property_uuid is not None and str(property_uuid).startswith("GPU-"):
        uuid = str(property_uuid)
    else:
        uuids = cuda._raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("cannot resolve visible CUDA device UUID")
        physical_index = int(cuda._get_nvml_device_index(0))
        if physical_index < 0 or physical_index >= len(uuids):
            raise RuntimeError("visible CUDA device UUID index is invalid")
        uuid = str(uuids[physical_index])
    free_bytes, total_bytes = cuda.mem_get_info(0)
    if name != authorization["cuda_device_name"]:
        raise RuntimeError("visible CUDA device name is not authorized")
    if uuid != authorization["cuda_device_uuid"]:
        raise RuntimeError("visible CUDA device UUID is not authorized")
    if int(free_bytes) < int(
        authorization["minimum_free_memory_bytes_before_load"]
    ):
        raise RuntimeError("authorized CUDA device is not sufficiently idle")
    if int(authorization["physical_micro_batch_pairs"]) != 1:
        raise RuntimeError("correctness smoke physical batch must equal one")
    return {
        "name": name,
        "uuid": uuid,
        "free_memory_bytes_before_load": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def _set_adapter_trainability(model: Any, adapter_name: str) -> None:
    model.set_adapter(adapter_name)
    trainable = 0
    for name, parameter in model.named_parameters():
        is_selected_lora = (
            "lora_" in name
            and (
                f".{adapter_name}." in name
                or name.endswith(f".{adapter_name}")
            )
        )
        parameter.requires_grad_(is_selected_lora)
        if is_selected_lora:
            trainable += parameter.numel()
    if trainable != 33030144:
        raise ValueError(
            f"policy trainable parameter count differs: {trainable}"
        )


def _pair_logps(model: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    import torch

    input_ids = torch.cat(
        [batch["chosen_input_ids"], batch["rejected_input_ids"]],
        dim=0,
    ).to(model.device)
    attention_mask = torch.cat(
        [
            batch["chosen_attention_mask"],
            batch["rejected_attention_mask"],
        ],
        dim=0,
    ).to(model.device)
    field_mask = torch.cat(
        [batch["chosen_field_mask"], batch["rejected_field_mask"]],
        dim=0,
    ).to(model.device)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    logps = field_mean_logps(logits, input_ids, field_mask)[
        "per_sequence_logp"
    ]
    if logps.shape != (2,):
        raise ValueError("smoke micro-batch must contain one chosen/rejected pair")
    return logps[0:1], logps[1:2]


def execute_one_smoke_step(
    *,
    arm: str,
    rows: list[dict[str, Any]],
    authorization: dict[str, Any],
    training_config_path: Path,
    checkpoint_lock_path: Path,
    base_training_configuration_path: Path,
) -> dict[str, Any]:
    """Run one optimizer step. Caller must verify authorization first."""
    base_path, _base_configuration = verify_base_model_snapshot(
        base_training_configuration_path
    )
    adapter_dir, adapter_hashes = _seed42_checkpoint(checkpoint_lock_path)

    import torch

    device_identity = verify_authorized_cuda_device(
        torch_module=torch,
        authorization=authorization,
    )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU smoke requires BF16 support")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    config = read_json(training_config_path)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=True,
    )
    model.load_adapter(
        str(adapter_dir),
        adapter_name="reference",
        is_trainable=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    collator = SORCDPOPairCollator(pad_token_id=151643, cutoff_len=2048)

    reference_logps = []
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for row in rows:
            batch = collator([row])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected = _pair_logps(model, batch)
            reference_logps.append(
                (chosen.detach().cpu(), rejected.detach().cpu())
            )

    _set_adapter_trainability(model, "default")
    model.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["optimization"]["learning_rate"]),
        betas=tuple(config["optimization"]["betas"]),
        eps=float(config["optimization"]["epsilon"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    beta = float(config["loss"]["beta"])
    for index, row in enumerate(rows):
        batch = collator([row])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy_chosen, policy_rejected = _pair_logps(model, batch)
            reference_chosen = reference_logps[index][0].to(model.device)
            reference_rejected = reference_logps[index][1].to(model.device)
            per_pair = field_dpo_per_pair(
                policy_chosen_logps=policy_chosen,
                policy_rejected_logps=policy_rejected,
                reference_chosen_logps=reference_chosen,
                reference_rejected_logps=reference_rejected,
                beta=beta,
                offsets=batch["odpo_offset"].to(model.device),
            )["per_pair_loss"]
            objective = weighted_objective(
                per_pair,
                batch["objective_weight"].to(model.device),
                normalization_pair_count=len(rows),
            )["loss"]
        if not torch.isfinite(objective):
            raise FloatingPointError("smoke loss is non-finite")
        objective.backward()
        total_loss += float(objective.detach().cpu())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        max_norm=float(config["optimization"]["max_grad_norm"]),
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
        raise FloatingPointError("smoke gradient norm is invalid")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(str(authorization["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter_output = output_dir / "adapter"
    model.set_adapter("default")
    model.save_pretrained(
        str(adapter_output),
        selected_adapters=["default"],
        safe_serialization=True,
    )
    result = {
        "schema_version": "exp54-sorc-dpo-gpu-smoke-result-v1",
        "status": "SORC_DPO_GPU_SMOKE_STEP_COMPLETE_REQUIRES_RESULT_AUDIT",
        "arm": arm,
        "seed": 42,
        "pair_count": len(rows),
        "optimizer_steps": 1,
        "loss": total_loss,
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_device_identity": device_identity,
        "base_model_path": str(base_path),
        "r3_adapter_hashes": adapter_hashes,
        "output_adapter_config_sha256": sha256_file(
            adapter_output / "adapter_config.json"
        ),
        "output_adapter_model_sha256": sha256_file(
            adapter_output / "adapter_model.safetensors"
        ),
        "dev_accessed": False,
        "test_accessed": False,
    }
    result_path = output_dir / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-lock", type=Path, default=DEFAULT_SMOKE_LOCK)
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        default=DEFAULT_AUTHORIZATION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, smoke_lock, smoke_plan = load_and_validate_smoke_rows(
        arm=args.arm,
        smoke_lock_path=args.smoke_lock,
        smoke_plan_path=args.smoke_plan,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SORC_DPO_SMOKE_CPU_VALIDATION_PASS",
                    "arm": args.arm,
                    "pairs": len(rows),
                    "smoke_lock_sha256": sha256_file(args.smoke_lock),
                    "smoke_plan_sha256": sha256_file(args.smoke_plan),
                    "model_loaded": False,
                    "cuda_initialized": False,
                    "forward_backward_executed": False,
                    "dev_accessed": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    authorization = verify_gpu_execution_authorization(
        path=args.authorization,
        arm=args.arm,
        smoke_lock_path=args.smoke_lock,
        smoke_plan_path=args.smoke_plan,
        training_config_path=args.training_config,
        checkpoint_lock_path=DEFAULT_CHECKPOINT_LOCK,
        base_training_configuration_path=(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
        frozen_training_lock_path=DEFAULT_FROZEN_TRAINING_LOCK,
        implementation_candidate_lock_path=(
            DEFAULT_IMPLEMENTATION_CANDIDATE_LOCK
        ),
    )
    result = execute_one_smoke_step(
        arm=args.arm,
        rows=rows,
        authorization=authorization,
        training_config_path=args.training_config,
        checkpoint_lock_path=DEFAULT_CHECKPOINT_LOCK,
        base_training_configuration_path=(
            DEFAULT_BASE_TRAINING_CONFIGURATION
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
