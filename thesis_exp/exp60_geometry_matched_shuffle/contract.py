"""Preflight and scientific-contract hashing for Exp60."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp60_geometry_matched_shuffle import (
    PREFLIGHT_SOURCE_LOCK_PATH,
    PROTOCOL_PATH,
    REAL_PREFLIGHT_DECISION_PATH,
    REPO_ROOT,
)


ALLOWED_POST_PREFLIGHT_PROTOCOL_CHANGES = (
    "status",
    "formal_runs.physical_gpu_bindings",
    "formal_freeze_timestamp",
)

FORMAL_SOURCE_LOCK_SCHEMA_VERSION = 1
FORMAL_MANDATORY_FILES = (
    "thesis_exp/configs/exp60_geometry_matched_shuffle/protocol_draft.json",
    "thesis_exp/exp60_geometry_matched_shuffle/train.py",
    "thesis_exp/exp60_geometry_matched_shuffle/analyze_confirmation.py",
    "thesis_exp/exp60_geometry_matched_shuffle/real_model_preflight.py",
    "thesis_exp/exp60_geometry_matched_shuffle/finalize_real_preflight.py",
    "thesis_exp/outputs/exp60_geometry_matched_shuffle/audit/max_mismatch_mapping.jsonl",
    "thesis_exp/outputs/exp60_geometry_matched_shuffle/decision/real_model_preflight_decision.json",
    "thesis_exp/configs/exp60_geometry_matched_shuffle/preflight_source_lock.json",
    "thesis_exp/outputs/exp60_geometry_matched_shuffle/real_model_preflight/seed_47/real_model_no_update_preflight.json",
    "thesis_exp/outputs/exp60_geometry_matched_shuffle/real_model_preflight/seed_48/real_model_no_update_preflight.json",
    "thesis_exp/outputs/exp60_geometry_matched_shuffle/real_model_preflight/seed_49/real_model_no_update_preflight.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_scientific_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(protocol)
    normalized.pop("status", None)
    normalized.pop("formal_freeze_timestamp", None)
    normalized["formal_runs"].pop("physical_gpu_bindings", None)
    return normalized


def normalized_scientific_protocol_sha256(protocol: dict[str, Any]) -> str:
    payload = json.dumps(
        normalized_scientific_protocol(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_finite_scalar(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"EXP60_NONFINITE_SCALAR:{name}")
    return numeric


def manifest_sha256(paths: list[str] | tuple[str, ...]) -> str:
    payload = "".join(f"{path}\n" for path in sorted(paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_gpu_uuid(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) == 16:
            raw = value.hex()
        else:
            try:
                raw = value.decode("ascii")
            except (UnicodeDecodeError, ValueError):
                raw = value.hex()
    else:
        raw = str(value).strip()
    if not raw or raw.upper() in {"N/A", "NONE"}:
        return None
    lowered = raw.lower()
    if lowered.startswith("mig-"):
        raise RuntimeError("EXP60_MIG_ENVIRONMENT_NOT_AUTHORIZED")
    if lowered.startswith("gpu-"):
        raw = raw[4:]
    normalized = "".join(character.lower() for character in raw if character.isalnum())
    return normalized or None


def stable_gpu_identity(torch: Any, device: Any) -> dict[str, Any]:
    """Resolve one physical GPU using stable hardware identity or fail closed."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise RuntimeError(
            "Exp60 requires exactly one explicit CUDA_VISIBLE_DEVICES identifier"
        )
    properties = torch.cuda.get_device_properties(device)
    torch_uuid_raw = getattr(properties, "uuid", None)
    torch_uuid = _normalize_gpu_uuid(torch_uuid_raw)
    smi_uuid_raw = pci_bus_id = mig_mode = None
    try:
        command = subprocess.run(
            [
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=uuid,pci.bus_id,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        command = None
    if command is not None and command.returncode == 0 and command.stdout.strip():
        parts = [part.strip() for part in command.stdout.strip().splitlines()[0].split(",")]
        if len(parts) == 3:
            smi_uuid_raw, pci_bus_id, mig_mode = parts
            if not pci_bus_id or pci_bus_id.upper() == "N/A":
                raise RuntimeError("EXP60_STABLE_GPU_PCI_BUS_ID_UNAVAILABLE")
    smi_uuid = _normalize_gpu_uuid(smi_uuid_raw)
    if mig_mode and mig_mode.strip().lower() == "enabled":
        raise RuntimeError("EXP60_MIG_ENVIRONMENT_NOT_AUTHORIZED")
    if torch_uuid and smi_uuid and torch_uuid != smi_uuid:
        raise RuntimeError("EXP60_GPU_UUID_SOURCES_DISAGREE")
    stable_uuid = torch_uuid or smi_uuid
    if not stable_uuid:
        raise RuntimeError("EXP60_STABLE_GPU_IDENTITY_UNAVAILABLE")
    return {
        "cuda_visible_devices": visible,
        "identity": f"GPU-{stable_uuid}",
        "stable_gpu_uuid": f"GPU-{stable_uuid}",
        "torch_uuid": str(torch_uuid_raw) if torch_uuid_raw else None,
        "nvidia_smi_uuid": smi_uuid_raw,
        "pci_bus_id": pci_bus_id,
        "mig_mode": mig_mode,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def validate_formal_source_lock(
    lock: dict[str, Any], protocol: dict[str, Any], decision: dict[str, Any]
) -> dict[str, str]:
    """Validate the final lock schema and all internal bindings before hashing files."""

    if lock.get("schema_version") != FORMAL_SOURCE_LOCK_SCHEMA_VERSION:
        raise RuntimeError("Invalid Exp60 formal source-lock schema")
    if lock.get("status") != "EXP60_FORMAL_SOURCE_LOCK":
        raise RuntimeError("Invalid Exp60 formal source-lock status")
    expected_protocol_sha = sha256_file(PROTOCOL_PATH)
    expected_decision_sha = sha256_file(REAL_PREFLIGHT_DECISION_PATH)
    expected_normalized_sha = normalized_scientific_protocol_sha256(protocol)
    required_scalars = {
        "protocol_sha256": expected_protocol_sha,
        "real_model_preflight_decision_sha256": expected_decision_sha,
        "normalized_scientific_protocol_sha256": expected_normalized_sha,
        "mandatory_file_manifest_sha256": manifest_sha256(FORMAL_MANDATORY_FILES),
    }
    for field, expected in required_scalars.items():
        if lock.get(field) != expected:
            raise RuntimeError(f"Invalid Exp60 formal source-lock binding: {field}")
    if lock.get("contains_frozen_analysis") is not True:
        raise RuntimeError("Exp60 formal source lock does not contain frozen analysis")
    if lock.get("physical_gpu_bindings_equal_preflight_devices") is not True:
        raise RuntimeError("Exp60 formal source lock lacks GPU/preflight binding")
    if lock.get("allowed_splits") != ["train", "dev"] or lock.get("test_access_count") != 0:
        raise RuntimeError("Invalid Exp60 formal source-lock split contract")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Exp60 formal source-lock file manifest is empty")
    missing = sorted(set(FORMAL_MANDATORY_FILES) - set(files))
    if missing:
        raise RuntimeError("Exp60 formal source lock misses mandatory files: " + ", ".join(missing))
    if int(lock.get("file_count", -1)) != len(files):
        raise RuntimeError("Exp60 formal source-lock file count mismatch")
    if decision.get("status") != "EXP60_REAL_MODEL_PREFLIGHT_ALL_SEEDS_PASS":
        raise RuntimeError("Exp60 preflight decision is not PASS")
    decision_binding = decision.get("preflight_contract_binding", {})
    if decision_binding.get("normalized_scientific_protocol_sha256") != expected_normalized_sha:
        raise RuntimeError("Exp60 decision/protocol scientific binding mismatch")
    gpu_bindings = decision.get("preflight_gpu_bindings")
    if not isinstance(gpu_bindings, dict) or set(gpu_bindings) != {
        "gpu_slot_0",
        "gpu_slot_1",
        "gpu_slot_2",
    }:
        raise RuntimeError("Exp60 decision lacks three GPU bindings")
    stable_uuids = [
        str(binding.get("stable_gpu_uuid") or "") for binding in gpu_bindings.values()
    ]
    if "" in stable_uuids or len(set(stable_uuids)) != 3:
        raise RuntimeError("Exp60 decision lacks three distinct stable GPU UUIDs")
    expected_visible_bindings = {
        slot: str(binding.get("cuda_visible_devices"))
        for slot, binding in gpu_bindings.items()
    }
    if protocol.get("formal_runs", {}).get("physical_gpu_bindings") != expected_visible_bindings:
        raise RuntimeError("Exp60 formal GPU bindings differ from preflight decision")
    return required_scalars


def verify_preflight_source_lock() -> dict[str, Any]:
    if not PREFLIGHT_SOURCE_LOCK_PATH.is_file():
        raise FileNotFoundError(PREFLIGHT_SOURCE_LOCK_PATH)
    lock = json.loads(PREFLIGHT_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "EXP60_PREFLIGHT_SOURCE_LOCK_NO_TRAINING_AUTHORITY":
        raise RuntimeError("Invalid Exp60 preflight source-lock status")
    for relative, expected in lock["files"].items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Exp60 preflight source-lock mismatch: {relative}: {actual} != {expected}"
            )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    normalized_sha = normalized_scientific_protocol_sha256(protocol)
    if normalized_sha != lock["normalized_scientific_protocol_sha256"]:
        raise RuntimeError("Exp60 normalized scientific protocol mismatch")
    return {
        "preflight_source_lock_sha256": sha256_file(PREFLIGHT_SOURCE_LOCK_PATH),
        "normalized_scientific_protocol_sha256": normalized_sha,
    }
