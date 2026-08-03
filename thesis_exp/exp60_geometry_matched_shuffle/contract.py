"""Preflight and scientific-contract hashing for Exp60."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from thesis_exp.exp60_geometry_matched_shuffle import (
    PREFLIGHT_SOURCE_LOCK_PATH,
    PROTOCOL_PATH,
    REPO_ROOT,
)


ALLOWED_POST_PREFLIGHT_PROTOCOL_CHANGES = (
    "status",
    "formal_runs.physical_gpu_bindings",
    "formal_freeze_timestamp",
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
