"""Source-lock helpers and formal-authorization gates for Exp61."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation import EXP61_ROOT, FROZEN_PROTOCOL, ROOT


SOURCE_LOCK = EXP61_ROOT / "configs/source_lock.json"
LOCKED_RELATIVE_FILES = (
    "thesis_exp/exp61_soft_sts15_external_confirmation/__init__.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/configs/stage0_protocol.json",
    "thesis_exp/exp61_soft_sts15_external_confirmation/configs/protocol.json",
    "thesis_exp/exp61_soft_sts15_external_confirmation/data.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/method.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/mapping.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/geometry.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/model.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/runtime.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/metrics.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/preflight.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/train.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/analysis.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/contract.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/build_source_lock.py",
    "thesis_exp/exp61_soft_sts15_external_confirmation/finalize_preflight.py",
    "thesis_exp/exp60_geometry_matched_shuffle/geometry.py",
    "thesis_exp/exp60_geometry_matched_shuffle/train.py",
    "thesis_exp/exp59_residual_geometry/geometry.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/tests/test_exp61_soft_sts15.py",
    "thesis_exp/outputs/exp61_soft_sts15_external_confirmation/data/split_manifest.jsonl",
    "thesis_exp/outputs/exp61_soft_sts15_external_confirmation/audit/train_maximum_mismatch_mapping.jsonl",
    "thesis_exp/outputs/exp61_soft_sts15_external_confirmation/audit/train_maximum_mismatch_mapping_audit.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = str(item.relative_to(path))
        if relative == ".msc":
            continue
        value = sha256_file(item)
        files[relative] = value
        digest.update(f"{relative}\t{value}\n".encode("utf-8"))
    return {"path": str(path), "manifest_sha256": digest.hexdigest(), "files": files}


def build_source_lock(model_path: Path) -> dict[str, Any]:
    files = {relative: sha256_file(ROOT / relative) for relative in LOCKED_RELATIVE_FILES}
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    return {
        "status": "EXP61_SOURCE_LOCK_CANDIDATE_AWAITING_FINAL_PREFLIGHT_REVIEW",
        "files": files,
        "protocol_sha256": sha256_file(FROZEN_PROTOCOL),
        "model": directory_manifest(model_path),
        "train_dataset_contract_sha256": "e4abd30af2c17937c03c8d8726a9ac61d88af72490f79b91dcc7206a43596f31",
        "dev_dataset_contract_sha256": "df15133be9843a5b0ee4853c0dfec82e166ff1246d0b2b2324da3c8a5a0dc871",
        "mapping_semantic_sha256": "ab7bcaee6532e93b5babfd0fbbe69d18dd2945f94859dd26da2c736f43d3e9f0",
        "test_access_count": 0,
        "formal_training_authorized": bool(protocol["authorization"]["formal_nine_run_gpu_training"]),
    }


def verify_source_lock(*, require_formal_authorization: bool) -> dict[str, Any]:
    if not SOURCE_LOCK.is_file():
        raise FileNotFoundError(SOURCE_LOCK)
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Exp61 source-lock mismatch: {relative}")
    if sha256_file(FROZEN_PROTOCOL) != lock["protocol_sha256"]:
        raise RuntimeError("Exp61 protocol changed after source lock")
    if lock.get("test_access_count") != 0:
        raise RuntimeError("Exp61 source lock records test access")
    if require_formal_authorization:
        protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
        if protocol["authorization"].get("formal_nine_run_gpu_training") is not True:
            raise PermissionError("Exp61 formal training has not passed final preflight review")
        if lock.get("status") != "EXP61_SOURCE_LOCK_FROZEN_AFTER_FINAL_PREFLIGHT_REVIEW":
            raise PermissionError("Exp61 source lock is still a review candidate")
    return lock


def verify_model_against_lock(model_path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    runtime = directory_manifest(model_path.resolve())
    expected = lock["model"]
    if runtime["manifest_sha256"] != expected["manifest_sha256"]:
        raise RuntimeError("Exp61 runtime model manifest differs from source lock")
    if runtime["files"] != expected["files"]:
        raise RuntimeError("Exp61 runtime model file set differs from source lock")
    return runtime
