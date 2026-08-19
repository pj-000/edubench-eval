"""Source/model lock for formal Exp62 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation import CONFIG_ROOT, OUTPUT_ROOT, REPO_ROOT, SEEDS
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import EXPECTED_ANNOTATION_SHA256
from thesis_exp.exp62_summeval_routing_confirmation.data import EXPECTED_MANIFEST_SHA256
from thesis_exp.exp62_summeval_routing_confirmation.train import (
    EXPECTED_DEV_CONTRACT_SHA256,
    EXPECTED_TRAIN_CONTRACT_SHA256,
    PROTOCOL_PATH,
)


SOURCE_LOCK_PATH = CONFIG_ROOT / "source_lock.json"
SOURCE_FILES = (
    "thesis_exp/exp62_summeval_routing_confirmation/__init__.py",
    "thesis_exp/exp62_summeval_routing_confirmation/audit_dataset.py",
    "thesis_exp/exp62_summeval_routing_confirmation/audit_token_lengths.py",
    "thesis_exp/exp62_summeval_routing_confirmation/configs/stage0_protocol.json",
    "thesis_exp/exp62_summeval_routing_confirmation/configs/protocol.json",
    "thesis_exp/exp62_summeval_routing_confirmation/build_source_lock.py",
    "thesis_exp/exp62_summeval_routing_confirmation/contract.py",
    "thesis_exp/exp62_summeval_routing_confirmation/data.py",
    "thesis_exp/exp62_summeval_routing_confirmation/geometry.py",
    "thesis_exp/exp62_summeval_routing_confirmation/metrics.py",
    "thesis_exp/exp62_summeval_routing_confirmation/model.py",
    "thesis_exp/exp62_summeval_routing_confirmation/preflight.py",
    "thesis_exp/exp62_summeval_routing_confirmation/runtime.py",
    "thesis_exp/exp62_summeval_routing_confirmation/train.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp59_residual_geometry/train.py",
    "thesis_exp/tests/test_exp62_summeval_stage0.py",
    "thesis_exp/outputs/exp62_summeval_routing_confirmation/stage0/split_manifest.jsonl",
    "thesis_exp/outputs/exp62_summeval_routing_confirmation/stage0/stage0_audit.json",
    "thesis_exp/outputs/exp62_summeval_routing_confirmation/stage0/token_length_audit.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = str(item.relative_to(path))
        if relative == ".msc":
            continue
        value = sha256_file(item)
        files[relative] = value
        digest.update(f"{relative}\t{value}\n".encode("utf-8"))
    return {"manifest_sha256": digest.hexdigest(), "files": files}


def build_source_lock(model_path: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP62_PROTOCOL_FROZEN_FORMAL_TRAINING_AUTHORIZED":
        raise RuntimeError("Exp62 protocol must be formally authorized before source locking")
    source_hashes = {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_FILES}
    preflight_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = (
            OUTPUT_ROOT
            / "preflight"
            / f"seed_{seed}"
            / "real_model_no_update_preflight.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "EXP62_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS":
            raise RuntimeError(f"Exp62 seed {seed} preflight did not pass")
        preflight_hashes[str(seed)] = sha256_file(path)
    return {
        "status": "EXP62_SOURCE_AND_MODEL_LOCK_FROZEN",
        "source_files": source_hashes,
        "preflight_sha256": preflight_hashes,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "model": directory_manifest(model_path.resolve()),
        "annotation_sha256": EXPECTED_ANNOTATION_SHA256,
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "train_contract_sha256": EXPECTED_TRAIN_CONTRACT_SHA256,
        "dev_contract_sha256": EXPECTED_DEV_CONTRACT_SHA256,
        "test_access_count": 0,
        "formal_training_authorized": True,
    }


def verify_source_lock(model_path: Path) -> dict[str, Any]:
    lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "EXP62_SOURCE_AND_MODEL_LOCK_FROZEN":
        raise RuntimeError("Exp62 source lock status is invalid")
    if set(lock.get("source_files", {})) != set(SOURCE_FILES):
        raise RuntimeError("Exp62 source-lock file set changed")
    for relative, expected in lock["source_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Exp62 source-lock mismatch: {relative}")
    if sha256_file(PROTOCOL_PATH) != lock.get("protocol_sha256"):
        raise RuntimeError("Exp62 protocol changed after locking")
    for seed, expected in lock.get("preflight_sha256", {}).items():
        path = (
            OUTPUT_ROOT
            / "preflight"
            / f"seed_{seed}"
            / "real_model_no_update_preflight.json"
        )
        if sha256_file(path) != expected:
            raise RuntimeError(f"Exp62 preflight changed for seed {seed}")
    runtime_model = directory_manifest(model_path.resolve())
    if runtime_model != lock.get("model"):
        raise RuntimeError("Exp62 runtime model differs from frozen model manifest")
    if lock.get("test_access_count") != 0 or lock.get("formal_training_authorized") is not True:
        raise RuntimeError("Exp62 source lock does not authorize formal training")
    return lock
