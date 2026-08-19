"""Build and verify the immutable Exp64 source/model/preflight lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    EXPECTED_ANNOTATION_SHA256,
    sha256_file,
)
from thesis_exp.exp62_summeval_routing_confirmation.contract import directory_manifest
from thesis_exp.exp62_summeval_routing_confirmation.data import EXPECTED_MANIFEST_SHA256
from thesis_exp.exp64_optimizer_state_residual import CONFIG_ROOT, OUTPUT_ROOT, REPO_ROOT


LOCK_PATH = CONFIG_ROOT / "source_lock.json"
FORMAL_LOCK_PATH = CONFIG_ROOT / "formal_source_lock.json"
PROTOCOL_PATH = CONFIG_ROOT / "protocol_draft.json"
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight" / "existing_exp63_seed67_epoch2.json"
SYNTHETIC_TEST_PATH = OUTPUT_ROOT / "preflight" / "synthetic_unit_tests.json"
PROBE_MANIFEST = OUTPUT_ROOT / "stage0" / "probe_manifest.jsonl"
PROBE_AUDIT = OUTPUT_ROOT / "stage0" / "probe_manifest_audit.json"
RNG_SUMMARY = (
    REPO_ROOT
    / "thesis_exp/outputs/exp63_same_state_counterfactual/decision/checkpoint_rng_audit_summary.json"
)
SOURCE_FILES = (
    "thesis_exp/exp64_optimizer_state_residual/__init__.py",
    "thesis_exp/exp64_optimizer_state_residual/analyze.py",
    "thesis_exp/exp64_optimizer_state_residual/build_probe_manifest.py",
    "thesis_exp/exp64_optimizer_state_residual/formal_counterfactual.py",
    "thesis_exp/exp64_optimizer_state_residual/mechanics.py",
    "thesis_exp/exp64_optimizer_state_residual/preflight_existing_exp63.py",
    "thesis_exp/exp64_optimizer_state_residual/source_lock.py",
    "thesis_exp/exp64_optimizer_state_residual/state.py",
    "thesis_exp/exp64_optimizer_state_residual/train_base.py",
    "thesis_exp/exp64_optimizer_state_residual/RESEARCH_SPEC.md",
    "thesis_exp/configs/exp64_optimizer_state_residual/protocol_draft.json",
    "thesis_exp/tests/test_exp64_optimizer_state_residual.py",
    "thesis_exp/exp62_summeval_routing_confirmation/audit_dataset.py",
    "thesis_exp/exp62_summeval_routing_confirmation/data.py",
    "thesis_exp/exp62_summeval_routing_confirmation/model.py",
    "thesis_exp/exp62_summeval_routing_confirmation/runtime.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp59_residual_geometry/train.py",
    "thesis_exp/outputs/exp62_summeval_routing_confirmation/stage0/split_manifest.jsonl",
    "thesis_exp/outputs/exp64_optimizer_state_residual/stage0/probe_manifest.jsonl",
    "thesis_exp/outputs/exp64_optimizer_state_residual/stage0/probe_manifest_audit.json",
    "thesis_exp/outputs/exp63_same_state_counterfactual/decision/checkpoint_rng_audit_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/existing_exp63_seed67_epoch2.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/synthetic_unit_tests.json",
)
FORMAL_ADDITIONAL_FILES = (
    "thesis_exp/configs/exp64_optimizer_state_residual/source_lock.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/base/seed_72/run_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/base/seed_73/run_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/base/seed_74/run_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/base/seed_75/run_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/base/seed_76/run_summary.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/formal_probe_batch_oom_revision.json",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/failed_formal_seed72_oom.log",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/failed_formal_seed73_oom.log",
    "thesis_exp/outputs/exp64_optimizer_state_residual/preflight/failed_formal_seed74_oom.log",
)
FORMAL_SOURCE_FILES = SOURCE_FILES + FORMAL_ADDITIONAL_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "formal"), default="base")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _source_hashes(files: tuple[str, ...] = SOURCE_FILES) -> dict[str, str]:
    result = {}
    for relative in files:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def build_formal_lock(annotations: Path, model_path: Path) -> dict[str, Any]:
    result = build_lock(annotations, model_path)
    base_lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    revision_path = (
        OUTPUT_ROOT / "preflight" / "formal_probe_batch_oom_revision.json"
    )
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    summaries = {}
    for seed in (72, 73, 74, 75, 76):
        path = OUTPUT_ROOT / "base" / f"seed_{seed}" / "run_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "status": summary.get("status")
            == "EXP64_BASE_TRAJECTORY_COMPLETE_NO_DEV_OUTCOMES_READ",
            "seed": summary.get("seed") == seed,
            "checkpoints": len(summary.get("checkpoints", ())) == 3,
            "dev_zero": summary.get("dev_outcomes_read") == 0,
            "test_zero": summary.get("test_access_count") == 0,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Exp64 base summary failed for seed {seed}: {checks}")
        summaries[str(seed)] = {
            "sha256": sha256_file(path),
            "checkpoint_sha256": [value["sha256"] for value in summary["checkpoints"]],
        }
    formal_hashes = _source_hashes(FORMAL_SOURCE_FILES)
    formal_checks = {
        "base_lock_status": base_lock.get("status")
        == "EXP64_SOURCE_MODEL_PROTOCOL_LOCK_FROZEN",
        "base_lock_hash": revision.get("base_trajectory_source_lock_sha256")
        == sha256_file(LOCK_PATH),
        "mechanical_revision_before_results": revision.get("status")
        == "EXP64_MECHANICAL_REVISION_BEFORE_ANY_FORMAL_STAGE_RESULT",
        "independent_outcomes_zero": revision.get("result_exposure", {}).get(
            "independent_model_outcomes_observed"
        )
        == 0,
        "formal_stage_files_zero": revision.get("result_exposure", {}).get(
            "formal_stage_json_files_written"
        )
        == 0,
        "test_zero": revision.get("result_exposure", {}).get("test_access_count") == 0,
    }
    if not all(formal_checks.values()):
        raise RuntimeError(f"Exp64 formal-lock gate failed: {formal_checks}")
    result.update(
        {
            "status": "EXP64_FORMAL_SOURCE_MODEL_PROTOCOL_LOCK_FROZEN",
            "checks": {**result["checks"], **formal_checks},
            "source_files": formal_hashes,
            "source_files_logical_sha256": _logical_hash(formal_hashes),
            "base_source_lock_sha256": sha256_file(LOCK_PATH),
            "base_trajectory_summaries": summaries,
            "mechanical_revision_sha256": sha256_file(revision_path),
        }
    )
    return result


def _logical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_lock(annotations: Path, model_path: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    synthetic = json.loads(SYNTHETIC_TEST_PATH.read_text(encoding="utf-8"))
    rng = json.loads(RNG_SUMMARY.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_AUDIT.read_text(encoding="utf-8"))
    evidence = protocol.get("freeze_evidence", {})
    checks = {
        "formal_protocol": protocol.get("status")
        == "EXP64_PROTOCOL_FROZEN_BEFORE_INDEPENDENT_OUTCOMES",
        "base_authorized": protocol.get("authorization", {}).get(
            "five_base_trajectory_gpu_training"
        )
        is True,
        "counterfactual_authorized": protocol.get("authorization", {}).get(
            "formal_counterfactual_updates"
        )
        is True,
        "test_forbidden": protocol.get("authorization", {}).get("test_evaluation") is False,
        "annotation": sha256_file(annotations) == EXPECTED_ANNOTATION_SHA256,
        "split_manifest": protocol.get("dataset", {}).get("split_manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "probe_manifest": sha256_file(PROBE_MANIFEST)
        == protocol.get("probe", {}).get("manifest_sha256"),
        "probe_audit": probe.get("status") == "EXP64_PROBE_MANIFEST_PASS",
        "rng_audit": rng.get("status")
        == "EXP63_RNG_AUDIT_EXACT_PASS_RETAIN_ORIGINAL_RESULTS",
        "real_model_preflight": preflight.get("status")
        == "EXP64_EXISTING_STATE_NO_OUTCOME_PREFLIGHT_PASS",
        "synthetic_unit_tests": synthetic.get("status")
        == "EXP64_SYNTHETIC_UNIT_TESTS_PASS",
        "freeze_before_independent_outcomes": evidence.get(
            "independent_seed_outcomes_read_before_freeze"
        )
        == 0,
        "rng_evidence_hash": evidence.get("rng_audit_summary_sha256")
        == sha256_file(RNG_SUMMARY),
        "probe_evidence_hash": evidence.get("probe_manifest_audit_sha256")
        == sha256_file(PROBE_AUDIT),
        "synthetic_evidence_hash": evidence.get("synthetic_unit_tests_sha256")
        == sha256_file(SYNTHETIC_TEST_PATH),
        "preflight_evidence_hash": evidence.get(
            "existing_state_no_outcome_preflight_sha256"
        )
        == sha256_file(PREFLIGHT_PATH),
        "no_test_access": all(
            value.get("test_access_count") == 0
            for value in (protocol, probe, rng, preflight, synthetic)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp64 source-lock gate failed: {checks}")
    source_hashes = _source_hashes()
    result = {
        "status": "EXP64_SOURCE_MODEL_PROTOCOL_LOCK_FROZEN",
        "checks": checks,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_files": source_hashes,
        "source_files_logical_sha256": _logical_hash(source_hashes),
        "annotation_sha256": sha256_file(annotations),
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "probe_manifest_sha256": sha256_file(PROBE_MANIFEST),
        "rng_audit_summary_sha256": sha256_file(RNG_SUMMARY),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "synthetic_unit_tests_sha256": sha256_file(SYNTHETIC_TEST_PATH),
        "model": directory_manifest(model_path.resolve()),
        "formal_training_authorized": True,
        "formal_counterfactual_authorized": True,
        "test_access_count": 0,
    }
    return result


def verify_source_lock(model_path: Path) -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "EXP64_SOURCE_MODEL_PROTOCOL_LOCK_FROZEN":
        raise RuntimeError("Exp64 source lock is not frozen")
    if set(lock.get("source_files", {})) != set(SOURCE_FILES):
        raise RuntimeError("Exp64 source-lock file set changed")
    for relative, expected in lock["source_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Exp64 source-lock mismatch: {relative}")
    if sha256_file(PROTOCOL_PATH) != lock.get("protocol_sha256"):
        raise RuntimeError("Exp64 protocol changed after source lock")
    if directory_manifest(model_path.resolve()) != lock.get("model"):
        raise RuntimeError("Exp64 runtime model differs from the frozen model")
    if lock.get("test_access_count") != 0:
        raise RuntimeError("Exp64 source lock reports test access")
    return lock


def verify_formal_source_lock(model_path: Path) -> dict[str, Any]:
    lock = json.loads(FORMAL_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "EXP64_FORMAL_SOURCE_MODEL_PROTOCOL_LOCK_FROZEN":
        raise RuntimeError("Exp64 formal source lock is not frozen")
    if set(lock.get("source_files", {})) != set(FORMAL_SOURCE_FILES):
        raise RuntimeError("Exp64 formal source-lock file set changed")
    for relative, expected in lock["source_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Exp64 formal source-lock mismatch: {relative}")
    if sha256_file(PROTOCOL_PATH) != lock.get("protocol_sha256"):
        raise RuntimeError("Exp64 protocol changed after formal source lock")
    if sha256_file(LOCK_PATH) != lock.get("base_source_lock_sha256"):
        raise RuntimeError("Exp64 base source lock changed before formal audit")
    if directory_manifest(model_path.resolve()) != lock.get("model"):
        raise RuntimeError("Exp64 formal runtime model differs from the frozen model")
    if lock.get("test_access_count") != 0:
        raise RuntimeError("Exp64 formal source lock reports test access")
    return lock


def main() -> None:
    args = parse_args()
    result = (
        build_formal_lock(args.annotations, args.model_name_or_path)
        if args.phase == "formal"
        else build_lock(args.annotations, args.model_name_or_path)
    )
    output = args.output or (FORMAL_LOCK_PATH if args.phase == "formal" else LOCK_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
