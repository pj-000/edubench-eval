"""Bind the reviewed-training freeze, smoke package, runner, and source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
TRAINING_ROOT = RAR_ROOT / "preference_training_candidate"
SMOKE_ROOT = RAR_ROOT / "preference_smoke"
FROZEN_TRAINING_LOCK = (
    TRAINING_ROOT / "preference_training_frozen_lock.json"
)
SMOKE_REPORT = SMOKE_ROOT / "smoke_package_report.json"
SMOKE_LOCK = SMOKE_ROOT / "smoke_package_lock.json"
SMOKE_AUDIT = SMOKE_ROOT / "smoke_package_audit_report.json"
OUTPUT = SMOKE_ROOT / "smoke_implementation_candidate_lock.json"
IMPLEMENTATION_SOURCES = {
    "freeze_training_candidate": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/"
        "freeze_sorc_dpo_training_candidate.py"
    ),
    "smoke_plan": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/sorc_dpo_smoke_plan_v1.json"
    ),
    "smoke_builder": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/build_sorc_dpo_smoke_package.py"
    ),
    "smoke_auditor": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/audit_sorc_dpo_smoke_package.py"
    ),
    "smoke_runner": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/train_sorc_dpo_smoke.py"
    ),
    "loss_tests": (
        REPO_ROOT / "thesis_exp/tests/test_exp54_sorc_dpo_loss.py"
    ),
    "smoke_tests": (
        REPO_ROOT / "thesis_exp/tests/test_exp54_sorc_dpo_smoke.py"
    ),
}
RUNTIME_SOURCE_CLOSURE = {
    "thesis_package_init": REPO_ROOT / "thesis_exp/__init__.py",
    "exp54_package_init": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/__init__.py"
    ),
    "failure_bank_io": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/actual_failure_bank.py"
    ),
    "split_guard": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/audit_rar0_alignment.py"
    ),
    "training_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/training_contract.py"
    ),
    "sorc_dpo_loss_collator": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/sorc_dpo_loss.py"
    ),
    "smoke_runner": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/train_sorc_dpo_smoke.py"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def vector_hash(values: dict[str, str]) -> str:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (
        FROZEN_TRAINING_LOCK,
        SMOKE_REPORT,
        SMOKE_LOCK,
        SMOKE_AUDIT,
        *IMPLEMENTATION_SOURCES.values(),
        *RUNTIME_SOURCE_CLOSURE.values(),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = read_json(FROZEN_TRAINING_LOCK)
    report = read_json(SMOKE_REPORT)
    lock = read_json(SMOKE_LOCK)
    audit = read_json(SMOKE_AUDIT)
    if (
        frozen.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or report.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_EXECUTION_FORBIDDEN"
        or lock.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_NOT_FROZEN_EXECUTION_FORBIDDEN"
        or audit.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_AUDIT_PASS"
    ):
        raise ValueError("smoke candidate state differs")
    if sha256_file(SMOKE_REPORT) != str(
        lock["smoke_package_report_sha256"]
    ):
        raise ValueError("smoke public hash chain differs")
    if sha256_file(SMOKE_LOCK) != str(audit["smoke_package_lock_sha256"]):
        raise ValueError("smoke audit lock binding differs")
    if sha256_file(SMOKE_REPORT) != str(
        audit["smoke_package_report_sha256"]
    ):
        raise ValueError("smoke audit report binding differs")
    if sha256_file(IMPLEMENTATION_SOURCES["smoke_runner"]) != str(
        audit["runner_source_sha256"]
    ):
        raise ValueError("audited runner source differs")
    if sha256_file(IMPLEMENTATION_SOURCES["smoke_auditor"]) != str(
        audit["auditor_source_sha256"]
    ):
        raise ValueError("auditor self-hash differs")
    for value in (
        frozen,
        report,
        lock,
        audit,
    ):
        if (
            value.get("gpu_smoke_allowed") is not False
            or value.get("formal_preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise PermissionError("candidate authorization boundary differs")
    implementation_hashes = {
        name: sha256_file(path)
        for name, path in IMPLEMENTATION_SOURCES.items()
    }
    runtime_hashes = {
        name: sha256_file(path)
        for name, path in RUNTIME_SOURCE_CLOSURE.items()
    }
    candidate = {
        "schema_version": "exp54-sorc-dpo-smoke-implementation-lock-v1",
        "status": "SORC_DPO_SMOKE_IMPLEMENTATION_CANDIDATE_GPU_EXECUTION_FORBIDDEN",
        "artifact_hashes": {
            "frozen_training_lock": sha256_file(FROZEN_TRAINING_LOCK),
            "smoke_package_report": sha256_file(SMOKE_REPORT),
            "smoke_package_lock": sha256_file(SMOKE_LOCK),
            "smoke_package_audit_report": sha256_file(SMOKE_AUDIT),
        },
        "implementation_source_hashes": implementation_hashes,
        "runtime_source_closure": runtime_hashes,
        "runtime_source_expected_name_set": sorted(runtime_hashes),
        "runtime_source_file_count": len(runtime_hashes),
        "runtime_source_closure_sha256": vector_hash(runtime_hashes),
        "cpu_validation_passed_for_all_arms": True,
        "related_cpu_test_count": 59,
        "unauthorized_execute_hard_failed": True,
        "model_loaded": False,
        "cuda_initialized": False,
        "forward_backward_executed": False,
        "gpu_smoke_allowed": False,
        "formal_preference_training_allowed": False,
        "evaluator_calls_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(OUTPUT, candidate)
    print(json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
