"""Freeze the reviewed SORC-DPO pair candidates without exposing private rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file


REVIEWED_COMMIT = "ad3e2dda3f05b99fb571137686c505367de4c4bd"
ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/preference_pairs"
)
REPORT = ROOT / "candidate_report.json"
CANDIDATE_LOCK = ROOT / "candidate_lock.json"
AUDIT_REPORT = ROOT / "audit_report.json"
OUTPUT = ROOT / "pair_protocol_frozen_lock.json"
SOURCES = {
    "pair_protocol": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/"
        "sorc_dpo_pair_protocol_v1.json"
    ),
    "pair_builder": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/build_sorc_dpo_pairs.py"
    ),
    "pair_auditor": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/audit_sorc_dpo_pairs.py"
    ),
    "failure_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/actual_failure_bank.py"
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def freeze() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = read_json(REPORT)
    lock = read_json(CANDIDATE_LOCK)
    audit = read_json(AUDIT_REPORT)
    if audit.get("status") != "SORC_DPO_PAIR_CANDIDATE_AUDIT_PASS":
        raise ValueError("pair candidate audit has not passed")
    if (
        report.get("status")
        != "SORC_DPO_PAIR_CANDIDATES_WRITTEN_NOT_FROZEN"
        or lock.get("status")
        != "PAIR_CANDIDATES_NOT_FROZEN_TRAINING_NOT_ALLOWED"
    ):
        raise ValueError("candidate state differs")
    for value in (report, lock, audit):
        if (
            value.get("preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise ValueError("candidate boundary is not fail-closed")
    if sha256_file(REPORT) != str(lock["candidate_report_sha256"]):
        raise ValueError("candidate report lock differs")
    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    private_hashes = dict(lock["private_output_hashes"])
    expected_names = {
        "score_pairs_hybrid",
        "score_pairs_synthetic_seed42",
        "rationale_pairs_r3_r2",
        "actual_rationale_candidates",
    }
    if set(private_hashes) != expected_names:
        raise ValueError("private pair artifact set differs")
    for name, expected_hash in private_hashes.items():
        path = ROOT / "private" / f"{name}.jsonl"
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"{name}: private pair hash differs")
    aggregate = audit["aggregate"]
    if (
        int(aggregate["hybrid_score_pairs"]) != 838
        or int(aggregate["synthetic_score_pairs"]) != 838
        or int(aggregate["rationale_pairs"]) != 1600
        or int(aggregate["actual_rationale_candidates"]) != 948
    ):
        raise ValueError("reviewed pair aggregate differs")
    frozen = {
        "schema_version": "exp54-sorc-dpo-pair-frozen-lock-v1",
        "status": "PAIR_PROTOCOL_FROZEN_IMPLEMENTATION_ALLOWED",
        "review_verdict": "PAIR_PROTOCOL_PASS",
        "reviewed_commit": REVIEWED_COMMIT,
        "aggregate": aggregate,
        "public_artifact_hashes": {
            "candidate_report": sha256_file(REPORT),
            "candidate_lock": sha256_file(CANDIDATE_LOCK),
            "audit_report": sha256_file(AUDIT_REPORT),
        },
        "private_artifact_hashes": private_hashes,
        "source_hashes": {
            name: sha256_file(path) for name, path in SOURCES.items()
        },
        "pair_protocol_frozen": True,
        "training_implementation_allowed": True,
        "evaluator_execution_allowed": False,
        "gpu_preference_training_allowed": False,
        "formal_preference_training_allowed": False,
        "actual_rationale_preference_labels_created": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(OUTPUT, frozen)
    return frozen


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
