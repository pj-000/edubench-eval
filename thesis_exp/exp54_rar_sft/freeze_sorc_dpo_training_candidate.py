"""Freeze the reviewed SORC-DPO training implementation candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file


REVIEWED_COMMIT = "82a22f842b2f5009962ba0019bcae61f172d5797"
REVIEW_VERDICT = "PREFERENCE_TRAINING_IMPLEMENTATION_PASS"
ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_training_candidate"
)
REPORT = ROOT / "candidate_report.json"
CANDIDATE_LOCK = ROOT / "candidate_lock.json"
AUDIT_REPORT = ROOT / "audit_report.json"
OUTPUT = ROOT / "preference_training_frozen_lock.json"
EXPECTED_PUBLIC_HASHES = {
    "candidate_report": (
        "c698bc97e34567a1517c3e7fae8a60699458d6fc27ad05c48968b794495543d1"
    ),
    "candidate_lock": (
        "8578259ba22444c4883134b9558b54323768f804bb9097021afd8758abe3aeb1"
    ),
    "audit_report": (
        "1b16e213ff39634c23204d550637c9593201251bc7f7a530907e9968f9b77567"
    ),
}
EXPECTED_MANIFESTS = {
    "P1_FIELD_DPO",
    "P2_SORC_SCORE",
    "P3_JOINT_SORC",
    "P1_SYN_SEED42",
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


def freeze() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    actual_public_hashes = {
        "candidate_report": sha256_file(REPORT),
        "candidate_lock": sha256_file(CANDIDATE_LOCK),
        "audit_report": sha256_file(AUDIT_REPORT),
    }
    if actual_public_hashes != EXPECTED_PUBLIC_HASHES:
        raise ValueError("reviewed public candidate hashes differ")
    report = read_json(REPORT)
    candidate_lock = read_json(CANDIDATE_LOCK)
    audit = read_json(AUDIT_REPORT)
    if (
        report.get("status")
        != "SORC_DPO_TRAINING_MANIFEST_CANDIDATE_NOT_AUTHORIZED"
        or candidate_lock.get("status")
        != "SORC_DPO_TRAINING_CANDIDATE_NOT_FROZEN_NOT_AUTHORIZED"
        or audit.get("status")
        != "SORC_DPO_TRAINING_CANDIDATE_AUDIT_PASS"
    ):
        raise ValueError("reviewed candidate states differ")
    if set(candidate_lock["private_manifest_hashes"]) != EXPECTED_MANIFESTS:
        raise ValueError("private manifest set differs")
    for arm, expected_hash in candidate_lock["private_manifest_hashes"].items():
        path = ROOT / "private" / f"{arm.lower()}.jsonl"
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"{arm}: private manifest differs")
    if (
        audit.get("all_arm_optimizer_step_counts_equal_27") is not True
        or audit.get("p1_p1_syn_same_record_and_block_vectors") is not True
        or audit.get(
            "p1_p1_syn_same_chosen_sequence_and_mask_vectors"
        )
        is not True
        or audit.get("padding_fixed_right_to_2048") is not True
    ):
        raise ValueError("reviewed control assertions differ")
    for value in (report, candidate_lock, audit):
        if (
            value.get("gpu_preference_training_allowed") is not False
            or value.get("formal_preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise PermissionError("reviewed authorization boundary differs")

    frozen = {
        "schema_version": "exp54-sorc-dpo-training-frozen-lock-v1",
        "status": "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED",
        "review_gate": {
            "verdict": REVIEW_VERDICT,
            "reviewed_commit": REVIEWED_COMMIT,
            "public_artifact_hashes": actual_public_hashes,
        },
        "private_manifest_hashes": dict(
            candidate_lock["private_manifest_hashes"]
        ),
        "source_hashes": dict(candidate_lock["source_hashes"]),
        "pair_artifact_hashes": dict(
            candidate_lock["pair_artifact_hashes"]
        ),
        "tokenizer_lock_sha256": str(
            candidate_lock["tokenizer_lock_sha256"]
        ),
        "loss_collator_manifests_and_config_frozen": True,
        "all_arm_optimizer_step_counts": 27,
        "padding": "fixed_right_padding_to_cutoff_2048",
        "smoke_subset_build_allowed": True,
        "smoke_runner_implementation_allowed": True,
        "gpu_smoke_allowed": False,
        "formal_preference_training_allowed": False,
        "evaluator_calls_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "invalidation_rule": (
            "Any reviewed public artifact, private manifest, loss/collator, "
            "training config, tokenizer, checkpoint, or upstream source hash "
            "change invalidates this freeze."
        ),
    }
    write_json(OUTPUT, frozen)
    return frozen


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(freeze(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
