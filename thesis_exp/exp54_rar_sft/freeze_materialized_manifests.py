"""Promote the reviewed materialized-manifest candidate to an immutable lock.

This command never rebuilds or edits a private artifact.  It verifies the
exact candidate reviewed at the MATERIALIZED_MANIFEST_PASS gate, re-hashes
every private manifest and source artifact, and writes one aggregate public
freeze lock.  Training remains forbidden by the resulting lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    reject_eval_path,
    write_json,
)
from thesis_exp.exp54_rar_sft.audit_training_manifests import (
    SOURCE_PATHS,
    UPSTREAM_LOCK_PATHS,
)
from thesis_exp.exp54_rar_sft.build_training_manifests import (
    DEFAULT_ALL_REFERENCES,
    DEFAULT_CONSISTENT_REFERENCES,
    DEFAULT_OUTPUT,
)
from thesis_exp.exp54_rar_sft.manifest_source_contract import ARMS
from thesis_exp.exp54_rar_sft.reference_schedule import FORMAL_SEEDS


REVIEWED_COMMIT = "dad35890a1f0a5111ae0c69a2b60dff26700c24a"
REVIEW_VERDICT = "MATERIALIZED_MANIFEST_PASS"
EXPECTED_CANDIDATE_REPORT_SHA256 = (
    "74dbf657a9b157f4c144258b953263ddb82b073f8062a33d7c11e7ad91649ee9"
)
EXPECTED_CANDIDATE_LOCK_SHA256 = (
    "2f94d2d2b34165335fca1ed44152653fadbc0b24338db619dc8800a8ebec9deb"
)
DEFAULT_CANDIDATE_REPORT = (
    DEFAULT_OUTPUT / "audit/materialized_manifest_candidate_report.json"
)
DEFAULT_CANDIDATE_LOCK = (
    DEFAULT_OUTPUT / "protocol/materialized_manifest_candidate_lock.json"
)
DEFAULT_FROZEN_LOCK = (
    DEFAULT_OUTPUT / "protocol/materialized_manifest_frozen_lock.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    reject_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_hash(path: Path, expected: str, *, label: str) -> str:
    reject_eval_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} differs from the reviewed candidate")
    return actual


def validate_reviewed_candidate(
    candidate_report_path: Path,
    candidate_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact public candidate reviewed by GPT-5.6 Pro."""
    _require_hash(
        candidate_report_path,
        EXPECTED_CANDIDATE_REPORT_SHA256,
        label="candidate report",
    )
    _require_hash(
        candidate_lock_path,
        EXPECTED_CANDIDATE_LOCK_SHA256,
        label="candidate lock",
    )
    report = _read_json(candidate_report_path)
    lock = _read_json(candidate_lock_path)
    expected_status = "MATERIALIZED_MANIFEST_CANDIDATE_AUDITED_NOT_FROZEN"
    if report.get("status") != expected_status or lock.get("status") != expected_status:
        raise ValueError("reviewed candidate status differs")
    if lock.get("candidate_report_sha256") != EXPECTED_CANDIDATE_REPORT_SHA256:
        raise ValueError("candidate lock does not bind the reviewed report")
    for artifact_name, artifact in (("report", report), ("lock", lock)):
        if (
            artifact.get("manifest_freeze_allowed")
            or artifact.get("smoke_training_allowed")
            or artifact.get("formal_training_allowed")
            or artifact.get("dev_accessed")
            or artifact.get("test_accessed")
            or artifact.get("training_used")
        ):
            raise PermissionError(
                f"candidate {artifact_name} crosses its authorization boundary"
            )
    for key in (
        "formal_seeds",
        "arms",
        "logical_epochs",
        "rows_per_logical_epoch",
        "private_artifact_hashes",
        "source_hashes",
        "upstream_lock_hashes",
        "tokenizer_lock_sha256",
        "config_sha256",
        "locked_train_score_vector_sha256",
    ):
        if report.get(key) != lock.get(key):
            raise ValueError(f"candidate report/lock differ at {key}")
    if report["formal_seeds"] != list(FORMAL_SEEDS):
        raise ValueError("candidate formal seeds differ")
    if report["arms"] != list(ARMS):
        raise ValueError("candidate arms differ")
    if not all(report["independent_reconstruction"].values()):
        raise ValueError("candidate independent reconstruction is incomplete")
    return report, lock


def verify_public_hash_chain(candidate_lock: dict[str, Any]) -> None:
    for name, path in SOURCE_PATHS.items():
        _require_hash(
            path,
            str(candidate_lock["source_hashes"][name]),
            label=f"reviewed source {name}",
        )
    for name, path in UPSTREAM_LOCK_PATHS.items():
        _require_hash(
            path,
            str(candidate_lock["upstream_lock_hashes"][name]),
            label=f"reviewed upstream lock {name}",
        )
    candidate_config = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/materialized_manifest_candidate.json"
    )
    _require_hash(
        candidate_config,
        str(candidate_lock["config_sha256"]),
        label="reviewed candidate config",
    )


def collect_private_hashes(output_dir: Path) -> dict[str, Any]:
    """Hash the exact private files without reading or publishing row content."""
    private: dict[str, Any] = {
        "shared_prompt_cache": file_sha256(
            output_dir / "data/shared_prompt_cache.jsonl"
        ),
        "all_rater_reference_inventory": file_sha256(DEFAULT_ALL_REFERENCES),
        "label_consistent_reference_inventory": file_sha256(
            DEFAULT_CONSISTENT_REFERENCES
        ),
        "manifests_by_seed": {},
        "source_artifacts_by_seed": {},
    }
    for seed in FORMAL_SEEDS:
        seed_key = f"seed{seed}"
        private["manifests_by_seed"][seed_key] = {
            arm: file_sha256(
                output_dir
                / "data"
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            for arm in ARMS
        }
        private["source_artifacts_by_seed"][seed_key] = {
            "base_schedule_sha256": file_sha256(
                output_dir / "data" / f"base_event_schedule_seed{seed}.jsonl"
            ),
            "event_donor_map_sha256": file_sha256(
                output_dir / "data" / f"r2_event_donor_map_seed{seed}.jsonl"
            ),
            "r2_event_mask_sha256": file_sha256(
                output_dir / "data" / f"r2_event_active_mask_seed{seed}.jsonl"
            ),
            "r3_event_mask_sha256": file_sha256(
                output_dir / "data" / f"r3_event_active_mask_seed{seed}.jsonl"
            ),
        }
    return private


def build_frozen_lock(
    *,
    candidate_report_path: Path,
    candidate_lock_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    report, candidate_lock = validate_reviewed_candidate(
        candidate_report_path,
        candidate_lock_path,
    )
    verify_public_hash_chain(candidate_lock)
    actual_private_hashes = collect_private_hashes(output_dir)
    if actual_private_hashes != candidate_lock["private_artifact_hashes"]:
        raise ValueError("private artifacts differ from the reviewed candidate")
    return {
        "status": "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED",
        "review_gate": {
            "verdict": REVIEW_VERDICT,
            "reviewed_commit": REVIEWED_COMMIT,
            "candidate_report_sha256": file_sha256(candidate_report_path),
            "candidate_lock_sha256": file_sha256(candidate_lock_path),
        },
        "formal_seeds": report["formal_seeds"],
        "arms": report["arms"],
        "logical_epochs": report["logical_epochs"],
        "rows_per_logical_epoch": report["rows_per_logical_epoch"],
        "private_artifact_hashes": actual_private_hashes,
        "source_hashes": candidate_lock["source_hashes"],
        "upstream_lock_hashes": candidate_lock["upstream_lock_hashes"],
        "tokenizer_lock_sha256": candidate_lock["tokenizer_lock_sha256"],
        "config_sha256": candidate_lock["config_sha256"],
        "locked_train_score_vector_sha256": candidate_lock[
            "locked_train_score_vector_sha256"
        ],
        "manifest_frozen": True,
        "manifest_freeze_allowed": True,
        "training_configuration_review_allowed": True,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "invalidation_rule": (
            "Any manifest, private source artifact, reviewed source, tokenizer, "
            "candidate config, or upstream lock hash change invalidates this freeze."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=DEFAULT_CANDIDATE_REPORT,
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        default=DEFAULT_CANDIDATE_LOCK,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frozen-lock", type=Path, default=DEFAULT_FROZEN_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = build_frozen_lock(
        candidate_report_path=args.candidate_report,
        candidate_lock_path=args.candidate_lock,
        output_dir=args.output_dir,
    )
    write_json(args.frozen_lock, frozen)


if __name__ == "__main__":
    main()
