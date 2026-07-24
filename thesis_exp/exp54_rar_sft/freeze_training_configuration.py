"""Freeze the exact training configuration approved at the narrow review gate.

This command only promotes already-audited public locks to an immutable
configuration freeze. It does not read private row content, install an
authorization trust anchor, load model weights, or execute training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.authorization_guard import (
    closure_sha256,
    runtime_source_closure,
    sha256_file,
)


REVIEWED_COMMIT = "d2762a427b8bed957459b04ba239a988ab3acea5"
REVIEW_VERDICT = "TRAINING_CONFIGURATION_PASS"
EXPECTED_CANDIDATE_REPORT_SHA256 = (
    "27186421a67aa5c0b324672e567280859295e3066201fbca5b97ae2cd07dc0aa"
)
EXPECTED_CANDIDATE_LOCK_SHA256 = (
    "a69b7c4594d86d8dbffa432550d0d69a0d39bd67785c89dc8045abf676e4ee0a"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
DEFAULT_FROZEN_MANIFEST_LOCK = (
    DEFAULT_OUTPUT / "protocol/materialized_manifest_frozen_lock.json"
)
DEFAULT_CANDIDATE_REPORT = (
    DEFAULT_OUTPUT / "audit/training_configuration_candidate_report.json"
)
DEFAULT_CANDIDATE_LOCK = (
    DEFAULT_OUTPUT / "protocol/training_configuration_candidate_lock.json"
)
DEFAULT_FROZEN_LOCK = (
    DEFAULT_OUTPUT / "protocol/training_configuration_frozen_lock.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_training_configuration_frozen_lock(
    *,
    config_path: Path,
    frozen_manifest_lock_path: Path,
    candidate_report_path: Path,
    candidate_lock_path: Path,
) -> dict[str, Any]:
    if sha256_file(candidate_report_path) != EXPECTED_CANDIDATE_REPORT_SHA256:
        raise ValueError("training-configuration candidate report differs")
    if sha256_file(candidate_lock_path) != EXPECTED_CANDIDATE_LOCK_SHA256:
        raise ValueError("training-configuration candidate lock differs")
    report = _read_object(candidate_report_path)
    candidate = _read_object(candidate_lock_path)
    expected_status = (
        "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED"
    )
    if report.get("status") != expected_status:
        raise ValueError("training-configuration report status differs")
    if candidate.get("status") != expected_status:
        raise ValueError("training-configuration lock status differs")
    if (
        candidate.get("candidate_report_sha256")
        != EXPECTED_CANDIDATE_REPORT_SHA256
    ):
        raise ValueError("candidate lock does not bind the reviewed report")
    for name, artifact in (("report", report), ("lock", candidate)):
        if (
            artifact.get("smoke_training_allowed")
            or artifact.get("formal_training_allowed")
            or artifact.get("dev_accessed")
            or artifact.get("test_accessed")
            or artifact.get("training_used")
        ):
            raise PermissionError(f"{name} crosses the review authorization")

    config_sha256 = sha256_file(config_path)
    frozen_manifest_lock_sha256 = sha256_file(frozen_manifest_lock_path)
    if report.get("configuration_sha256") != config_sha256:
        raise ValueError("candidate report does not bind the configuration")
    if candidate.get("configuration_sha256") != config_sha256:
        raise ValueError("candidate lock does not bind the configuration")
    if (
        report.get("manifest_freeze", {}).get("frozen_lock_sha256")
        != frozen_manifest_lock_sha256
    ):
        raise ValueError("candidate report does not bind frozen manifests")
    if (
        candidate.get("manifest_frozen_lock_sha256")
        != frozen_manifest_lock_sha256
    ):
        raise ValueError("candidate lock does not bind frozen manifests")

    closure = runtime_source_closure()
    closure_digest = closure_sha256(closure)
    if report.get("runtime_source_closure") != closure:
        raise ValueError("candidate report runtime closure differs")
    if candidate.get("runtime_source_closure") != closure:
        raise ValueError("candidate lock runtime closure differs")
    if report.get("runtime_source_closure_sha256") != closure_digest:
        raise ValueError("candidate report runtime closure digest differs")
    if candidate.get("runtime_source_closure_sha256") != closure_digest:
        raise ValueError("candidate lock runtime closure digest differs")
    if len(closure) != 16:
        raise ValueError("reviewed runtime closure must contain 16 files")

    return {
        "status": (
            "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED"
        ),
        "review_gate": {
            "verdict": REVIEW_VERDICT,
            "reviewed_commit": REVIEWED_COMMIT,
            "candidate_report_sha256": EXPECTED_CANDIDATE_REPORT_SHA256,
            "candidate_lock_sha256": EXPECTED_CANDIDATE_LOCK_SHA256,
        },
        "configuration_sha256": config_sha256,
        "materialized_manifest_frozen_lock_sha256": (
            frozen_manifest_lock_sha256
        ),
        "runtime_source_closure": closure,
        "runtime_source_closure_sha256": closure_digest,
        "runtime_source_closure_file_count": len(closure),
        "configuration_frozen": True,
        "smoke_package_build_allowed": True,
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "invalidation_rule": (
            "Any configuration, frozen-manifest lock, reviewed candidate, "
            "or runtime-source hash change invalidates this freeze."
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--frozen-manifest-lock",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST_LOCK,
    )
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
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=DEFAULT_FROZEN_LOCK,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = build_training_configuration_frozen_lock(
        config_path=args.config,
        frozen_manifest_lock_path=args.frozen_manifest_lock,
        candidate_report_path=args.candidate_report,
        candidate_lock_path=args.candidate_lock,
    )
    write_json(args.frozen_lock, frozen)


if __name__ == "__main__":
    main()
