"""Independently audit the non-installed Exp54 smoke authorization candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.authorization_guard import (
    TRUSTED_AUTHORIZATION_DIGEST_PATH,
    closure_sha256,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
    smoke_runtime_source_closure,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    DEFAULT_SMOKE_CLAIM_ROOT,
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_OUTPUT_ROOT,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    REVIEWED_SMOKE_PACKAGE_COMMIT,
    SMOKE_ARMS,
    SMOKE_MAX_INVOCATIONS_PER_ARM,
    SMOKE_OPTIMIZER_STEPS_PER_ARM,
    SMOKE_SCHEMA_VERSION,
    SMOKE_SOURCE_SEED,
)


REVIEWED_SMOKE_GUARD_COMMIT = (
    "b2e616e324ed9de7c8aaf8eb1dee5a2f0d7a53bb"
)
DEFAULT_CAMPAIGN_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "smoke_execution_campaign_candidate.json"
)
DEFAULT_AUTHORIZATION_CANDIDATE = (
    DEFAULT_SMOKE_OUTPUT_ROOT.parent
    / "protocol/smoke_execution_authorization_candidate.json"
)
DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE = (
    DEFAULT_SMOKE_OUTPUT_ROOT.parent
    / "protocol/smoke_execution_authorization_candidate.sha256"
)
DEFAULT_AUTHORIZATION_REPORT = (
    DEFAULT_SMOKE_OUTPUT_ROOT.parent
    / "audit/smoke_execution_authorization_candidate_report.json"
)
DEFAULT_AUTHORIZATION_INSTALL_PATH = Path(
    "/etc/edubench/exp54_smoke_authorization.json"
)
DEFAULT_STAGED_TRUST_ANCHOR_PATH = Path(
    "/etc/edubench/exp54_smoke_authorization.sha256.staged"
)
EXPECTED_EXECUTION_OUTPUT_ROOT = Path(
    "/home/jpang/edubench-eval-exp2/thesis_exp/outputs/"
    "exp54_rar_sft/rar_v2/smoke_runs"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PermissionError(f"{label} must remain absent")


def _expected_campaign_plan() -> dict[str, Any]:
    campaign_id = "exp54-smoke-b2e616e-v1"
    return {
        "status": "SMOKE_EXECUTION_CAMPAIGN_CANDIDATE_NOT_INSTALLED",
        "schema_version": (
            "exp54-smoke-execution-campaign-candidate-v1"
        ),
        "reviewed_smoke_guard_commit": REVIEWED_SMOKE_GUARD_COMMIT,
        "smoke_campaign_id": campaign_id,
        "run_id_by_arm": {
            arm: f"{campaign_id}-{arm.lower()}" for arm in SMOKE_ARMS
        },
        "max_invocations_per_arm": SMOKE_MAX_INVOCATIONS_PER_ARM,
        "training_user": "jpang",
        "training_uid": 1025,
        "training_group": "jpang",
        "training_gid": 1025,
        "filesystem_type": "xfs",
        "claim_root": str(DEFAULT_SMOKE_CLAIM_ROOT),
        "claim_root_directory_owner": "root",
        "claim_root_directory_group": "jpang",
        "claim_root_directory_mode": "0750",
        "claim_campaign_directory_owner": "root",
        "claim_campaign_directory_group": "jpang",
        "claim_campaign_directory_mode": "0770",
        "claim_campaign_must_be_append_only": True,
        "output_root": str(EXPECTED_EXECUTION_OUTPUT_ROOT),
        "output_campaign_directory_owner": "jpang",
        "output_campaign_directory_group": "jpang",
        "output_campaign_directory_mode": "0750",
        "output_final_arm_directories_must_be_absent": True,
        "authorization_install_path": str(
            DEFAULT_AUTHORIZATION_INSTALL_PATH
        ),
        "staged_trust_anchor_path": str(
            DEFAULT_STAGED_TRUST_ANCHOR_PATH
        ),
        "trust_anchor_install_path": str(
            TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH
        ),
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "hyperparameter_selection_allowed": False,
        "test_accessed": False,
        "training_used": False,
    }


def audit_smoke_authorization_candidate(
    *,
    campaign_plan_path: Path,
    smoke_lock_path: Path,
    smoke_plan_path: Path,
    configuration_lock_path: Path,
    materialized_lock_path: Path,
    authorization_path: Path,
    digest_path: Path,
    report_path: Path,
    authorization_install_path: Path = DEFAULT_AUTHORIZATION_INSTALL_PATH,
    trusted_digest_install_path: Path = (
        TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH
    ),
    staged_digest_install_path: Path = DEFAULT_STAGED_TRUST_ANCHOR_PATH,
    formal_trusted_digest_path: Path = TRUSTED_AUTHORIZATION_DIGEST_PATH,
    claim_root_path: Path = DEFAULT_SMOKE_CLAIM_ROOT,
) -> dict[str, Any]:
    """Reconstruct the exact candidate without importing its builder."""
    for path, label in (
        (authorization_install_path, "installed authorization"),
        (trusted_digest_install_path, "installed smoke trust anchor"),
        (staged_digest_install_path, "staged smoke trust anchor"),
        (formal_trusted_digest_path, "installed formal trust anchor"),
        (claim_root_path, "smoke claim root"),
    ):
        _require_absent(path, label=label)
    plan = _read_object(campaign_plan_path)
    if plan != _expected_campaign_plan():
        raise ValueError("campaign plan differs from independent contract")
    output_campaign_path = (
        Path(str(plan["output_root"])) / str(plan["smoke_campaign_id"])
    )
    _require_absent(
        output_campaign_path,
        label="smoke output campaign directory",
    )
    smoke_lock = _read_object(smoke_lock_path)
    if smoke_lock.get("status") != (
        "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("smoke package is not frozen")
    for field in (
        "trust_anchor_install_allowed",
        "forward_backward_allowed",
        "smoke_training_allowed",
        "formal_training_allowed",
        "dev_accessed",
        "test_accessed",
        "training_used",
    ):
        if smoke_lock.get(field) is not False:
            raise PermissionError(f"smoke lock crosses boundary at {field}")
    closure = smoke_runtime_source_closure()
    closure_digest = closure_sha256(closure)
    expected_upstream = {
        "smoke_plan_sha256": sha256_file(smoke_plan_path),
        "training_configuration_frozen_lock_sha256": sha256_file(
            configuration_lock_path
        ),
        "materialized_manifest_frozen_lock_sha256": sha256_file(
            materialized_lock_path
        ),
        "smoke_runtime_source_closure_sha256": closure_digest,
        "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
    }
    for field, expected in expected_upstream.items():
        if smoke_lock.get(field) != expected:
            raise ValueError(f"smoke lock differs at {field}")
    if smoke_lock.get("smoke_runtime_source_closure") != closure:
        raise ValueError("smoke runtime closure differs")
    if plan["output_root"] != smoke_lock.get("smoke_output_root"):
        raise ValueError("campaign output root differs from frozen smoke lock")
    if plan["claim_root"] != smoke_lock.get("smoke_claim_root"):
        raise ValueError("campaign claim root differs from frozen smoke lock")

    campaign_id = str(plan["smoke_campaign_id"])
    run_ids = dict(plan["run_id_by_arm"])
    output_root = Path(str(plan["output_root"]))
    output_dirs = {
        arm: str(output_root / campaign_id / run_ids[arm])
        for arm in SMOKE_ARMS
    }
    expected_authorization = {
        "schema_version": "exp54-smoke-execution-authorization-v1",
        "authorization_mode": "smoke",
        "review_verdict": "SMOKE_PACKAGE_PASS",
        "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
        "smoke_package_frozen_lock_sha256": sha256_file(smoke_lock_path),
        "training_configuration_frozen_lock_sha256": sha256_file(
            configuration_lock_path
        ),
        "smoke_plan_sha256": sha256_file(smoke_plan_path),
        "materialized_manifest_frozen_lock_sha256": sha256_file(
            materialized_lock_path
        ),
        "smoke_runtime_source_closure_sha256": closure_digest,
        "smoke_schema_version": SMOKE_SCHEMA_VERSION,
        "allowed_arms": list(SMOKE_ARMS),
        "allowed_seed": SMOKE_SOURCE_SEED,
        "optimizer_steps_per_arm": SMOKE_OPTIMIZER_STEPS_PER_ARM,
        "smoke_campaign_id": campaign_id,
        "run_id_by_arm": run_ids,
        "output_dir_by_arm": output_dirs,
        "max_invocations_per_arm": SMOKE_MAX_INVOCATIONS_PER_ARM,
        "claim_root": str(plan["claim_root"]),
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "hyperparameter_selection_allowed": False,
    }
    authorization = _read_object(authorization_path)
    if authorization != expected_authorization:
        differing = sorted(
            key
            for key in set(authorization) | set(expected_authorization)
            if authorization.get(key) != expected_authorization.get(key)
        )
        raise ValueError(f"authorization differs: {differing}")
    authorization_digest = hashlib.sha256(
        authorization_path.read_bytes()
    ).hexdigest()
    if digest_path.read_text(encoding="ascii") != authorization_digest + "\n":
        raise ValueError("authorization digest candidate differs")

    report = _read_object(report_path)
    source_paths = {
        "authorization_builder": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/"
            "build_smoke_authorization_candidate.py"
        ),
        "authorization_auditor": Path(__file__),
        "installed_authorization_auditor": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/"
            "audit_installed_smoke_authorization.py"
        ),
        "installation_plan": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/"
            "SMOKE_AUTHORIZATION_INSTALLATION_PLAN.md"
        ),
        "campaign_plan": campaign_plan_path,
    }
    expected_report = {
        "status": "SMOKE_AUTHORIZATION_CANDIDATE_NOT_INSTALLED",
        "reviewed_smoke_guard_commit": REVIEWED_SMOKE_GUARD_COMMIT,
        "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
        "authorization_candidate_sha256": authorization_digest,
        "authorization_digest_candidate_sha256": sha256_file(digest_path),
        "smoke_package_frozen_lock_sha256": sha256_file(smoke_lock_path),
        "training_configuration_frozen_lock_sha256": sha256_file(
            configuration_lock_path
        ),
        "smoke_plan_sha256": sha256_file(smoke_plan_path),
        "materialized_manifest_frozen_lock_sha256": sha256_file(
            materialized_lock_path
        ),
        "smoke_runtime_source_closure": closure,
        "smoke_runtime_source_closure_sha256": closure_digest,
        "smoke_runtime_source_closure_file_count": len(closure),
        "campaign_plan_sha256": sha256_file(campaign_plan_path),
        "smoke_campaign_id": campaign_id,
        "run_id_by_arm": run_ids,
        "output_dir_by_arm": output_dirs,
        "claim_root": str(plan["claim_root"]),
        "authorization_install_path": str(authorization_install_path),
        "trust_anchor_install_path": str(trusted_digest_install_path),
        "staged_trust_anchor_path": str(staged_digest_install_path),
        "source_hashes": {
            name: sha256_file(path)
            for name, path in sorted(source_paths.items())
        },
        "authorization_fields_exactly_reconstructed": True,
        "authorization_digest_matches_exact_bytes": True,
        "installed_authorization_lstat_checked_absent": True,
        "installed_trust_anchor_lstat_checked_absent": True,
        "staged_trust_anchor_lstat_checked_absent": True,
        "formal_trust_anchor_lstat_checked_absent": True,
        "claim_root_lstat_checked_absent": True,
        "output_campaign_lstat_checked_absent": True,
        "preactivation_preflight_required": True,
        "active_anchor_must_be_absent_during_preflight": True,
        "formal_anchor_must_be_absent_during_preflight": True,
        "trust_anchor_install_allowed": False,
        "claim_creation_allowed": False,
        "model_loading_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    if report != expected_report:
        differing = sorted(
            key
            for key in set(report) | set(expected_report)
            if report.get(key) != expected_report.get(key)
        )
        raise ValueError(f"authorization report differs: {differing}")
    return {
        "status": "SMOKE_AUTHORIZATION_CANDIDATE_AUDIT_PASS",
        "authorization_candidate_sha256": authorization_digest,
        "authorization_fields_exactly_reconstructed": True,
        "authorization_digest_matches_exact_bytes": True,
        "runtime_source_closure_verified": True,
        "installation_targets_absent": True,
        "trust_anchor_install_allowed": False,
        "claim_creation_allowed": False,
        "model_loading_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-plan",
        type=Path,
        default=DEFAULT_CAMPAIGN_PLAN,
    )
    parser.add_argument(
        "--smoke-package-lock",
        type=Path,
        default=DEFAULT_SMOKE_FROZEN_LOCK,
    )
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
    parser.add_argument(
        "--training-configuration-frozen-lock",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    )
    parser.add_argument(
        "--materialized-manifest-frozen-lock",
        type=Path,
        default=DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    )
    parser.add_argument(
        "--authorization-candidate",
        type=Path,
        default=DEFAULT_AUTHORIZATION_CANDIDATE,
    )
    parser.add_argument(
        "--authorization-digest-candidate",
        type=Path,
        default=DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_AUTHORIZATION_REPORT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_smoke_authorization_candidate(
        campaign_plan_path=args.campaign_plan,
        smoke_lock_path=args.smoke_package_lock,
        smoke_plan_path=args.smoke_plan,
        configuration_lock_path=(
            args.training_configuration_frozen_lock
        ),
        materialized_lock_path=args.materialized_manifest_frozen_lock,
        authorization_path=args.authorization_candidate,
        digest_path=args.authorization_digest_candidate,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
