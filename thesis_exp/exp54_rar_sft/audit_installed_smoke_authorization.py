"""Preflight a staged Exp54 smoke authorization before activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.audit_smoke_authorization_candidate import (
    DEFAULT_AUTHORIZATION_CANDIDATE,
    DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_STAGED_TRUST_ANCHOR_PATH,
    _expected_campaign_plan,
)
from thesis_exp.exp54_rar_sft.authorization_guard import (
    TRUSTED_AUTHORIZATION_DIGEST_PATH,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
    _directory_has_append_only_flag,
    verify_smoke_authorization,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    SMOKE_ARMS,
)


DEFAULT_AUTHORIZATION_INSTALL_PATH = Path(
    "/etc/edubench/exp54_smoke_authorization.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_regular_root_file(
    path: Path,
    *,
    expected_bytes: bytes,
) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{path}: expected a real regular file")
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise PermissionError(f"{path}: expected root:root ownership")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise PermissionError(f"{path}: expected mode 0444")
    if path.read_bytes() != expected_bytes:
        raise ValueError(f"{path}: installed bytes differ from candidate")


def _require_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{path}: expected a real directory")
    if metadata.st_uid != uid or metadata.st_gid != gid:
        raise PermissionError(f"{path}: directory ownership differs")
    if stat.S_IMODE(metadata.st_mode) != mode:
        raise PermissionError(f"{path}: directory mode differs")


def _require_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PermissionError(f"{label} must remain absent")


def audit_installed_smoke_authorization(
    *,
    campaign_plan_path: Path,
    authorization_candidate_path: Path,
    digest_candidate_path: Path,
    authorization_install_path: Path,
    staged_digest_install_path: Path,
    active_digest_install_path: Path,
    formal_digest_install_path: Path,
    config_path: Path,
    smoke_plan_path: Path,
    smoke_lock_path: Path,
    configuration_lock_path: Path,
    claim_root_path: Path | None = None,
    output_root_path: Path | None = None,
) -> dict[str, Any]:
    """Verify staged installation while both production anchors are absent."""
    _require_absent(
        active_digest_install_path,
        label="active smoke trust anchor",
    )
    _require_absent(
        formal_digest_install_path,
        label="formal-training trust anchor",
    )
    plan = _read_object(campaign_plan_path)
    if plan != _expected_campaign_plan():
        raise ValueError("pre-activation campaign plan differs")
    authorization_candidate_bytes = authorization_candidate_path.read_bytes()
    digest_candidate_bytes = digest_candidate_path.read_bytes()
    expected_authorization_digest = hashlib.sha256(
        authorization_candidate_bytes
    ).hexdigest()
    if digest_candidate_bytes != (
        expected_authorization_digest + "\n"
    ).encode("ascii"):
        raise ValueError("candidate digest does not bind candidate bytes")
    _require_regular_root_file(
        authorization_install_path,
        expected_bytes=authorization_candidate_bytes,
    )
    _require_regular_root_file(
        staged_digest_install_path,
        expected_bytes=digest_candidate_bytes,
    )

    campaign_id = str(plan["smoke_campaign_id"])
    claim_root = (
        Path(str(plan["claim_root"]))
        if claim_root_path is None
        else claim_root_path
    )
    claim_campaign = claim_root / campaign_id
    _require_directory(claim_root, uid=0, gid=1025, mode=0o750)
    _require_directory(claim_campaign, uid=0, gid=1025, mode=0o770)
    if not _directory_has_append_only_flag(claim_campaign):
        raise PermissionError("claim campaign directory is not append-only")
    for arm in SMOKE_ARMS:
        _require_absent(
            claim_campaign / f"{arm.lower()}.claimed",
            label=f"{arm} claim",
        )

    output_root = (
        Path(str(plan["output_root"]))
        if output_root_path is None
        else output_root_path
    )
    output_campaign = output_root / campaign_id
    _require_directory(output_root, uid=1025, gid=1025, mode=0o750)
    _require_directory(output_campaign, uid=1025, gid=1025, mode=0o750)
    for arm in SMOKE_ARMS:
        _require_absent(
            output_campaign / str(plan["run_id_by_arm"][arm]),
            label=f"{arm} final output directory",
        )

    context_hashes: set[tuple[str, str, str]] = set()
    for arm in SMOKE_ARMS:
        context = verify_smoke_authorization(
            authorization_path=authorization_install_path,
            config_path=config_path,
            smoke_plan_path=smoke_plan_path,
            smoke_package_lock_path=smoke_lock_path,
            training_configuration_frozen_lock_path=(
                configuration_lock_path
            ),
            arm=arm,
            trusted_digest_path=staged_digest_install_path,
        )
        context_hashes.add(
            (
                context.authorization_sha256,
                context.config_sha256,
                context.smoke_lock_sha256,
            )
        )
    if len(context_hashes) != 1:
        raise AssertionError("arm authorization contexts differ")
    return {
        "status": "SMOKE_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS",
        "reviewed_smoke_guard_commit": plan[
            "reviewed_smoke_guard_commit"
        ],
        "smoke_campaign_id": campaign_id,
        "arms_verified_without_claim": list(SMOKE_ARMS),
        "authorization_sha256": expected_authorization_digest,
        "active_smoke_anchor_absent": True,
        "formal_training_anchor_absent": True,
        "staged_digest_verified": True,
        "claim_root_verified": True,
        "claim_campaign_append_only_verified": True,
        "all_claims_absent": True,
        "fixed_output_parents_verified": True,
        "all_final_output_directories_absent": True,
        "activation_performed": False,
        "claim_created": False,
        "model_loaded": False,
        "forward_backward_executed": False,
        "smoke_training_executed": False,
        "formal_training_executed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-plan",
        type=Path,
        default=DEFAULT_CAMPAIGN_PLAN,
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
        "--authorization-install-path",
        type=Path,
        default=DEFAULT_AUTHORIZATION_INSTALL_PATH,
    )
    parser.add_argument(
        "--staged-digest-install-path",
        type=Path,
        default=DEFAULT_STAGED_TRUST_ANCHOR_PATH,
    )
    parser.add_argument(
        "--active-digest-install-path",
        type=Path,
        default=TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
    )
    parser.add_argument(
        "--formal-digest-install-path",
        type=Path,
        default=TRUSTED_AUTHORIZATION_DIGEST_PATH,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
    parser.add_argument(
        "--smoke-package-lock",
        type=Path,
        default=DEFAULT_SMOKE_FROZEN_LOCK,
    )
    parser.add_argument(
        "--training-configuration-frozen-lock",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_installed_smoke_authorization(
        campaign_plan_path=args.campaign_plan,
        authorization_candidate_path=args.authorization_candidate,
        digest_candidate_path=args.authorization_digest_candidate,
        authorization_install_path=args.authorization_install_path,
        staged_digest_install_path=args.staged_digest_install_path,
        active_digest_install_path=args.active_digest_install_path,
        formal_digest_install_path=args.formal_digest_install_path,
        config_path=args.config,
        smoke_plan_path=args.smoke_plan,
        smoke_lock_path=args.smoke_package_lock,
        configuration_lock_path=(
            args.training_configuration_frozen_lock
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
