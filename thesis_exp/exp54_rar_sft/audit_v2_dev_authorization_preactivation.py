"""Read-only pre-activation audit for the Exp54 V2 dev campaign."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.v2_dev_authorization_guard import (
    ARMS,
    EPOCHS,
    SEEDS,
    DEFAULT_V2_AUTHORIZATION_PATH,
    DEFAULT_V2_CLAIM_ROOT,
    DEFAULT_V2_OUTPUT_ROOT,
    DEFAULT_V2_STAGED_DIGEST_PATH,
    DEFAULT_V2_TRAINING_ROOT,
    DEFAULT_V2_TRUSTED_DIGEST_PATH,
    DEFAULT_V2_WHEEL_ROOT,
    _read_regular_once,
    _require_private_directory,
    authenticate_v2_dev_authorization,
)


DEFAULT_FORMAL_TRAINING_ANCHOR = Path(
    "/etc/edubench/exp54_authorization.sha256"
)
PASS_STATUS = "V2_DEV_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS"


def _require_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PermissionError(f"{path}: must be absent before activation")


def _require_readonly_owned_regular(path: Path) -> None:
    material = _read_regular_once(path)
    if material.uid != os.getuid():
        raise PermissionError(f"{path}: owner differs")
    if stat.S_IMODE(material.mode) != 0o444:
        raise PermissionError(f"{path}: expected exact mode 0444")


def audit_v2_dev_authorization_preactivation(
    *,
    authorization_path: Path = DEFAULT_V2_AUTHORIZATION_PATH,
    staged_digest_path: Path = DEFAULT_V2_STAGED_DIGEST_PATH,
    active_digest_path: Path = DEFAULT_V2_TRUSTED_DIGEST_PATH,
    formal_training_anchor: Path = DEFAULT_FORMAL_TRAINING_ANCHOR,
    claim_root: Path = DEFAULT_V2_CLAIM_ROOT,
    training_root: Path = DEFAULT_V2_TRAINING_ROOT,
    output_root: Path = DEFAULT_V2_OUTPUT_ROOT,
    wheel_root: Path = DEFAULT_V2_WHEEL_ROOT,
    repository_commit: str | None = None,
) -> dict[str, Any]:
    """Validate the complete campaign while production remains disabled."""
    _require_absent(active_digest_path)
    _require_absent(formal_training_anchor)
    _require_readonly_owned_regular(authorization_path)
    _require_readonly_owned_regular(staged_digest_path)
    _require_private_directory(claim_root)
    _require_private_directory(output_root)

    first = authenticate_v2_dev_authorization(
        arm=ARMS[0],
        seed=SEEDS[0],
        epoch=EPOCHS[0],
        authorization_path=authorization_path,
        trusted_digest_path=staged_digest_path,
        claim_root=claim_root,
        training_root=training_root,
        output_root=output_root,
        wheel_root=wheel_root,
        repository_commit=repository_commit,
    )
    campaign_dir = claim_root / first.campaign_id
    _require_private_directory(campaign_dir)
    if any(campaign_dir.iterdir()):
        raise PermissionError("V2 campaign claim directory is not empty")

    verified_task_ids = []
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in EPOCHS:
                verified = authenticate_v2_dev_authorization(
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                    authorization_path=authorization_path,
                    trusted_digest_path=staged_digest_path,
                    claim_root=claim_root,
                    training_root=training_root,
                    output_root=output_root,
                    wheel_root=wheel_root,
                    repository_commit=repository_commit,
                )
                if verified.campaign_id != first.campaign_id:
                    raise PermissionError(
                        "V2 authorization campaign changed"
                    )
                _require_absent(verified.output_dir)
                verified_task_ids.append(verified.task_id)

    _require_absent(active_digest_path)
    if any(campaign_dir.iterdir()):
        raise PermissionError("preflight created a V2 campaign claim")
    return {
        "status": PASS_STATUS,
        "campaign_id": first.campaign_id,
        "authorization_sha256": first.authorization_sha256,
        "verified_task_count": len(verified_task_ids),
        "active_digest_path": str(active_digest_path),
        "staged_digest_path": str(staged_digest_path),
        "activation_performed": False,
        "claim_created": False,
        "output_created": False,
        "dev_loader_called": False,
        "model_loaded": False,
        "cuda_initialized": False,
        "test_accessed": False,
    }


def main() -> None:
    report = audit_v2_dev_authorization_preactivation()
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
