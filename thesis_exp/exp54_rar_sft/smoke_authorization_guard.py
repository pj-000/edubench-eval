"""Fail-closed external authorization for one-step Exp54 smoke execution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.authorization_guard import (
    EXPECTED_RUNTIME_SOURCE_NAMES,
    closure_sha256,
    read_json_object,
    read_trusted_authorization_digest,
    runtime_source_closure,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    SMOKE_ARMS,
    SMOKE_OPTIMIZER_STEPS_PER_ARM,
    SMOKE_SCHEMA_VERSION,
    SMOKE_SOURCE_SEED,
)


TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH = Path(
    "/etc/edubench/exp54_smoke_authorization.sha256"
)
SMOKE_RUNTIME_SOURCE_PATHS = {
    "smoke_authorization_guard": Path(__file__),
    "smoke_training_entrypoint": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/train_rar_sft_smoke.py"
    ),
    "smoke_training_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/smoke_training_contract.py"
    ),
    "smoke_launcher": (
        REPO_ROOT / "thesis_exp/scripts/run_exp54_rar_sft_smoke.sh"
    ),
}
EXPECTED_SMOKE_RUNTIME_SOURCE_NAMES = {
    *(f"formal::{name}" for name in EXPECTED_RUNTIME_SOURCE_NAMES),
    *SMOKE_RUNTIME_SOURCE_PATHS.keys(),
}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def smoke_runtime_source_closure() -> dict[str, str]:
    formal = {
        f"formal::{name}": digest
        for name, digest in runtime_source_closure().items()
    }
    smoke_missing = [
        str(path)
        for path in SMOKE_RUNTIME_SOURCE_PATHS.values()
        if not path.is_file()
    ]
    if smoke_missing:
        raise FileNotFoundError(
            f"smoke runtime source closure is incomplete: {smoke_missing}"
        )
    closure = {
        **formal,
        **{
            name: sha256_file(path)
            for name, path in sorted(SMOKE_RUNTIME_SOURCE_PATHS.items())
        },
    }
    if set(closure) != EXPECTED_SMOKE_RUNTIME_SOURCE_NAMES:
        raise RuntimeError("smoke runtime source closure definition differs")
    if len(closure) != 20:
        raise RuntimeError("smoke runtime source closure must contain 20 files")
    return dict(sorted(closure.items()))


def verify_smoke_authorization(
    *,
    authorization_path: Path | None,
    config_path: Path,
    smoke_package_lock_path: Path,
    training_configuration_frozen_lock_path: Path,
    arm: str,
    trusted_digest_path: Path = (
        TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH
    ),
    require_root_owned_digest: bool = True,
) -> dict[str, Any]:
    """Authenticate a future review artifact permitting one smoke step."""
    if authorization_path is None:
        raise PermissionError("smoke execution is not authorized")
    if arm not in SMOKE_ARMS:
        raise PermissionError(f"unsupported smoke arm: {arm}")

    trusted_digest = read_trusted_authorization_digest(
        trusted_digest_path,
        require_root_owned=require_root_owned_digest,
    )
    if sha256_file(authorization_path) != trusted_digest:
        raise PermissionError(
            "smoke authorization does not match the external digest"
        )
    authorization = read_json_object(authorization_path)
    smoke_lock = read_json_object(smoke_package_lock_path)
    configuration_lock = read_json_object(
        training_configuration_frozen_lock_path
    )

    if (
        authorization.get("schema_version")
        != "exp54-smoke-execution-authorization-v1"
    ):
        raise PermissionError("smoke authorization schema differs")
    if authorization.get("authorization_mode") != "smoke":
        raise PermissionError("authorization mode is not smoke")
    if authorization.get("review_verdict") != "SMOKE_PACKAGE_PASS":
        raise PermissionError("smoke authorization lacks the review verdict")
    if not _COMMIT_RE.fullmatch(
        str(authorization.get("reviewed_smoke_package_commit", ""))
    ):
        raise PermissionError("reviewed smoke-package commit is invalid")
    if smoke_lock.get("status") != (
        "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("smoke package is not frozen")
    if configuration_lock.get("status") != (
        "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("training configuration is not frozen")
    for name, artifact in (
        ("smoke package", smoke_lock),
        ("training configuration", configuration_lock),
    ):
        if (
            artifact.get("trust_anchor_install_allowed")
            or artifact.get("forward_backward_allowed")
            or artifact.get("smoke_training_allowed")
            or artifact.get("formal_training_allowed")
            or artifact.get("dev_accessed")
            or artifact.get("test_accessed")
            or artifact.get("training_used")
        ):
            raise PermissionError(f"{name} crosses its authorization boundary")

    expected_bindings = {
        "smoke_package_frozen_lock_sha256": sha256_file(
            smoke_package_lock_path
        ),
        "training_configuration_frozen_lock_sha256": sha256_file(
            training_configuration_frozen_lock_path
        ),
        "smoke_plan_sha256": smoke_lock.get("smoke_plan_sha256"),
        "materialized_manifest_frozen_lock_sha256": smoke_lock.get(
            "materialized_manifest_frozen_lock_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"smoke authorization does not bind {field}")
    config_sha256 = sha256_file(config_path)
    if configuration_lock.get("configuration_sha256") != config_sha256:
        raise PermissionError("frozen configuration does not bind this config")
    if smoke_lock.get("configuration_sha256") != config_sha256:
        raise PermissionError("smoke package does not bind this config")

    closure = smoke_runtime_source_closure()
    closure_digest = closure_sha256(closure)
    if smoke_lock.get("smoke_runtime_source_closure") != closure:
        raise PermissionError("smoke package runtime closure differs")
    if (
        smoke_lock.get("smoke_runtime_source_closure_sha256")
        != closure_digest
    ):
        raise PermissionError("smoke package runtime closure digest differs")
    if authorization.get("smoke_runtime_source_closure_sha256") != (
        closure_digest
    ):
        raise PermissionError("smoke authorization runtime closure differs")
    if authorization.get("smoke_schema_version") != SMOKE_SCHEMA_VERSION:
        raise PermissionError("smoke authorization contract version differs")
    if authorization.get("allowed_arms") != list(SMOKE_ARMS):
        raise PermissionError("smoke authorization arms differ")
    if authorization.get("allowed_seed") != SMOKE_SOURCE_SEED:
        raise PermissionError("smoke authorization seed differs")
    if (
        authorization.get("optimizer_steps_per_arm")
        != SMOKE_OPTIMIZER_STEPS_PER_ARM
    ):
        raise PermissionError("smoke authorization step budget differs")
    if authorization.get("formal_training_allowed") is not False:
        raise PermissionError("smoke authorization permits formal training")
    if authorization.get("dev_accessed") is not False:
        raise PermissionError("smoke authorization claims dev access")
    if authorization.get("test_accessed") is not False:
        raise PermissionError("smoke authorization claims test access")
    if authorization.get("hyperparameter_selection_allowed") is not False:
        raise PermissionError("smoke authorization permits model selection")
    return authorization
