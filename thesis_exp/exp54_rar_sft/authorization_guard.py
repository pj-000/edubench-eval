"""Independent external-digest authorization gate for Exp54 training.

This module deliberately uses only the Python standard library. It does not
trust experiment hashing or JSON helpers. A repository-local pair of
self-consistent lock files is insufficient: the exact authorization-file
SHA-256 must also arrive through a repository-external runtime channel.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_AUTHORIZATION_DIGEST_PATH = Path(
    "/etc/edubench/exp54_authorization.sha256"
)
RUNTIME_SOURCE_PATHS = {
    "authorization_guard": Path(__file__),
    "training_entrypoint": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/train_rar_sft.py"
    ),
    "block_loss_and_collator": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
    ),
    "training_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/training_contract.py"
    ),
    "inference_contract": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/inference_contract.py"
    ),
    "manifest_io_and_split_guard": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/audit_rar0_alignment.py"
    ),
    "exp54_package_init": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/__init__.py"
    ),
    "prompt_cleaning": (
        REPO_ROOT / "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py"
    ),
    "exp02_package_init": (
        REPO_ROOT / "thesis_exp/src/edujudge/exp02/__init__.py"
    ),
    "shared_io": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/io.py"
    ),
    "text_normalization": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/text_norm.py"
    ),
    "utils_package_init": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/__init__.py"
    ),
    "thesis_package_init": REPO_ROOT / "thesis_exp/__init__.py",
    "thesis_src_package_init": (
        REPO_ROOT / "thesis_exp/src/__init__.py"
    ),
    "edujudge_package_init": (
        REPO_ROOT / "thesis_exp/src/edujudge/__init__.py"
    ),
    "launcher": REPO_ROOT / "thesis_exp/scripts/run_exp54_rar_sft.sh",
}
EXPECTED_RUNTIME_SOURCE_NAMES = {
    "authorization_guard",
    "training_entrypoint",
    "block_loss_and_collator",
    "training_contract",
    "inference_contract",
    "manifest_io_and_split_guard",
    "exp54_package_init",
    "prompt_cleaning",
    "exp02_package_init",
    "shared_io",
    "text_normalization",
    "utils_package_init",
    "thesis_package_init",
    "thesis_src_package_init",
    "edujudge_package_init",
    "launcher",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PermissionError(f"{path}: authorization input must be an object")
    return value


def runtime_source_closure() -> dict[str, str]:
    if set(RUNTIME_SOURCE_PATHS) != EXPECTED_RUNTIME_SOURCE_NAMES:
        raise RuntimeError("runtime source closure definition differs")
    missing = [
        str(path)
        for path in RUNTIME_SOURCE_PATHS.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"runtime source closure is incomplete: {missing}")
    return {
        name: sha256_file(path)
        for name, path in sorted(RUNTIME_SOURCE_PATHS.items())
    }


def closure_sha256(closure: dict[str, str]) -> str:
    payload = json.dumps(
        sorted(closure.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_false_state(value: Any, *, source: str) -> None:
    expected = {
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    if value != expected:
        raise PermissionError(f"{source} sealed-data state is invalid")


def read_trusted_authorization_digest(
    path: Path = TRUSTED_AUTHORIZATION_DIGEST_PATH,
    *,
    require_root_owned: bool = True,
) -> str:
    """Read an immutable repository-external trust anchor."""
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise PermissionError(
            "trusted external authorization digest file is unavailable"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise PermissionError(
            "trusted authorization digest must be a regular non-symlink file"
        )
    if require_root_owned and metadata.st_uid != 0:
        raise PermissionError(
            "trusted authorization digest must be owned by root"
        )
    if metadata.st_mode & 0o022:
        raise PermissionError(
            "trusted authorization digest must not be group/other writable"
        )
    digest = path.read_text(encoding="ascii").strip()
    if not _SHA256_RE.fullmatch(digest):
        raise PermissionError("trusted authorization digest is invalid")
    return digest


def verify_external_authorization(
    *,
    authorization_lock_path: Path | None,
    config_path: Path,
    frozen_lock_path: Path,
    training_config_lock_path: Path,
    arm: str,
    seed: int,
    trusted_digest_path: Path = TRUSTED_AUTHORIZATION_DIGEST_PATH,
    require_root_owned_digest: bool = True,
) -> dict[str, Any]:
    """Authenticate and validate a future formal-training authorization."""
    if authorization_lock_path is None:
        raise PermissionError(
            "training is not authorized: an external authorization is required"
        )
    trusted_digest = read_trusted_authorization_digest(
        trusted_digest_path,
        require_root_owned=require_root_owned_digest,
    )
    actual_authorization_digest = sha256_file(authorization_lock_path)
    if actual_authorization_digest != trusted_digest:
        raise PermissionError(
            "authorization file does not match the trusted external digest"
        )

    authorization = read_json_object(authorization_lock_path)
    configuration_lock = read_json_object(training_config_lock_path)
    if authorization.get("schema_version") != "exp54-external-authorization-v1":
        raise PermissionError("authorization schema differs")
    if authorization.get("authorization_mode") != "formal":
        raise PermissionError("authorization does not permit formal training")
    if authorization.get("review_verdict") != "TRAINING_CONFIGURATION_PASS":
        raise PermissionError("authorization lacks the required review verdict")
    if not _COMMIT_RE.fullmatch(
        str(authorization.get("reviewed_training_configuration_commit", ""))
    ):
        raise PermissionError("authorization reviewed commit is invalid")

    if (
        configuration_lock.get("status")
        != "TRAINING_CONFIGURATION_CANDIDATE_AUDITED_NOT_AUTHORIZED"
        or configuration_lock.get("smoke_training_allowed")
        or configuration_lock.get("formal_training_allowed")
        or configuration_lock.get("dev_accessed")
        or configuration_lock.get("test_accessed")
        or configuration_lock.get("training_used")
    ):
        raise PermissionError("training-configuration audit lock is invalid")

    config_sha256 = sha256_file(config_path)
    frozen_sha256 = sha256_file(frozen_lock_path)
    config_lock_sha256 = sha256_file(training_config_lock_path)
    expected_bindings = {
        "configuration_sha256": config_sha256,
        "training_configuration_lock_sha256": config_lock_sha256,
        "candidate_report_sha256": configuration_lock.get(
            "candidate_report_sha256"
        ),
        "frozen_manifest_lock_sha256": frozen_sha256,
    }
    for field, expected in expected_bindings.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"authorization does not bind {field}")
    if configuration_lock.get("configuration_sha256") != config_sha256:
        raise PermissionError("configuration audit does not bind this config")
    if configuration_lock.get("manifest_frozen_lock_sha256") != frozen_sha256:
        raise PermissionError(
            "configuration audit does not bind this manifest freeze"
        )

    actual_closure = runtime_source_closure()
    actual_closure_sha256 = closure_sha256(actual_closure)
    if configuration_lock.get("runtime_source_closure") != actual_closure:
        raise PermissionError(
            "configuration audit does not bind the runtime source closure"
        )
    if (
        configuration_lock.get("runtime_source_closure_sha256")
        != actual_closure_sha256
    ):
        raise PermissionError(
            "configuration audit source-closure digest differs"
        )
    if authorization.get("runtime_source_closure_sha256") != actual_closure_sha256:
        raise PermissionError(
            "authorization does not bind the runtime source closure"
        )
    if arm not in authorization.get("allowed_arms", []):
        raise PermissionError(f"authorization does not permit arm {arm}")
    if seed not in authorization.get("allowed_seeds", []):
        raise PermissionError(f"authorization does not permit seed {seed}")
    _require_false_state(
        authorization.get("sealed_data_state"),
        source="authorization",
    )
    return authorization
