"""Read-once authorization and one-use claims for Exp54 smoke execution."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.authorization_guard import (
    EXPECTED_RUNTIME_SOURCE_NAMES,
    closure_sha256,
    runtime_source_closure,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_SMOKE_CLAIM_ROOT,
    DEFAULT_SMOKE_OUTPUT_ROOT,
    REVIEWED_SMOKE_PACKAGE_COMMIT,
    SMOKE_ARMS,
    SMOKE_MAX_INVOCATIONS_PER_ARM,
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_FS_IOC_GETFLAGS = 0x80086601
_FS_APPEND_FL = 0x00000020
_MAX_PUBLIC_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ReadOnceBytes:
    path: Path
    payload: bytes
    sha256: str
    mode: int
    uid: int
    gid: int
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ReadOnceJson:
    file: ReadOnceBytes
    value: dict[str, Any]


@dataclass(frozen=True)
class AuthenticatedSmokeContext:
    authorization: dict[str, Any]
    authorization_sha256: str
    config: dict[str, Any]
    config_sha256: str
    smoke_plan: dict[str, Any]
    smoke_plan_sha256: str
    smoke_lock: dict[str, Any]
    smoke_lock_sha256: str
    training_configuration_lock: dict[str, Any]
    training_configuration_lock_sha256: str


@dataclass(frozen=True)
class SmokeInvocationClaim:
    campaign_id: str
    arm: str
    run_id: str
    claim_path: Path
    output_dir: Path


def _open_read_once(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required for smoke authorization")
    flags |= nofollow
    try:
        return os.open(path, flags)
    except OSError as exc:
        if exc.errno in {
            errno.ELOOP,
            errno.ENXIO,
            errno.ENOTDIR,
            errno.ENOENT,
        }:
            raise PermissionError(
                f"{path}: read-once regular file is unavailable"
            ) from exc
        raise


def read_regular_bytes_once(
    path: Path,
    *,
    max_bytes: int = _MAX_PUBLIC_BYTES,
) -> ReadOnceBytes:
    """Read one regular non-symlink inode once; hash and users share payload."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    fd = _open_read_once(path)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"{path}: expected a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise PermissionError(f"{path}: file exceeds read-once limit")
        payload = b"".join(chunks)
        final_metadata = os.fstat(fd)
        initial_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        final_identity = (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        )
        if initial_identity != final_identity or len(payload) != metadata.st_size:
            raise PermissionError(f"{path}: file changed during read-once input")
        return ReadOnceBytes(
            path=path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            mode=metadata.st_mode,
            uid=metadata.st_uid,
            gid=metadata.st_gid,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
    finally:
        os.close(fd)


def read_regular_json_once(path: Path) -> ReadOnceJson:
    material = read_regular_bytes_once(path)
    try:
        value = json.loads(material.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(f"{path}: invalid authenticated JSON") from exc
    if not isinstance(value, dict):
        raise PermissionError(f"{path}: authenticated JSON must be an object")
    return ReadOnceJson(file=material, value=value)


def _require_path_still_same(material: ReadOnceBytes) -> None:
    try:
        metadata = material.path.lstat()
    except FileNotFoundError as exc:
        raise PermissionError(
            f"{material.path}: authenticated path disappeared"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(
            f"{material.path}: authenticated path type changed"
        )
    observed = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    expected = (
        material.device,
        material.inode,
        material.size,
        material.mtime_ns,
    )
    if observed != expected:
        raise PermissionError(
            f"{material.path}: authenticated path changed during verification"
        )


def _read_trusted_digest_once(
    path: Path,
    *,
    require_root_owned: bool,
) -> str:
    material = read_regular_bytes_once(path, max_bytes=1024)
    if require_root_owned and material.uid != 0:
        raise PermissionError("smoke trust anchor must be owned by root")
    if material.mode & 0o022:
        raise PermissionError(
            "smoke trust anchor must not be group/other writable"
        )
    try:
        digest = material.payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PermissionError("smoke trust anchor is not ASCII") from exc
    if not _SHA256_RE.fullmatch(digest):
        raise PermissionError("smoke trust anchor digest is invalid")
    return digest


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


def _validated_campaign(
    authorization: dict[str, Any],
    *,
    expected_claim_root: Path,
    expected_output_root: Path,
) -> None:
    campaign_id = authorization.get("smoke_campaign_id")
    if not isinstance(campaign_id, str) or not _CAMPAIGN_RE.fullmatch(
        campaign_id
    ):
        raise PermissionError("smoke campaign ID is invalid")
    if authorization.get("max_invocations_per_arm") != (
        SMOKE_MAX_INVOCATIONS_PER_ARM
    ):
        raise PermissionError("smoke invocation budget differs")
    if authorization.get("claim_root") != str(expected_claim_root):
        raise PermissionError("smoke claim root differs")

    run_ids = authorization.get("run_id_by_arm")
    output_dirs = authorization.get("output_dir_by_arm")
    if not isinstance(run_ids, dict) or set(run_ids) != set(SMOKE_ARMS):
        raise PermissionError("smoke run-ID mapping differs")
    if not isinstance(output_dirs, dict) or set(output_dirs) != set(SMOKE_ARMS):
        raise PermissionError("smoke output-directory mapping differs")
    if len(set(run_ids.values())) != len(SMOKE_ARMS):
        raise PermissionError("smoke run IDs must be unique")
    if len(set(output_dirs.values())) != len(SMOKE_ARMS):
        raise PermissionError("smoke output directories must be unique")
    for arm in SMOKE_ARMS:
        run_id = run_ids[arm]
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise PermissionError(f"{arm}: smoke run ID is invalid")
        expected_output = expected_output_root / campaign_id / run_id
        if output_dirs[arm] != str(expected_output):
            raise PermissionError(f"{arm}: smoke output directory differs")


def verify_smoke_authorization(
    *,
    authorization_path: Path | None,
    config_path: Path,
    smoke_plan_path: Path,
    smoke_package_lock_path: Path,
    training_configuration_frozen_lock_path: Path,
    arm: str,
    trusted_digest_path: Path = TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
    require_root_owned_digest: bool = True,
    expected_claim_root: Path = DEFAULT_SMOKE_CLAIM_ROOT,
    expected_output_root: Path = DEFAULT_SMOKE_OUTPUT_ROOT,
) -> AuthenticatedSmokeContext:
    """Read every authenticated input once and return the exact execution view."""
    if authorization_path is None:
        raise PermissionError("smoke execution is not authorized")
    if arm not in SMOKE_ARMS:
        raise PermissionError(f"unsupported smoke arm: {arm}")

    trusted_digest = _read_trusted_digest_once(
        trusted_digest_path,
        require_root_owned=require_root_owned_digest,
    )
    authorization_input = read_regular_json_once(authorization_path)
    if authorization_input.file.sha256 != trusted_digest:
        raise PermissionError(
            "smoke authorization does not match the external digest"
        )
    config_input = read_regular_json_once(config_path)
    plan_input = read_regular_json_once(smoke_plan_path)
    smoke_lock_input = read_regular_json_once(smoke_package_lock_path)
    configuration_lock_input = read_regular_json_once(
        training_configuration_frozen_lock_path
    )
    authorization = authorization_input.value
    smoke_lock = smoke_lock_input.value
    configuration_lock = configuration_lock_input.value

    if (
        authorization.get("schema_version")
        != "exp54-smoke-execution-authorization-v1"
    ):
        raise PermissionError("smoke authorization schema differs")
    if authorization.get("authorization_mode") != "smoke":
        raise PermissionError("authorization mode is not smoke")
    if authorization.get("review_verdict") != "SMOKE_PACKAGE_PASS":
        raise PermissionError("smoke authorization lacks the review verdict")
    if authorization.get("reviewed_smoke_package_commit") != (
        REVIEWED_SMOKE_PACKAGE_COMMIT
    ):
        raise PermissionError("reviewed smoke-package commit differs")
    if smoke_lock.get("reviewed_smoke_package_commit") != (
        REVIEWED_SMOKE_PACKAGE_COMMIT
    ):
        raise PermissionError("smoke lock reviewed commit differs")
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
        "smoke_package_frozen_lock_sha256": smoke_lock_input.file.sha256,
        "training_configuration_frozen_lock_sha256": (
            configuration_lock_input.file.sha256
        ),
        "smoke_plan_sha256": plan_input.file.sha256,
        "materialized_manifest_frozen_lock_sha256": smoke_lock.get(
            "materialized_manifest_frozen_lock_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if authorization.get(field) != expected:
            raise PermissionError(f"smoke authorization does not bind {field}")
    if smoke_lock.get("smoke_plan_sha256") != plan_input.file.sha256:
        raise PermissionError("smoke lock does not bind the read-once plan")
    if configuration_lock.get("configuration_sha256") != (
        config_input.file.sha256
    ):
        raise PermissionError("frozen configuration does not bind this config")
    if smoke_lock.get("configuration_sha256") != config_input.file.sha256:
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
    _validated_campaign(
        authorization,
        expected_claim_root=expected_claim_root,
        expected_output_root=expected_output_root,
    )
    for authenticated_input in (
        authorization_input,
        config_input,
        plan_input,
        smoke_lock_input,
        configuration_lock_input,
    ):
        _require_path_still_same(authenticated_input.file)
    return AuthenticatedSmokeContext(
        authorization=authorization,
        authorization_sha256=authorization_input.file.sha256,
        config=config_input.value,
        config_sha256=config_input.file.sha256,
        smoke_plan=plan_input.value,
        smoke_plan_sha256=plan_input.file.sha256,
        smoke_lock=smoke_lock,
        smoke_lock_sha256=smoke_lock_input.file.sha256,
        training_configuration_lock=configuration_lock,
        training_configuration_lock_sha256=(
            configuration_lock_input.file.sha256
        ),
    )


def _directory_has_append_only_flag(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        buffer = bytearray(struct.pack("I", 0))
        fcntl.ioctl(fd, _FS_IOC_GETFLAGS, buffer, True)
        flags = struct.unpack("I", buffer)[0]
        return bool(flags & _FS_APPEND_FL)
    finally:
        os.close(fd)


def _validate_claim_directory(
    claim_root: Path,
    campaign_dir: Path,
    *,
    require_root_owned: bool,
    require_append_only: bool,
) -> None:
    for label, path in (("claim root", claim_root), ("campaign", campaign_dir)):
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise PermissionError(f"smoke {label} directory is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"smoke {label} must be a real directory")
        if require_root_owned and metadata.st_uid != 0:
            raise PermissionError(f"smoke {label} must be root-owned")
        if metadata.st_mode & 0o002:
            raise PermissionError(f"smoke {label} must not be world-writable")
    if require_append_only and not _directory_has_append_only_flag(campaign_dir):
        raise PermissionError("smoke campaign directory must be append-only")


def claim_smoke_invocation(
    context: AuthenticatedSmokeContext,
    *,
    arm: str,
    claim_root: Path = DEFAULT_SMOKE_CLAIM_ROOT,
    require_root_owned: bool = True,
    require_append_only: bool = True,
) -> SmokeInvocationClaim:
    """Atomically consume one arm-specific invocation before model or CUDA."""
    if arm not in SMOKE_ARMS:
        raise PermissionError(f"unsupported smoke arm: {arm}")
    authorization = context.authorization
    campaign_id = str(authorization["smoke_campaign_id"])
    if authorization.get("claim_root") != str(claim_root):
        raise PermissionError("runtime smoke claim root differs")
    campaign_dir = claim_root / campaign_id
    _validate_claim_directory(
        claim_root,
        campaign_dir,
        require_root_owned=require_root_owned,
        require_append_only=require_append_only,
    )
    run_id = str(authorization["run_id_by_arm"][arm])
    output_dir = Path(str(authorization["output_dir_by_arm"][arm]))
    claim_path = campaign_dir / f"{arm.lower()}.claimed"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required for smoke claims")
    flags |= nofollow
    payload = json.dumps(
        {
            "schema_version": "exp54-smoke-claim-v1",
            "smoke_campaign_id": campaign_id,
            "arm": arm,
            "run_id": run_id,
            "output_dir": str(output_dir),
            "authorization_sha256": context.authorization_sha256,
            "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        fd = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise PermissionError(f"{arm}: smoke authorization already consumed") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o400)
    finally:
        os.close(fd)
    return SmokeInvocationClaim(
        campaign_id=campaign_id,
        arm=arm,
        run_id=run_id,
        claim_path=claim_path,
        output_dir=output_dir,
    )


def reserve_smoke_output_directory(claim: SmokeInvocationClaim) -> Path:
    """Atomically reserve the authorization-bound output after claim."""
    campaign_output = claim.output_dir.parent
    campaign_output.mkdir(parents=True, exist_ok=True)
    if campaign_output.is_symlink() or not campaign_output.is_dir():
        raise PermissionError("smoke campaign output parent is invalid")
    claim.output_dir.mkdir(exist_ok=False)
    if claim.output_dir.is_symlink() or not claim.output_dir.is_dir():
        raise PermissionError("reserved smoke output is invalid")
    return claim.output_dir
