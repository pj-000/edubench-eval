"""Fail-closed read-once authorization for one Exp54 V2 dev checkpoint."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.v2_dev_runtime_contract import (
    observed_installed_distribution,
    require_runtime_versions,
    sha256_file,
    v2_runtime_source_closure,
    v2_runtime_source_closure_sha256,
    verify_wheel_file,
)


DEFAULT_V2_AUTHORIZATION_PATH = (
    Path.home() / ".config/edubench/exp54_v2_dev_authorization.json"
)
DEFAULT_V2_TRUSTED_DIGEST_PATH = (
    Path.home() / ".config/edubench/exp54_v2_dev_authorization.sha256"
)
DEFAULT_V2_STAGED_DIGEST_PATH = Path(
    f"{DEFAULT_V2_TRUSTED_DIGEST_PATH}.staged"
)
DEFAULT_V2_CLAIM_ROOT = (
    Path.home() / ".local/state/edubench/exp54-v2-dev"
)
DEFAULT_V2_OUTPUT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/dev_runs_v2"
)
DEFAULT_V2_TRAINING_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
)
DEFAULT_V2_WHEEL_ROOT = REPO_ROOT / ".cache/exp54_xgrammar"
DEFAULT_PROTOCOL_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "inference_protocol_v2_candidate.json"
)
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json"
)
DEFAULT_CANDIDATE_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "inference_protocol_v2_candidate_report.json"
)
DEFAULT_CANDIDATE_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "inference_protocol_v2_candidate_lock.json"
)
DEFAULT_DEV_PATH = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
EPOCHS = (1, 2, 3)
FORMAL_BATCH_SIZE = 16
MAX_NEW_TOKENS = 256
EXPECTED_TASK_COUNT = 36
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_MAX_AUTHORIZATION_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ReadOnceMaterial:
    path: Path
    payload: bytes
    sha256: str
    uid: int
    mode: int
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class VerifiedV2DevAuthorization:
    authorization: dict[str, Any]
    authorization_sha256: str
    campaign_id: str
    task_id: str
    arm: str
    seed: int
    epoch: int
    batch_size: int
    max_new_tokens: int
    output_dir: Path
    training_root: Path


@dataclass(frozen=True)
class V2DevAuthorizationContext(VerifiedV2DevAuthorization):
    claim_path: Path


def _read_regular_once(
    path: Path,
    *,
    max_bytes: int = _MAX_AUTHORIZATION_BYTES,
) -> ReadOnceMaterial:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required for V2 authorization")
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        if exc.errno in {
            errno.ENOENT,
            errno.ELOOP,
            errno.ENOTDIR,
            errno.ENXIO,
        }:
            raise PermissionError(
                f"{path}: authenticated file is unavailable"
            ) from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PermissionError(f"{path}: expected a regular file")
        if metadata.st_size > max_bytes:
            raise PermissionError(f"{path}: authenticated file is too large")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        final = os.fstat(descriptor)
        identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        )
        if identity != final_identity or len(payload) != metadata.st_size:
            raise PermissionError(
                f"{path}: authenticated file changed while reading"
            )
        return ReadOnceMaterial(
            path=path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            uid=metadata.st_uid,
            mode=metadata.st_mode,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
    finally:
        os.close(descriptor)


def _require_private_regular_file(material: ReadOnceMaterial) -> None:
    if material.uid != os.getuid():
        raise PermissionError(
            f"{material.path}: authenticated file owner differs"
        )
    if material.mode & 0o022:
        raise PermissionError(
            f"{material.path}: authenticated file is group/other writable"
        )


def _read_json(material: ReadOnceMaterial) -> dict[str, Any]:
    try:
        value = json.loads(material.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(
            f"{material.path}: authenticated JSON is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise PermissionError(
            f"{material.path}: authenticated JSON must be an object"
        )
    return value


def _current_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _COMMIT_RE.fullmatch(value):
        raise PermissionError("repository HEAD is invalid")
    return value


def _task_id(arm: str, seed: int, epoch: int) -> str:
    return f"{arm.lower()}-seed{seed}-epoch{epoch}"


def _expected_task_ids() -> set[str]:
    return {
        _task_id(arm, seed, epoch)
        for arm in ARMS
        for seed in SEEDS
        for epoch in EPOCHS
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PermissionError(f"{path}: expected a JSON object")
    return value


def _require_private_directory(path: Path) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{path}: expected a real directory")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{path}: directory owner differs")
    if metadata.st_mode & 0o022:
        raise PermissionError(
            f"{path}: directory is group/other writable"
        )


def _checkpoint_artifacts(
    training_root: Path,
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> dict[str, str]:
    checkpoint = (
        training_root
        / f"seed{seed}"
        / arm.lower()
        / f"checkpoint-logical-epoch-{epoch}"
    )
    paths = {
        "adapter_config": checkpoint / "adapter/adapter_config.json",
        "adapter_model": checkpoint / "adapter/adapter_model.safetensors",
        "trainer_state": checkpoint / "trainer_state.json",
    }
    return {
        name: sha256_file(path) for name, path in sorted(paths.items())
    }


def _verify_static_bindings(
    authorization: dict[str, Any],
    *,
    repository_commit: str,
    training_root: Path,
    output_root: Path,
    wheel_root: Path,
    requested_task_id: str,
) -> None:
    exact = {
        "schema_version": "exp54-v2-dev-authorization-v1",
        "status": "EXP54_V2_DEV_AUTHORIZATION_REVIEWED",
        "review_verdict": "INFERENCE_PROTOCOL_V2_PASS",
        "reviewed_commit": repository_commit,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "batch_size": FORMAL_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "test_allowed": False,
        "max_invocations_per_checkpoint": 1,
        "training_root": str(training_root),
        "output_root": str(output_root),
    }
    for key, expected in exact.items():
        if authorization.get(key) != expected:
            raise PermissionError(f"V2 authorization differs at {key}")
    campaign_id = str(authorization.get("campaign_id") or "")
    if not _CAMPAIGN_RE.fullmatch(campaign_id):
        raise PermissionError("V2 authorization campaign ID is invalid")
    protocol = _read_object(DEFAULT_PROTOCOL_CONFIG)
    training_config = _read_object(DEFAULT_TRAINING_CONFIG)
    if protocol["selection_and_access"]["formal_v2_dev_allowed"]:
        raise PermissionError("repository protocol must remain not-authorized")
    if protocol["generation"]["max_new_tokens"] != MAX_NEW_TOKENS:
        raise PermissionError("V2 protocol token budget differs")
    if protocol["generation"]["formal_batch_size"] != FORMAL_BATCH_SIZE:
        raise PermissionError("V2 protocol formal batch size differs")
    report = _read_object(DEFAULT_CANDIDATE_REPORT)
    lock = _read_object(DEFAULT_CANDIDATE_LOCK)
    expected_hashes = {
        "candidate_report_sha256": sha256_file(DEFAULT_CANDIDATE_REPORT),
        "candidate_lock_sha256": sha256_file(DEFAULT_CANDIDATE_LOCK),
        "protocol_config_sha256": sha256_file(DEFAULT_PROTOCOL_CONFIG),
        "grammar_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf"
        ),
        "decoder_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/structured_decoder_v2.py"
        ),
        "parser_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/inference_contract.py"
        ),
        "dev_split_sha256": sha256_file(DEFAULT_DEV_PATH),
    }
    for key, expected in expected_hashes.items():
        if authorization.get(key) != expected:
            raise PermissionError(
                f"V2 authorization does not bind {key}"
            )
    if (
        report.get("formal_v2_dev_allowed")
        or report.get("formal_test_allowed")
        or report.get("test_accessed")
        or lock.get("formal_v2_dev_allowed")
        or lock.get("formal_test_allowed")
        or lock.get("test_accessed")
    ):
        raise PermissionError("V2 candidate audit state is invalid")
    closure = v2_runtime_source_closure()
    closure_digest = v2_runtime_source_closure_sha256(closure)
    if authorization.get("runtime_source_closure") != closure:
        raise PermissionError("V2 authorization source closure differs")
    if (
        authorization.get("runtime_source_closure_sha256")
        != closure_digest
    ):
        raise PermissionError("V2 authorization closure digest differs")
    runtime_versions = require_runtime_versions(protocol, training_config)
    if authorization.get("runtime_versions") != runtime_versions:
        raise PermissionError("V2 authorization runtime versions differ")
    backend = protocol["backend"]
    wheel_specs = [
        {
            "distribution": "xgrammar",
            "filename": backend["wheel_filename"],
            "sha256": backend["wheel_sha256"],
        },
        {
            "distribution": "apache-tvm-ffi",
            "filename": backend["dependency_wheels"][0][
                "wheel_filename"
            ],
            "sha256": backend["dependency_wheels"][0]["wheel_sha256"],
        },
    ]
    observed_wheels = {
        spec["distribution"]: verify_wheel_file(
            wheel_root / spec["filename"],
            expected_filename=spec["filename"],
            expected_sha256=spec["sha256"],
        )
        for spec in wheel_specs
    }
    if authorization.get("observed_wheels") != observed_wheels:
        raise PermissionError("V2 authorization wheel evidence differs")
    distributions = {
        name: observed_installed_distribution(name)
        for name in ("xgrammar", "apache-tvm-ffi")
    }
    if authorization.get("installed_distributions") != distributions:
        raise PermissionError(
            "V2 authorization installed distributions differ"
        )
    tasks = authorization.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASK_COUNT:
        raise PermissionError("V2 authorization task count differs")
    task_by_id = {
        str(task.get("task_id") or ""): task
        for task in tasks
        if isinstance(task, dict)
    }
    if set(task_by_id) != _expected_task_ids():
        raise PermissionError("V2 authorization task set differs")
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in EPOCHS:
                task_id = _task_id(arm, seed, epoch)
                task = task_by_id[task_id]
                expected_task = {
                    "task_id": task_id,
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "output_dir": str(
                        output_root
                        / arm.lower()
                        / f"seed{seed}"
                        / f"epoch{epoch}"
                    ),
                    "max_invocations": 1,
                }
                if (
                    set(task)
                    != set(expected_task) | {"checkpoint_artifacts"}
                    or {
                        key: task.get(key) for key in expected_task
                    }
                    != expected_task
                ):
                    raise PermissionError(
                        f"V2 authorization task differs: {task_id}"
                    )
                artifacts = task.get("checkpoint_artifacts")
                if (
                    not isinstance(artifacts, dict)
                    or set(artifacts)
                    != {
                        "adapter_config",
                        "adapter_model",
                        "trainer_state",
                    }
                    or any(
                        not isinstance(value, str)
                        or not _SHA256_RE.fullmatch(value)
                        for value in artifacts.values()
                    )
                ):
                    raise PermissionError(
                        "V2 checkpoint artifact binding is invalid: "
                        f"{task_id}"
                    )
                if (
                    task_id == requested_task_id
                    and artifacts
                    != _checkpoint_artifacts(
                        training_root,
                        arm=arm,
                        seed=seed,
                        epoch=epoch,
                    )
                ):
                    raise PermissionError(
                        "V2 checkpoint artifact bytes differ: "
                        f"{task_id}"
                    )


def _claim_task(
    *,
    claim_root: Path,
    campaign_id: str,
    task_id: str,
    authorization_sha256: str,
) -> Path:
    _require_private_directory(claim_root)
    campaign_dir = claim_root / campaign_id
    _require_private_directory(campaign_dir)
    claim_path = campaign_dir / f"{task_id}.claimed"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW is required for V2 claims")
    try:
        descriptor = os.open(claim_path, flags | nofollow, 0o444)
    except FileExistsError as exc:
        raise PermissionError(
            f"V2 dev checkpoint was already claimed: {task_id}"
        ) from exc
    try:
        payload = (
            json.dumps(
                {
                    "authorization_sha256": authorization_sha256,
                    "campaign_id": campaign_id,
                    "task_id": task_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim_path


def _reserve_output_directory(
    *,
    output_root: Path,
    output_dir: Path,
    arm: str,
    seed: int,
    epoch: int,
) -> None:
    _require_private_directory(output_root)
    expected = (
        output_root
        / arm.lower()
        / f"seed{seed}"
        / f"epoch{epoch}"
    )
    if output_dir != expected:
        raise PermissionError("V2 output directory differs")
    current = output_root
    for component in (arm.lower(), f"seed{seed}"):
        current = current / component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _require_private_directory(current)
    try:
        output_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise PermissionError(
            f"V2 output directory already exists: {output_dir}"
        ) from exc
    _require_private_directory(output_dir)


def authenticate_v2_dev_authorization(
    *,
    arm: str,
    seed: int,
    epoch: int,
    authorization_path: Path = DEFAULT_V2_AUTHORIZATION_PATH,
    trusted_digest_path: Path = DEFAULT_V2_TRUSTED_DIGEST_PATH,
    claim_root: Path = DEFAULT_V2_CLAIM_ROOT,
    training_root: Path = DEFAULT_V2_TRAINING_ROOT,
    output_root: Path = DEFAULT_V2_OUTPUT_ROOT,
    wheel_root: Path = DEFAULT_V2_WHEEL_ROOT,
    repository_commit: str | None = None,
) -> VerifiedV2DevAuthorization:
    """Authenticate one task without creating a claim or output."""
    if arm not in ARMS or seed not in SEEDS or epoch not in EPOCHS:
        raise PermissionError("requested V2 dev task is outside the campaign")
    digest_material = _read_regular_once(
        trusted_digest_path,
        max_bytes=1024,
    )
    authorization_material = _read_regular_once(authorization_path)
    _require_private_regular_file(digest_material)
    _require_private_regular_file(authorization_material)
    try:
        trusted_digest = digest_material.payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PermissionError("V2 trusted digest is not ASCII") from exc
    if not _SHA256_RE.fullmatch(trusted_digest):
        raise PermissionError("V2 trusted digest is invalid")
    if authorization_material.sha256 != trusted_digest:
        raise PermissionError("V2 authorization digest differs")
    authorization = _read_json(authorization_material)
    commit = repository_commit or _current_commit()
    task_id = _task_id(arm, seed, epoch)
    _verify_static_bindings(
        authorization,
        repository_commit=commit,
        training_root=training_root,
        output_root=output_root,
        wheel_root=wheel_root,
        requested_task_id=task_id,
    )
    task = next(
        task
        for task in authorization["tasks"]
        if task["task_id"] == task_id
    )
    campaign_id = authorization["campaign_id"]
    return VerifiedV2DevAuthorization(
        authorization=authorization,
        authorization_sha256=authorization_material.sha256,
        campaign_id=campaign_id,
        task_id=task_id,
        arm=arm,
        seed=seed,
        epoch=epoch,
        batch_size=FORMAL_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        output_dir=Path(task["output_dir"]),
        training_root=training_root,
    )


def require_v2_dev_authorization(
    *,
    arm: str,
    seed: int,
    epoch: int,
    authorization_path: Path = DEFAULT_V2_AUTHORIZATION_PATH,
    trusted_digest_path: Path = DEFAULT_V2_TRUSTED_DIGEST_PATH,
    claim_root: Path = DEFAULT_V2_CLAIM_ROOT,
    training_root: Path = DEFAULT_V2_TRAINING_ROOT,
    output_root: Path = DEFAULT_V2_OUTPUT_ROOT,
    wheel_root: Path = DEFAULT_V2_WHEEL_ROOT,
    repository_commit: str | None = None,
) -> V2DevAuthorizationContext:
    """Authenticate, validate and consume one formal V2 dev invocation."""
    verified = authenticate_v2_dev_authorization(
        arm=arm,
        seed=seed,
        epoch=epoch,
        authorization_path=authorization_path,
        trusted_digest_path=trusted_digest_path,
        claim_root=claim_root,
        training_root=training_root,
        output_root=output_root,
        wheel_root=wheel_root,
        repository_commit=repository_commit,
    )
    claim_path = _claim_task(
        claim_root=claim_root,
        campaign_id=verified.campaign_id,
        task_id=verified.task_id,
        authorization_sha256=verified.authorization_sha256,
    )
    _reserve_output_directory(
        output_root=output_root,
        output_dir=verified.output_dir,
        arm=verified.arm,
        seed=verified.seed,
        epoch=verified.epoch,
    )
    return V2DevAuthorizationContext(
        authorization=verified.authorization,
        authorization_sha256=verified.authorization_sha256,
        campaign_id=verified.campaign_id,
        task_id=verified.task_id,
        arm=verified.arm,
        seed=verified.seed,
        epoch=verified.epoch,
        batch_size=verified.batch_size,
        max_new_tokens=verified.max_new_tokens,
        output_dir=verified.output_dir,
        training_root=verified.training_root,
        claim_path=claim_path,
    )
