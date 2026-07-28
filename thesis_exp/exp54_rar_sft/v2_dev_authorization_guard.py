"""Fail-closed read-once authorization for one Exp54 V2 dev checkpoint."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    parse_v2_protocol_payload,
)
from thesis_exp.exp54_rar_sft.v2_dev_runtime_contract import (
    require_runtime_versions,
    sha256_file,
    verify_installed_distribution_matches_wheel,
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
    Path("/var/lib/edubench/exp54-v2-dev")
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
_MAX_ADAPTER_BYTES = 4 * 1024 * 1024 * 1024
_EXPECTED_DEV_ROWS = 664
_LABELS = frozenset(range(1, 6))
_FS_IOC_GETFLAGS = 0x80086601
_FS_APPEND_FL = 0x00000020
_CLAIM_ROOT_MODE = 0o755
_CAMPAIGN_DIRECTORY_MODE = 0o1730


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
    protocol_material: ReadOnceMaterial
    protocol: dict[str, Any]
    grammar_material: ReadOnceMaterial
    training_config_material: ReadOnceMaterial
    training_config: dict[str, Any]
    dev_material: ReadOnceMaterial
    dev_rows: tuple[dict[str, Any], ...]
    checkpoint_materials: dict[str, ReadOnceMaterial]
    trainer_state: dict[str, Any]


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


def _read_jsonl_rows(
    material: ReadOnceMaterial,
) -> tuple[dict[str, Any], ...]:
    try:
        lines = material.payload.decode("utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(
            f"{material.path}: authenticated JSONL is invalid"
        ) from exc
    if any(not isinstance(value, dict) for value in values):
        raise PermissionError(
            f"{material.path}: authenticated JSONL row is not an object"
        )
    if len(values) != _EXPECTED_DEV_ROWS:
        raise PermissionError("authenticated dev row count differs")
    record_ids = [str(row.get("record_id") or "") for row in values]
    if any(not value for value in record_ids):
        raise PermissionError("authenticated dev contains an empty record ID")
    if len(set(record_ids)) != len(record_ids):
        raise PermissionError("authenticated dev contains duplicate record IDs")
    if any(int(row["label_5"]) not in _LABELS for row in values):
        raise PermissionError("authenticated dev label is outside 1-5")
    return tuple(values)


def verify_read_once_material_unchanged(
    material: ReadOnceMaterial,
) -> None:
    """Reopen and compare identity plus complete bytes to an authenticated read."""
    current = _read_regular_once(
        material.path,
        max_bytes=max(material.size, 1),
    )
    expected_identity = (
        material.device,
        material.inode,
        material.size,
        material.mtime_ns,
    )
    current_identity = (
        current.device,
        current.inode,
        current.size,
        current.mtime_ns,
    )
    if (
        current_identity != expected_identity
        or current.sha256 != material.sha256
        or current.payload != material.payload
    ):
        raise PermissionError(
            f"{material.path}: authenticated material was replaced"
        )


def verify_execution_context_materials_unchanged(
    context: VerifiedV2DevAuthorization,
) -> None:
    materials = (
        context.protocol_material,
        context.grammar_material,
        context.training_config_material,
        context.dev_material,
        *context.checkpoint_materials.values(),
    )
    for material in materials:
        verify_read_once_material_unchanged(material)


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


def _directory_has_append_only_flag(path: Path) -> bool:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    try:
        raw = fcntl.ioctl(
            descriptor,
            _FS_IOC_GETFLAGS,
            struct.pack("I", 0),
        )
    except OSError as exc:
        raise PermissionError(
            f"{path}: append-only filesystem flag cannot be verified"
        ) from exc
    finally:
        os.close(descriptor)
    return bool(struct.unpack("I", raw)[0] & _FS_APPEND_FL)


def _require_root_managed_claim_root(path: Path) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{path}: expected a real claim-root directory")
    if metadata.st_uid != 0:
        raise PermissionError(f"{path}: claim root must be root-owned")
    if stat.S_IMODE(metadata.st_mode) != _CLAIM_ROOT_MODE:
        raise PermissionError(
            f"{path}: claim-root mode must be "
            f"{oct(_CLAIM_ROOT_MODE)}"
        )


def _require_nonreplayable_campaign_directory(path: Path) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"{path}: expected a real campaign directory")
    if metadata.st_uid != 0:
        raise PermissionError(f"{path}: campaign directory must be root-owned")
    if metadata.st_gid not in set(os.getgroups()) | {os.getgid()}:
        raise PermissionError(
            f"{path}: execution user is outside the training group"
        )
    if stat.S_IMODE(metadata.st_mode) != _CAMPAIGN_DIRECTORY_MODE:
        raise PermissionError(
            f"{path}: campaign mode must be "
            f"{oct(_CAMPAIGN_DIRECTORY_MODE)}"
        )
    if not _directory_has_append_only_flag(path):
        raise PermissionError(
            f"{path}: campaign directory is not append-only"
        )


def _checkpoint_paths(
    training_root: Path,
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> dict[str, Path]:
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
    return dict(sorted(paths.items()))


def _checkpoint_artifacts(
    training_root: Path,
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in _checkpoint_paths(
            training_root,
            arm=arm,
            seed=seed,
            epoch=epoch,
        ).items()
    }


def _validated_trainer_state(
    material: ReadOnceMaterial,
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    state = _read_json(material)
    expected = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": arm,
        "seed": seed,
        "logical_epoch_number": epoch,
        "global_optimizer_step": epoch * 332,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise PermissionError(
                f"authenticated trainer state differs at {key}"
            )
    return state


def _verify_static_bindings(
    authorization: dict[str, Any],
    *,
    protocol_material: ReadOnceMaterial,
    grammar_material: ReadOnceMaterial,
    training_config_material: ReadOnceMaterial,
    candidate_report_material: ReadOnceMaterial,
    candidate_lock_material: ReadOnceMaterial,
    dev_material: ReadOnceMaterial,
    checkpoint_materials: dict[str, ReadOnceMaterial],
    repository_commit: str,
    training_root: Path,
    output_root: Path,
    wheel_root: Path,
    requested_task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    protocol = parse_v2_protocol_payload(
        protocol_material.payload,
        grammar_material.payload,
    )
    training_config = _read_json(training_config_material)
    if protocol["selection_and_access"]["formal_v2_dev_allowed"]:
        raise PermissionError("repository protocol must remain not-authorized")
    if protocol["generation"]["max_new_tokens"] != MAX_NEW_TOKENS:
        raise PermissionError("V2 protocol token budget differs")
    if protocol["generation"]["formal_batch_size"] != FORMAL_BATCH_SIZE:
        raise PermissionError("V2 protocol formal batch size differs")
    report = _read_json(candidate_report_material)
    lock = _read_json(candidate_lock_material)
    expected_hashes = {
        "candidate_report_sha256": candidate_report_material.sha256,
        "candidate_lock_sha256": candidate_lock_material.sha256,
        "protocol_config_sha256": protocol_material.sha256,
        "grammar_sha256": grammar_material.sha256,
        "decoder_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/structured_decoder_v2.py"
        ),
        "parser_sha256": sha256_file(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/inference_contract.py"
        ),
        "dev_split_sha256": dev_material.sha256,
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
    distributions = {}
    for spec in wheel_specs:
        name = spec["distribution"]
        distributions[name] = verify_installed_distribution_matches_wheel(
            name,
            wheel_root / spec["filename"],
            expected_wheel_sha256=spec["sha256"],
        )
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
                    != {
                        name: material.sha256
                        for name, material in checkpoint_materials.items()
                    }
                ):
                    raise PermissionError(
                        "V2 checkpoint artifact bytes differ: "
                        f"{task_id}"
                    )
    return protocol, training_config


def _claim_task(
    *,
    claim_root: Path,
    campaign_id: str,
    task_id: str,
    authorization_sha256: str,
) -> Path:
    _require_root_managed_claim_root(claim_root)
    campaign_dir = claim_root / campaign_id
    _require_nonreplayable_campaign_directory(campaign_dir)
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
    materialize_dev_rows: bool = True,
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
    protocol_material = _read_regular_once(DEFAULT_PROTOCOL_CONFIG)
    grammar_material = _read_regular_once(
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf"
    )
    training_config_material = _read_regular_once(DEFAULT_TRAINING_CONFIG)
    candidate_report_material = _read_regular_once(DEFAULT_CANDIDATE_REPORT)
    candidate_lock_material = _read_regular_once(DEFAULT_CANDIDATE_LOCK)
    dev_material = _read_regular_once(DEFAULT_DEV_PATH)
    checkpoint_materials = {
        name: _read_regular_once(
            path,
            max_bytes=(
                _MAX_ADAPTER_BYTES
                if name == "adapter_model"
                else _MAX_AUTHORIZATION_BYTES
            ),
        )
        for name, path in _checkpoint_paths(
            training_root,
            arm=arm,
            seed=seed,
            epoch=epoch,
        ).items()
    }
    protocol, training_config = _verify_static_bindings(
        authorization,
        protocol_material=protocol_material,
        grammar_material=grammar_material,
        training_config_material=training_config_material,
        candidate_report_material=candidate_report_material,
        candidate_lock_material=candidate_lock_material,
        dev_material=dev_material,
        checkpoint_materials=checkpoint_materials,
        repository_commit=commit,
        training_root=training_root,
        output_root=output_root,
        wheel_root=wheel_root,
        requested_task_id=task_id,
    )
    dev_rows = (
        _read_jsonl_rows(dev_material)
        if materialize_dev_rows
        else ()
    )
    task = next(
        task
        for task in authorization["tasks"]
        if task["task_id"] == task_id
    )
    campaign_id = authorization["campaign_id"]
    trainer_state = _validated_trainer_state(
        checkpoint_materials["trainer_state"],
        arm=arm,
        seed=seed,
        epoch=epoch,
    )
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
        protocol_material=protocol_material,
        protocol=protocol,
        grammar_material=grammar_material,
        training_config_material=training_config_material,
        training_config=training_config,
        dev_material=dev_material,
        dev_rows=dev_rows,
        checkpoint_materials=checkpoint_materials,
        trainer_state=trainer_state,
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
        protocol_material=verified.protocol_material,
        protocol=verified.protocol,
        grammar_material=verified.grammar_material,
        training_config_material=verified.training_config_material,
        training_config=verified.training_config,
        dev_material=verified.dev_material,
        dev_rows=verified.dev_rows,
        checkpoint_materials=verified.checkpoint_materials,
        trainer_state=verified.trainer_state,
        claim_path=claim_path,
    )
