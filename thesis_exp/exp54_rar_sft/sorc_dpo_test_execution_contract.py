"""Fail-closed execution contract for the Exp54 one-time test campaign."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import argparse
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.run_dev_inference import file_sha256


ARMS = (
    "P0_R3_SFT",
    "P1_FIELD_DPO",
    "P2_SORC_SCORE",
    "P3_JOINT_SORC",
)
SEEDS = (42, 43, 44)
EXPECTED_RUN_KEYS = tuple(
    f"{arm}/seed_{seed}" for arm in ARMS for seed in SEEDS
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "sorc_dpo_one_time_test_preregistration_v1.json"
)
PREREGISTRATION_SHA256 = (
    "bb05b883349b048d9bc1f889036f2fb547cbd7155641cd1e6c8303d42e3ea551"
)
EXPECTED_TEST_BLOB_SHA1 = "7749c2c0f166186cc840409d64424b5a78e7222a"
EXPECTED_XGRAMMAR_SHA256 = (
    "2e2e649417b9570fc3d22e6a7413d0f5b47ace7afe002b21ad4d4efae679bba6"
)
EXPECTED_ADAPTER_CONFIG_SHA256 = {
    ("P0_R3_SFT", 42): (
        "66460105dd0140d94e7ea3c01c93b675fb62a0ae5226a300f9eb57017a9b6b08"
    ),
    ("P0_R3_SFT", 43): (
        "0e0af839f61c7020e55d14ebaf0850375cc9a0f9e5825f5f65fd527fbf46e5aa"
    ),
    ("P0_R3_SFT", 44): (
        "b259400d76c4e85a288c8f63261b23e1034aaa3a5f79a1fe18c73b40672217d9"
    ),
    ("P1_FIELD_DPO", 42): (
        "d5247c2f11e09d3f9c38056da0cede7614669fbcc7621b70134c0840a7f52206"
    ),
    ("P1_FIELD_DPO", 43): (
        "06525440c7cc5bde992459f9de8c8089ef9334bb7edba713a5aa642f8a87f13f"
    ),
    ("P1_FIELD_DPO", 44): (
        "2fabe160bdd0560571a7ea4ad9cbf018b8611b4732432f6115756cdb5f4e7235"
    ),
    ("P2_SORC_SCORE", 42): (
        "dbd4ec6e7510e35ab500386494d9459ed776b9cd902a99a08432203445595a9c"
    ),
    ("P2_SORC_SCORE", 43): (
        "2f07e07c25afca514e482a9c7613d213d927fee4459e2b6b7cdca43d9902defb"
    ),
    ("P2_SORC_SCORE", 44): (
        "962f7053bbee4c107163e7b2fc763cfef435ed868c93ea47647d48046a6e1aab"
    ),
    ("P3_JOINT_SORC", 42): (
        "42cc4998b4cd87ec7920a46038b9dd4dc9f5399ff4af3bebb7671f3d09748180"
    ),
    ("P3_JOINT_SORC", 43): (
        "d4536532156d64a0f70a94dca4f8152170cdbe0d8860383bd705d1886fa6a87c"
    ),
    ("P3_JOINT_SORC", 44): (
        "56659e1b69e50c76c614e47c316a3969ffe5e195c8f643f510eb60d30591f84b"
    ),
}
P0_RELATIVE_TEMPLATE = (
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs/"
    "seed{seed}/r3/checkpoint-logical-epoch-3/adapter"
)
PREFERENCE_RELATIVE_TEMPLATE = (
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/train/{arm}/seed_{seed}/adapter"
)


def regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path}: expected regular non-symlink file")
    return path.read_bytes()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(regular_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> dict[str, Any]:
    payload = regular_bytes(path)
    if hashlib.sha256(payload).hexdigest() != PREREGISTRATION_SHA256:
        raise ValueError("one-time test preregistration SHA-256 differs")
    value = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "exp54-sorc-dpo-one-time-test-preregistration-v1"
        or value.get("status") != "CANDIDATE_NOT_AUTHORIZED"
        or value.get("test_accessed") is not False
        or value.get("test_execution_allowed") is not False
        or tuple(value.get("statistical_protocol", {}).get("seed_replicates", ()))
        != SEEDS
        or value.get("test_source_binding", {}).get("git_index_blob_sha1")
        != EXPECTED_TEST_BLOB_SHA1
        or value.get("inference_protocol", {}).get("protocol_id")
        != "RAR_SFT_VLLM_COMPACT_JSON_V1"
    ):
        raise ValueError("one-time test preregistration contract differs")
    if tuple(value.get("arms", {})) != ARMS:
        raise ValueError("one-time test arm order differs")
    return value


def checkpoint_path(
    *,
    repo_root: Path,
    arm: str,
    seed: int,
) -> Path:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("checkpoint arm or seed is outside the frozen grid")
    if arm == "P0_R3_SFT":
        relative = P0_RELATIVE_TEMPLATE.format(seed=seed)
    else:
        relative = PREFERENCE_RELATIVE_TEMPLATE.format(
            arm=arm.lower(),
            seed=seed,
        )
    return repo_root / relative


def validate_checkpoint(
    *,
    repo_root: Path,
    preregistration: dict[str, Any],
    arm: str,
    seed: int,
) -> dict[str, Any]:
    adapter_path = checkpoint_path(repo_root=repo_root, arm=arm, seed=seed)
    config_path = adapter_path / "adapter_config.json"
    model_path = adapter_path / "adapter_model.safetensors"
    config_payload = regular_bytes(config_path)
    model_payload = regular_bytes(model_path)
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    if (
        repo_root.resolve() == REPO_ROOT.resolve()
        and config_sha256 != EXPECTED_ADAPTER_CONFIG_SHA256[(arm, seed)]
    ):
        raise ValueError(f"{arm}/seed_{seed}: adapter config SHA-256 differs")
    expected = preregistration["arms"][arm][
        f"seed_{seed}_adapter_model_sha256"
    ]
    actual = hashlib.sha256(model_payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{arm}/seed_{seed}: adapter SHA-256 differs")

    if arm == "P0_R3_SFT":
        state_path = adapter_path.parent / "trainer_state.json"
        state = read_object(state_path)
        if (
            state.get("status") != "EXP54_FORMAL_CHECKPOINT_UNEVALUATED"
            or state.get("arm") != "R3"
            or state.get("seed") != seed
            or state.get("logical_epoch_number") != 3
            or state.get("global_optimizer_step") != 996
            or state.get("test_accessed") is not False
        ):
            raise ValueError(f"{arm}/seed_{seed}: SFT checkpoint state differs")
        source_state_sha256 = file_sha256(state_path)
    else:
        result_path = adapter_path.parent / "result.json"
        result = read_object(result_path)
        if (
            result.get("status") != "SORC_DPO_FORMAL_TRAINING_COMPLETE"
            or result.get("arm") != arm
            or result.get("seed") != seed
            or result.get("optimizer_steps") != 27
            or result.get("dev_accessed") is not False
            or result.get("test_accessed") is not False
            or result.get("output_adapter_model_sha256") != actual
        ):
            raise ValueError(
                f"{arm}/seed_{seed}: preference checkpoint state differs"
            )
        source_state_sha256 = file_sha256(result_path)
    return {
        "run_key": f"{arm}/seed_{seed}",
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": config_sha256,
        "adapter_model_sha256": actual,
        "source_state_sha256": source_state_sha256,
    }


def validate_all_checkpoints(
    *,
    repo_root: Path,
    preregistration: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        validate_checkpoint(
            repo_root=repo_root,
            preregistration=preregistration,
            arm=arm,
            seed=seed,
        )
        for arm in ARMS
        for seed in SEEDS
    ]
    if tuple(row["run_key"] for row in rows) != EXPECTED_RUN_KEYS:
        raise AssertionError("validated checkpoint grid differs")
    return rows


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def materialize_test_blob(
    *,
    repo_root: Path,
    destination: Path,
    expected_blob_sha1: str = EXPECTED_TEST_BLOB_SHA1,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        ["git", "cat-file", "blob", expected_blob_sha1],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout
    if git_blob_sha1(payload) != expected_blob_sha1:
        raise ValueError("materialized test Git blob identity differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {
        "git_blob_sha1": expected_blob_sha1,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "path": str(destination),
    }


def load_test_rows(path: Path) -> list[dict[str, Any]]:
    return parse_test_rows(regular_bytes(path))


def parse_test_rows(payload: bytes) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("isolated test blob is empty")
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if any(not value for value in record_ids):
        raise ValueError("isolated test contains an empty record ID")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("isolated test contains duplicate record IDs")
    if any(int(row.get("label_5", 0)) not in (1, 2, 3, 4, 5) for row in rows):
        raise ValueError("isolated test label is outside 1-5")
    return rows


def verify_completion_receipts(output_root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    observed_paths = set()
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = output_root / arm.lower() / f"seed_{seed}"
            receipt_path = run_dir / "completion_receipt.json"
            receipt = read_object(receipt_path)
            predictions_path = run_dir / "predictions.jsonl"
            protocol_path = run_dir / "protocol.json"
            expected = {
                "status": "EXP54_SORC_DPO_ONE_TIME_TEST_RUN_COMPLETE",
                "arm": arm,
                "seed": seed,
                "dev_accessed": False,
                "test_accessed": True,
                "scientific_metrics_computed": False,
            }
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{arm}/seed_{seed}: completion receipt differs")
            if (
                file_sha256(predictions_path)
                != receipt.get("predictions_sha256")
                or file_sha256(protocol_path)
                != receipt.get("protocol_sha256")
            ):
                raise ValueError(f"{arm}/seed_{seed}: output hash differs")
            receipts.append(receipt)
            observed_paths.add(run_dir.resolve())
    extra = {
        path.parent.resolve()
        for path in output_root.glob("*/seed_*/completion_receipt.json")
    } - observed_paths
    if extra:
        raise ValueError("one-time test output contains extra run receipts")
    if len(receipts) != 12:
        raise AssertionError("one-time test receipt count differs")
    return receipts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repo-root", type=Path, required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--destination", type=Path, required=True)
    receipts = subparsers.add_parser("receipts")
    receipts.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.command == "preflight":
        preregistration = load_preregistration()
        from thesis_exp.exp54_rar_sft.train_rar_sft import (
            DEFAULT_CONFIG,
            _read_object,
            validate_training_configuration,
            verify_model_snapshot,
        )

        config = _read_object(DEFAULT_CONFIG)
        validate_training_configuration(config)
        model_hashes = verify_model_snapshot(config)
        rows = validate_all_checkpoints(
            repo_root=args.repo_root,
            preregistration=preregistration,
        )
        print(
            json.dumps(
                {
                    "status": "EXP54_ONE_TIME_TEST_PREFLIGHT_PASS",
                    "checkpoint_count": len(rows),
                    "base_model_file_count": len(model_hashes),
                    "test_accessed": False,
                },
                sort_keys=True,
            )
        )
    elif args.command == "materialize":
        load_preregistration()
        binding = materialize_test_blob(
            repo_root=args.repo_root,
            destination=args.destination,
        )
        print(
            json.dumps(
                {
                    "status": "EXP54_ONE_TIME_TEST_BLOB_MATERIALIZED",
                    **binding,
                    "test_accessed": True,
                },
                sort_keys=True,
            )
        )
    else:
        rows = verify_completion_receipts(args.output_root)
        print(
            json.dumps(
                {
                    "status": "EXP54_ONE_TIME_TEST_ALL_RUNS_COMPLETE",
                    "completion_receipt_count": len(rows),
                    "scientific_metrics_read": False,
                    "test_accessed": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    _main()
