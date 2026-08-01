"""Independently audit the frozen Exp54 SFT dev checkpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
REVIEWED_COMMIT = "a0c987b6973c5e03130fdee009df347b35d48080"
REVIEW_VERDICT = "SFT_DEV_RESULT_PASS"
DEFAULT_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_TRAINING_ROOT = DEFAULT_ROOT / "formal_runs"
DEFAULT_DEV_ROOT = DEFAULT_ROOT / "dev_runs_vllm"
DEFAULT_LOCK = (
    DEFAULT_ROOT
    / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
)
DEFAULT_REPORT = (
    DEFAULT_ROOT
    / "audit/sft_dev_checkpoint_selection_freeze_report.json"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path}: expected regular non-symlink file")
    return path.read_bytes()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(regular_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def canonical_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def independent_backend_contract(backend: Any) -> dict[str, Any]:
    if not isinstance(backend, dict):
        raise ValueError("inference backend metadata is not an object")
    expected_keys = {
        "budget_boundary_policy",
        "compact_json_whitespace_disabled",
        "cuda",
        "forced_completion_count",
        "name",
        "torch",
        "version",
        "xgrammar_source_sha256",
    }
    if set(backend) != expected_keys:
        raise ValueError("inference backend metadata fields differ")
    if not isinstance(backend["forced_completion_count"], int):
        raise ValueError("forced-completion count is not an integer")
    return {
        key: backend[key]
        for key in sorted(expected_keys - {"forced_completion_count"})
    }


def verify_binding(binding: dict[str, Any]) -> None:
    path = REPO_ROOT / str(binding["path"])
    payload = regular_bytes(path)
    if len(payload) != int(binding["bytes"]):
        raise ValueError(f"{path}: byte count differs")
    if sha256_bytes(payload) != binding["sha256"]:
        raise ValueError(f"{path}: SHA-256 differs")


def independent_tree_binding(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"{path}: directory symlink is forbidden")
            continue
        payload = regular_bytes(path)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "canonical_file_vector_sha256": canonical_digest(rows),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    lock = read_object(args.lock)
    if (
        lock.get("schema_version")
        != "exp54-sft-dev-selection-freeze-v1"
        or lock.get("status")
        != "SFT_DEV_CHECKPOINT_SELECTION_FROZEN"
        or lock.get("checkpoint_selection_frozen") is not True
    ):
        raise ValueError("SFT dev selection lock status differs")
    if (
        lock.get("selected_epoch") != 3
        or lock.get("selected_checkpoint_count") != 12
        or lock.get("arms") != list(ARMS)
        or lock.get("seeds") != list(SEEDS)
    ):
        raise ValueError("frozen selection grid differs")
    if lock.get("review_gate") != {
        "verdict": REVIEW_VERDICT,
        "reviewed_commit": REVIEWED_COMMIT,
    }:
        raise ValueError("review gate differs")
    if (
        lock.get("test_accessed") is not False
        or lock.get("test_execution_allowed") is not False
        or lock.get("preference_training_allowed") is not False
        or lock.get("rationale_blind_audit_execution_allowed") is not False
    ):
        raise PermissionError("freeze crosses its authorization boundary")

    checkpoints = lock["checkpoint_bindings"]
    results = lock["selected_dev_result_bindings"]
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    for name, rows in (("checkpoint", checkpoints), ("result", results)):
        observed = {
            (str(row["arm"]), int(row["seed"])) for row in rows
        }
        if len(rows) != 12 or observed != expected:
            raise ValueError(f"{name} binding grid differs")
    if canonical_digest(checkpoints) != lock[
        "selected_checkpoint_vector_sha256"
    ]:
        raise ValueError("checkpoint vector digest differs")
    if canonical_digest(results) != lock[
        "selected_dev_result_vector_sha256"
    ]:
        raise ValueError("selected-result vector digest differs")

    for row in checkpoints:
        arm = str(row["arm"])
        seed = int(row["seed"])
        expected_relative = (
            f"seed{seed}/{arm.lower()}/checkpoint-logical-epoch-3"
        )
        if row["checkpoint_relative_path"] != expected_relative:
            raise ValueError("checkpoint path differs from arm/seed")
        for binding in row["adapter_files"].values():
            verify_binding(binding)
        verify_binding(row["trainer_state"])
        state = read_object(REPO_ROOT / row["trainer_state"]["path"])
        if (
            state.get("status") != "EXP54_FORMAL_CHECKPOINT_UNEVALUATED"
            or state.get("arm") != arm
            or state.get("seed") != seed
            or state.get("logical_epoch_number") != 3
            or state.get("global_optimizer_step") != 996
            or state.get("dev_accessed") is not False
            or state.get("test_accessed") is not False
        ):
            raise ValueError("checkpoint trainer state differs")

    selected_prediction_rows = 0
    backend_contracts = []
    for row in results:
        for name in ("metrics", "protocol", "run_state"):
            verify_binding(row[name])
        verify_binding(row["predictions_jsonl"])
        selected_prediction_rows += int(
            row["predictions_jsonl"]["rows"]
        )
        if int(row["predictions_jsonl"]["rows"]) != 664:
            raise ValueError("selected prediction row count differs")
        prediction_payload = regular_bytes(
            REPO_ROOT / row["predictions_jsonl"]["path"]
        )
        prediction_lines = [
            line for line in prediction_payload.splitlines() if line.strip()
        ]
        if len(prediction_lines) != 664:
            raise ValueError("selected prediction payload row count differs")
        if any(
            json.loads(line.decode("utf-8")).get("parse_success") is not True
            for line in prediction_lines
        ):
            raise ValueError("selected prediction parse result differs")
        metrics = read_object(REPO_ROOT / row["metrics"]["path"])
        protocol = read_object(REPO_ROOT / row["protocol"]["path"])
        backend_contracts.append(
            independent_backend_contract(protocol.get("backend"))
        )
        if (
            metrics.get("execution", {}).get("strict_parse_rate") != 1.0
            or metrics.get("test_accessed") is not False
        ):
            raise ValueError("selected dev result status differs")
    if (
        any(
            value != backend_contracts[0]
            for value in backend_contracts[1:]
        )
        or backend_contracts[0] != lock["inference_backend_contract"]
    ):
        raise ValueError("selected inference backend contract differs")

    for binding in lock["public_summary_bindings"].values():
        verify_binding(binding)
    if canonical_digest(lock["public_summary_bindings"]) != lock[
        "public_summary_vector_sha256"
    ]:
        raise ValueError("summary vector digest differs")
    for binding in lock["source_bindings"].values():
        verify_binding(binding)
    if canonical_digest(lock["source_bindings"]) != lock[
        "source_vector_sha256"
    ]:
        raise ValueError("source vector digest differs")
    for binding in lock["upstream_bindings"].values():
        verify_binding(binding)
    if canonical_digest(lock["upstream_bindings"]) != lock[
        "upstream_vector_sha256"
    ]:
        raise ValueError("upstream vector digest differs")
    if independent_tree_binding(args.dev_root) != lock[
        "all_dev_artifact_tree"
    ]:
        raise ValueError("complete dev artifact tree differs")

    return {
        "schema_version": "exp54-sft-dev-selection-freeze-audit-v1",
        "status": "SFT_DEV_CHECKPOINT_SELECTION_FREEZE_AUDIT_PASS",
        "lock_sha256": sha256_bytes(regular_bytes(args.lock)),
        "selected_checkpoint_count": len(checkpoints),
        "selected_prediction_rows": selected_prediction_rows,
        "all_selected_epochs_equal_three": True,
        "strict_parse_rate_all_selected": 1.0,
        "checkpoint_hashes_verified": True,
        "selected_dev_artifact_hashes_verified": True,
        "complete_dev_tree_verified": True,
        "summary_source_and_upstream_hashes_verified": True,
        "rationale_blind_audit_planning_allowed": True,
        "rationale_blind_audit_execution_allowed": False,
        "test_accessed": False,
        "test_execution_allowed": False,
        "preference_training_allowed": False,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--training-root", type=Path, default=DEFAULT_TRAINING_ROOT
    )
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.report, audit(args))


if __name__ == "__main__":
    main()
