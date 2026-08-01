"""Freeze the 12 reviewed Exp54 epoch-3 checkpoints and dev evidence.

This command is read-only with respect to model checkpoints and dev outputs.
It publishes hashes and aggregate metadata only; it never publishes dev rows,
record IDs, predictions, or rationale text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
SELECTED_EPOCH = 3
EXPECTED_DEV_ROWS = 664
REVIEWED_COMMIT = "a0c987b6973c5e03130fdee009df347b35d48080"
REVIEW_VERDICT = "SFT_DEV_RESULT_PASS"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_TRAINING_ROOT = DEFAULT_OUTPUT_ROOT / "formal_runs"
DEFAULT_DEV_ROOT = DEFAULT_OUTPUT_ROOT / "dev_runs_vllm"
DEFAULT_SUMMARY_ROOT = DEFAULT_OUTPUT_ROOT / "dev_summary_vllm"
DEFAULT_LOCK = (
    DEFAULT_OUTPUT_ROOT
    / "protocol/sft_dev_checkpoint_selection_frozen_lock.json"
)
SOURCE_PATHS = (
    "thesis_exp/exp54_rar_sft/freeze_sft_dev_selection.py",
    "thesis_exp/exp54_rar_sft/audit_sft_dev_selection_freeze.py",
    "thesis_exp/exp54_rar_sft/training_contract.py",
    "thesis_exp/exp54_rar_sft/inference_contract.py",
    "thesis_exp/exp54_rar_sft/run_dev_inference_vllm.py",
    "thesis_exp/exp54_rar_sft/collect_dev_results.py",
    "thesis_exp/scripts/run_exp54_vllm_dev_campaign.sh",
)
UPSTREAM_PATHS = (
    "thesis_exp/exp54_rar_sft/configs/canonical_rubric_registry.json",
    "thesis_exp/exp54_rar_sft/configs/qwen_tokenizer_lock_spec.json",
    "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json",
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json",
    "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_frozen_lock.json",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path}: expected regular non-symlink file")
    return path.read_bytes()


def sha256_file(path: Path) -> str:
    return sha256_bytes(read_regular_bytes(path))


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def relative_name(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def file_binding(path: Path) -> dict[str, Any]:
    payload = read_regular_bytes(path)
    return {
        "path": relative_name(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def inference_backend_contract(backend: Any) -> dict[str, Any]:
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


def tree_binding(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"{path}: directory symlink is forbidden")
            continue
        payload = read_regular_bytes(path)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    if not rows:
        raise ValueError(f"{root}: empty artifact tree")
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "canonical_file_vector_sha256": canonical_digest(rows),
    }


def load_selection(path: Path) -> list[dict[str, Any]]:
    rows = list(
        csv.DictReader(
            read_regular_bytes(path).decode("utf-8").splitlines()
        )
    )
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = {
        (str(row["arm"]), int(row["seed"]))
        for row in rows
    }
    if len(rows) != 12 or observed != expected:
        raise ValueError("selected-checkpoint table is not the 12-arm/seed grid")
    if any(int(row["selected_epoch"]) != SELECTED_EPOCH for row in rows):
        raise ValueError("not every reviewed checkpoint selects epoch 3")
    if any(
        row["selection_rule"] != "max Exact, lower MAE, earlier epoch"
        for row in rows
    ):
        raise ValueError("checkpoint selection rule differs")
    return sorted(rows, key=lambda row: (row["arm"], int(row["seed"])))


def checkpoint_binding(
    *,
    training_root: Path,
    dev_root: Path,
    arm: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = (
        training_root
        / f"seed{seed}"
        / arm.lower()
        / f"checkpoint-logical-epoch-{SELECTED_EPOCH}"
    )
    adapter = checkpoint / "adapter"
    state_path = checkpoint / "trainer_state.json"
    state = read_object(state_path)
    expected_state = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": arm,
        "seed": seed,
        "logical_epoch_number": SELECTED_EPOCH,
        "global_optimizer_step": 996,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for key, value in expected_state.items():
        if state.get(key) != value:
            raise ValueError(f"{state_path}: state differs at {key}")
    adapter_files = {
        name: file_binding(adapter / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    if adapter_files["adapter_model.safetensors"]["bytes"] < 1_000_000:
        raise ValueError(f"{adapter}: adapter payload is unexpectedly small")

    run_dir = (
        dev_root
        / arm.lower()
        / f"seed{seed}"
        / f"epoch{SELECTED_EPOCH}"
    )
    metrics = read_object(run_dir / "metrics.json")
    protocol = read_object(run_dir / "protocol.json")
    run_state = read_object(run_dir / "run_state.json")
    for artifact in (metrics, protocol, run_state):
        if (
            artifact.get("arm") != arm
            or artifact.get("seed") != seed
            or artifact.get("epoch") != SELECTED_EPOCH
            or artifact.get("test_accessed") is not False
        ):
            raise ValueError(f"{run_dir}: selected dev metadata differs")
    prediction_payload = read_regular_bytes(run_dir / "predictions.jsonl")
    prediction_lines = [
        line for line in prediction_payload.splitlines() if line.strip()
    ]
    if len(prediction_lines) != EXPECTED_DEV_ROWS:
        raise ValueError(f"{run_dir}: selected prediction count differs")
    for row_number, line in enumerate(prediction_lines, start=1):
        prediction = json.loads(line.decode("utf-8"))
        if (
            not isinstance(prediction, dict)
            or prediction.get("parse_success") is not True
        ):
            raise ValueError(
                f"{run_dir}: prediction {row_number} is not strictly parsed"
            )
    if metrics.get("execution", {}).get("strict_parse_rate") != 1.0:
        raise ValueError(f"{run_dir}: strict parse rate is not one")
    checkpoint_row = {
        "arm": arm,
        "seed": seed,
        "selected_epoch": SELECTED_EPOCH,
        "checkpoint_relative_path": checkpoint.relative_to(
            training_root
        ).as_posix(),
        "adapter_files": adapter_files,
        "trainer_state": file_binding(state_path),
        "training_bindings": {
            key: state[key]
            for key in (
                "config_sha256",
                "frozen_manifest_lock_sha256",
                "manifest_sha256",
            )
        },
    }
    result_row = {
        "arm": arm,
        "seed": seed,
        "selected_epoch": SELECTED_EPOCH,
        "metrics": file_binding(run_dir / "metrics.json"),
        "protocol": file_binding(run_dir / "protocol.json"),
        "run_state": file_binding(run_dir / "run_state.json"),
        "predictions_jsonl": {
            "path": relative_name(run_dir / "predictions.jsonl"),
            "bytes": len(prediction_payload),
            "sha256": sha256_bytes(prediction_payload),
            "rows": len(prediction_lines),
        },
    }
    return checkpoint_row, result_row


def build_lock(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.summary_root / "selected_checkpoints.csv"
    selection = load_selection(selection_path)
    checkpoints = []
    selected_results = []
    backends = []
    for row in selection:
        arm = str(row["arm"])
        seed = int(row["seed"])
        checkpoint, result = checkpoint_binding(
            training_root=args.training_root,
            dev_root=args.dev_root,
            arm=arm,
            seed=seed,
        )
        checkpoints.append(checkpoint)
        selected_results.append(result)
        protocol = read_object(REPO_ROOT / result["protocol"]["path"])
        backends.append(inference_backend_contract(protocol["backend"]))
    if any(value != backends[0] for value in backends[1:]):
        raise ValueError(
            "selected runs do not share one inference backend contract"
        )

    summary_files = {
        path.name: file_binding(path)
        for path in sorted(args.summary_root.iterdir())
        if path.is_file()
    }
    required_summary = {
        "all_epoch_metrics.csv",
        "final_results.json",
        "multiseed_summary.csv",
        "paired_bootstrap.json",
        "report.md",
        "selected_checkpoints.csv",
    }
    if set(summary_files) != required_summary:
        raise ValueError("public dev summary file set differs")
    final_results = read_object(args.summary_root / "final_results.json")
    if (
        final_results.get("status") != "EXP54_DEV_STUDY_COMPLETE"
        or final_results.get("dev_accessed") is not True
        or final_results.get("test_accessed") is not False
    ):
        raise ValueError("final dev result status differs")

    source_bindings = {
        path: file_binding(REPO_ROOT / path) for path in SOURCE_PATHS
    }
    upstream_bindings = {
        path: file_binding(REPO_ROOT / path) for path in UPSTREAM_PATHS
    }
    return {
        "schema_version": "exp54-sft-dev-selection-freeze-v1",
        "status": "SFT_DEV_CHECKPOINT_SELECTION_FROZEN",
        "review_gate": {
            "verdict": REVIEW_VERDICT,
            "reviewed_commit": REVIEWED_COMMIT,
        },
        "selection_rule": "maximum Exact, lower MAE, earlier epoch",
        "selected_epoch": SELECTED_EPOCH,
        "selected_checkpoint_count": len(checkpoints),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "selection_table": file_binding(selection_path),
        "checkpoint_bindings": checkpoints,
        "selected_dev_result_bindings": selected_results,
        "selected_checkpoint_vector_sha256": canonical_digest(checkpoints),
        "selected_dev_result_vector_sha256": canonical_digest(
            selected_results
        ),
        "all_dev_artifact_tree": tree_binding(args.dev_root),
        "public_summary_bindings": summary_files,
        "public_summary_vector_sha256": canonical_digest(summary_files),
        "source_bindings": source_bindings,
        "source_vector_sha256": canonical_digest(source_bindings),
        "upstream_bindings": upstream_bindings,
        "upstream_vector_sha256": canonical_digest(upstream_bindings),
        "inference_backend_contract": backends[0],
        "output_schema_location": (
            "OUTPUT_SCHEMA embedded in run_dev_inference_vllm.py"
        ),
        "checkpoint_selection_frozen": True,
        "rationale_blind_audit_planning_allowed": True,
        "rationale_blind_audit_execution_allowed": False,
        "test_accessed": False,
        "test_execution_allowed": False,
        "preference_training_allowed": False,
        "invalidation_rule": (
            "Any selected adapter, trainer state, selected dev artifact, "
            "complete dev tree, summary, source, inference backend, or "
            "upstream lock hash change invalidates this freeze."
        ),
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
    parser.add_argument(
        "--training-root", type=Path, default=DEFAULT_TRAINING_ROOT
    )
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument(
        "--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.lock, build_lock(args))


if __name__ == "__main__":
    main()
