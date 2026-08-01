"""Freeze the first Exp54 dev execution as a format-only failure.

The raw predictions, record IDs, and row-level diagnostics remain private on
the training server.  The public report contains only aggregate counts,
cryptographic hashes, and protocol/source identities.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
V1_ATTEMPTS = (
    ("S0", 42, 1, 664),
    ("S0", 42, 2, 576),
    ("S0", 42, 3, 664),
)
EXPECTED_DEV_ROWS = 664
EXPECTED_FINAL_CHECKPOINTS = len(ARMS) * len(SEEDS)
SOURCE_NAMES = (
    "thesis_exp/exp54_rar_sft/run_dev_inference.py",
    "thesis_exp/exp54_rar_sft/inference_contract.py",
    "thesis_exp/exp54_rar_sft/block_loss.py",
    "thesis_exp/exp54_rar_sft/training_contract.py",
    "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json",
)
PUBLIC_FORBIDDEN_KEYS = {
    "answer",
    "label_5",
    "prediction",
    "question",
    "rationale",
    "raw_output",
    "record_id",
    "record_ids",
    "rubric",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected JSON objects")
    return rows


def relative(path: Path, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def file_entry(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative(path, repo_root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def observed_mtime_utc(path: Path) -> str:
    return datetime.datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=datetime.timezone.utc,
    ).isoformat()


def repository_head(repo_root: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError("repository HEAD is not a 40-character lowercase SHA")
    return value


def checkpoint_entry(
    *,
    formal_root: Path,
    repo_root: Path,
    arm: str,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    checkpoint = (
        formal_root
        / f"seed{seed}"
        / arm.lower()
        / f"checkpoint-logical-epoch-{epoch}"
    )
    files = [
        file_entry(checkpoint / "adapter/adapter_config.json", repo_root),
        file_entry(
            checkpoint / "adapter/adapter_model.safetensors",
            repo_root,
        ),
        file_entry(checkpoint / "trainer_state.json", repo_root),
    ]
    state = read_object(checkpoint / "trainer_state.json")
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
            raise ValueError(
                f"{checkpoint}: trainer state differs at {key}"
            )
    return {
        "arm": arm,
        "seed": seed,
        "logical_epoch_number": epoch,
        "files": files,
        "aggregate_sha256": sha256_bytes(canonical_bytes(files)),
    }


def classify_parse_errors(rows: list[dict[str, Any]]) -> dict[str, int]:
    errors = Counter()
    for row in rows:
        if bool(row.get("parse_success")):
            errors["strict_parse_success"] += 1
        else:
            error = str(row.get("parse_error") or "")
            if error == "generated review is not valid JSON":
                errors["invalid_json"] += 1
            elif error == "generated review contains trailing non-whitespace":
                errors["trailing_non_whitespace"] += 1
            elif error:
                errors["other_strict_parse_failure"] += 1
            else:
                errors["missing_parse_error"] += 1
    return dict(sorted(errors.items()))


def attempt_entry(
    *,
    dev_root: Path,
    repo_root: Path,
    arm: str,
    seed: int,
    epoch: int,
    expected_failures: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = (
        dev_root / arm.lower() / f"seed{seed}" / f"epoch{epoch}"
    )
    predictions_path = directory / "predictions.jsonl"
    protocol_path = directory / "protocol.json"
    failures_path = directory / "parse_failures.json"
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        raise ValueError(
            f"{metrics_path}: scientific metrics must not exist for V1"
        )
    predictions = read_jsonl(predictions_path)
    if len(predictions) != EXPECTED_DEV_ROWS:
        raise ValueError(
            f"{predictions_path}: expected {EXPECTED_DEV_ROWS} rows"
        )
    positions = [int(row["row_position"]) for row in predictions]
    if positions != list(range(EXPECTED_DEV_ROWS)):
        raise ValueError(f"{predictions_path}: row order differs")
    record_ids = [str(row.get("record_id") or "") for row in predictions]
    if any(not value for value in record_ids):
        raise ValueError(f"{predictions_path}: empty record ID")
    if len(set(record_ids)) != EXPECTED_DEV_ROWS:
        raise ValueError(f"{predictions_path}: duplicate record ID")
    failure_count = sum(
        not bool(row.get("parse_success")) for row in predictions
    )
    if failure_count != expected_failures:
        raise ValueError(
            f"{predictions_path}: failures {failure_count} "
            f"!= {expected_failures}"
        )
    protocol = read_object(protocol_path)
    if int(protocol.get("parse_failure_count", -1)) != failure_count:
        raise ValueError(f"{protocol_path}: parse count differs")
    if protocol.get("test_accessed") is not False:
        raise ValueError(f"{protocol_path}: test boundary differs")
    failures = read_object(failures_path)
    if int(failures.get("count", -1)) != failure_count:
        raise ValueError(f"{failures_path}: parse count differs")
    failure_ids = list(failures.get("record_ids") or [])
    expected_failure_ids = [
        row["record_id"]
        for row in predictions
        if not bool(row.get("parse_success"))
    ]
    if failure_ids != expected_failure_ids:
        raise ValueError(f"{failures_path}: failure IDs differ")
    files = [
        file_entry(protocol_path, repo_root),
        file_entry(predictions_path, repo_root),
        file_entry(failures_path, repo_root),
    ]
    public = {
        "arm": arm,
        "seed": seed,
        "logical_epoch_number": epoch,
        "protocol_file_observed_mtime_utc": observed_mtime_utc(
            protocol_path
        ),
        "rows": len(predictions),
        "strict_parse_success_count": len(predictions) - failure_count,
        "strict_parse_failure_count": failure_count,
        "parse_error_categories": classify_parse_errors(predictions),
        "scientific_metrics_written": False,
        "files": files,
        "record_id_vector_sha256": sha256_bytes(
            canonical_bytes(record_ids)
        ),
        "failure_record_id_vector_sha256": sha256_bytes(
            canonical_bytes(failure_ids)
        ),
    }
    private = {
        **public,
        "record_ids": record_ids,
        "failure_record_ids": failure_ids,
    }
    return public, private


def reject_public_leakage(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PUBLIC_FORBIDDEN_KEYS:
                raise ValueError(f"public report leaks {key} at {path}")
            reject_public_leakage(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_public_leakage(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/"):
            raise ValueError(f"public report contains absolute path at {path}")


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def build_reports(
    *,
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dev_root = output_root / "dev_runs"
    formal_root = output_root / "formal_runs"
    public_attempts = []
    private_attempts = []
    for arm, seed, epoch, failures in V1_ATTEMPTS:
        public, private = attempt_entry(
            dev_root=dev_root,
            repo_root=repo_root,
            arm=arm,
            seed=seed,
            epoch=epoch,
            expected_failures=failures,
        )
        public_attempts.append(public)
        private_attempts.append(private)
    record_hashes = {
        item["record_id_vector_sha256"] for item in public_attempts
    }
    if len(record_hashes) != 1:
        raise ValueError("V1 attempts used different ordered dev rows")

    final_checkpoints = [
        checkpoint_entry(
            formal_root=formal_root,
            repo_root=repo_root,
            arm=arm,
            seed=seed,
            epoch=3,
        )
        for arm in ARMS
        for seed in SEEDS
    ]
    if len(final_checkpoints) != EXPECTED_FINAL_CHECKPOINTS:
        raise AssertionError("final checkpoint count differs")
    v1_checkpoints = [
        checkpoint_entry(
            formal_root=formal_root,
            repo_root=repo_root,
            arm=arm,
            seed=seed,
            epoch=epoch,
        )
        for arm, seed, epoch, _ in V1_ATTEMPTS
    ]
    sources = [
        file_entry(repo_root / name, repo_root)
        for name in SOURCE_NAMES
    ]
    public = {
        "status": "DEV_EXECUTION_ATTEMPT_V1_FORMAT_EXECUTION_FAILURE",
        "schema_version": "exp54-dev-execution-attempt-v1-freeze-v1",
        "scientific_metrics_valid": False,
        "checkpoint_selection_allowed": False,
        "dev_rerun_allowed": False,
        "test_accessed": False,
        "repository_head_observed_at_freeze": repository_head(repo_root),
        "execution_source_identity": {
            "complete_identity_method": "per-file SHA-256 bindings",
            "fully_represented_by_repository_commit": False,
            "reason": (
                "the V1 runner was not committed at execution time; the "
                "observed repository HEAD is contextual rather than a "
                "complete execution-source identity"
            ),
        },
        "failure_scope": (
            "strict JSON serialization and termination under unconstrained "
            "greedy decoding"
        ),
        "root_cause_classification": (
            "field-content-only training loss plus unconstrained decoding "
            "created a train-inference serialization mismatch"
        ),
        "attempts": public_attempts,
        "ordered_dev_row_vector_sha256": next(iter(record_hashes)),
        "v1_checkpoint_bindings": v1_checkpoints,
        "all_final_checkpoint_bindings": final_checkpoints,
        "all_final_checkpoint_count": len(final_checkpoints),
        "runtime_source_bindings": sources,
        "runtime_source_aggregate_sha256": sha256_bytes(
            canonical_bytes(sources)
        ),
        "cross_arm_read_only_diagnostic": {
            "scope": "seed42 logical epoch 3, first 16 ordered dev rows",
            "formal_metrics_written": False,
            "sample_id_vector_sha256": sha256_bytes(
                canonical_bytes(
                    private_attempts[0]["record_ids"][:16]
                )
            ),
            "R1": {
                "rows": 16,
                "strict_parse_success_count": 3,
                "strict_parse_failure_count": 13,
                "max_token_hit_count": 5,
            },
            "R2": {
                "rows": 16,
                "strict_parse_success_count": 0,
                "strict_parse_failure_count": 16,
                "max_token_hit_count": 15,
            },
            "R3": {
                "rows": 16,
                "strict_parse_success_count": 0,
                "strict_parse_failure_count": 16,
                "max_token_hit_count": 4,
            },
            "note": (
                "Interactive read-only diagnostic established that the "
                "format failure crosses rationale arms; no score or "
                "rationale semantic metric was computed."
            ),
        },
        "privacy": {
            "raw_outputs_public": False,
            "row_level_identifiers_public": False,
            "labels_public": False,
            "questions_answers_rubrics_public": False,
            "private_evidence_retained": True,
        },
        "protocol_boundary": {
            "v1_outputs_must_not_be_deleted_or_overwritten": True,
            "no_posthoc_score_extraction": True,
            "no_lenient_parser": True,
            "v2_must_freeze_outside_dev": True,
            "independent_review_required_before_dev_rerun": True,
        },
    }
    private = {
        **public,
        "attempts": private_attempts,
    }
    reject_public_leakage(public)
    return public, private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            default_repo
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
        ),
    )
    parser.add_argument(
        "--public-report",
        type=Path,
        default=(
            default_repo
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit"
            / "dev_execution_attempt_v1_report.json"
        ),
    )
    parser.add_argument(
        "--private-lock",
        type=Path,
        default=(
            default_repo
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
            / "private_dev_execution_evidence"
            / "dev_execution_attempt_v1_private_lock.json"
        ),
    )
    args = parser.parse_args()
    public, private = build_reports(
        repo_root=args.repo_root,
        output_root=args.output_root,
    )
    write_new_json(args.private_lock, private)
    write_new_json(args.public_report, public)
    print(
        json.dumps(
            {
                "status": public["status"],
                "public_report": str(args.public_report),
                "public_report_sha256": sha256_file(args.public_report),
                "private_lock": str(args.private_lock),
                "private_lock_sha256": sha256_file(args.private_lock),
                "test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
