"""Independently reconstruct and audit the frozen train-only smoke package.

This auditor does not import the production smoke builder. It recomputes the
slot-wise selection from the frozen full manifests, compares the private
subset rows and prompt cache exactly, and verifies the aggregate public lock.
It never imports a model framework or executes training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
)
from thesis_exp.exp54_rar_sft.authorization_guard import (
    TRUSTED_AUTHORIZATION_DIGEST_PATH,
    closure_sha256,
    runtime_source_closure,
)
from thesis_exp.exp54_rar_sft.freeze_training_configuration import (
    EXPECTED_CANDIDATE_LOCK_SHA256,
    EXPECTED_CANDIDATE_REPORT_SHA256,
    REVIEWED_COMMIT,
    REVIEW_VERDICT,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
    smoke_runtime_source_closure,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    DEFAULT_PRIVATE_SMOKE_DIR,
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_SMOKE_REPORT,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    DEFAULT_TRAINING_CONFIG,
    SMOKE_ARMS,
    SMOKE_EVENTS_PER_ARM,
    SMOKE_SELECTION_SLOTS,
    SMOKE_SOURCE_EPOCH_INDEX,
    selector_digest,
    smoke_manifest_path,
    smoke_prompt_cache_path,
    validate_smoke_plan,
    vector_sha256,
)


DEFAULT_FULL_DATA_DIR = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data"
)


def _read_object(path: Path) -> dict[str, Any]:
    reject_eval_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _coordinate(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("seed"),
        row.get("epoch_index"),
        row.get("epoch_number"),
        row.get("row_position"),
        row.get("base_event_id"),
        row.get("record_id"),
        row.get("prompt_cache_id"),
        row.get("prompt_token_ids_sha256"),
        row.get("score_target"),
        row.get("score_loss_active"),
        row.get("cutoff_len"),
        row.get("packing"),
        row.get("truncated"),
    )


def _require_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PermissionError(f"{label} must remain uninstalled")


def _independent_selected_indices(
    manifests: dict[str, list[dict[str, Any]]],
) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(manifests["S0"])
        if row["epoch_index"] == SMOKE_SOURCE_EPOCH_INDEX
    ]
    selected: list[int] = []
    for expected_active, expected_score in SMOKE_SELECTION_SLOTS:
        candidates = []
        for index in eligible:
            r2_active = bool(manifests["R2"][index]["rationale_active"])
            r3_active = bool(manifests["R3"][index]["rationale_active"])
            if r2_active != r3_active:
                raise ValueError("R2/R3 activity differs during reconstruction")
            score_vectors = {
                int(manifests[arm][index]["score_target"])
                for arm in SMOKE_ARMS
            }
            if len(score_vectors) != 1:
                raise ValueError("score targets differ during reconstruction")
            score = next(iter(score_vectors))
            if r3_active is expected_active and score == expected_score:
                candidates.append(index)
        if not candidates:
            raise ValueError(
                "no event satisfies independently reconstructed smoke slot"
            )
        selected.append(
            min(
                candidates,
                key=lambda index: selector_digest(
                    str(manifests["S0"][index]["base_event_id"])
                ),
            )
        )
    selected.sort(
        key=lambda index: selector_digest(
            str(manifests["S0"][index]["base_event_id"])
        )
    )
    if len(selected) != SMOKE_EVENTS_PER_ARM:
        raise AssertionError("independent smoke selection count differs")
    if len(set(selected)) != len(selected):
        raise AssertionError("independent smoke selection contains duplicates")
    return selected


def audit_smoke_training_package(
    *,
    full_data_dir: Path,
    private_smoke_dir: Path,
    smoke_plan_path: Path,
    training_configuration_frozen_lock_path: Path,
    materialized_manifest_frozen_lock_path: Path,
    report_path: Path,
    frozen_lock_path: Path,
) -> dict[str, Any]:
    for path in (
        full_data_dir,
        private_smoke_dir,
        smoke_plan_path,
        training_configuration_frozen_lock_path,
        materialized_manifest_frozen_lock_path,
        report_path,
        frozen_lock_path,
    ):
        reject_eval_path(path)
    _require_absent(
        TRUSTED_AUTHORIZATION_DIGEST_PATH,
        label="formal trust anchor",
    )
    _require_absent(
        TRUSTED_SMOKE_AUTHORIZATION_DIGEST_PATH,
        label="smoke trust anchor",
    )
    plan = _read_object(smoke_plan_path)
    validate_smoke_plan(plan)
    configuration = _read_object(training_configuration_frozen_lock_path)
    materialized = _read_object(materialized_manifest_frozen_lock_path)
    report = _read_object(report_path)
    frozen = _read_object(frozen_lock_path)
    if configuration.get("status") != (
        "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("training configuration is not frozen")
    if materialized.get("status") != (
        "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
    ):
        raise PermissionError("materialized manifests are not frozen")
    if configuration.get("review_gate") != {
        "verdict": REVIEW_VERDICT,
        "reviewed_commit": REVIEWED_COMMIT,
        "candidate_report_sha256": EXPECTED_CANDIDATE_REPORT_SHA256,
        "candidate_lock_sha256": EXPECTED_CANDIDATE_LOCK_SHA256,
    }:
        raise ValueError("training-configuration review gate differs")
    if configuration.get("configuration_sha256") != file_sha256(
        DEFAULT_TRAINING_CONFIG
    ):
        raise ValueError("training-configuration config binding differs")
    formal_closure = runtime_source_closure()
    if configuration.get("runtime_source_closure") != formal_closure:
        raise ValueError("training-configuration runtime closure differs")
    if configuration.get("runtime_source_closure_sha256") != closure_sha256(
        formal_closure
    ):
        raise ValueError(
            "training-configuration runtime closure digest differs"
        )
    expected_status = (
        "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
    )
    if report.get("status") != expected_status:
        raise ValueError("smoke report status differs")
    if frozen.get("status") != expected_status:
        raise ValueError("smoke frozen-lock status differs")
    if frozen.get("candidate_report_sha256") != file_sha256(report_path):
        raise ValueError("smoke frozen lock does not bind the public report")
    if frozen.get("training_configuration_frozen_lock_sha256") != (
        file_sha256(training_configuration_frozen_lock_path)
    ):
        raise ValueError("smoke lock does not bind frozen configuration")
    if frozen.get("materialized_manifest_frozen_lock_sha256") != (
        file_sha256(materialized_manifest_frozen_lock_path)
    ):
        raise ValueError("smoke lock does not bind frozen manifests")

    expected_manifest_hashes = materialized["private_artifact_hashes"][
        "manifests_by_seed"
    ]["seed42"]
    manifests: dict[str, list[dict[str, Any]]] = {}
    for arm in SMOKE_ARMS:
        source_path = (
            full_data_dir
            / f"training_manifest_{arm.lower()}_seed42.jsonl"
        )
        if file_sha256(source_path) != expected_manifest_hashes[arm]:
            raise ValueError(f"source {arm} manifest hash differs")
        rows = read_jsonl(source_path, protect_split=True)
        if len(rows) != 7_962:
            raise ValueError(f"source {arm} row count differs")
        manifests[arm] = rows
    reference_coordinates = [
        _coordinate(row) for row in manifests["S0"]
    ]
    for arm in SMOKE_ARMS[1:]:
        if [_coordinate(row) for row in manifests[arm]] != (
            reference_coordinates
        ):
            raise ValueError(f"source {arm} coordinates differ")

    selected_indices = _independent_selected_indices(manifests)
    expected_by_arm = {
        arm: [manifests[arm][index] for index in selected_indices]
        for arm in SMOKE_ARMS
    }
    private_hashes = frozen["private_artifact_hashes"]
    for arm in SMOKE_ARMS:
        private_path = smoke_manifest_path(private_smoke_dir, arm)
        if file_sha256(private_path) != private_hashes["manifests_by_arm"][arm]:
            raise ValueError(f"private {arm} smoke hash differs")
        if read_jsonl(private_path, protect_split=True) != expected_by_arm[arm]:
            raise ValueError(f"private {arm} smoke rows differ")

    full_prompt_path = full_data_dir / "shared_prompt_cache.jsonl"
    if file_sha256(full_prompt_path) != materialized[
        "private_artifact_hashes"
    ]["shared_prompt_cache"]:
        raise ValueError("source prompt-cache hash differs")
    full_prompts = read_jsonl(full_prompt_path, protect_split=True)
    prompt_by_id = {
        str(row["prompt_cache_id"]): row for row in full_prompts
    }
    if len(prompt_by_id) != len(full_prompts):
        raise ValueError("source prompt cache contains duplicates")
    expected_prompts = [
        prompt_by_id[str(row["prompt_cache_id"])]
        for row in expected_by_arm["S0"]
    ]
    private_prompt_path = smoke_prompt_cache_path(private_smoke_dir)
    if file_sha256(private_prompt_path) != private_hashes["prompt_cache"]:
        raise ValueError("private smoke prompt-cache hash differs")
    if read_jsonl(private_prompt_path, protect_split=True) != expected_prompts:
        raise ValueError("private smoke prompt-cache rows differ")

    selected_event_ids = [
        str(row["base_event_id"]) for row in expected_by_arm["S0"]
    ]
    selected_activity = [
        bool(row["rationale_active"]) for row in expected_by_arm["R3"]
    ]
    expected_histogram = {
        str(score): count
        for score, count in sorted(
            Counter(
                int(row["score_target"])
                for row in expected_by_arm["S0"]
            ).items()
        )
    }
    expected_active_by_arm = {
        arm: sum(
            bool(row["rationale_active"]) for row in expected_by_arm[arm]
        )
        for arm in SMOKE_ARMS
    }
    expected_tokens = {
        arm: {
            "unpadded_sequence_tokens": sum(
                int(row["sequence_token_count"])
                for row in expected_by_arm[arm]
            ),
            "fixed_padded_tokens": sum(
                int(row["cutoff_len"]) for row in expected_by_arm[arm]
            ),
            "score_supervised_tokens": sum(
                len(row["score_token_positions"])
                for row in expected_by_arm[arm]
            ),
            "rationale_supervised_tokens": sum(
                len(row["rationale_token_positions"])
                for row in expected_by_arm[arm]
            ),
        }
        for arm in SMOKE_ARMS
    }
    expected_fields = {
        "selected_event_vector_sha256": vector_sha256(selected_event_ids),
        "selected_activity_vector_sha256": vector_sha256(selected_activity),
        "score_target_histogram": expected_histogram,
        "rationale_active_events_by_arm": expected_active_by_arm,
        "token_totals_by_arm": expected_tokens,
        "smoke_plan_sha256": file_sha256(smoke_plan_path),
        "training_configuration_frozen_lock_sha256": file_sha256(
            training_configuration_frozen_lock_path
        ),
        "materialized_manifest_frozen_lock_sha256": file_sha256(
            materialized_manifest_frozen_lock_path
        ),
        "private_artifact_hashes": private_hashes,
    }
    for name, expected in expected_fields.items():
        if report.get(name) != expected:
            raise ValueError(f"smoke report differs at {name}")
        if frozen.get(name) != expected:
            raise ValueError(f"smoke lock differs at {name}")

    closure = smoke_runtime_source_closure()
    if report.get("smoke_runtime_source_closure") != closure:
        raise ValueError("smoke report runtime closure differs")
    if frozen.get("smoke_runtime_source_closure") != closure:
        raise ValueError("smoke lock runtime closure differs")
    if report.get("smoke_runtime_source_closure_sha256") != closure_sha256(
        closure
    ):
        raise ValueError("smoke report runtime closure digest differs")
    source_paths = {
        "package_builder": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/build_smoke_training_package.py"
        ),
        "package_auditor": Path(__file__),
        "training_configuration_freezer": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/freeze_training_configuration.py"
        ),
        "smoke_contract": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/smoke_training_contract.py"
        ),
        "smoke_authorization_guard": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/smoke_authorization_guard.py"
        ),
        "smoke_entrypoint": (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/train_rar_sft_smoke.py"
        ),
        "smoke_launcher": (
            REPO_ROOT / "thesis_exp/scripts/run_exp54_rar_sft_smoke.sh"
        ),
    }
    actual_source_hashes = {
        name: file_sha256(path) for name, path in source_paths.items()
    }
    if report.get("source_hashes") != actual_source_hashes:
        raise ValueError("smoke report source hashes differ")
    if frozen.get("source_hashes") != actual_source_hashes:
        raise ValueError("smoke lock source hashes differ")

    for name, artifact in (("report", report), ("lock", frozen)):
        if (
            artifact.get("trust_anchor_install_allowed")
            or artifact.get("forward_backward_allowed")
            or artifact.get("smoke_training_allowed")
            or artifact.get("formal_training_allowed")
            or artifact.get("dev_accessed")
            or artifact.get("test_accessed")
            or artifact.get("training_used")
        ):
            raise PermissionError(f"smoke {name} crosses authorization boundary")
    return {
        "status": "SMOKE_TRAINING_PACKAGE_INDEPENDENT_AUDIT_PASS",
        "events_per_arm": SMOKE_EVENTS_PER_ARM,
        "private_rows_exactly_reconstructed": True,
        "prompt_rows_exactly_reconstructed": True,
        "public_hash_chain_verified": True,
        "runtime_source_closure_verified": True,
        "trust_anchors_absent": True,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-data-dir",
        type=Path,
        default=DEFAULT_FULL_DATA_DIR,
    )
    parser.add_argument(
        "--private-smoke-dir",
        type=Path,
        default=DEFAULT_PRIVATE_SMOKE_DIR,
    )
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
    parser.add_argument(
        "--training-configuration-frozen-lock",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    )
    parser.add_argument(
        "--materialized-manifest-frozen-lock",
        type=Path,
        default=DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_SMOKE_REPORT)
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=DEFAULT_SMOKE_FROZEN_LOCK,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_smoke_training_package(
        full_data_dir=args.full_data_dir,
        private_smoke_dir=args.private_smoke_dir,
        smoke_plan_path=args.smoke_plan,
        training_configuration_frozen_lock_path=(
            args.training_configuration_frozen_lock
        ),
        materialized_manifest_frozen_lock_path=(
            args.materialized_manifest_frozen_lock
        ),
        report_path=args.report,
        frozen_lock_path=args.frozen_lock,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
