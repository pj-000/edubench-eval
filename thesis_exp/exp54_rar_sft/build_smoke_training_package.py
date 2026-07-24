"""Build and freeze the deterministic, private, train-only Exp54 smoke subset.

The command verifies the already-frozen full manifests, selects eight seed-42
epoch-0 events with a fixed hash rule, and copies the original materialized
rows byte-for-semantics into private smoke files. Public outputs contain only
aggregate counts and cryptographic hashes. No model, CUDA, forward, backward,
dev, or test access is involved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_json,
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
    SMOKE_ACTIVE_EVENTS,
    SMOKE_ARMS,
    SMOKE_EVENTS_PER_ARM,
    SMOKE_INACTIVE_EVENTS,
    SMOKE_OPTIMIZER_STEPS_PER_ARM,
    SMOKE_SCHEMA_VERSION,
    SMOKE_SELECTION_SLOTS,
    SMOKE_SOURCE_EPOCH_INDEX,
    SMOKE_SOURCE_SEED,
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


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _write_private_exact(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    reject_eval_path(path)
    payload = _jsonl_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace different private smoke artifact: {path}"
            )
        return
    path.write_bytes(payload)


def _require_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise PermissionError(f"{label} must remain uninstalled")


def _validate_upstream_locks(
    *,
    training_configuration_frozen_lock_path: Path,
    materialized_manifest_frozen_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configuration = _read_object(training_configuration_frozen_lock_path)
    materialized = _read_object(materialized_manifest_frozen_lock_path)
    if configuration.get("status") != (
        "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED"
    ):
        raise PermissionError("training configuration is not frozen")
    if materialized.get("status") != (
        "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
    ):
        raise PermissionError("materialized manifests are not frozen")
    if (
        configuration.get("materialized_manifest_frozen_lock_sha256")
        != file_sha256(materialized_manifest_frozen_lock_path)
    ):
        raise ValueError("training configuration binds another manifest freeze")
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
        raise ValueError("training-configuration freeze binds another config")
    formal_closure = runtime_source_closure()
    if configuration.get("runtime_source_closure") != formal_closure:
        raise ValueError("training-configuration runtime closure differs")
    if configuration.get("runtime_source_closure_sha256") != closure_sha256(
        formal_closure
    ):
        raise ValueError(
            "training-configuration runtime closure digest differs"
        )
    if configuration.get("runtime_source_closure_file_count") != 16:
        raise ValueError("training-configuration runtime closure count differs")
    if (
        configuration.get("configuration_frozen") is not True
        or configuration.get("smoke_package_build_allowed") is not True
        or materialized.get("manifest_frozen") is not True
    ):
        raise PermissionError("upstream locks do not permit smoke packaging")
    for name, artifact in (
        ("training configuration", configuration),
        ("materialized manifest", materialized),
    ):
        if (
            artifact.get("smoke_training_allowed")
            or artifact.get("formal_training_allowed")
            or artifact.get("dev_accessed")
            or artifact.get("test_accessed")
            or artifact.get("training_used")
        ):
            raise PermissionError(f"{name} crosses its authorization boundary")
    return configuration, materialized


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


def _load_and_validate_full_manifests(
    *,
    full_data_dir: Path,
    materialized_lock: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    manifests: dict[str, list[dict[str, Any]]] = {}
    expected_hashes = materialized_lock["private_artifact_hashes"][
        "manifests_by_seed"
    ]["seed42"]
    for arm in SMOKE_ARMS:
        path = (
            full_data_dir
            / f"training_manifest_{arm.lower()}_seed42.jsonl"
        )
        reject_eval_path(path)
        if file_sha256(path) != expected_hashes[arm]:
            raise ValueError(f"full {arm} manifest differs from frozen hash")
        rows = read_jsonl(path, protect_split=True)
        if len(rows) != 7_962:
            raise ValueError(f"full {arm} manifest row count differs")
        for row in rows:
            if row.get("arm") != arm or row.get("seed") != SMOKE_SOURCE_SEED:
                raise ValueError(f"full {arm} manifest metadata differs")
            if row.get("score_loss_active") is not True:
                raise ValueError(f"full {arm} manifest disables score loss")
        manifests[arm] = rows

    reference_coordinates = [
        _coordinate(row) for row in manifests["S0"]
    ]
    for arm in SMOKE_ARMS[1:]:
        if [_coordinate(row) for row in manifests[arm]] != (
            reference_coordinates
        ):
            raise ValueError(f"{arm} event coordinates differ from S0")
    r2_activity = [bool(row["rationale_active"]) for row in manifests["R2"]]
    r3_activity = [bool(row["rationale_active"]) for row in manifests["R3"]]
    if r2_activity != r3_activity:
        raise ValueError("full R2/R3 rationale activity vectors differ")
    return manifests


def _select_indices(
    manifests: dict[str, list[dict[str, Any]]],
) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(manifests["R3"])
        if row["epoch_index"] == SMOKE_SOURCE_EPOCH_INDEX
    ]
    if len(eligible) != 2_654:
        raise ValueError("smoke source epoch row count differs")
    key = lambda index: selector_digest(
        str(manifests["S0"][index]["base_event_id"])
    )
    selected = []
    for active, score_label in SMOKE_SELECTION_SLOTS:
        slot = [
            index
            for index in eligible
            if bool(manifests["R3"][index]["rationale_active"]) is active
            and int(manifests["S0"][index]["score_target"]) == score_label
        ]
        if not slot:
            raise ValueError(
                "insufficient smoke candidates for slot "
                f"(active={active}, score={score_label})"
            )
        selected.append(min(slot, key=key))
    selected = sorted(selected, key=key)
    if len(selected) != SMOKE_EVENTS_PER_ARM or len(set(selected)) != len(
        selected
    ):
        raise AssertionError("smoke selection is not eight unique events")
    return selected


def _select_prompt_rows(
    *,
    full_data_dir: Path,
    materialized_lock: dict[str, Any],
    selected_manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = full_data_dir / "shared_prompt_cache.jsonl"
    reject_eval_path(path)
    expected_hash = materialized_lock["private_artifact_hashes"][
        "shared_prompt_cache"
    ]
    if file_sha256(path) != expected_hash:
        raise ValueError("full prompt cache differs from frozen hash")
    rows = read_jsonl(path, protect_split=True)
    by_id = {str(row["prompt_cache_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("full prompt cache contains duplicate IDs")
    selected = []
    for manifest_row in selected_manifest_rows:
        prompt = by_id.get(str(manifest_row["prompt_cache_id"]))
        if prompt is None:
            raise ValueError("selected smoke prompt is missing")
        if prompt["record_id"] != manifest_row["record_id"]:
            raise ValueError("selected smoke prompt belongs to another record")
        if (
            prompt["prompt_token_ids_sha256"]
            != manifest_row["prompt_token_ids_sha256"]
        ):
            raise ValueError("selected smoke prompt hash differs")
        selected.append(prompt)
    if len({row["prompt_cache_id"] for row in selected}) != len(selected):
        raise ValueError("smoke prompt cache contains duplicate IDs")
    return selected


def build_smoke_training_package(
    *,
    full_data_dir: Path,
    private_smoke_dir: Path,
    smoke_plan_path: Path,
    training_configuration_frozen_lock_path: Path,
    materialized_manifest_frozen_lock_path: Path,
    report_path: Path,
    frozen_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    configuration_lock, materialized_lock = _validate_upstream_locks(
        training_configuration_frozen_lock_path=(
            training_configuration_frozen_lock_path
        ),
        materialized_manifest_frozen_lock_path=(
            materialized_manifest_frozen_lock_path
        ),
    )
    manifests = _load_and_validate_full_manifests(
        full_data_dir=full_data_dir,
        materialized_lock=materialized_lock,
    )
    selected_indices = _select_indices(manifests)
    selected_by_arm = {
        arm: [manifests[arm][index] for index in selected_indices]
        for arm in SMOKE_ARMS
    }
    selected_coordinates = [
        _coordinate(row) for row in selected_by_arm["S0"]
    ]
    for arm in SMOKE_ARMS[1:]:
        if [_coordinate(row) for row in selected_by_arm[arm]] != (
            selected_coordinates
        ):
            raise AssertionError("selected smoke coordinates differ by arm")

    prompt_rows = _select_prompt_rows(
        full_data_dir=full_data_dir,
        materialized_lock=materialized_lock,
        selected_manifest_rows=selected_by_arm["S0"],
    )
    for arm in SMOKE_ARMS:
        _write_private_exact(
            smoke_manifest_path(private_smoke_dir, arm),
            selected_by_arm[arm],
        )
    _write_private_exact(
        smoke_prompt_cache_path(private_smoke_dir),
        prompt_rows,
    )

    for arm in SMOKE_ARMS:
        if read_jsonl(
            smoke_manifest_path(private_smoke_dir, arm),
            protect_split=True,
        ) != selected_by_arm[arm]:
            raise AssertionError("private smoke manifest round-trip differs")
    if read_jsonl(
        smoke_prompt_cache_path(private_smoke_dir),
        protect_split=True,
    ) != prompt_rows:
        raise AssertionError("private smoke prompt-cache round-trip differs")

    private_hashes = {
        "prompt_cache": file_sha256(
            smoke_prompt_cache_path(private_smoke_dir)
        ),
        "manifests_by_arm": {
            arm: file_sha256(smoke_manifest_path(private_smoke_dir, arm))
            for arm in SMOKE_ARMS
        },
    }
    selected_event_ids = [
        str(row["base_event_id"]) for row in selected_by_arm["S0"]
    ]
    selected_activity = [
        bool(row["rationale_active"]) for row in selected_by_arm["R3"]
    ]
    rationale_active_by_arm = {
        arm: sum(
            bool(row["rationale_active"]) for row in selected_by_arm[arm]
        )
        for arm in SMOKE_ARMS
    }
    score_histogram = {
        str(score): count
        for score, count in sorted(
            Counter(
                int(row["score_target"])
                for row in selected_by_arm["S0"]
            ).items()
        )
    }
    token_totals = {
        arm: {
            "unpadded_sequence_tokens": sum(
                int(row["sequence_token_count"])
                for row in selected_by_arm[arm]
            ),
            "fixed_padded_tokens": sum(
                int(row["cutoff_len"]) for row in selected_by_arm[arm]
            ),
            "score_supervised_tokens": sum(
                len(row["score_token_positions"])
                for row in selected_by_arm[arm]
            ),
            "rationale_supervised_tokens": sum(
                len(row["rationale_token_positions"])
                for row in selected_by_arm[arm]
            ),
        }
        for arm in SMOKE_ARMS
    }
    closure = smoke_runtime_source_closure()
    source_hashes = {
        "package_builder": file_sha256(Path(__file__)),
        "package_auditor": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/audit_smoke_training_package.py"
        ),
        "training_configuration_freezer": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/freeze_training_configuration.py"
        ),
        "smoke_contract": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/smoke_training_contract.py"
        ),
        "smoke_authorization_guard": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/smoke_authorization_guard.py"
        ),
        "smoke_entrypoint": file_sha256(
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/train_rar_sft_smoke.py"
        ),
        "smoke_launcher": file_sha256(
            REPO_ROOT / "thesis_exp/scripts/run_exp54_rar_sft_smoke.sh"
        ),
    }
    common = {
        "status": (
            "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED"
        ),
        "schema_version": SMOKE_SCHEMA_VERSION,
        "source_seed": SMOKE_SOURCE_SEED,
        "source_epoch_index": SMOKE_SOURCE_EPOCH_INDEX,
        "arms": list(SMOKE_ARMS),
        "events_per_arm": SMOKE_EVENTS_PER_ARM,
        "optimizer_steps_per_arm": SMOKE_OPTIMIZER_STEPS_PER_ARM,
        "selected_active_events": sum(selected_activity),
        "selected_inactive_events": (
            len(selected_activity) - sum(selected_activity)
        ),
        "selected_event_vector_sha256": vector_sha256(selected_event_ids),
        "selected_activity_vector_sha256": vector_sha256(
            selected_activity
        ),
        "same_event_vector_and_order_across_arms": True,
        "same_score_vector_across_arms": True,
        "score_target_histogram": score_histogram,
        "rationale_active_events_by_arm": rationale_active_by_arm,
        "token_totals_by_arm": token_totals,
        "fixed_padded_token_budget_equal_across_arms": (
            len(
                {
                    token_totals[arm]["fixed_padded_tokens"]
                    for arm in SMOKE_ARMS
                }
            )
            == 1
        ),
        "private_artifact_hashes": private_hashes,
        "smoke_plan_sha256": file_sha256(smoke_plan_path),
        "training_configuration_frozen_lock_sha256": file_sha256(
            training_configuration_frozen_lock_path
        ),
        "materialized_manifest_frozen_lock_sha256": file_sha256(
            materialized_manifest_frozen_lock_path
        ),
        "configuration_sha256": configuration_lock["configuration_sha256"],
        "source_hashes": source_hashes,
        "smoke_runtime_source_closure": closure,
        "smoke_runtime_source_closure_sha256": closure_sha256(closure),
        "smoke_runtime_source_closure_file_count": len(closure),
        "source_full_manifest_rows_per_arm": 7_962,
        "full_source_manifest_hashes_verified": True,
        "full_r2_r3_activity_vectors_equal": True,
        "private_rows_copied_without_materialized_field_changes": True,
        "train_only_path_guard_enforced": True,
        "formal_trust_anchor_lstat_checked_absent": True,
        "smoke_trust_anchor_lstat_checked_absent": True,
        "smoke_subset_frozen": True,
        "diagnostic_only": True,
        "hyperparameter_selection_allowed": False,
        "checkpoint_selection_allowed": False,
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    if not common["fixed_padded_token_budget_equal_across_arms"]:
        raise AssertionError("smoke padded-token budgets differ")
    if common["selected_active_events"] != SMOKE_ACTIVE_EVENTS:
        raise AssertionError("selected active-event count differs")
    if common["selected_inactive_events"] != SMOKE_INACTIVE_EVENTS:
        raise AssertionError("selected inactive-event count differs")

    report = {
        **common,
        "privacy": {
            "human_rationale_text_published": False,
            "record_ids_published": False,
            "reference_ids_published": False,
            "event_ids_published": False,
            "row_level_token_ids_published": False,
            "private_smoke_artifacts_version_controlled": False,
        },
        "review_scope": (
            "deterministic train-only subset, exact one-step budget, "
            "fail-closed external smoke authorization, and diagnostic runner"
        ),
    }
    write_json(report_path, report)
    frozen_lock = {
        **common,
        "candidate_report_sha256": file_sha256(report_path),
        "invalidation_rule": (
            "Any plan, upstream lock, private subset, runtime source, "
            "selector, or package-builder hash change invalidates this freeze."
        ),
    }
    write_json(frozen_lock_path, frozen_lock)
    return report, frozen_lock


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
    parser.add_argument(
        "--smoke-plan",
        type=Path,
        default=DEFAULT_SMOKE_PLAN,
    )
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
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_SMOKE_REPORT,
    )
    parser.add_argument(
        "--frozen-lock",
        type=Path,
        default=DEFAULT_SMOKE_FROZEN_LOCK,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_smoke_training_package(
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


if __name__ == "__main__":
    main()
