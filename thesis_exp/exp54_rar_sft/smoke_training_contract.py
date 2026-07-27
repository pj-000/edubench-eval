"""Immutable contract for the train-only Exp54 smoke package and execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT


SMOKE_SCHEMA_VERSION = "exp54-smoke-training-v1"
SMOKE_SOURCE_SEED = 42
SMOKE_SOURCE_EPOCH_INDEX = 0
SMOKE_ACTIVE_EVENTS = 4
SMOKE_INACTIVE_EVENTS = 4
SMOKE_EVENTS_PER_ARM = SMOKE_ACTIVE_EVENTS + SMOKE_INACTIVE_EVENTS
SMOKE_MICRO_BATCH_SIZE = 2
SMOKE_GRADIENT_ACCUMULATION_STEPS = 4
SMOKE_OPTIMIZER_STEPS_PER_ARM = 1
SMOKE_MAX_INVOCATIONS_PER_ARM = 1
SMOKE_SELECTOR_NAMESPACE = "exp54-smoke-selector-v1"
REVIEWED_SMOKE_PACKAGE_COMMIT = (
    "e3c642abca96f3f88034caecfae30fda596c2827"
)
SMOKE_ARMS = ("S0", "R1", "R2", "R3")
SMOKE_SELECTION_SLOTS = (
    (True, 1),
    (True, 2),
    (True, 3),
    (True, 5),
    (False, 1),
    (False, 2),
    (False, 4),
    (False, 5),
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_PRIVATE_SMOKE_DIR = DEFAULT_OUTPUT / "data/smoke_v1_stratified"
DEFAULT_SMOKE_OUTPUT_ROOT = DEFAULT_OUTPUT / "smoke_runs"
DEFAULT_SMOKE_CLAIM_ROOT = Path("/var/lib/edubench/exp54-smoke")
DEFAULT_SMOKE_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/smoke_training_plan.json"
)
DEFAULT_TRAINING_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
DEFAULT_TRAINING_CONFIG_FROZEN_LOCK = (
    DEFAULT_OUTPUT / "protocol/training_configuration_frozen_lock.json"
)
DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK = (
    DEFAULT_OUTPUT / "protocol/materialized_manifest_frozen_lock.json"
)
DEFAULT_SMOKE_REPORT = (
    DEFAULT_OUTPUT / "audit/smoke_training_package_report.json"
)
DEFAULT_SMOKE_FROZEN_LOCK = (
    DEFAULT_OUTPUT / "protocol/smoke_training_package_frozen_lock.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def selector_digest(base_event_id: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [
                SMOKE_SELECTOR_NAMESPACE,
                SMOKE_SOURCE_SEED,
                SMOKE_SOURCE_EPOCH_INDEX,
                base_event_id,
            ]
        )
    ).hexdigest()


def vector_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def smoke_manifest_path(private_dir: Path, arm: str) -> Path:
    if arm not in SMOKE_ARMS:
        raise ValueError(f"unsupported smoke arm: {arm}")
    return private_dir / f"smoke_manifest_{arm.lower()}_seed42.jsonl"


def smoke_prompt_cache_path(private_dir: Path) -> Path:
    return private_dir / "smoke_prompt_cache_seed42.jsonl"


def validate_smoke_plan(plan: dict[str, Any]) -> None:
    expected = {
        "status": "SMOKE_PLAN_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "schema_version": SMOKE_SCHEMA_VERSION,
        "source_seed": SMOKE_SOURCE_SEED,
        "source_epoch_index": SMOKE_SOURCE_EPOCH_INDEX,
        "selector_namespace": SMOKE_SELECTOR_NAMESPACE,
        "active_events": SMOKE_ACTIVE_EVENTS,
        "inactive_events": SMOKE_INACTIVE_EVENTS,
        "events_per_arm": SMOKE_EVENTS_PER_ARM,
        "arms": list(SMOKE_ARMS),
        "micro_batch_size_per_device": SMOKE_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": (
            SMOKE_GRADIENT_ACCUMULATION_STEPS
        ),
        "optimizer_steps_per_arm": SMOKE_OPTIMIZER_STEPS_PER_ARM,
        "max_invocations_per_arm": SMOKE_MAX_INVOCATIONS_PER_ARM,
        "shuffle": False,
        "drop_last": False,
        "packing": False,
        "truncation": False,
        "fixed_padding_length": 2048,
        "independent_base_reload_per_arm": True,
        "same_event_vector_across_arms": True,
        "same_training_budget_across_arms": True,
        "read_once_authenticated_context_required": True,
        "atomic_per_arm_claim_required": True,
        "authorization_fixed_output_directory_required": True,
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
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"smoke plan differs at {key}")
    if set(plan) != set(expected) | {
        "selection_rule",
        "acceptance_criteria",
    }:
        raise ValueError("smoke plan fields differ")
    if plan.get("selection_rule") != (
        "Within seed 42 epoch 0, require identical R2/R3 activity; "
        "fill the eight frozen (activity, score-label) slots "
        "[(active,1),(active,2),(active,3),(active,5),"
        "(inactive,1),(inactive,2),(inactive,4),(inactive,5)] "
        "with the minimum canonical selector SHA-256 in each slot, "
        "then order the selected events by digest."
    ):
        raise ValueError("smoke selection rule differs")
    expected_acceptance = {
        "one_optimizer_step_exactly": True,
        "finite_total_score_and_rationale_losses": True,
        "finite_nonzero_preclip_gradient_norm": True,
        "score_supervision_present_for_every_event": True,
        "rationale_mask_matches_activity_for_every_event": True,
        "adapter_only_model_checkpoint": True,
        "adapter_tensor_key_shape_dtype_exact_match": True,
        "no_base_weight_file_in_output": True,
        "no_dev_or_test_access": True,
        "result_not_usable_for_model_or_hyperparameter_selection": True,
    }
    if plan.get("acceptance_criteria") != expected_acceptance:
        raise ValueError("smoke acceptance criteria differ")
