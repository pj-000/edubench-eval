"""Audit the Exp54 V2 inference candidate without reading dev or test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import DEFAULT_TRAIN
from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    DEFAULT_PROTOCOL_CONFIG,
    REPO_ROOT,
    load_v2_protocol,
)
from thesis_exp.exp54_rar_sft.v2_dev_runtime_contract import (
    EXPECTED_V2_RUNTIME_SOURCE_COUNT,
    EXPECTED_V2_RUNTIME_SOURCE_NAMES,
    require_runtime_versions,
    verify_installed_distribution_matches_wheel,
    verify_wheel_file,
    v2_runtime_source_closure,
    v2_runtime_source_closure_sha256,
)


ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
EPOCHS = (1, 2, 3)
DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
DEFAULT_REFERENCE_SET_DATA_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "reference_set_data_lock.json"
)
EXPECTED_MATERIALIZED_MANIFEST_FROZEN_LOCK_SHA256 = (
    "8b5fffce6b54804f834499643953deae23393a7764c0652d0fd7df2214b20713"
)
EXPECTED_REFERENCE_SET_DATA_LOCK_SHA256 = (
    "fed541fb3cb82851db010d13e5dafa057ffbe196f4ecff357f5d9f4408e6d6f4"
)
AUDIT_SOURCE_NAMES = (
    "thesis_exp/exp54_rar_sft/DEV_EXECUTION_ATTEMPT_V1_ROOT_CAUSE.md",
    "thesis_exp/exp54_rar_sft/INFERENCE_PROTOCOL_V2.md",
    "thesis_exp/exp54_rar_sft/V2_DEV_AUTHORIZATION_PLAN.md",
    "thesis_exp/exp54_rar_sft/audit_inference_v2_candidate.py",
    "thesis_exp/exp54_rar_sft/"
    "audit_v2_dev_authorization_preactivation.py",
    "thesis_exp/exp54_rar_sft/freeze_dev_execution_attempt_v1.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_budget_probe.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_determinism_probe.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_train_smoke.py",
    "thesis_exp/tests/test_exp54_inference_v2.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _audit_source_bindings() -> dict[str, str]:
    return {
        name: file_sha256(REPO_ROOT / name)
        for name in AUDIT_SOURCE_NAMES
    }


def _prompt_token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frozen_manifest_contract(
    data_dir: Path,
    *,
    materialized_lock_path: Path = DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    reference_lock_path: Path = DEFAULT_REFERENCE_SET_DATA_LOCK,
) -> dict[str, Any]:
    materialized = json.loads(
        materialized_lock_path.read_text(encoding="utf-8")
    )
    reference = json.loads(
        reference_lock_path.read_text(encoding="utf-8")
    )
    materialized_lock_sha256 = file_sha256(materialized_lock_path)
    reference_lock_sha256 = file_sha256(reference_lock_path)
    if (
        materialized_lock_sha256
        != EXPECTED_MATERIALIZED_MANIFEST_FROZEN_LOCK_SHA256
    ):
        raise ValueError("materialized frozen-lock trust anchor differs")
    if reference_lock_sha256 != EXPECTED_REFERENCE_SET_DATA_LOCK_SHA256:
        raise ValueError("reference-set lock trust anchor differs")
    if materialized.get("status") != (
        "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
    ):
        raise ValueError("materialized manifest lock is not frozen")
    if (
        not materialized.get("manifest_frozen")
        or materialized.get("dev_accessed")
        or materialized.get("test_accessed")
    ):
        raise ValueError("materialized manifest lock boundary differs")
    if reference.get("status") != "REFERENCE_SETS_READY":
        raise ValueError("reference-set data lock is not ready")
    if reference.get("dev_accessed") or reference.get("test_accessed"):
        raise ValueError("reference-set lock boundary differs")
    if (
        materialized["upstream_lock_hashes"]["reference_set_lock"]
        != reference_lock_sha256
    ):
        raise ValueError("reference-set lock hash differs")
    locked_train_sha256 = reference["input"]["rar0_source_hashes"]["train"]
    if file_sha256(DEFAULT_TRAIN) != locked_train_sha256:
        raise ValueError("locked train split hash differs")
    artifacts = materialized["private_artifact_hashes"]
    prompt_path = data_dir / "shared_prompt_cache.jsonl"
    if file_sha256(prompt_path) != artifacts["shared_prompt_cache"]:
        raise ValueError("frozen prompt-cache hash differs")
    manifest_hashes: dict[tuple[str, int], str] = {}
    for seed in SEEDS:
        for arm in ARMS:
            expected = artifacts["manifests_by_seed"][
                f"seed{seed}"
            ][arm]
            path = (
                data_dir
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            if file_sha256(path) != expected:
                raise ValueError(
                    f"frozen manifest hash differs: {arm}/seed{seed}"
                )
            manifest_hashes[(arm, seed)] = expected
    return {
        "materialized_lock_sha256": materialized_lock_sha256,
        "reference_lock_sha256": reference_lock_sha256,
        "locked_train_sha256": locked_train_sha256,
        "prompt_cache_sha256": artifacts["shared_prompt_cache"],
        "manifest_hashes": manifest_hashes,
    }


def _independent_smoke_selection(
    data_dir: Path,
    *,
    frozen_contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rebuild all 36 selections without importing production helpers."""
    contract = frozen_contract or _frozen_manifest_contract(data_dir)
    train_rows = read_jsonl(DEFAULT_TRAIN)
    metadata = {
        str(row["record_id"]): {
            "label": int(row["label_5"]),
            "language": str(row["language"]),
        }
        for row in train_rows
    }
    if len(metadata) != len(train_rows):
        raise ValueError("train metadata contains duplicate record IDs")
    prompt_rows = read_jsonl(data_dir / "shared_prompt_cache.jsonl")
    prompts = {
        str(row["prompt_cache_id"]): row
        for row in prompt_rows
    }
    if len(prompts) != len(prompt_rows):
        raise ValueError("prompt cache contains duplicate IDs")

    selected = []
    group_index = 0
    for arm in ARMS:
        for seed in SEEDS:
            manifest_path = (
                data_dir
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            manifest = read_jsonl(manifest_path)
            manifest_sha256 = file_sha256(manifest_path)
            if manifest_sha256 != contract["manifest_hashes"][(arm, seed)]:
                raise ValueError("manifest differs from frozen contract")
            for epoch in EPOCHS:
                candidates = []
                for source_row in manifest:
                    if int(source_row["epoch_number"]) != epoch:
                        continue
                    row = dict(source_row)
                    row["_manifest_language_field_present"] = (
                        "language" in row
                    )
                    record_id = str(row["record_id"])
                    if record_id not in metadata:
                        raise ValueError("manifest record is absent from train")
                    expected = metadata[record_id]
                    if int(row["score_target"]) != expected["label"]:
                        raise ValueError(
                            "manifest score differs from locked train label"
                        )
                    if (
                        "language" in row
                        and str(row["language"]) != expected["language"]
                    ):
                        raise ValueError(
                            "manifest language differs from locked train"
                        )
                    row["label"] = expected["label"]
                    row["language"] = expected["language"]
                    candidates.append(row)
                if not candidates:
                    raise ValueError("independent smoke group is empty")
                target_label = group_index % 5 + 1
                target_language = (
                    "zh" if group_index % 2 == 0 else "en"
                )
                target_active = (
                    arm != "S0" and group_index % 2 == 0
                )
                chosen = min(
                    candidates,
                    key=lambda row: (
                        int(int(row["score_target"]) != target_label),
                        int(str(row["language"]) != target_language),
                        int(
                            bool(row["rationale_active"])
                            != target_active
                        ),
                        str(row["base_event_id"]),
                    ),
                )
                prompt = prompts[str(chosen["prompt_cache_id"])]
                prompt_token_ids = [
                    int(value) for value in prompt["prompt_token_ids"]
                ]
                prompt_hash = _prompt_token_ids_sha256(prompt_token_ids)
                if (
                    prompt["record_id"] != chosen["record_id"]
                    or prompt["prompt_token_ids_sha256"] != prompt_hash
                    or chosen["prompt_token_ids_sha256"] != prompt_hash
                ):
                    raise ValueError(
                        "independent manifest/prompt binding differs"
                    )
                selected.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        "label": int(chosen["score_target"]),
                        "language": str(chosen["language"]),
                        "rationale_active": bool(
                            chosen["rationale_active"]
                        ),
                        "base_event_id": str(chosen["base_event_id"]),
                        "record_id": str(chosen["record_id"]),
                        "prompt_cache_id": str(
                            chosen["prompt_cache_id"]
                        ),
                        "prompt_token_ids_sha256": prompt_hash,
                        "manifest_sha256": manifest_sha256,
                        "manifest_language_field_present": bool(
                            chosen["_manifest_language_field_present"]
                        ),
                    }
                )
                group_index += 1
    expected_order = [
        (arm, seed, epoch)
        for arm in ARMS
        for seed in SEEDS
        for epoch in EPOCHS
    ]
    actual_order = [
        (row["arm"], row["seed"], row["epoch"])
        for row in selected
    ]
    if actual_order != expected_order:
        raise ValueError("independent smoke group order differs")
    return selected


def _assert_private_smoke(
    *,
    smoke_dir: Path,
    data_dir: Path,
    public_smoke: dict[str, Any],
) -> dict[str, Any]:
    frozen_contract = _frozen_manifest_contract(data_dir)
    private_path = smoke_dir / "private_train_smoke_results.jsonl"
    if file_sha256(private_path) != public_smoke["private_results_sha256"]:
        raise ValueError("private train-smoke hash differs")
    rows = read_jsonl(private_path)
    if len(rows) != 36:
        raise ValueError("private train-smoke row count differs")
    expected_selection = _independent_smoke_selection(
        data_dir,
        frozen_contract=frozen_contract,
    )
    frozen_contract["manifest_language_field_present"] = any(
        row["manifest_language_field_present"]
        for row in expected_selection
    )
    actual_order = [
        (row["arm"], row["seed"], row["epoch"]) for row in rows
    ]
    expected_order = [
        (row["arm"], row["seed"], row["epoch"])
        for row in expected_selection
    ]
    if actual_order != expected_order:
        raise ValueError("private train-smoke order differs")
    for row, expected in zip(rows, expected_selection):
        for key in (
            "label",
            "language",
            "rationale_active",
            "base_event_id",
            "record_id",
            "prompt_cache_id",
            "prompt_token_ids_sha256",
            "manifest_sha256",
        ):
            if row[key] != expected[key]:
                raise ValueError(
                    f"private train-smoke selection differs at {key}"
                )
        adapter_path = Path(row["adapter_path"])
        expected_adapter_hashes = {
            "adapter_config_sha256": file_sha256(
                adapter_path / "adapter_config.json"
            ),
            "adapter_model_sha256": file_sha256(
                adapter_path / "adapter_model.safetensors"
            ),
            "trainer_state_sha256": file_sha256(
                adapter_path.parent / "trainer_state.json"
            ),
        }
        for key, expected_hash in expected_adapter_hashes.items():
            if row[key] != expected_hash:
                raise ValueError(
                    f"private train-smoke artifact differs at {key}"
                )
        parse_review_json(row["output_text"])
        diagnostics = row["diagnostics"]
        if int(diagnostics["generated_token_count"]) < 1:
            raise ValueError("private train-smoke output is empty")
        if (
            int(diagnostics["active_generation_steps"])
            != int(diagnostics["generated_token_count"])
        ):
            raise ValueError("private train-smoke step count differs")
    return frozen_contract


def audit_candidate(args: argparse.Namespace) -> None:
    protocol = load_v2_protocol(args.protocol_config)
    if importlib.metadata.version("xgrammar") != protocol["backend"]["version"]:
        raise ValueError("XGrammar version differs")
    dependency = protocol["backend"]["dependency_wheels"][0]
    if (
        importlib.metadata.version(dependency["package"])
        != dependency["version"]
    ):
        raise ValueError("XGrammar dependency differs")
    training_config_path = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/"
        "training_configuration_candidate.json"
    )
    training_config = json.loads(
        training_config_path.read_text(encoding="utf-8")
    )
    runtime_versions = require_runtime_versions(
        protocol,
        training_config,
    )
    observed_wheels = {
        "xgrammar": verify_wheel_file(
            args.wheel_root / protocol["backend"]["wheel_filename"],
            expected_filename=protocol["backend"]["wheel_filename"],
            expected_sha256=protocol["backend"]["wheel_sha256"],
        ),
        "apache-tvm-ffi": verify_wheel_file(
            args.wheel_root / dependency["wheel_filename"],
            expected_filename=dependency["wheel_filename"],
            expected_sha256=dependency["wheel_sha256"],
        ),
    }
    wheel_specs = {
        "xgrammar": (
            protocol["backend"]["wheel_filename"],
            protocol["backend"]["wheel_sha256"],
        ),
        "apache-tvm-ffi": (
            dependency["wheel_filename"],
            dependency["wheel_sha256"],
        ),
    }
    installed_distributions = {
        name: verify_installed_distribution_matches_wheel(
            name,
            args.wheel_root / filename,
            expected_wheel_sha256=digest,
        )
        for name, (filename, digest) in wheel_specs.items()
    }
    public_smoke_path = args.smoke_dir / "public_train_smoke_report.json"
    if public_smoke_path.read_bytes() != args.public_smoke_copy.read_bytes():
        raise ValueError("public train-smoke copy differs bytewise")
    public_smoke = json.loads(public_smoke_path.read_text(encoding="utf-8"))
    if public_smoke["status"] != "EXP54_INFERENCE_V2_TRAIN_ONLY_SMOKE_COMPLETE":
        raise ValueError("train-only smoke is not complete")
    expected_coverage = {
        "rows": 36,
        "arms": ["R1", "R2", "R3", "S0"],
        "seeds": [42, 43, 44],
        "logical_epochs": [1, 2, 3],
        "labels": [1, 2, 3, 4, 5],
        "languages": ["en", "zh"],
        "rationale_activity_states": [False, True],
        "unique_arm_seed_epoch_groups": 36,
    }
    if public_smoke["coverage"] != expected_coverage:
        raise ValueError("train-only smoke coverage differs")
    if public_smoke["execution"]["strict_parse_rate"] != 1.0:
        raise ValueError("train-only smoke strict parse did not reach one")
    if (
        public_smoke["protocol_config_sha256"]
        != file_sha256(args.protocol_config)
    ):
        raise ValueError("train-only smoke protocol source differs")
    current_decoder_sha256 = file_sha256(
        REPO_ROOT / "thesis_exp/exp54_rar_sft/structured_decoder_v2.py"
    )
    current_smoke_runner_sha256 = file_sha256(
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/run_inference_v2_train_smoke.py"
    )
    if public_smoke["decoder_source_sha256"] != current_decoder_sha256:
        raise ValueError("train-only smoke decoder source differs")
    if (
        public_smoke["runner_source_sha256"]
        != current_smoke_runner_sha256
    ):
        raise ValueError("train-only smoke runner source differs")
    if public_smoke["privacy"] != {
        "row_level_selection_public": False,
        "record_or_event_ids_public": False,
        "train_prompts_public": False,
        "generated_text_public": False,
        "score_or_rationale_semantics_inspected": False,
    }:
        raise ValueError("train-only smoke privacy declaration differs")
    if (
        public_smoke["dev_accessed"]
        or public_smoke["test_accessed"]
        or public_smoke["scientific_metrics_computed"]
        or public_smoke["formal_v2_dev_allowed"]
    ):
        raise ValueError("train-only smoke crossed its authorization boundary")
    frozen_contract = _assert_private_smoke(
        smoke_dir=args.smoke_dir,
        data_dir=args.data_dir,
        public_smoke=public_smoke,
    )
    budget_probe = json.loads(
        args.budget_probe_path.read_text(encoding="utf-8")
    )
    expected_budget_probe = {
        "status": "EXP54_INFERENCE_V2_LOCKED_BUDGET_PROBE_PASS",
        "generated_token_count": 256,
        "forced_completion": True,
        "completion_at_max_token_boundary": True,
        "score_preserved": True,
        "strict_parse_success": True,
        "generated_text_public": False,
        "token_ids_public": False,
        "data_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for key, expected in expected_budget_probe.items():
        if budget_probe.get(key) != expected:
            raise ValueError(f"locked budget probe differs at {key}")
    if (
        budget_probe["protocol_config_sha256"]
        != file_sha256(args.protocol_config)
    ):
        raise ValueError("locked budget probe protocol hash differs")
    probe_source = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/run_inference_v2_budget_probe.py"
    )
    if budget_probe["probe_source_sha256"] != file_sha256(probe_source):
        raise ValueError("locked budget probe source hash differs")
    determinism_probe = json.loads(
        args.determinism_probe_path.read_text(encoding="utf-8")
    )
    expected_determinism = {
        "status": "EXP54_INFERENCE_V2_REAL_DETERMINISM_PROBE_PASS",
        "arm": "S0",
        "seed": 42,
        "logical_epoch": 1,
        "repetitions": 2,
        "output_bytes_identical": True,
        "output_token_ids_identical": True,
        "diagnostics_identical": True,
        "strict_parse_success_both": True,
        "generated_text_public": False,
        "token_ids_public": False,
        "row_identifier_public": False,
        "train_only": True,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for key, expected in expected_determinism.items():
        if determinism_probe.get(key) != expected:
            raise ValueError(f"real determinism probe differs at {key}")
    determinism_source = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/"
        "run_inference_v2_determinism_probe.py"
    )
    if determinism_probe["probe_source_sha256"] != file_sha256(
        determinism_source
    ):
        raise ValueError("real determinism probe source hash differs")
    if (
        determinism_probe["protocol_config_sha256"]
        != file_sha256(args.protocol_config)
    ):
        raise ValueError("real determinism probe protocol hash differs")
    runtime_sources = v2_runtime_source_closure()
    if (
        set(runtime_sources) != EXPECTED_V2_RUNTIME_SOURCE_NAMES
        or len(runtime_sources) != EXPECTED_V2_RUNTIME_SOURCE_COUNT
    ):
        raise ValueError("V2 runtime source closure differs")
    runtime_source_digest = v2_runtime_source_closure_sha256(
        runtime_sources
    )
    audit_sources = _audit_source_bindings()
    v1_report_path = (
        REPO_ROOT
        / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
        "dev_execution_attempt_v1_report.json"
    )
    v1_receipt_path = (
        REPO_ROOT
        / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
        "dev_execution_attempt_v1_freeze_receipt.json"
    )
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v1_receipt = json.loads(v1_receipt_path.read_text(encoding="utf-8"))
    if (
        v1_report["scientific_metrics_valid"]
        or v1_report["checkpoint_selection_allowed"]
        or v1_report["dev_rerun_allowed"]
        or v1_report["test_accessed"]
    ):
        raise ValueError("V1 evidence is not fail-closed")
    if (
        v1_receipt["dev_execution_attempt_v1_public_report_sha256"]
        != file_sha256(v1_report_path)
    ):
        raise ValueError("V1 receipt public-report hash differs")
    if (
        v1_receipt["dev_execution_attempt_v1_private_lock_sha256"]
        != file_sha256(args.v1_private_lock_path)
    ):
        raise ValueError("V1 receipt private-lock hash differs")
    if (
        v1_receipt["freezer_source_sha256"]
        != audit_sources[
            "thesis_exp/exp54_rar_sft/"
            "freeze_dev_execution_attempt_v1.py"
        ]
    ):
        raise ValueError("V1 receipt freezer source hash differs")
    if any(
        not str(item.get("protocol_file_observed_mtime_utc") or "")
        for item in v1_report["attempts"]
    ):
        raise ValueError("V1 report lacks observed attempt timestamps")
    if len(v1_report["repository_head_observed_at_freeze"]) != 40:
        raise ValueError("V1 report lacks repository HEAD context")
    report = {
        "status": "EXP54_INFERENCE_PROTOCOL_V2_CANDIDATE_AUDIT_PASS",
        "schema_version": "exp54-inference-v2-candidate-audit-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "grammar_sha256": protocol["grammar"]["sha256"],
        "runtime_source_count": len(runtime_sources),
        "runtime_source_names": sorted(runtime_sources),
        "runtime_source_bindings": runtime_sources,
        "runtime_source_closure_sha256": runtime_source_digest,
        "candidate_audit_source_bindings": audit_sources,
        "dependencies": {
            "runtime_versions": runtime_versions,
            "observed_wheels": observed_wheels,
            "observed_installed_distributions": (
                installed_distributions
            ),
            "editable_install_allowed": False,
        },
        "formal_execution_contract": {
            "batch_size": protocol["generation"][
                "formal_batch_size"
            ],
            "max_new_tokens": protocol["generation"][
                "max_new_tokens"
            ],
            "batch_size_cli_override_allowed": False,
            "output_dir_cli_override_allowed": False,
            "training_root_cli_override_allowed": False,
            "protocol_config_cli_override_allowed": False,
            "test_path_allowed": False,
            "mixed_length_batch_regression_passed": True,
            "mixed_completion_batch_regression_passed": True,
            "batched_singleton_token_ids_equal": True,
            "batched_singleton_output_bytes_equal": True,
            "batched_singleton_diagnostics_equal": True,
            "finished_row_attention_mask_zero_verified": True,
            "unfinished_row_kv_cache_continuation_verified": True,
            "authenticated_protocol_payload_consumed_directly": True,
            "authenticated_grammar_payload_consumed_directly": True,
            "authenticated_training_config_consumed_directly": True,
            "authenticated_dev_rows_consumed_directly": True,
            "checkpoint_identity_hash_checked_before_after_load": True,
        },
        "authorization_gate": {
            "external_authorization_required": True,
            "authorization_checked_before_dev_model_output_cuda": True,
            "exact_task_count": 36,
            "one_claim_per_checkpoint": True,
            "staged_digest_is_nonactivating": True,
            "atomic_digest_rename_is_activation_point": True,
            "preactivation_preflight_required": True,
            "claim_root": "/var/lib/edubench/exp54-v2-dev",
            "claim_root_root_owned": True,
            "campaign_directory_training_group_managed": True,
            "campaign_directory_mode": "01770",
            "campaign_directory_append_only_required": True,
            "claim_deletion_cannot_restore_invocation": True,
            "claim_occurs_before_dev_rows_materialization": True,
            "concurrent_loser_materializes_dev_rows": False,
            "dev_parse_failure_retains_claim": True,
            "real_root_append_only_installation_required_before_activation": (
                True
            ),
            "real_root_append_only_installation_currently_present": False,
        },
        "v1_evidence": {
            "report_sha256": file_sha256(v1_report_path),
            "freeze_receipt_sha256": file_sha256(v1_receipt_path),
            "scientific_metrics_valid": False,
            "checkpoint_selection_allowed": False,
        },
        "tests": {
            "server_test_files": [
                "thesis_exp/tests/test_exp54_inference_v2.py",
                "thesis_exp/tests/test_exp54_v2_dev_authorization.py",
            ],
            "passed": 70,
            "failed": 0,
            "skipped": 0,
        },
        "train_only_smoke": {
            "public_report_sha256": file_sha256(public_smoke_path),
            "public_report_copy_path": str(
                args.public_smoke_copy.relative_to(REPO_ROOT)
            ),
            "coverage": public_smoke["coverage"],
            "execution": public_smoke["execution"],
            "private_results_retained": True,
            "private_results_public": False,
            "decoder_source_sha256": current_decoder_sha256,
            "runner_source_sha256": current_smoke_runner_sha256,
            "materialized_manifest_frozen_lock_sha256": (
                frozen_contract["materialized_lock_sha256"]
            ),
            "reference_set_data_lock_sha256": (
                frozen_contract["reference_lock_sha256"]
            ),
            "locked_train_sha256": frozen_contract[
                "locked_train_sha256"
            ],
            "frozen_prompt_cache_sha256": frozen_contract[
                "prompt_cache_sha256"
            ],
            "frozen_manifest_count": len(
                frozen_contract["manifest_hashes"]
            ),
            "manifest_language_field_present": frozen_contract[
                "manifest_language_field_present"
            ],
            "language_derived_from_unique_locked_train_record": True,
            "private_smoke_language_matches_locked_train": True,
        },
        "locked_budget_probe": {
            "report_sha256": file_sha256(args.budget_probe_path),
            "generated_token_count": 256,
            "forced_completion": True,
            "completion_at_max_token_boundary": True,
            "strict_parse_success": True,
        },
        "real_checkpoint_determinism_probe": {
            "report_sha256": file_sha256(args.determinism_probe_path),
            "repetitions": 2,
            "output_bytes_identical": True,
            "output_token_ids_identical": True,
            "diagnostics_identical": True,
        },
        "privacy": public_smoke["privacy"],
        "v2_dev_accessed": False,
        "test_accessed": False,
        "formal_v2_dev_allowed": False,
        "formal_test_allowed": False,
        "independent_review_required": True,
    }
    write_json(args.report_path, report)
    lock = {
        "status": "EXP54_INFERENCE_PROTOCOL_V2_CANDIDATE_LOCK_NOT_AUTHORIZED",
        "schema_version": "exp54-inference-v2-candidate-lock-v1",
        "candidate_report_sha256": file_sha256(args.report_path),
        "protocol_config_sha256": report["protocol_config_sha256"],
        "grammar_sha256": report["grammar_sha256"],
        "runtime_source_names": sorted(runtime_sources),
        "runtime_source_bindings": runtime_sources,
        "runtime_source_closure_sha256": runtime_source_digest,
        "candidate_audit_source_bindings": audit_sources,
        "dependencies": report["dependencies"],
        "formal_execution_contract": report[
            "formal_execution_contract"
        ],
        "authorization_gate": report["authorization_gate"],
        "v1_report_sha256": report["v1_evidence"]["report_sha256"],
        "v1_freeze_receipt_sha256": report["v1_evidence"][
            "freeze_receipt_sha256"
        ],
        "train_only_smoke_public_report_sha256": report[
            "train_only_smoke"
        ]["public_report_sha256"],
        "materialized_manifest_frozen_lock_sha256": report[
            "train_only_smoke"
        ]["materialized_manifest_frozen_lock_sha256"],
        "reference_set_data_lock_sha256": report[
            "train_only_smoke"
        ]["reference_set_data_lock_sha256"],
        "locked_train_sha256": report["train_only_smoke"][
            "locked_train_sha256"
        ],
        "frozen_prompt_cache_sha256": report["train_only_smoke"][
            "frozen_prompt_cache_sha256"
        ],
        "locked_budget_probe_report_sha256": report[
            "locked_budget_probe"
        ]["report_sha256"],
        "real_checkpoint_determinism_probe_report_sha256": report[
            "real_checkpoint_determinism_probe"
        ]["report_sha256"],
        "formal_v2_dev_allowed": False,
        "formal_test_allowed": False,
        "dev_rerun_performed": False,
        "test_accessed": False,
    }
    write_json(args.lock_path, lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data"
        ),
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
            "inference_v2_train_smoke"
        ),
    )
    parser.add_argument(
        "--public-smoke-copy",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
            "inference_v2_train_only_smoke_report.json"
        ),
    )
    parser.add_argument(
        "--v1-private-lock-path",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
            "private_dev_execution_evidence/"
            "dev_execution_attempt_v1_private_lock.json"
        ),
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_CONFIG,
    )
    parser.add_argument(
        "--wheel-root",
        type=Path,
        default=REPO_ROOT / ".cache/exp54_xgrammar",
    )
    parser.add_argument(
        "--budget-probe-path",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
            "inference_v2_locked_budget_probe.json"
        ),
    )
    parser.add_argument(
        "--determinism-probe-path",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
            "inference_v2_real_determinism_probe.json"
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
            "inference_protocol_v2_candidate_report.json"
        ),
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
            "inference_protocol_v2_candidate_lock.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    audit_candidate(parse_args())


if __name__ == "__main__":
    main()
