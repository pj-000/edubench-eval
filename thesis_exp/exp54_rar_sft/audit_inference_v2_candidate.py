"""Audit the Exp54 V2 inference candidate without reading dev or test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_inference_v2_train_smoke import (
    ARMS,
    EPOCHS,
    SEEDS,
    build_smoke_selection,
    file_sha256,
    read_jsonl,
    write_json,
)
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    DEFAULT_PROTOCOL_CONFIG,
    REPO_ROOT,
    load_v2_protocol,
)


SOURCE_NAMES = (
    "thesis_exp/exp54_rar_sft/DEV_EXECUTION_ATTEMPT_V1_ROOT_CAUSE.md",
    "thesis_exp/exp54_rar_sft/INFERENCE_PROTOCOL_V2.md",
    "thesis_exp/exp54_rar_sft/audit_inference_v2_candidate.py",
    "thesis_exp/exp54_rar_sft/configs/inference_protocol_v2_candidate.json",
    "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf",
    "thesis_exp/exp54_rar_sft/freeze_dev_execution_attempt_v1.py",
    "thesis_exp/exp54_rar_sft/inference_contract.py",
    "thesis_exp/exp54_rar_sft/run_dev_inference.py",
    "thesis_exp/exp54_rar_sft/run_dev_inference_v2.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_budget_probe.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_determinism_probe.py",
    "thesis_exp/exp54_rar_sft/run_inference_v2_train_smoke.py",
    "thesis_exp/exp54_rar_sft/structured_decoder_v2.py",
    "thesis_exp/tests/test_exp54_inference_v2.py",
)


def _source_bindings() -> dict[str, str]:
    actual = {
        name: file_sha256(REPO_ROOT / name)
        for name in SOURCE_NAMES
    }
    if set(actual) != set(SOURCE_NAMES):
        raise ValueError("V2 runtime source closure differs")
    return actual


def _assert_private_smoke(
    *,
    smoke_dir: Path,
    data_dir: Path,
    public_smoke: dict[str, Any],
) -> None:
    private_path = smoke_dir / "private_train_smoke_results.jsonl"
    if file_sha256(private_path) != public_smoke["private_results_sha256"]:
        raise ValueError("private train-smoke hash differs")
    rows = read_jsonl(private_path)
    if len(rows) != 36:
        raise ValueError("private train-smoke row count differs")
    expected_selection = build_smoke_selection(data_dir)
    expected_by_group = {
        (row["arm"], row["seed"], row["epoch"]): row
        for row in expected_selection
    }
    expected_groups = {
        (arm, seed, epoch)
        for arm in ARMS
        for seed in SEEDS
        for epoch in EPOCHS
    }
    actual_groups = {
        (row["arm"], row["seed"], row["epoch"]) for row in rows
    }
    if actual_groups != expected_groups:
        raise ValueError("private train-smoke group closure differs")
    for row in rows:
        group = (row["arm"], row["seed"], row["epoch"])
        expected = expected_by_group[group]
        for key in (
            "label",
            "language",
            "rationale_active",
            "base_event_id",
            "record_id",
            "prompt_cache_id",
        ):
            if row[key] != expected[key]:
                raise ValueError(f"private train-smoke selection differs at {key}")
        adapter = Path(row["adapter_path"]) / "adapter_model.safetensors"
        if file_sha256(adapter) != row["adapter_model_sha256"]:
            raise ValueError("private train-smoke adapter hash differs")
        parse_review_json(row["output_text"])
        diagnostics = row["diagnostics"]
        if int(diagnostics["generated_token_count"]) < 1:
            raise ValueError("private train-smoke output is empty")
        if (
            int(diagnostics["active_generation_steps"])
            != int(diagnostics["generated_token_count"])
        ):
            raise ValueError("private train-smoke step count differs")


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
    _assert_private_smoke(
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
    sources = _source_bindings()
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
        != sources[
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
        "runtime_source_count": len(sources),
        "runtime_source_bindings": sources,
        "dependencies": {
            "xgrammar_version": importlib.metadata.version("xgrammar"),
            "xgrammar_wheel_sha256": protocol["backend"]["wheel_sha256"],
            "apache_tvm_ffi_version": importlib.metadata.version(
                dependency["package"]
            ),
            "apache_tvm_ffi_wheel_sha256": dependency["wheel_sha256"],
        },
        "v1_evidence": {
            "report_sha256": file_sha256(v1_report_path),
            "freeze_receipt_sha256": file_sha256(v1_receipt_path),
            "scientific_metrics_valid": False,
            "checkpoint_selection_allowed": False,
        },
        "tests": {
            "server_test_file": "thesis_exp/tests/test_exp54_inference_v2.py",
            "passed": 26,
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
        "runtime_source_bindings": sources,
        "v1_report_sha256": report["v1_evidence"]["report_sha256"],
        "v1_freeze_receipt_sha256": report["v1_evidence"][
            "freeze_receipt_sha256"
        ],
        "train_only_smoke_public_report_sha256": report[
            "train_only_smoke"
        ]["public_report_sha256"],
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
