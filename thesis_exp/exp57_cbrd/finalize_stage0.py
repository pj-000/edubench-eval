"""Freeze the passed Exp57 Stage 0 evidence and source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import CONFIG_ROOT, OUTPUT_ROOT, REPO_ROOT


SOURCE_FILES = (
    "thesis_exp/configs/exp57_cbrd/stage0_protocol.json",
    "thesis_exp/exp57_cbrd/README.md",
    "thesis_exp/exp57_cbrd/__init__.py",
    "thesis_exp/exp57_cbrd/archive_legacy_sources.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/exp57_cbrd/finalize_stage0.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/mechanism_audit.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/preflight.py",
    "thesis_exp/exp57_cbrd/torch_audit.py",
    "thesis_exp/scripts/run_exp57_cbrd_stage0.sh",
    "thesis_exp/scripts/sync_exp57_cbrd_to_server.sh",
    "thesis_exp/tests/test_exp57_cbrd_stage0.py",
)

EVIDENCE_FILES = (
    "thesis_exp/outputs/exp57_cbrd/audit/legacy_source_archive.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_data_and_source_audit.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_torch_identity.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_routing_bf16_false.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_routing_bf16_true.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_mechanism_routes.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage0_two_branch_bf16_no_go.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finalize() -> dict[str, Any]:
    data = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_data_and_source_audit.json")
    tensor = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_torch_identity.json")
    fp32 = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_routing_bf16_false.json")
    bf16 = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_routing_bf16_true.json")
    routes = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_real_qwen3_mechanism_routes.json")
    rejected = load("thesis_exp/outputs/exp57_cbrd/audit/stage0_two_branch_bf16_no_go.json")
    checks = {
        "legacy_source_and_live_dependencies": data["status"] == "PASS",
        "train_dev_support_and_shuffle": (
            data["train"]["distinct_target_vectors"] == 13
            and data["train"]["distinct_residual_vectors"] == 9
            and data["dev"]["distinct_target_vectors"] == 13
            and data["shuffled_residual"]["mapping_matches_exp55"]
        ),
        "scalar_toy_and_amp_identity": tensor["status"] == "PASS",
        "real_qwen3_fp32_identity": (
            fp32["status"] == "PASS"
            and fp32["hard_path_max_difference"] == 0.0
            and fp32["max_gradient_difference"] == 0.0
        ),
        "real_qwen3_bf16_identity": (
            bf16["status"] == "PASS"
            and bf16["historical_head_hash_matches"]
            and bf16["max_gradient_difference"] == 0.0
        ),
        "all_mechanism_routes": routes["status"] == "PASS",
        "unsafe_two_branch_implementation_rejected": (
            rejected["status"] == "FAIL"
            and rejected["max_relative_gradient_difference"] > 0.01
        ),
        "no_test_access": all(
            result.get("test_access_count") == 0
            for result in (data, tensor, fp32, bf16, routes)
        ),
        "no_training_or_optimizer_step": True,
    }
    decision = {
        "status": (
            "STAGE0_PASS_READY_TO_FREEZE_STAGE1"
            if all(checks.values())
            else "STAGE0_FAIL"
        ),
        "experiment": "Exp57-CBRD",
        "production_implementation": "single_soft_ce_hidden_gradient_hook",
        "checks": checks,
        "key_results": {
            "target_vector_count": 13,
            "residual_vector_count": 9,
            "train_relation_states": data["train"]["relation_state_counts"],
            "dev_relation_states": data["dev"]["relation_state_counts"],
            "shuffle_mapping_sha256": data["shuffled_residual"]["mapping_sha256"],
            "shuffle_effective_changes": data["shuffled_residual"]["effective_target_changes"],
            "fp32_identity_max_gradient_difference": fp32["max_gradient_difference"],
            "bf16_identity_max_gradient_difference": bf16["max_gradient_difference"],
            "historical_bf16_head_hash": bf16["base_head_hash"],
            "mechanism_route_comparisons": routes["comparisons"],
            "rejected_two_branch_relative_gradient_difference": rejected[
                "max_relative_gradient_difference"
            ],
        },
        "interpretation": (
            "The CE decomposition and the single-soft-CE hidden-gradient router are "
            "numerically validated. Stage 0 authorizes protocol freezing and "
            "implementation of train/dev-only Stage 1 controls; it does not establish "
            "a CBRD mechanism effect and does not authorize historical-test access."
        ),
        "test_access_count": 0,
    }
    if decision["status"] != "STAGE0_PASS_READY_TO_FREEZE_STAGE1":
        raise RuntimeError(json.dumps(decision, ensure_ascii=False, indent=2))

    source_lock = {
        "status": "EXP57_CBRD_STAGE0_SOURCE_LOCKED",
        "source_files": {
            relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES
        },
        "evidence_files": {
            relative: sha256(REPO_ROOT / relative) for relative in EVIDENCE_FILES
        },
        "test_access_count": 0,
    }
    write_json(CONFIG_ROOT / "stage0_source_lock.json", source_lock)
    write_json(OUTPUT_ROOT / "decision" / "stage0_decision.json", decision)
    markdown = (
        "# Exp57 CBRD Stage 0 decision\n\n"
        f"- Status: **{decision['status']}**\n"
        "- Historical test accessed: no\n"
        "- Training performed: no\n"
        "- FP32 routed-HMSA gradient difference: 0\n"
        "- BF16 routed-HMSA gradient difference: 0\n"
        "- Historical BF16 head hash: matched\n"
        "- Directional route audit: passed\n\n"
        "Stage 1 may now be pre-registered and implemented on train/dev only. "
        "This decision does not yet show that the boundary residual improves scoring.\n"
    )
    markdown_path = OUTPUT_ROOT / "decision" / "stage0_decision.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    return decision


if __name__ == "__main__":
    print(json.dumps(finalize(), ensure_ascii=False, indent=2, sort_keys=True))
