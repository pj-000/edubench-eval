"""Freeze Exp57 Stage 1 source hashes after all pre-training gates pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import CONFIG_ROOT, OUTPUT_ROOT, REPO_ROOT
from thesis_exp.exp57_cbrd.data_audit import write_json


SOURCE_FILES = (
    "thesis_exp/exp57_cbrd/__init__.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/metrics.py",
    "thesis_exp/exp57_cbrd/train.py",
    "thesis_exp/exp57_cbrd/gate.py",
    "thesis_exp/configs/exp57_cbrd/stage0_source_lock.json",
    "thesis_exp/configs/exp57_cbrd/stage1_protocol.json",
    "thesis_exp/configs/exp57_cbrd/protocol_deviation_a6000_20260802.json",
    "thesis_exp/configs/exp57_cbrd/protocol_deviation_parallel_seeds_3090_20260802.json",
    "thesis_exp/outputs/exp57_cbrd/decision/stage0_decision.json",
    "thesis_exp/scripts/run_exp57_cbrd_stage1_job.sh",
    "thesis_exp/scripts/run_exp57_cbrd_stage1_a6000_job.sh",
    "thesis_exp/tests/test_exp57_cbrd_stage0.py",
    "thesis_exp/tests/test_exp57_cbrd_stage1.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
    "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    stage0_path = OUTPUT_ROOT / "decision" / "stage0_decision.json"
    accumulation_path = OUTPUT_ROOT / "audit" / "stage1_real_qwen3_accumulation_scale_routes.json"
    stage0 = read_json(stage0_path)
    accumulation = read_json(accumulation_path)
    smoke_paths = {
        "consensus_only": OUTPUT_ROOT / "smoke_remote" / "smoke" / "consensus_only" / "run_summary.json",
        "dual_hard": OUTPUT_ROOT / "smoke_remote" / "smoke" / "dual_hard" / "run_summary.json",
        "routed_hmsa": OUTPUT_ROOT / "smoke_remote" / "smoke" / "routed_hmsa" / "run_summary.json",
        "shuffled_residual": OUTPUT_ROOT / "smoke_remote" / "smoke" / "shuffled_residual" / "run_summary.json",
        "detached_soft": OUTPUT_ROOT / "smoke_remote" / "smoke" / "detached_soft" / "run_summary.json",
        "residual_only": OUTPUT_ROOT / "smoke_remote" / "smoke32" / "residual_only" / "run_summary.json",
        "sign_flipped": OUTPUT_ROOT / "smoke_remote" / "smoke32" / "sign_flipped" / "run_summary.json",
    }
    smoke = {variant: read_json(path) for variant, path in smoke_paths.items()}
    checks = {
        "stage0_passed": stage0.get("status") == "STAGE0_PASS_READY_TO_FREEZE_STAGE1",
        "accumulation_scale_route_audit_passed": (
            accumulation.get("status") == "PASS"
            and float(accumulation.get("loss_scale")) == 1.0 / 32.0
        ),
        "all_seven_smoke_paths_completed": all(
            value.get("status") == "COMPLETED" for value in smoke.values()
        ),
        "all_smoke_paths_train_dev_only": all(
            int(value.get("test_access_count", -1)) == 0 for value in smoke.values()
        ),
        "shuffled_smoke_mapping_matches": (
            smoke["shuffled_residual"]["shuffle_contract"]["mapping_sha256"]
            == "b4e96c49607700be99816582c1b85a8085b8c5abb260ddafbad4e9ee0dc25ad4"
        ),
        "source_files_present": all((REPO_ROOT / relative).is_file() for relative in SOURCE_FILES),
        "no_test_access": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage 1 cannot be frozen: {checks}")
    files = {relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}
    lock = {
        "status": "EXP57_STAGE1_SOURCE_LOCK_FROZEN",
        "files": files,
        "stage0_decision_sha256": sha256(stage0_path),
        "accumulation_route_audit_sha256": sha256(accumulation_path),
        "protocol_sha256": sha256(CONFIG_ROOT / "stage1_protocol.json"),
        "test_access_count": 0,
    }
    lock_path = CONFIG_ROOT / "stage1_source_lock.json"
    write_json(lock_path, lock)
    decision = {
        "status": "STAGE1_PILOT_AUTHORIZED",
        "authorized_runs": {
            "scientific_variants": [
                "dual_hard",
                "consensus_only",
                "routed_hmsa",
                "residual_only",
                "sign_flipped",
                "shuffled_residual",
            ],
            "seed": 42,
            "technical_control": "detached_soft seed 42",
        },
        "checks": checks,
        "source_lock_sha256": sha256(lock_path),
        "interpretation": "Formal seed 42 may screen only integrity and catastrophic failure. It may not be used to prune a small effect or claim the CBRD mechanism.",
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "decision" / "stage1_preflight_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
