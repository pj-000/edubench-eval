"""Freeze the Exp59 implementation closure before any real-model result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp59_residual_geometry import (
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REPO_ROOT,
    SOURCE_LOCK_PATH,
)


SOURCE_FILES = (
    "thesis_exp/configs/exp59_residual_geometry/protocol.json",
    "thesis_exp/configs/exp59_residual_geometry/implementation_amendment_v2.json",
    "thesis_exp/outputs/exp59_residual_geometry/decision/seed42_v1_implementation_stop.json",
    "thesis_exp/exp59_residual_geometry/__init__.py",
    "thesis_exp/exp59_residual_geometry/README.md",
    "thesis_exp/exp59_residual_geometry/geometry.py",
    "thesis_exp/exp59_residual_geometry/train.py",
    "thesis_exp/exp59_residual_geometry/preflight.py",
    "thesis_exp/exp59_residual_geometry/endpoint_parity.py",
    "thesis_exp/scripts/run_exp59_geometry_job.sh",
    "thesis_exp/tests/test_exp59_residual_geometry.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/train.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/exp57_cbrd/preflight.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
    "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl",
    "thesis_exp/configs/exp57_cbrd/stage1_protocol.json",
    "thesis_exp/configs/exp57_cbrd/stage1_source_lock.json",
    "thesis_exp/outputs/exp57_cbrd/decision/stage1_confirmation_decision.json",
    "thesis_exp/outputs/exp58_matched_clipping/decision/seed42_stop_decision.json",
    "thesis_exp/outputs/exp59_residual_geometry/audit/endpoint_parity.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP59_RESIDUAL_GEOMETRY_PROTOCOL_FROZEN_BEFORE_REAL_MODEL_RESULTS":
        raise RuntimeError("Exp59 protocol is not frozen")
    missing = [relative for relative in SOURCE_FILES if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    lock = {
        "status": "EXP59_SOURCE_LOCK_V2_REFROZEN_AFTER_NUMERICAL_AMENDMENT_BEFORE_SEED42_RERUN",
        "files": {relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES},
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "test_access_count": 0,
    }
    write_json(SOURCE_LOCK_PATH, lock)
    decision = {
        "status": "EXP59_V2_CPU_GATES_PASS_READY_FOR_REAL_MODEL_PREFLIGHT",
        "source_lock_sha256": sha256(SOURCE_LOCK_PATH),
        "formal_training_authorized": False,
        "next_required_step": "repeat real Qwen3 dual-precision full-accumulation no-update preflight for the FP64 scalar-reduction amendment",
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "decision" / "cpu_preflight_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
