"""Freeze the Exp58 implementation closure before any formal result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp58_matched_clipping import (
    OUTPUT_ROOT,
    PROTOCOL_PATH,
    REPO_ROOT,
    SOURCE_LOCK_PATH,
)


SOURCE_FILES = (
    "thesis_exp/configs/exp58_matched_clipping/protocol.json",
    "thesis_exp/exp58_matched_clipping/__init__.py",
    "thesis_exp/exp58_matched_clipping/README.md",
    "thesis_exp/exp58_matched_clipping/matched_update.py",
    "thesis_exp/exp58_matched_clipping/train.py",
    "thesis_exp/exp58_matched_clipping/preflight.py",
    "thesis_exp/scripts/run_exp58_matched_job.sh",
    "thesis_exp/tests/test_exp58_matched_clipping.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/train.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
    "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl",
    "thesis_exp/outputs/exp57_cbrd/decision/stage1_confirmation_decision.json",
    "thesis_exp/outputs/exp57_cbrd/audit/stage1_clip_gradient_audit.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP58_MATCHED_CLIPPING_PROTOCOL_FROZEN_BEFORE_RESULTS":
        raise RuntimeError("Protocol is not frozen")
    missing = [relative for relative in SOURCE_FILES if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    files = {relative: sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}
    lock = {
        "status": "EXP58_SOURCE_LOCK_FROZEN_BEFORE_PREFLIGHT_AND_RESULTS",
        "files": files,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "test_access_count": 0,
    }
    write_json(SOURCE_LOCK_PATH, lock)
    decision = {
        "status": "EXP58_CPU_GATES_PASS_READY_FOR_REAL_MODEL_PREFLIGHT",
        "source_lock_sha256": sha256(SOURCE_LOCK_PATH),
        "formal_training_authorized": False,
        "next_required_step": "real Qwen3 full-accumulation no-update preflight",
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "decision" / "cpu_preflight_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
