"""Create the formal Exp60 source lock only after reviewed GPU preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    OUTPUT_ROOT,
    PREFLIGHT_SOURCE_LOCK_PATH,
    PROTOCOL_PATH,
    REAL_PREFLIGHT_DECISION_PATH,
    REPO_ROOT,
    SOURCE_LOCK_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.contract import (
    normalized_scientific_protocol_sha256,
)


STATIC_FILES = (
    "thesis_exp/__init__.py",
    "thesis_exp/src/__init__.py",
    "thesis_exp/src/edujudge/__init__.py",
    "thesis_exp/src/edujudge/utils/__init__.py",
    "thesis_exp/configs/exp60_geometry_matched_shuffle/protocol_draft.json",
    "thesis_exp/exp60_geometry_matched_shuffle/__init__.py",
    "thesis_exp/exp60_geometry_matched_shuffle/contract.py",
    "thesis_exp/exp60_geometry_matched_shuffle/freeze_preflight_contract.py",
    "thesis_exp/exp60_geometry_matched_shuffle/mapping.py",
    "thesis_exp/exp60_geometry_matched_shuffle/geometry.py",
    "thesis_exp/exp60_geometry_matched_shuffle/preflight.py",
    "thesis_exp/exp60_geometry_matched_shuffle/real_model_preflight.py",
    "thesis_exp/exp60_geometry_matched_shuffle/finalize_real_preflight.py",
    "thesis_exp/exp60_geometry_matched_shuffle/freeze_source_lock.py",
    "thesis_exp/exp60_geometry_matched_shuffle/train.py",
    "thesis_exp/exp60_geometry_matched_shuffle/analyze_confirmation.py",
    "thesis_exp/exp60_geometry_matched_shuffle/trainer_static_audit.py",
    "thesis_exp/exp60_geometry_matched_shuffle/source_closure_audit.py",
    "thesis_exp/tests/test_exp60_geometry_matched_shuffle.py",
    "thesis_exp/tests/test_exp60_torch_geometry.py",
    "thesis_exp/exp57_cbrd/__init__.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/metrics.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/train.py",
    "thesis_exp/exp59_residual_geometry/geometry.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/exp02/__init__.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/src/edujudge/utils/text_norm.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP60_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS":
        raise RuntimeError("Protocol must be frozen exactly once before source locking")
    bindings = protocol["formal_runs"].get("physical_gpu_bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings) != {"gpu_slot_0", "gpu_slot_1", "gpu_slot_2"}
        or len({str(value) for value in bindings.values()}) != 3
    ):
        raise RuntimeError("Three distinct physical GPU bindings must be frozen")
    decision = json.loads(REAL_PREFLIGHT_DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "EXP60_REAL_MODEL_PREFLIGHT_ALL_SEEDS_PASS":
        raise RuntimeError("All-seed real-model preflight has not passed")
    preflight_lock = json.loads(PREFLIGHT_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if sha256_file(PREFLIGHT_SOURCE_LOCK_PATH) != decision[
        "preflight_contract_binding"
    ]["preflight_source_lock_sha256"]:
        raise RuntimeError("Preflight source lock differs from reviewed preflight reports")
    normalized_sha = normalized_scientific_protocol_sha256(protocol)
    if normalized_sha != decision["preflight_contract_binding"][
        "normalized_scientific_protocol_sha256"
    ] or normalized_sha != preflight_lock["normalized_scientific_protocol_sha256"]:
        raise RuntimeError("Scientific protocol changed after real-model preflight")
    expected_bindings = {
        slot: str(identity["cuda_visible_devices"])
        for slot, identity in decision["preflight_gpu_bindings"].items()
    }
    if bindings != expected_bindings:
        raise RuntimeError(
            "Formal physical GPU bindings must equal the three preflighted devices"
        )
    protocol_relative = str(PROTOCOL_PATH.relative_to(REPO_ROOT))
    for relative, expected in preflight_lock["files"].items():
        if relative == protocol_relative:
            continue
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Source changed after real-model preflight: {relative}")
    generated = [
        str(MAPPING_PATH.relative_to(REPO_ROOT)),
        str(MAPPING_AUDIT_PATH.relative_to(REPO_ROOT)),
        str(REAL_PREFLIGHT_DECISION_PATH.relative_to(REPO_ROOT)),
        str(PREFLIGHT_SOURCE_LOCK_PATH.relative_to(REPO_ROOT)),
        *(
            str(
                (
                    OUTPUT_ROOT
                    / "real_model_preflight"
                    / f"seed_{seed}"
                    / "real_model_no_update_preflight.json"
                ).relative_to(REPO_ROOT)
            )
            for seed in (47, 48, 49)
        ),
    ]
    relative_files = list(dict.fromkeys([*STATIC_FILES, *generated]))
    missing = [relative for relative in relative_files if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing Exp60 lock inputs: " + ", ".join(missing))
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in relative_files}
    if "thesis_exp/exp60_geometry_matched_shuffle/analyze_confirmation.py" not in files:
        raise AssertionError("Frozen analysis must be source-locked")
    lock = {
        "status": "EXP60_FORMAL_SOURCE_LOCK",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "real_model_preflight_decision_sha256": sha256_file(
            REAL_PREFLIGHT_DECISION_PATH
        ),
        "mapping_sha256": decision.get("mapping_sha256")
        or json.loads(MAPPING_AUDIT_PATH.read_text(encoding="utf-8"))["mapping_sha256"],
        "files": files,
        "file_count": len(files),
        "contains_frozen_analysis": True,
        "normalized_scientific_protocol_sha256": normalized_sha,
        "physical_gpu_bindings_equal_preflight_devices": True,
        "contains_model_and_environment_manifests_via_seed_reports": True,
        "allowed_splits": ["train", "dev"],
        "test_access_count": 0,
    }
    write_json(SOURCE_LOCK_PATH, lock)
    return lock


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
