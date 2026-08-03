"""Generate the no-training-authority source lock used by real preflight."""

from __future__ import annotations

import json
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    PREFLIGHT_SOURCE_LOCK_PATH,
    PROTOCOL_PATH,
    REPO_ROOT,
)
from thesis_exp.exp60_geometry_matched_shuffle.contract import (
    ALLOWED_POST_PREFLIGHT_PROTOCOL_CHANGES,
    normalized_scientific_protocol_sha256,
    sha256_file,
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
    "thesis_exp/src/edujudge/exp02/__init__.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/src/edujudge/utils/text_norm.py",
)


def run() -> dict[str, Any]:
    if PREFLIGHT_SOURCE_LOCK_PATH.exists():
        raise FileExistsError(
            "Exp60 preflight source lock already exists and is immutable; "
            "a revised experiment requires an explicitly reviewed new lock"
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP60_DRAFT_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_TRAINING":
        raise RuntimeError("Preflight lock requires the no-training-authority draft")
    generated = (
        str(MAPPING_PATH.relative_to(REPO_ROOT)),
        str(MAPPING_AUDIT_PATH.relative_to(REPO_ROOT)),
    )
    relative_files = list(dict.fromkeys((*STATIC_FILES, *generated)))
    missing = [relative for relative in relative_files if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing Exp60 preflight-lock inputs: " + ", ".join(missing))
    files = {relative: sha256_file(REPO_ROOT / relative) for relative in relative_files}
    lock = {
        "status": "EXP60_PREFLIGHT_SOURCE_LOCK_NO_TRAINING_AUTHORITY",
        "authority": "real-model no-update preflight only; does not authorize optimizer construction, optimizer step, checkpointing or formal training",
        "protocol_sha256_at_preflight": sha256_file(PROTOCOL_PATH),
        "normalized_scientific_protocol_sha256": normalized_scientific_protocol_sha256(protocol),
        "allowed_post_preflight_protocol_changes": list(
            ALLOWED_POST_PREFLIGHT_PROTOCOL_CHANGES
        ),
        "mapping_sha256": protocol["mapping"]["canonical_sha256"],
        "files": files,
        "file_count": len(files),
        "allowed_splits": ["train"],
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    write_json(PREFLIGHT_SOURCE_LOCK_PATH, lock)
    return lock


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
