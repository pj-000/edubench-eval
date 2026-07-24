import copy
import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.freeze_materialized_manifests import (
    EXPECTED_CANDIDATE_LOCK_SHA256,
    EXPECTED_CANDIDATE_REPORT_SHA256,
    REVIEWED_COMMIT,
    REVIEW_VERDICT,
    validate_reviewed_candidate,
)


def test_exact_reviewed_candidate_is_bound_to_pass() -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    report_path = base / "audit/materialized_manifest_candidate_report.json"
    lock_path = base / "protocol/materialized_manifest_candidate_lock.json"
    report, lock = validate_reviewed_candidate(report_path, lock_path)
    assert report["status"].endswith("AUDITED_NOT_FROZEN")
    assert lock["candidate_report_sha256"] == EXPECTED_CANDIDATE_REPORT_SHA256
    assert len(EXPECTED_CANDIDATE_LOCK_SHA256) == 64
    assert REVIEW_VERDICT == "MATERIALIZED_MANIFEST_PASS"
    assert REVIEWED_COMMIT == (
        "dad35890a1f0a5111ae0c69a2b60dff26700c24a"
    )


def test_reviewed_candidate_rejects_changed_public_bytes(tmp_path: Path) -> None:
    base = Path("thesis_exp/outputs/exp54_rar_sft/rar_v2")
    report_path = base / "audit/materialized_manifest_candidate_report.json"
    lock_path = base / "protocol/materialized_manifest_candidate_lock.json"
    changed_report = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered["manifest_freeze_allowed"] = True
    changed_report.write_text(
        json.dumps(tampered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate report"):
        validate_reviewed_candidate(changed_report, lock_path)
