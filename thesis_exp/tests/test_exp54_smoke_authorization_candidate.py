import json
from pathlib import Path

import pytest

from thesis_exp.exp54_rar_sft.audit_smoke_authorization_candidate import (
    audit_smoke_authorization_candidate,
)
from thesis_exp.exp54_rar_sft.build_smoke_authorization_candidate import (
    DEFAULT_CAMPAIGN_PLAN,
    build_smoke_authorization_candidate,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
)


def _candidate_paths(root: Path) -> dict[str, Path]:
    return {
        "authorization_path": root / "authorization.json",
        "digest_path": root / "authorization.sha256",
        "report_path": root / "report.json",
        "authorization_install_path": root / "installed-authorization.json",
        "trusted_digest_install_path": root / "installed-digest.sha256",
        "formal_trusted_digest_path": root / "formal-digest.sha256",
        "claim_root_path": root / "claim-root",
    }


def _build(root: Path) -> dict[str, Path]:
    paths = _candidate_paths(root)
    build_smoke_authorization_candidate(
        campaign_plan_path=DEFAULT_CAMPAIGN_PLAN,
        smoke_lock_path=DEFAULT_SMOKE_FROZEN_LOCK,
        smoke_plan_path=DEFAULT_SMOKE_PLAN,
        configuration_lock_path=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
        materialized_lock_path=DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
        **paths,
    )
    return paths


def _audit(paths: dict[str, Path]) -> dict:
    return audit_smoke_authorization_candidate(
        campaign_plan_path=DEFAULT_CAMPAIGN_PLAN,
        smoke_lock_path=DEFAULT_SMOKE_FROZEN_LOCK,
        smoke_plan_path=DEFAULT_SMOKE_PLAN,
        configuration_lock_path=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
        materialized_lock_path=DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
        **paths,
    )


def test_candidate_is_independently_reconstructed(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    result = _audit(paths)
    assert result["status"] == "SMOKE_AUTHORIZATION_CANDIDATE_AUDIT_PASS"
    assert result["installation_targets_absent"] is True
    assert result["trust_anchor_install_allowed"] is False
    assert result["claim_creation_allowed"] is False
    assert result["model_loading_allowed"] is False
    assert result["forward_backward_allowed"] is False
    assert result["smoke_training_allowed"] is False


def test_candidate_authorization_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _build(tmp_path)
    authorization = json.loads(
        paths["authorization_path"].read_text(encoding="utf-8")
    )
    authorization["optimizer_steps_per_arm"] = 2
    paths["authorization_path"].write_text(
        json.dumps(authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="authorization differs"):
        _audit(paths)


def test_candidate_digest_tampering_is_rejected(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    paths["digest_path"].write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="digest candidate differs"):
        _audit(paths)


@pytest.mark.parametrize(
    "target",
    [
        "authorization_install_path",
        "trusted_digest_install_path",
        "formal_trusted_digest_path",
        "claim_root_path",
    ],
)
def test_candidate_audit_requires_installation_targets_absent(
    tmp_path: Path,
    target: str,
) -> None:
    paths = _build(tmp_path)
    paths[target].write_text("unexpected", encoding="utf-8")
    with pytest.raises(PermissionError, match="must remain absent"):
        _audit(paths)


def test_candidate_builder_rejects_changed_smoke_lock(
    tmp_path: Path,
) -> None:
    smoke_lock_path = tmp_path / "smoke-lock.json"
    smoke_lock = json.loads(
        DEFAULT_SMOKE_FROZEN_LOCK.read_text(encoding="utf-8")
    )
    smoke_lock["smoke_runtime_source_closure_sha256"] = "0" * 64
    smoke_lock_path.write_text(
        json.dumps(smoke_lock, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="closure digest differs"):
        build_smoke_authorization_candidate(
            campaign_plan_path=DEFAULT_CAMPAIGN_PLAN,
            smoke_lock_path=smoke_lock_path,
            smoke_plan_path=DEFAULT_SMOKE_PLAN,
            configuration_lock_path=DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
            materialized_lock_path=DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
            **_candidate_paths(tmp_path / "outputs"),
        )
