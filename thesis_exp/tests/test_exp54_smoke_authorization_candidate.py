import ast
import inspect
import json
import os
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest

import thesis_exp.exp54_rar_sft.audit_installed_smoke_authorization as preflight
import thesis_exp.exp54_rar_sft.train_rar_sft_smoke as smoke_runner
from thesis_exp.exp54_rar_sft.authorization_guard import sha256_file
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    verify_smoke_authorization,
)
from thesis_exp.exp54_rar_sft.audit_smoke_authorization_candidate import (
    audit_smoke_authorization_candidate,
)
from thesis_exp.exp54_rar_sft.build_smoke_authorization_candidate import (
    DEFAULT_AUTHORIZATION_CANDIDATE,
    DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE,
    DEFAULT_CAMPAIGN_PLAN,
    build_smoke_authorization_candidate,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_MATERIALIZED_MANIFEST_FROZEN_LOCK,
    DEFAULT_SMOKE_FROZEN_LOCK,
    DEFAULT_SMOKE_PLAN,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
    SMOKE_ARMS,
)


def _candidate_paths(root: Path) -> dict[str, Path]:
    return {
        "authorization_path": root / "authorization.json",
        "digest_path": root / "authorization.sha256",
        "report_path": root / "report.json",
        "authorization_install_path": root / "installed-authorization.json",
        "trusted_digest_install_path": root / "installed-digest.sha256",
        "staged_digest_install_path": root / "staged-digest.sha256",
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
        "staged_digest_install_path",
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


def _preflight_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "preactivation"
    root.mkdir()
    authorization = root / "installed-authorization.json"
    staged = root / "authorization.sha256.staged"
    shutil.copyfile(DEFAULT_AUTHORIZATION_CANDIDATE, authorization)
    shutil.copyfile(DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE, staged)
    authorization.chmod(0o444)
    staged.chmod(0o444)
    claim_root = root / "claims"
    claim_campaign = claim_root / "exp54-smoke-b2e616e-v1"
    claim_campaign.mkdir(parents=True)
    output_root = root / "outputs"
    output_campaign = output_root / "exp54-smoke-b2e616e-v1"
    output_campaign.mkdir(parents=True)
    return {
        "authorization": authorization,
        "staged": staged,
        "active": root / "authorization.sha256",
        "formal": root / "formal-authorization.sha256",
        "claim_root": claim_root,
        "output_root": output_root,
    }


def _preflight_kwargs(paths: dict[str, Path]) -> dict:
    return {
        "campaign_plan_path": DEFAULT_CAMPAIGN_PLAN,
        "authorization_candidate_path": DEFAULT_AUTHORIZATION_CANDIDATE,
        "digest_candidate_path": DEFAULT_AUTHORIZATION_DIGEST_CANDIDATE,
        "authorization_install_path": paths["authorization"],
        "staged_digest_install_path": paths["staged"],
        "active_digest_install_path": paths["active"],
        "formal_digest_install_path": paths["formal"],
        "config_path": DEFAULT_TRAINING_CONFIG,
        "smoke_plan_path": DEFAULT_SMOKE_PLAN,
        "smoke_lock_path": DEFAULT_SMOKE_FROZEN_LOCK,
        "configuration_lock_path": DEFAULT_TRAINING_CONFIG_FROZEN_LOCK,
        "claim_root_path": paths["claim_root"],
        "output_root_path": paths["output_root"],
    }


def _mock_local_preactivation_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def require_exact_bytes(path: Path, *, expected_bytes: bytes) -> None:
        if path.read_bytes() != expected_bytes:
            raise ValueError("staged bytes differ")

    def require_existing_directory(
        path: Path,
        *,
        uid: int,
        gid: int,
        mode: int,
    ) -> None:
        del uid, gid, mode
        if not path.is_dir() or path.is_symlink():
            raise PermissionError("test directory is invalid")

    campaign_plan = json.loads(DEFAULT_CAMPAIGN_PLAN.read_text())
    expected_claim_root = Path(campaign_plan["claim_root"])
    expected_output_root = Path(campaign_plan["output_root"])

    def verify_with_local_staged_digest(**kwargs):
        return verify_smoke_authorization(
            **kwargs,
            require_root_owned_digest=False,
            expected_claim_root=expected_claim_root,
            expected_output_root=expected_output_root,
        )

    monkeypatch.setattr(
        preflight,
        "_require_regular_root_file",
        require_exact_bytes,
    )
    monkeypatch.setattr(
        preflight,
        "_require_directory",
        require_existing_directory,
    )
    monkeypatch.setattr(
        preflight,
        "_directory_has_append_only_flag",
        lambda _path: True,
    )
    monkeypatch.setattr(
        preflight,
        "verify_smoke_authorization",
        verify_with_local_staged_digest,
    )


def test_production_runner_cannot_authenticate_before_preflight_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _preflight_fixture(tmp_path)
    reached = {"claim": False, "model": False}
    campaign_plan = json.loads(DEFAULT_CAMPAIGN_PLAN.read_text())

    def verify_against_absent_active(**kwargs):
        return verify_smoke_authorization(
            **kwargs,
            trusted_digest_path=paths["active"],
            require_root_owned_digest=False,
            expected_claim_root=Path(campaign_plan["claim_root"]),
            expected_output_root=Path(campaign_plan["output_root"]),
        )

    monkeypatch.setattr(
        smoke_runner,
        "verify_smoke_authorization",
        verify_against_absent_active,
    )
    monkeypatch.setattr(
        smoke_runner,
        "claim_smoke_invocation",
        lambda *_args, **_kwargs: reached.__setitem__("claim", True),
    )
    monkeypatch.setattr(
        smoke_runner,
        "verify_model_snapshot",
        lambda _config: reached.__setitem__("model", True),
    )
    with pytest.raises(PermissionError, match="unavailable"):
        smoke_runner.run_smoke_training(
            arm="S0",
            config_path=DEFAULT_TRAINING_CONFIG,
            training_configuration_frozen_lock_path=(
                DEFAULT_TRAINING_CONFIG_FROZEN_LOCK
            ),
            smoke_plan_path=DEFAULT_SMOKE_PLAN,
            smoke_package_lock_path=DEFAULT_SMOKE_FROZEN_LOCK,
            smoke_authorization_path=paths["authorization"],
            private_smoke_dir=tmp_path / "private",
        )
    assert reached == {"claim": False, "model": False}


def test_preactivation_preflight_requires_active_anchor_absent(
    tmp_path: Path,
) -> None:
    paths = _preflight_fixture(tmp_path)
    paths["active"].write_text("active\n", encoding="ascii")
    with pytest.raises(PermissionError, match="active smoke trust anchor"):
        preflight.audit_installed_smoke_authorization(
            **_preflight_kwargs(paths)
        )


def test_preactivation_preflight_rejects_formal_anchor_present(
    tmp_path: Path,
) -> None:
    paths = _preflight_fixture(tmp_path)
    paths["formal"].write_text("formal\n", encoding="ascii")
    with pytest.raises(PermissionError, match="formal-training trust anchor"):
        preflight.audit_installed_smoke_authorization(
            **_preflight_kwargs(paths)
        )


def test_failed_preflight_leaves_active_anchor_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _preflight_fixture(tmp_path)
    paths["staged"].unlink()
    _mock_local_preactivation_environment(monkeypatch)
    with pytest.raises(FileNotFoundError):
        preflight.audit_installed_smoke_authorization(
            **_preflight_kwargs(paths)
        )
    assert not paths["active"].exists()


def test_failed_preflight_creates_no_claim_or_output(
    tmp_path: Path,
) -> None:
    paths = _preflight_fixture(tmp_path)
    paths["active"].write_text("active\n", encoding="ascii")
    claim = (
        paths["claim_root"]
        / "exp54-smoke-b2e616e-v1"
        / "s0.claimed"
    )
    output = (
        paths["output_root"]
        / "exp54-smoke-b2e616e-v1"
        / "exp54-smoke-b2e616e-v1-s0"
    )
    with pytest.raises(PermissionError):
        preflight.audit_installed_smoke_authorization(
            **_preflight_kwargs(paths)
        )
    assert not claim.exists()
    assert not output.exists()


def test_successful_preflight_then_atomic_activation_allows_all_four_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _preflight_fixture(tmp_path)
    _mock_local_preactivation_environment(monkeypatch)
    result = preflight.audit_installed_smoke_authorization(
        **_preflight_kwargs(paths)
    )
    assert result["status"] == (
        "SMOKE_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS"
    )
    assert result["activation_performed"] is False
    assert not paths["active"].exists()
    os.replace(paths["staged"], paths["active"])
    assert paths["active"].is_file()

    plan = json.loads(DEFAULT_CAMPAIGN_PLAN.read_text())
    observed = set()
    for arm in SMOKE_ARMS:
        context = verify_smoke_authorization(
            authorization_path=paths["authorization"],
            config_path=DEFAULT_TRAINING_CONFIG,
            smoke_plan_path=DEFAULT_SMOKE_PLAN,
            smoke_package_lock_path=DEFAULT_SMOKE_FROZEN_LOCK,
            training_configuration_frozen_lock_path=(
                DEFAULT_TRAINING_CONFIG_FROZEN_LOCK
            ),
            arm=arm,
            trusted_digest_path=paths["active"],
            require_root_owned_digest=False,
            expected_claim_root=Path(plan["claim_root"]),
            expected_output_root=Path(plan["output_root"]),
        )
        observed.add(context.authorization_sha256)
    assert observed == {sha256_file(paths["authorization"])}


def test_active_anchor_is_installed_only_after_preflight_pass() -> None:
    text = (
        Path(preflight.__file__).with_name(
            "SMOKE_AUTHORIZATION_INSTALLATION_PLAN.md"
        )
        .read_text(encoding="utf-8")
    )
    staged = text.index("### 4. Install only the staged digest")
    preactivation = text.index("### 5. Run pre-activation preflight")
    activation = text.index("### 6. Activate only after preflight PASS")
    assert staged < preactivation < activation
    assert "single authorization activation point" in text


def test_preactivation_auditor_has_no_execution_imports_or_calls() -> None:
    tree = ast.parse(inspect.getsource(preflight))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    banned = {
        "claim_smoke_invocation",
        "reserve_smoke_output_directory",
        "verify_model_snapshot",
        "AutoModelForCausalLM",
        "torch",
        "cuda",
    }
    assert not (imported & banned)
    assert not (called & banned)
    assert not (
        attribute_calls
        & {
            "mkdir",
            "open",
            "remove",
            "replace",
            "symlink_to",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
    )
