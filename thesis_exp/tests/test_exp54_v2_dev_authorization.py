from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from thesis_exp.exp54_rar_sft import (
    v2_dev_authorization_guard as guard,
)
from thesis_exp.exp54_rar_sft import v2_dev_runtime_contract as runtime


COMMIT = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _prepare_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    repository_root = tmp_path / "repo"
    protocol_path = repository_root / "protocol.json"
    training_config_path = repository_root / "training.json"
    report_path = repository_root / "report.json"
    lock_path = repository_root / "lock.json"
    dev_path = repository_root / "sealed-dev.jsonl"
    grammar_path = (
        repository_root
        / "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf"
    )
    decoder_path = (
        repository_root
        / "thesis_exp/exp54_rar_sft/structured_decoder_v2.py"
    )
    parser_path = (
        repository_root
        / "thesis_exp/exp54_rar_sft/inference_contract.py"
    )
    for path, payload in (
        (grammar_path, "root ::= \"x\"\n"),
        (decoder_path, "decoder\n"),
        (parser_path, "parser\n"),
        (dev_path, '{"sealed":true}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    protocol = {
        "selection_and_access": {
            "formal_v2_dev_allowed": False,
        },
        "generation": {
            "max_new_tokens": 256,
            "formal_batch_size": 16,
        },
        "backend": {
            "wheel_filename": "xgrammar.whl",
            "wheel_sha256": "1" * 64,
            "dependency_wheels": [
                {
                    "wheel_filename": "tvm.whl",
                    "wheel_sha256": "2" * 64,
                }
            ],
        },
    }
    _write_json(protocol_path, protocol)
    _write_json(training_config_path, {"runtime": {}})
    _write_json(
        report_path,
        {
            "formal_v2_dev_allowed": False,
            "formal_test_allowed": False,
            "test_accessed": False,
        },
    )
    _write_json(
        lock_path,
        {
            "formal_v2_dev_allowed": False,
            "formal_test_allowed": False,
            "test_accessed": False,
        },
    )
    monkeypatch.setattr(guard, "REPO_ROOT", repository_root)
    monkeypatch.setattr(
        guard,
        "DEFAULT_PROTOCOL_CONFIG",
        protocol_path,
    )
    monkeypatch.setattr(
        guard,
        "DEFAULT_TRAINING_CONFIG",
        training_config_path,
    )
    monkeypatch.setattr(
        guard,
        "DEFAULT_CANDIDATE_REPORT",
        report_path,
    )
    monkeypatch.setattr(
        guard,
        "DEFAULT_CANDIDATE_LOCK",
        lock_path,
    )
    monkeypatch.setattr(guard, "DEFAULT_DEV_PATH", dev_path)
    closure = {"runtime-source": "3" * 64}
    monkeypatch.setattr(
        guard,
        "v2_runtime_source_closure",
        lambda: closure,
    )
    monkeypatch.setattr(
        guard,
        "v2_runtime_source_closure_sha256",
        lambda value: "4" * 64,
    )
    runtime_versions = {"python": "locked"}
    monkeypatch.setattr(
        guard,
        "require_runtime_versions",
        lambda protocol, training: runtime_versions,
    )

    def fake_wheel(
        path: Path,
        *,
        expected_filename: str,
        expected_sha256: str,
    ) -> dict[str, object]:
        return {
            "path": str(path),
            "filename": expected_filename,
            "size": 7,
            "sha256": expected_sha256,
        }

    monkeypatch.setattr(guard, "verify_wheel_file", fake_wheel)
    monkeypatch.setattr(
        guard,
        "observed_installed_distribution",
        lambda name: {
            "distribution": name,
            "version": "locked",
            "file_count": 1,
            "file_tree_sha256": "5" * 64,
            "direct_url_sha256": None,
        },
    )

    training_root = tmp_path / "training"
    output_root = tmp_path / "outputs"
    wheel_root = tmp_path / "wheels"
    claim_root = tmp_path / "claims"
    campaign_id = "reviewed-v2-dev"
    claim_root.mkdir(mode=0o700)
    (claim_root / campaign_id).mkdir(mode=0o700)
    output_root.mkdir(mode=0o700)
    wheel_root.mkdir(mode=0o700)
    tasks = []
    for arm in guard.ARMS:
        for seed in guard.SEEDS:
            for epoch in guard.EPOCHS:
                adapter = (
                    training_root
                    / f"seed{seed}"
                    / arm.lower()
                    / f"checkpoint-logical-epoch-{epoch}"
                    / "adapter"
                )
                adapter.mkdir(parents=True)
                artifacts = {
                    "adapter_config": adapter / "adapter_config.json",
                    "adapter_model": (
                        adapter / "adapter_model.safetensors"
                    ),
                    "trainer_state": (
                        adapter.parent / "trainer_state.json"
                    ),
                }
                for name, path in artifacts.items():
                    path.write_bytes(
                        f"{arm}|{seed}|{epoch}|{name}".encode("ascii")
                    )
                task_id = guard._task_id(arm, seed, epoch)
                tasks.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        "output_dir": str(
                            output_root
                            / arm.lower()
                            / f"seed{seed}"
                            / f"epoch{epoch}"
                        ),
                        "checkpoint_artifacts": (
                            guard._checkpoint_artifacts(
                                training_root,
                                arm=arm,
                                seed=seed,
                                epoch=epoch,
                            )
                        ),
                        "max_invocations": 1,
                    }
                )
    observed_wheels = {
        "xgrammar": fake_wheel(
            wheel_root / "xgrammar.whl",
            expected_filename="xgrammar.whl",
            expected_sha256="1" * 64,
        ),
        "apache-tvm-ffi": fake_wheel(
            wheel_root / "tvm.whl",
            expected_filename="tvm.whl",
            expected_sha256="2" * 64,
        ),
    }
    distributions = {
        name: guard.observed_installed_distribution(name)
        for name in ("xgrammar", "apache-tvm-ffi")
    }
    authorization = {
        "schema_version": "exp54-v2-dev-authorization-v1",
        "status": "EXP54_V2_DEV_AUTHORIZATION_REVIEWED",
        "review_verdict": "INFERENCE_PROTOCOL_V2_PASS",
        "reviewed_commit": COMMIT,
        "campaign_id": campaign_id,
        "arms": list(guard.ARMS),
        "seeds": list(guard.SEEDS),
        "epochs": list(guard.EPOCHS),
        "batch_size": 16,
        "max_new_tokens": 256,
        "test_allowed": False,
        "max_invocations_per_checkpoint": 1,
        "training_root": str(training_root),
        "output_root": str(output_root),
        "candidate_report_sha256": guard.sha256_file(report_path),
        "candidate_lock_sha256": guard.sha256_file(lock_path),
        "protocol_config_sha256": guard.sha256_file(protocol_path),
        "grammar_sha256": guard.sha256_file(grammar_path),
        "decoder_sha256": guard.sha256_file(decoder_path),
        "parser_sha256": guard.sha256_file(parser_path),
        "dev_split_sha256": guard.sha256_file(dev_path),
        "runtime_source_closure": closure,
        "runtime_source_closure_sha256": "4" * 64,
        "runtime_versions": runtime_versions,
        "observed_wheels": observed_wheels,
        "installed_distributions": distributions,
        "tasks": tasks,
    }
    authorization_path = tmp_path / "authorization.json"
    trusted_digest_path = tmp_path / "authorization.sha256"

    def install(value: dict[str, object]) -> None:
        _write_json(authorization_path, value)
        digest = hashlib.sha256(
            authorization_path.read_bytes()
        ).hexdigest()
        trusted_digest_path.write_text(digest + "\n", encoding="ascii")
        authorization_path.chmod(0o600)
        trusted_digest_path.chmod(0o600)

    install(authorization)
    return {
        "authorization": authorization,
        "authorization_path": authorization_path,
        "trusted_digest_path": trusted_digest_path,
        "claim_root": claim_root,
        "training_root": training_root,
        "output_root": output_root,
        "wheel_root": wheel_root,
        "install": install,
    }


def _authorize(
    setup: dict[str, object],
    *,
    arm: str = "S0",
    seed: int = 42,
    epoch: int = 1,
):
    return guard.require_v2_dev_authorization(
        arm=arm,
        seed=seed,
        epoch=epoch,
        authorization_path=setup["authorization_path"],
        trusted_digest_path=setup["trusted_digest_path"],
        claim_root=setup["claim_root"],
        training_root=setup["training_root"],
        output_root=setup["output_root"],
        wheel_root=setup["wheel_root"],
        repository_commit=COMMIT,
    )


def test_missing_authorization_fails_before_runtime_or_dev(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    touched = []
    monkeypatch.setattr(
        guard,
        "_verify_static_bindings",
        lambda *args, **kwargs: touched.append("static"),
    )
    with pytest.raises(PermissionError):
        guard.require_v2_dev_authorization(
            arm="S0",
            seed=42,
            epoch=1,
            authorization_path=tmp_path / "missing.json",
            trusted_digest_path=tmp_path / "missing.sha256",
            claim_root=tmp_path / "claims",
            repository_commit=COMMIT,
        )
    assert touched == []


@pytest.mark.parametrize(
    "tamper",
    ["batch_size", "source", "checkpoint"],
)
def test_tampered_authorization_fails_without_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    authorization = json.loads(
        json.dumps(setup["authorization"])
    )
    if tamper == "batch_size":
        authorization["batch_size"] = 8
    elif tamper == "source":
        authorization["runtime_source_closure"][
            "runtime-source"
        ] = "9" * 64
    else:
        authorization["tasks"][0]["checkpoint_artifacts"][
            "adapter_model"
        ] = "9" * 64
    setup["install"](authorization)
    with pytest.raises(PermissionError):
        _authorize(setup)
    assert not list(
        (setup["claim_root"] / "reviewed-v2-dev").iterdir()
    )


def test_authorization_is_one_use_per_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    context = _authorize(setup)
    assert context.batch_size == 16
    assert context.max_new_tokens == 256
    assert context.claim_path.is_file()
    with pytest.raises(PermissionError):
        _authorize(setup)


def test_runner_checks_authorization_before_dev_model_or_output(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    from thesis_exp.exp54_rar_sft import run_dev_inference_v2 as runner

    touched = []

    def deny(**kwargs):
        touched.append("authorization")
        raise PermissionError("not reviewed")

    monkeypatch.setattr(runner, "require_v2_dev_authorization", deny)
    monkeypatch.setattr(
        runner,
        "load_v2_protocol",
        lambda *args, **kwargs: touched.append("protocol"),
    )
    monkeypatch.setattr(
        runner,
        "load_dev_rows",
        lambda *args, **kwargs: touched.append("dev"),
    )
    monkeypatch.setattr(
        runner,
        "verify_model_snapshot",
        lambda *args, **kwargs: touched.append("model"),
    )
    with pytest.raises(PermissionError):
        runner.run_inference_v2(
            SimpleNamespace(arm="S0", seed=42, epoch=1)
        )
    assert touched == ["authorization"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "1"],
        ["--output-dir", "/tmp/unreviewed"],
        ["--training-root", "/tmp/unreviewed"],
        ["--protocol-config", "/tmp/unreviewed.json"],
        ["--split", "test"],
    ],
)
def test_formal_cli_has_no_test_or_execution_override(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
):
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    from thesis_exp.exp54_rar_sft import run_dev_inference_v2 as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dev_inference_v2",
            "--arm",
            "S0",
            "--seed",
            "42",
            "--epoch",
            "1",
            *extra,
        ],
    )
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_runtime_closure_has_fixed_explicit_names_and_detects_each_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert set(runtime.V2_RUNTIME_SOURCE_PATHS) == (
        runtime.EXPECTED_V2_RUNTIME_SOURCE_NAMES
    )
    assert len(runtime.V2_RUNTIME_SOURCE_PATHS) == 26
    baseline = runtime.v2_runtime_source_closure_sha256()
    original = dict(runtime.V2_RUNTIME_SOURCE_PATHS)
    for index, name in enumerate(sorted(original)):
        replacement = tmp_path / f"changed-{index}.txt"
        replacement.write_text(
            f"changed source {name}\n",
            encoding="utf-8",
        )
        changed = dict(original)
        changed[name] = replacement
        monkeypatch.setattr(
            runtime,
            "V2_RUNTIME_SOURCE_PATHS",
            changed,
        )
        assert runtime.v2_runtime_source_closure_sha256() != baseline
    monkeypatch.setattr(
        runtime,
        "V2_RUNTIME_SOURCE_PATHS",
        original,
    )


def test_runtime_closure_rejects_missing_or_extra_name(
    monkeypatch: pytest.MonkeyPatch,
):
    original = dict(runtime.V2_RUNTIME_SOURCE_PATHS)
    missing = dict(original)
    missing.pop(next(iter(missing)))
    monkeypatch.setattr(runtime, "V2_RUNTIME_SOURCE_PATHS", missing)
    with pytest.raises(RuntimeError):
        runtime.v2_runtime_source_closure()
    extra = dict(original)
    extra["unreviewed_source"] = next(iter(original.values()))
    monkeypatch.setattr(runtime, "V2_RUNTIME_SOURCE_PATHS", extra)
    with pytest.raises(RuntimeError):
        runtime.v2_runtime_source_closure()


def test_exact_wheel_bytes_are_required(tmp_path: Path):
    wheel = tmp_path / "locked.whl"
    wheel.write_bytes(b"reviewed wheel")
    expected = hashlib.sha256(wheel.read_bytes()).hexdigest()
    runtime.verify_wheel_file(
        wheel,
        expected_filename="locked.whl",
        expected_sha256=expected,
    )
    wheel.write_bytes(b"different bytes, same claimed version")
    with pytest.raises(PermissionError):
        runtime.verify_wheel_file(
            wheel,
            expected_filename="locked.whl",
            expected_sha256=expected,
        )


def test_editable_distribution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_file = tmp_path / "package.py"
    package_file.write_text("value = 1\n", encoding="utf-8")

    class FakeDistribution:
        version = "0.2.5"
        files = [Path("package.py")]

        def read_text(self, name: str):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "file:///source",
                    "dir_info": {"editable": True},
                }
            )

        def locate_file(self, relative: str):
            return package_file

    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )
    with pytest.raises(PermissionError, match="editable"):
        runtime.observed_installed_distribution("xgrammar")


def test_candidate_auditor_does_not_import_production_selector():
    source = (
        Path(__file__).resolve().parents[1]
        / "exp54_rar_sft/audit_inference_v2_candidate.py"
    ).read_text(encoding="utf-8")
    assert "run_inference_v2_train_smoke import" not in source
    assert "build_smoke_selection" not in source


def test_preactivation_pass_is_read_only_then_atomic_activation_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from thesis_exp.exp54_rar_sft import (
        audit_v2_dev_authorization_preactivation as preflight,
    )

    setup = _prepare_authorization(tmp_path, monkeypatch)
    authorization_path = setup["authorization_path"]
    staged_path = setup["trusted_digest_path"]
    active_path = tmp_path / "authorization.active.sha256"
    formal_anchor = tmp_path / "formal-training.sha256"
    authorization_path.chmod(0o444)
    staged_path.chmod(0o444)

    with pytest.raises(PermissionError):
        guard.require_v2_dev_authorization(
            arm="S0",
            seed=42,
            epoch=1,
            authorization_path=authorization_path,
            trusted_digest_path=active_path,
            claim_root=setup["claim_root"],
            training_root=setup["training_root"],
            output_root=setup["output_root"],
            wheel_root=setup["wheel_root"],
            repository_commit=COMMIT,
        )
    report = preflight.audit_v2_dev_authorization_preactivation(
        authorization_path=authorization_path,
        staged_digest_path=staged_path,
        active_digest_path=active_path,
        formal_training_anchor=formal_anchor,
        claim_root=setup["claim_root"],
        training_root=setup["training_root"],
        output_root=setup["output_root"],
        wheel_root=setup["wheel_root"],
        repository_commit=COMMIT,
    )
    assert report["status"] == preflight.PASS_STATUS
    assert report["verified_task_count"] == 36
    assert report["claim_created"] is False
    assert report["output_created"] is False
    assert not list(
        (setup["claim_root"] / "reviewed-v2-dev").iterdir()
    )
    assert not any(setup["output_root"].iterdir())

    os.replace(staged_path, active_path)
    context = guard.require_v2_dev_authorization(
        arm="S0",
        seed=42,
        epoch=1,
        authorization_path=authorization_path,
        trusted_digest_path=active_path,
        claim_root=setup["claim_root"],
        training_root=setup["training_root"],
        output_root=setup["output_root"],
        wheel_root=setup["wheel_root"],
        repository_commit=COMMIT,
    )
    assert context.claim_path.is_file()


@pytest.mark.parametrize("blocker", ["active", "formal"])
def test_preactivation_rejects_active_or_formal_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocker: str,
):
    from thesis_exp.exp54_rar_sft import (
        audit_v2_dev_authorization_preactivation as preflight,
    )

    setup = _prepare_authorization(tmp_path, monkeypatch)
    authorization_path = setup["authorization_path"]
    staged_path = setup["trusted_digest_path"]
    active_path = tmp_path / "authorization.active.sha256"
    formal_anchor = tmp_path / "formal-training.sha256"
    authorization_path.chmod(0o444)
    staged_path.chmod(0o444)
    target = active_path if blocker == "active" else formal_anchor
    target.write_text("present\n", encoding="utf-8")

    with pytest.raises(PermissionError):
        preflight.audit_v2_dev_authorization_preactivation(
            authorization_path=authorization_path,
            staged_digest_path=staged_path,
            active_digest_path=active_path,
            formal_training_anchor=formal_anchor,
            claim_root=setup["claim_root"],
            training_root=setup["training_root"],
            output_root=setup["output_root"],
            wheel_root=setup["wheel_root"],
            repository_commit=COMMIT,
        )
    assert not list(
        (setup["claim_root"] / "reviewed-v2-dev").iterdir()
    )
