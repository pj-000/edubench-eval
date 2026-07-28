from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thesis_exp.exp54_rar_sft import (
    v2_dev_authorization_guard as guard,
)
from thesis_exp.exp54_rar_sft import v2_dev_runtime_contract as runtime
from thesis_exp.exp54_rar_sft import audit_inference_v2_candidate as auditor


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
        (
            dev_path,
            '{"record_id":"dev-1","label_5":3,'
            '"metric_id":"m","language":"en"}\n',
        ),
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
    monkeypatch.setattr(guard, "_EXPECTED_DEV_ROWS", 1)
    monkeypatch.setattr(
        guard,
        "parse_v2_protocol_payload",
        lambda protocol_payload, grammar_payload: json.loads(
            protocol_payload.decode("utf-8")
        ),
    )
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
        "verify_installed_distribution_matches_wheel",
        lambda name, path, expected_wheel_sha256: {
            "distribution": name,
            "version": "locked",
            "installed_distribution_matches_reviewed_wheel": True,
            "source_or_vcs_install_rejected": True,
            "wheel_payload_file_count": 1,
            "installed_payload_file_count": 1,
            "wheel_payload_file_tree_sha256": "5" * 64,
            "installed_payload_file_tree_sha256": "5" * 64,
            "direct_url_sha256": None,
        },
    )
    monkeypatch.setattr(
        guard,
        "_require_root_managed_claim_root",
        lambda path: None,
    )
    monkeypatch.setattr(
        guard,
        "_require_nonreplayable_campaign_directory",
        lambda path: None,
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
                    if name == "trainer_state":
                        _write_json(
                            path,
                            {
                                "status": (
                                    "EXP54_FORMAL_CHECKPOINT_UNEVALUATED"
                                ),
                                "arm": arm,
                                "seed": seed,
                                "logical_epoch_number": epoch,
                                "global_optimizer_step": epoch * 332,
                                "dev_accessed": False,
                                "test_accessed": False,
                            },
                        )
                    else:
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
        name: guard.verify_installed_distribution_matches_wheel(
            name,
            wheel_root / ("xgrammar.whl" if name == "xgrammar" else "tvm.whl"),
            expected_wheel_sha256=(
                "1" * 64 if name == "xgrammar" else "2" * 64
            ),
        )
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


def test_claim_occurs_before_dev_rows_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    events: list[str] = []
    original_claim = guard._claim_task
    original_materialize = guard._read_jsonl_rows

    def claim(**kwargs):
        events.append("claim")
        return original_claim(**kwargs)

    def materialize(material):
        events.append("materialize")
        return original_materialize(material)

    monkeypatch.setattr(guard, "_claim_task", claim)
    monkeypatch.setattr(guard, "_read_jsonl_rows", materialize)
    context = _authorize(setup)
    assert len(context.dev_rows) == 1
    assert events == ["claim", "materialize"]


def test_concurrent_loser_never_receives_materialized_dev_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    original_materialize = guard._read_jsonl_rows
    materialization_count = 0
    count_lock = threading.Lock()

    def materialize(material):
        nonlocal materialization_count
        with count_lock:
            materialization_count += 1
        return original_materialize(material)

    monkeypatch.setattr(guard, "_read_jsonl_rows", materialize)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def authorize() -> None:
        barrier.wait()
        try:
            context = _authorize(setup)
            outcomes.append(
                "won-materialized"
                if context.dev_rows
                else "won-empty"
            )
        except PermissionError:
            outcomes.append("lost")

    threads = [threading.Thread(target=authorize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["lost", "won-materialized"]
    assert materialization_count == 1


def test_dev_parse_failure_retains_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    calls = 0

    def fail(material):
        nonlocal calls
        calls += 1
        raise PermissionError("authenticated dev parse failed")

    monkeypatch.setattr(guard, "_read_jsonl_rows", fail)
    with pytest.raises(PermissionError, match="dev parse failed"):
        _authorize(setup)
    claim_path = (
        setup["claim_root"]
        / "reviewed-v2-dev"
        / "s0-seed42-epoch1.claimed"
    )
    assert claim_path.is_file()
    with pytest.raises(PermissionError, match="already claimed"):
        _authorize(setup)
    assert calls == 1


def test_campaign_mode_01770_allows_group_preflight_read(
    tmp_path: Path,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir(mode=0o1770)
    campaign.chmod(0o1770)
    assert guard._CAMPAIGN_DIRECTORY_MODE == 0o1770
    descriptor = os.open(
        campaign,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    os.close(descriptor)
    assert list(campaign.iterdir()) == []


@pytest.mark.parametrize(
    ("uid", "mode", "append_only", "message"),
    [
        (1000, 0o1770, True, "root-owned"),
        (0, 0o1730, True, "mode"),
        (0, 0o1770, False, "append-only"),
    ],
)
def test_campaign_policy_rejects_unsafe_installation_state(
    uid: int,
    mode: int,
    append_only: bool,
    message: str,
):
    metadata = SimpleNamespace(
        st_mode=stat.S_IFDIR | mode,
        st_uid=uid,
        st_gid=os.getgid(),
    )
    with pytest.raises(PermissionError, match=message):
        guard._validate_campaign_directory_policy(
            path=Path("/var/lib/edubench/exp54-v2-dev/campaign"),
            metadata=metadata,
            execution_groups={os.getgid()},
            append_only=append_only,
        )


def test_campaign_policy_accepts_root_group_01770_append_only():
    metadata = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o1770,
        st_uid=0,
        st_gid=os.getgid(),
    )
    guard._validate_campaign_directory_policy(
        path=Path("/var/lib/edubench/exp54-v2-dev/campaign"),
        metadata=metadata,
        execution_groups={os.getgid()},
        append_only=True,
    )


def _assert_authenticated_input_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: bytes,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    verified = guard.authenticate_v2_dev_authorization(
        arm="S0",
        seed=42,
        epoch=1,
        authorization_path=setup["authorization_path"],
        trusted_digest_path=setup["trusted_digest_path"],
        claim_root=setup["claim_root"],
        training_root=setup["training_root"],
        output_root=setup["output_root"],
        wheel_root=setup["wheel_root"],
        repository_commit=COMMIT,
    )
    material = getattr(verified, field)
    material.path.write_bytes(replacement)
    with pytest.raises(PermissionError, match="replaced"):
        guard.verify_read_once_material_unchanged(material)


def test_protocol_replacement_after_authorization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _assert_authenticated_input_replacement_is_rejected(
        tmp_path,
        monkeypatch,
        "protocol_material",
        b'{"replaced":true}\n',
    )


def test_dev_replacement_after_authorization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _assert_authenticated_input_replacement_is_rejected(
        tmp_path,
        monkeypatch,
        "dev_material",
        b'{"record_id":"other","label_5":1}\n',
    )


def test_checkpoint_replacement_after_authorization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _prepare_authorization(tmp_path, monkeypatch)
    verified = guard.authenticate_v2_dev_authorization(
        arm="S0",
        seed=42,
        epoch=1,
        authorization_path=setup["authorization_path"],
        trusted_digest_path=setup["trusted_digest_path"],
        claim_root=setup["claim_root"],
        training_root=setup["training_root"],
        output_root=setup["output_root"],
        wheel_root=setup["wheel_root"],
        repository_commit=COMMIT,
    )
    material = verified.checkpoint_materials["adapter_model"]
    material.path.write_bytes(b"replaced adapter")
    with pytest.raises(PermissionError, match="replaced"):
        guard.verify_read_once_material_unchanged(material)


def test_runner_uses_direct_operator_workflow_without_claim_gate():
    source = (
        Path(__file__).resolve().parents[1]
        / "exp54_rar_sft/run_dev_inference_v2.py"
    ).read_text(encoding="utf-8")
    assert "require_v2_dev_authorization" not in source
    assert "claim" not in source
    assert "trusted_digest" not in source
    assert "load_v2_protocol()" in source
    assert "load_dev_rows()" in source
    assert "_read_object(DEFAULT_CONFIG)" in source
    assert "DIRECT_RUNTIME_BATCH_SIZE = 32" in source
    assert '"execution_mode": "operator_direct"' in source


def test_concurrent_checkpoint_claim_has_exactly_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "claims"
    campaign = root / "campaign"
    campaign.mkdir(parents=True)
    monkeypatch.setattr(
        guard, "_require_root_managed_claim_root", lambda path: None
    )
    monkeypatch.setattr(
        guard,
        "_require_nonreplayable_campaign_directory",
        lambda path: None,
    )
    outcomes: list[str] = []
    barrier = threading.Barrier(8)

    def claim() -> None:
        barrier.wait()
        try:
            guard._claim_task(
                claim_root=root,
                campaign_id="campaign",
                task_id="s0-seed42-epoch1",
                authorization_sha256="a" * 64,
            )
            outcomes.append("won")
        except PermissionError:
            outcomes.append("lost")

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 7


def test_training_user_cannot_delete_or_replay_claim(tmp_path: Path):
    root = tmp_path / "user-owned"
    root.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="root-owned"):
        guard._require_root_managed_claim_root(root)


def test_deleted_output_does_not_restore_invocation_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "claims"
    (root / "campaign").mkdir(parents=True)
    monkeypatch.setattr(
        guard, "_require_root_managed_claim_root", lambda path: None
    )
    monkeypatch.setattr(
        guard,
        "_require_nonreplayable_campaign_directory",
        lambda path: None,
    )
    kwargs = {
        "claim_root": root,
        "campaign_id": "campaign",
        "task_id": "s0-seed42-epoch1",
        "authorization_sha256": "a" * 64,
    }
    guard._claim_task(**kwargs)
    output = tmp_path / "output"
    output.mkdir()
    output.rmdir()
    with pytest.raises(PermissionError, match="already claimed"):
        guard._claim_task(**kwargs)


def test_failed_claimed_task_requires_new_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "claims"
    (root / "campaign-a").mkdir(parents=True)
    (root / "campaign-b").mkdir()
    monkeypatch.setattr(
        guard, "_require_root_managed_claim_root", lambda path: None
    )
    monkeypatch.setattr(
        guard,
        "_require_nonreplayable_campaign_directory",
        lambda path: None,
    )
    base = {
        "claim_root": root,
        "task_id": "s0-seed42-epoch1",
        "authorization_sha256": "a" * 64,
    }
    guard._claim_task(campaign_id="campaign-a", **base)
    with pytest.raises(PermissionError):
        guard._claim_task(campaign_id="campaign-a", **base)
    assert guard._claim_task(
        campaign_id="campaign-b", **base
    ).is_file()


def test_runner_reserves_output_before_dev_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    from thesis_exp.exp54_rar_sft import run_dev_inference_v2 as runner

    touched = []

    def stop_before_dev():
        touched.append("protocol")
        raise RuntimeError("stop")

    monkeypatch.setattr(runner, "load_v2_protocol", stop_before_dev)
    monkeypatch.setattr(
        runner,
        "load_dev_rows",
        lambda: touched.append("dev"),
    )
    monkeypatch.setattr(
        runner,
        "verify_model_snapshot",
        lambda *args, **kwargs: touched.append("model"),
    )
    output_root = tmp_path / "dev-results"
    with pytest.raises(RuntimeError, match="stop"):
        runner.run_inference_v2(
            SimpleNamespace(
                arm="S0",
                seed=42,
                epoch=1,
                training_root=tmp_path / "training",
                output_root=output_root,
            )
        )
    assert touched == ["protocol"]
    assert (output_root / "s0/seed42/epoch1").is_dir()
    with pytest.raises(FileExistsError, match="duplicate"):
        runner.run_inference_v2(
            SimpleNamespace(
                arm="S0",
                seed=42,
                epoch=1,
                training_root=tmp_path / "training",
                output_root=output_root,
            )
        )
    assert touched == ["protocol"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "1"],
        ["--output-dir", "/tmp/unreviewed"],
        ["--protocol-config", "/tmp/unreviewed.json"],
        ["--split", "test"],
        ["--authorization", "/tmp/unreviewed.json"],
    ],
)
def test_direct_cli_has_no_test_protocol_or_authorization_override(
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


def _reviewed_wheel_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    direct_url: dict[str, object] | None = None,
    installed_package_payload: bytes = b"value = 1\n",
) -> tuple[Path, str]:
    wheel = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel_files = {
        "demo.py": b"value = 1\n",
        "demo-1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
        ),
        "demo-1.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
        ),
        "demo-1.0.dist-info/RECORD": b"",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in wheel_files.items():
            archive.writestr(name, payload)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    installed_root = tmp_path / "installed"
    for name, payload in wheel_files.items():
        if name.endswith("/RECORD"):
            continue
        path = installed_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            installed_package_payload if name == "demo.py" else payload
        )
    generated_script = tmp_path / "bin/demo"
    generated_script.parent.mkdir(parents=True, exist_ok=True)
    generated_script.write_text("#!/bin/sh\n", encoding="utf-8")

    resolved_direct_url = direct_url
    if resolved_direct_url is None:
        resolved_direct_url = {
            "url": wheel.as_uri(),
            "archive_info": {"hash": f"sha256={digest}"},
        }

    class FakeDistribution:
        version = "1.0"
        files = [
            Path("demo.py"),
            Path("demo-1.0.dist-info/METADATA"),
            Path("demo-1.0.dist-info/WHEEL"),
            Path("../../../bin/demo"),
        ]

        def read_text(self, name: str):
            assert name == "direct_url.json"
            return json.dumps(resolved_direct_url)

        def locate_file(self, relative: str):
            if str(relative) == "../../../bin/demo":
                return generated_script
            return installed_root / str(relative)

    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution(),
    )
    return wheel, digest


def test_exact_reviewed_wheel_install_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    wheel, digest = _reviewed_wheel_fixture(tmp_path, monkeypatch)
    evidence = runtime.verify_installed_distribution_matches_wheel(
        "demo",
        wheel,
        expected_wheel_sha256=digest,
    )
    assert evidence["installed_distribution_matches_reviewed_wheel"]
    assert (
        evidence["wheel_payload_file_tree_sha256"]
        == evidence["installed_payload_file_tree_sha256"]
    )


def test_noneditable_source_install_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    wheel, digest = _reviewed_wheel_fixture(
        tmp_path,
        monkeypatch,
        direct_url={"url": "file:///unreviewed/source", "dir_info": {}},
    )
    with pytest.raises(PermissionError, match="source-directory"):
        runtime.verify_installed_distribution_matches_wheel(
            "demo", wheel, expected_wheel_sha256=digest
        )


def test_vcs_install_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    wheel, digest = _reviewed_wheel_fixture(
        tmp_path,
        monkeypatch,
        direct_url={
            "url": "https://example.invalid/demo.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
        },
    )
    with pytest.raises(PermissionError, match="VCS"):
        runtime.verify_installed_distribution_matches_wheel(
            "demo", wheel, expected_wheel_sha256=digest
        )


def test_installed_payload_differing_from_wheel_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    wheel, digest = _reviewed_wheel_fixture(
        tmp_path,
        monkeypatch,
        installed_package_payload=b"value = 2\n",
    )
    with pytest.raises(PermissionError, match="differs"):
        runtime.verify_installed_distribution_matches_wheel(
            "demo", wheel, expected_wheel_sha256=digest
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
    with pytest.raises(PermissionError, match="source-directory"):
        runtime.observed_installed_distribution("xgrammar")


def test_candidate_auditor_does_not_import_production_selector():
    source = (
        Path(__file__).resolve().parents[1]
        / "exp54_rar_sft/audit_inference_v2_candidate.py"
    ).read_text(encoding="utf-8")
    assert "run_inference_v2_train_smoke import" not in source
    assert "build_smoke_selection" not in source


def _frozen_auditor_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train = tmp_path / "train.jsonl"
    train_row = {
        "record_id": "record-1",
        "label_5": 3,
        "language": "en",
    }
    train.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
    monkeypatch.setattr(auditor, "DEFAULT_TRAIN", train)
    prompt = {
        "prompt_cache_id": "prompt-1",
        "record_id": "record-1",
        "prompt_token_ids": [1, 2, 3],
        "prompt_token_ids_sha256": auditor._prompt_token_ids_sha256(
            [1, 2, 3]
        ),
    }
    prompt_path = data_dir / "shared_prompt_cache.jsonl"
    prompt_path.write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    manifest_hashes: dict[str, dict[str, str]] = {}
    manifest_paths: dict[tuple[str, int], Path] = {}
    for seed in auditor.SEEDS:
        seed_hashes: dict[str, str] = {}
        for arm in auditor.ARMS:
            path = (
                data_dir
                / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            rows = [
                {
                    "epoch_number": epoch,
                    "record_id": "record-1",
                    "score_target": 3,
                    "language": "en",
                    "rationale_active": arm != "S0",
                    "base_event_id": (
                        f"{arm.lower()}-{seed}-{epoch}"
                    ),
                    "prompt_cache_id": "prompt-1",
                    "prompt_token_ids_sha256": prompt[
                        "prompt_token_ids_sha256"
                    ],
                }
                for epoch in auditor.EPOCHS
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            seed_hashes[arm] = auditor.file_sha256(path)
            manifest_paths[(arm, seed)] = path
        manifest_hashes[f"seed{seed}"] = seed_hashes
    reference_lock = tmp_path / "reference-lock.json"
    _write_json(
        reference_lock,
        {
            "status": "REFERENCE_SETS_READY",
            "dev_accessed": False,
            "test_accessed": False,
            "input": {
                "rar0_source_hashes": {
                    "train": auditor.file_sha256(train)
                }
            },
        },
    )
    materialized_lock = tmp_path / "materialized-lock.json"
    _write_json(
        materialized_lock,
        {
            "status": (
                "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED"
            ),
            "manifest_frozen": True,
            "dev_accessed": False,
            "test_accessed": False,
            "upstream_lock_hashes": {
                "reference_set_lock": auditor.file_sha256(reference_lock)
            },
            "private_artifact_hashes": {
                "shared_prompt_cache": auditor.file_sha256(prompt_path),
                "manifests_by_seed": manifest_hashes,
            },
        },
    )
    monkeypatch.setattr(
        auditor,
        "EXPECTED_MATERIALIZED_MANIFEST_FROZEN_LOCK_SHA256",
        auditor.file_sha256(materialized_lock),
    )
    monkeypatch.setattr(
        auditor,
        "EXPECTED_REFERENCE_SET_DATA_LOCK_SHA256",
        auditor.file_sha256(reference_lock),
    )
    return {
        "data_dir": data_dir,
        "materialized_lock": materialized_lock,
        "reference_lock": reference_lock,
        "manifest_paths": manifest_paths,
    }


def test_independent_auditor_binds_frozen_manifest_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _frozen_auditor_fixture(tmp_path, monkeypatch)
    contract = auditor._frozen_manifest_contract(
        setup["data_dir"],
        materialized_lock_path=setup["materialized_lock"],
        reference_lock_path=setup["reference_lock"],
    )
    assert len(contract["manifest_hashes"]) == 12
    target = setup["manifest_paths"][("S0", 42)]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen manifest hash"):
        auditor._frozen_manifest_contract(
            setup["data_dir"],
            materialized_lock_path=setup["materialized_lock"],
            reference_lock_path=setup["reference_lock"],
        )


def test_independent_auditor_rejects_rewritten_upstream_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _frozen_auditor_fixture(tmp_path, monkeypatch)
    path = setup["materialized_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["private_artifact_hashes"]["shared_prompt_cache"] = "0" * 64
    _write_json(path, value)
    with pytest.raises(ValueError, match="trust anchor"):
        auditor._frozen_manifest_contract(
            setup["data_dir"],
            materialized_lock_path=path,
            reference_lock_path=setup["reference_lock"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("score_target", 5, "score differs"),
        ("language", "zh", "language differs"),
    ],
)
def test_independent_auditor_rejects_semantic_manifest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
):
    setup = _frozen_auditor_fixture(tmp_path, monkeypatch)
    contract = auditor._frozen_manifest_contract(
        setup["data_dir"],
        materialized_lock_path=setup["materialized_lock"],
        reference_lock_path=setup["reference_lock"],
    )
    target = setup["manifest_paths"][("S0", 42)]
    rows = [
        json.loads(line) for line in target.read_text().splitlines()
    ]
    rows[0][field] = value
    target.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    contract["manifest_hashes"][("S0", 42)] = auditor.file_sha256(
        target
    )
    with pytest.raises(ValueError, match=message):
        auditor._independent_smoke_selection(
            setup["data_dir"],
            frozen_contract=contract,
        )


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
    monkeypatch.setattr(
        preflight,
        "_require_readonly_owned_regular",
        lambda path: None,
    )
    monkeypatch.setattr(
        preflight,
        "_require_root_managed_claim_root",
        lambda path: None,
    )
    monkeypatch.setattr(
        preflight,
        "_require_nonreplayable_campaign_directory",
        lambda path: None,
    )
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
