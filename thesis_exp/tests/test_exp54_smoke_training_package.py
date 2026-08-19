import copy
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

import thesis_exp.exp54_rar_sft.audit_smoke_training_package as auditor
import thesis_exp.exp54_rar_sft.build_smoke_training_package as builder
import thesis_exp.exp54_rar_sft.smoke_authorization_guard as smoke_auth
import thesis_exp.exp54_rar_sft.train_rar_sft_smoke as smoke_runner
from thesis_exp.exp54_rar_sft.authorization_guard import (
    closure_sha256,
    runtime_source_closure,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.build_smoke_training_package import (
    build_smoke_training_package,
)
from thesis_exp.exp54_rar_sft.audit_smoke_training_package import (
    audit_smoke_training_package,
)
from thesis_exp.exp54_rar_sft.freeze_training_configuration import (
    build_training_configuration_frozen_lock,
)
from thesis_exp.exp54_rar_sft.smoke_authorization_guard import (
    AuthenticatedSmokeContext,
    claim_smoke_invocation,
    read_regular_bytes_once,
    read_regular_json_once,
    reserve_smoke_output_directory,
    smoke_runtime_source_closure,
    verify_smoke_authorization,
)
from thesis_exp.exp54_rar_sft.smoke_training_contract import (
    DEFAULT_SMOKE_CLAIM_ROOT,
    REVIEWED_SMOKE_PACKAGE_COMMIT,
    SMOKE_ARMS,
    SMOKE_EVENTS_PER_ARM,
    selector_digest,
    smoke_manifest_path,
    smoke_prompt_cache_path,
    validate_smoke_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/smoke_training_plan.json"
)
REAL_CONFIG = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/training_configuration_candidate.json"
)
REAL_CANDIDATE_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "training_configuration_candidate_report.json"
)
REAL_CANDIDATE_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_candidate_lock.json"
)
REAL_MATERIALIZED_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "materialized_manifest_frozen_lock.json"
)
REAL_TRAINING_CONFIGURATION_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "training_configuration_frozen_lock.json"
)
REAL_SMOKE_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/audit/"
    "smoke_training_package_report.json"
)
REAL_SMOKE_FROZEN_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/protocol/"
    "smoke_training_package_frozen_lock.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _synthetic_full_inputs(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "train-data"
    prompt_rows = [
        {
            "prompt_cache_id": f"prompt-{position}",
            "record_id": f"record-{position}",
            "prompt_token_ids_sha256": f"prompt-hash-{position}",
        }
        for position in range(2_654)
    ]
    prompt_path = data_dir / "shared_prompt_cache.jsonl"
    _write_jsonl(prompt_path, prompt_rows)
    manifest_hashes = {}
    for arm in SMOKE_ARMS:
        rows = []
        for epoch_index in range(3):
            for position in range(2_654):
                active = position % 2 == 0
                rows.append(
                    {
                        "arm": arm,
                        "seed": 42,
                        "epoch_index": epoch_index,
                        "epoch_number": epoch_index + 1,
                        "row_position": position,
                        "base_event_id": f"event-{epoch_index}-{position}",
                        "record_id": f"record-{position}",
                        "prompt_cache_id": f"prompt-{position}",
                        "prompt_token_ids_sha256": f"prompt-hash-{position}",
                        "score_target": position % 5 + 1,
                        "score_loss_active": True,
                        "cutoff_len": 2_048,
                        "packing": False,
                        "truncated": False,
                        "rationale_active": (
                            False
                            if arm == "S0"
                            else True
                            if arm == "R1"
                            else active
                        ),
                        "sequence_token_count": 100 + position % 7,
                        "score_token_positions": [10],
                        "rationale_token_positions": (
                            [11, 12]
                            if (
                                arm == "R1"
                                or (arm in {"R2", "R3"} and active)
                            )
                            else []
                        ),
                    }
                )
        path = data_dir / f"training_manifest_{arm.lower()}_seed42.jsonl"
        _write_jsonl(path, rows)
        manifest_hashes[arm] = sha256_file(path)
    materialized_path = root / "materialized-lock.json"
    materialized = {
        "status": "MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED",
        "manifest_frozen": True,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
        "private_artifact_hashes": {
            "shared_prompt_cache": sha256_file(prompt_path),
            "manifests_by_seed": {"seed42": manifest_hashes},
        },
    }
    _write_json(materialized_path, materialized)
    configuration_path = root / "configuration-frozen-lock.json"
    configuration = {
        "status": "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "review_gate": {
            "verdict": "TRAINING_CONFIGURATION_PASS",
            "reviewed_commit": (
                "d2762a427b8bed957459b04ba239a988ab3acea5"
            ),
            "candidate_report_sha256": (
                "27186421a67aa5c0b324672e567280859295e3066201fbca5b97ae2cd07dc0aa"
            ),
            "candidate_lock_sha256": (
                "a69b7c4594d86d8dbffa432550d0d69a0d39bd67785c89dc8045abf676e4ee0a"
            ),
        },
        "configuration_frozen": True,
        "smoke_package_build_allowed": True,
        "materialized_manifest_frozen_lock_sha256": sha256_file(
            materialized_path
        ),
        "configuration_sha256": sha256_file(REAL_CONFIG),
        "runtime_source_closure": runtime_source_closure(),
        "runtime_source_closure_sha256": closure_sha256(
            runtime_source_closure()
        ),
        "runtime_source_closure_file_count": 16,
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(configuration_path, configuration)
    return data_dir, materialized_path, configuration_path


def test_reviewed_training_configuration_freezes_exact_hash_chain() -> None:
    frozen = build_training_configuration_frozen_lock(
        config_path=REAL_CONFIG,
        frozen_manifest_lock_path=REAL_MATERIALIZED_FROZEN_LOCK,
        candidate_report_path=REAL_CANDIDATE_REPORT,
        candidate_lock_path=REAL_CANDIDATE_LOCK,
    )
    assert frozen["configuration_frozen"] is True
    assert frozen["smoke_package_build_allowed"] is True
    assert frozen["runtime_source_closure_file_count"] == 16
    assert frozen["review_gate"]["reviewed_commit"] == (
        "d2762a427b8bed957459b04ba239a988ab3acea5"
    )
    assert frozen["trust_anchor_install_allowed"] is False
    assert frozen["forward_backward_allowed"] is False
    assert frozen["smoke_training_allowed"] is False
    assert frozen["formal_training_allowed"] is False


def test_smoke_plan_and_selector_are_fixed() -> None:
    plan = json.loads(REAL_PLAN.read_text(encoding="utf-8"))
    validate_smoke_plan(plan)
    assert selector_digest("event-a") == selector_digest("event-a")
    assert selector_digest("event-a") != selector_digest("event-b")
    assert len(smoke_runtime_source_closure()) == 20
    changed = copy.deepcopy(plan)
    changed["events_per_arm"] = 10
    with pytest.raises(ValueError, match="events_per_arm"):
        validate_smoke_plan(changed)


def test_smoke_builder_freezes_same_eight_events_across_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, materialized_path, configuration_path = (
        _synthetic_full_inputs(tmp_path)
    )
    monkeypatch.setattr(builder, "_require_absent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        auditor,
        "_require_absent",
        lambda *_args, **_kwargs: None,
    )
    private_dir = tmp_path / "private-smoke"
    report_path = tmp_path / "smoke-report.json"
    lock_path = tmp_path / "smoke-lock.json"
    report, lock = build_smoke_training_package(
        full_data_dir=data_dir,
        private_smoke_dir=private_dir,
        smoke_plan_path=REAL_PLAN,
        training_configuration_frozen_lock_path=configuration_path,
        materialized_manifest_frozen_lock_path=materialized_path,
        report_path=report_path,
        frozen_lock_path=lock_path,
    )
    assert report["events_per_arm"] == SMOKE_EVENTS_PER_ARM
    assert report["selected_active_events"] == 4
    assert report["selected_inactive_events"] == 4
    assert report["same_event_vector_and_order_across_arms"] is True
    assert report["fixed_padded_token_budget_equal_across_arms"] is True
    assert report["rationale_active_events_by_arm"] == {
        "S0": 0,
        "R1": 8,
        "R2": 4,
        "R3": 4,
    }
    assert report["score_target_histogram"] == {
        "1": 2,
        "2": 2,
        "3": 1,
        "4": 1,
        "5": 2,
    }
    assert lock["candidate_report_sha256"] == sha256_file(report_path)
    assert lock["smoke_training_allowed"] is False
    assert lock["forward_backward_allowed"] is False
    vectors = []
    for arm in SMOKE_ARMS:
        rows = [
            json.loads(line)
            for line in smoke_manifest_path(private_dir, arm)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        vectors.append([row["base_event_id"] for row in rows])
    assert all(vector == vectors[0] for vector in vectors[1:])
    prompt_rows = smoke_prompt_cache_path(private_dir).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(prompt_rows) == SMOKE_EVENTS_PER_ARM
    audit = audit_smoke_training_package(
        full_data_dir=data_dir,
        private_smoke_dir=private_dir,
        smoke_plan_path=REAL_PLAN,
        training_configuration_frozen_lock_path=configuration_path,
        materialized_manifest_frozen_lock_path=materialized_path,
        report_path=report_path,
        frozen_lock_path=lock_path,
    )
    assert audit["private_rows_exactly_reconstructed"] is True
    assert audit["prompt_rows_exactly_reconstructed"] is True
    assert audit["forward_backward_allowed"] is False


@dataclass(frozen=True)
class _AuthorizationFiles:
    config_path: Path
    plan_path: Path
    configuration_lock_path: Path
    smoke_lock_path: Path
    authorization_path: Path
    trusted_digest_path: Path
    claim_root: Path
    output_root: Path
    campaign_id: str


def _synthetic_smoke_authorization_files(root: Path) -> _AuthorizationFiles:
    config_path = root / "training-config.json"
    config_path.write_bytes(REAL_CONFIG.read_bytes())
    plan_path = root / "smoke-plan.json"
    plan_path.write_bytes(REAL_PLAN.read_bytes())
    configuration_lock_path = root / "configuration-frozen.json"
    configuration_lock = {
        "status": "TRAINING_CONFIGURATION_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "configuration_sha256": sha256_file(config_path),
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(configuration_lock_path, configuration_lock)
    closure = smoke_runtime_source_closure()
    smoke_lock_path = root / "smoke-package.json"
    smoke_lock = {
        "status": "SMOKE_TRAINING_PACKAGE_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
        "smoke_plan_sha256": sha256_file(plan_path),
        "materialized_manifest_frozen_lock_sha256": "m" * 64,
        "configuration_sha256": sha256_file(config_path),
        "smoke_runtime_source_closure": closure,
        "smoke_runtime_source_closure_sha256": closure_sha256(closure),
        "rationale_active_events_by_arm": {
            "S0": 0,
            "R1": 4,
            "R2": 4,
            "R3": 4,
        },
        "trust_anchor_install_allowed": False,
        "forward_backward_allowed": False,
        "smoke_training_allowed": False,
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    _write_json(smoke_lock_path, smoke_lock)
    campaign_id = "campaign-001"
    claim_root = root / "claims"
    (claim_root / campaign_id).mkdir(parents=True)
    output_root = root / "outputs"
    run_ids = {arm: f"{campaign_id}-{arm.lower()}" for arm in SMOKE_ARMS}
    output_dirs = {
        arm: str(output_root / campaign_id / run_ids[arm])
        for arm in SMOKE_ARMS
    }
    authorization_path = root / "smoke-authorization.json"
    authorization = {
        "schema_version": "exp54-smoke-execution-authorization-v1",
        "authorization_mode": "smoke",
        "review_verdict": "SMOKE_PACKAGE_PASS",
        "reviewed_smoke_package_commit": REVIEWED_SMOKE_PACKAGE_COMMIT,
        "smoke_package_frozen_lock_sha256": sha256_file(smoke_lock_path),
        "training_configuration_frozen_lock_sha256": sha256_file(
            configuration_lock_path
        ),
        "smoke_plan_sha256": sha256_file(plan_path),
        "materialized_manifest_frozen_lock_sha256": "m" * 64,
        "smoke_runtime_source_closure_sha256": closure_sha256(closure),
        "smoke_schema_version": "exp54-smoke-training-v1",
        "allowed_arms": list(SMOKE_ARMS),
        "allowed_seed": 42,
        "optimizer_steps_per_arm": 1,
        "smoke_campaign_id": campaign_id,
        "run_id_by_arm": run_ids,
        "output_dir_by_arm": output_dirs,
        "max_invocations_per_arm": 1,
        "claim_root": str(claim_root),
        "formal_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "hyperparameter_selection_allowed": False,
    }
    _write_json(authorization_path, authorization)
    trusted_digest_path = root / "trusted-smoke-digest"
    trusted_digest_path.write_text(
        sha256_file(authorization_path) + "\n",
        encoding="ascii",
    )
    return _AuthorizationFiles(
        config_path=config_path,
        plan_path=plan_path,
        configuration_lock_path=configuration_lock_path,
        smoke_lock_path=smoke_lock_path,
        authorization_path=authorization_path,
        trusted_digest_path=trusted_digest_path,
        claim_root=claim_root,
        output_root=output_root,
        campaign_id=campaign_id,
    )


def _verify_fixture(files: _AuthorizationFiles) -> AuthenticatedSmokeContext:
    return verify_smoke_authorization(
        authorization_path=files.authorization_path,
        config_path=files.config_path,
        smoke_plan_path=files.plan_path,
        smoke_package_lock_path=files.smoke_lock_path,
        training_configuration_frozen_lock_path=(
            files.configuration_lock_path
        ),
        arm="R3",
        trusted_digest_path=files.trusted_digest_path,
        require_root_owned_digest=False,
        expected_claim_root=files.claim_root,
        expected_output_root=files.output_root,
    )


def _rewrite_authorization(
    files: _AuthorizationFiles,
    mutator,
) -> None:
    authorization = json.loads(
        files.authorization_path.read_text(encoding="utf-8")
    )
    mutator(authorization)
    _write_json(files.authorization_path, authorization)
    files.trusted_digest_path.write_text(
        sha256_file(files.authorization_path) + "\n",
        encoding="ascii",
    )


def test_read_once_json_hashes_and_parses_the_same_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    payload = b'{"value":"authenticated"}'
    path.write_bytes(payload)
    observed = read_regular_json_once(path)
    path.write_bytes(b'{"value":"replaced"}')
    assert observed.file.sha256 == hashlib.sha256(payload).hexdigest()
    assert observed.value == {"value": "authenticated"}


def test_private_jsonl_parser_uses_the_authenticated_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private.jsonl"
    path.write_bytes(b'{"value":"authenticated"}\n')
    observed = read_regular_bytes_once(path)
    path.write_bytes(b'{"value":"replaced"}\n')
    assert smoke_runner._parse_jsonl_payload(
        observed,
        label="private probe",
    ) == [{"value": "authenticated"}]


def test_authorization_rejects_fifo_and_symlink(tmp_path: Path) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    fifo = tmp_path / "authorization.fifo"
    os.mkfifo(fifo)
    with pytest.raises(PermissionError, match="regular file"):
        verify_smoke_authorization(
            authorization_path=fifo,
            config_path=files.config_path,
            smoke_plan_path=files.plan_path,
            smoke_package_lock_path=files.smoke_lock_path,
            training_configuration_frozen_lock_path=(
                files.configuration_lock_path
            ),
            arm="R3",
            trusted_digest_path=files.trusted_digest_path,
            require_root_owned_digest=False,
            expected_claim_root=files.claim_root,
            expected_output_root=files.output_root,
        )
    link = tmp_path / "authorization-link.json"
    link.symlink_to(files.authorization_path)
    with pytest.raises(PermissionError, match="unavailable"):
        verify_smoke_authorization(
            authorization_path=link,
            config_path=files.config_path,
            smoke_plan_path=files.plan_path,
            smoke_package_lock_path=files.smoke_lock_path,
            training_configuration_frozen_lock_path=(
                files.configuration_lock_path
            ),
            arm="R3",
            trusted_digest_path=files.trusted_digest_path,
            require_root_owned_digest=False,
            expected_claim_root=files.claim_root,
            expected_output_root=files.output_root,
        )


@pytest.mark.parametrize("target_name", ["config_path", "smoke_lock_path"])
def test_authorization_rejects_path_replacement_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    target = getattr(files, target_name)
    original_reader = smoke_auth.read_regular_json_once

    def replacing_reader(path: Path):
        observed = original_reader(path)
        if path == target:
            replacement = tmp_path / f"{target.name}.replacement"
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
        return observed

    monkeypatch.setattr(
        smoke_auth,
        "read_regular_json_once",
        replacing_reader,
    )
    with pytest.raises(PermissionError, match="changed during verification"):
        _verify_fixture(files)


@pytest.mark.parametrize(
    "commit",
    [
        "a" * 40,
        REVIEWED_SMOKE_PACKAGE_COMMIT.upper(),
        REVIEWED_SMOKE_PACKAGE_COMMIT[:12],
    ],
)
def test_authorization_rejects_any_other_reviewed_commit(
    tmp_path: Path,
    commit: str,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    _rewrite_authorization(
        files,
        lambda authorization: authorization.__setitem__(
            "reviewed_smoke_package_commit",
            commit,
        ),
    )
    with pytest.raises(PermissionError, match="reviewed.*commit"):
        _verify_fixture(files)


def test_smoke_lock_and_authorization_commit_must_both_match(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    smoke_lock = json.loads(files.smoke_lock_path.read_text(encoding="utf-8"))
    smoke_lock["reviewed_smoke_package_commit"] = "b" * 40
    _write_json(files.smoke_lock_path, smoke_lock)
    _rewrite_authorization(
        files,
        lambda authorization: authorization.update(
            {
                "smoke_package_frozen_lock_sha256": sha256_file(
                    files.smoke_lock_path
                ),
                "reviewed_smoke_package_commit": "b" * 40,
            }
        ),
    )
    with pytest.raises(PermissionError, match="reviewed.*commit"):
        _verify_fixture(files)


def test_authorization_rejects_unbound_output_directory(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    _rewrite_authorization(
        files,
        lambda authorization: authorization["output_dir_by_arm"].__setitem__(
            "R3",
            str(tmp_path / "arbitrary-second-output"),
        ),
    )
    with pytest.raises(PermissionError, match="output directory differs"):
        _verify_fixture(files)


def test_smoke_authorization_returns_exact_read_once_context(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)
    assert context.authorization["optimizer_steps_per_arm"] == 1
    assert context.authorization["reviewed_smoke_package_commit"] == (
        REVIEWED_SMOKE_PACKAGE_COMMIT
    )
    assert context.config_sha256 == sha256_file(files.config_path)
    assert context.smoke_lock_sha256 == sha256_file(files.smoke_lock_path)


def test_same_arm_authorization_can_be_claimed_only_once(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)
    first = claim_smoke_invocation(
        context,
        arm="R3",
        claim_root=files.claim_root,
        require_root_owned=False,
        require_append_only=False,
    )
    assert first.arm == "R3"
    with pytest.raises(PermissionError, match="already consumed"):
        claim_smoke_invocation(
            context,
            arm="R3",
            claim_root=files.claim_root,
            require_root_owned=False,
            require_append_only=False,
        )


def test_authorized_output_directory_is_reserved_atomically(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)
    claim = claim_smoke_invocation(
        context,
        arm="R1",
        claim_root=files.claim_root,
        require_root_owned=False,
        require_append_only=False,
    )
    output = reserve_smoke_output_directory(claim)
    assert output == Path(
        context.authorization["output_dir_by_arm"]["R1"]
    )
    with pytest.raises(FileExistsError):
        reserve_smoke_output_directory(claim)


def test_concurrent_claim_allows_exactly_one_winner(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)

    def attempt() -> bool:
        try:
            claim_smoke_invocation(
                context,
                arm="R2",
                claim_root=files.claim_root,
                require_root_owned=False,
                require_append_only=False,
            )
        except PermissionError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _value: attempt(), range(2)))
    assert sorted(outcomes) == [False, True]


def test_four_arms_each_claim_once_and_fifth_claim_fails(
    tmp_path: Path,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)
    claims = {
        arm: claim_smoke_invocation(
            context,
            arm=arm,
            claim_root=files.claim_root,
            require_root_owned=False,
            require_append_only=False,
        )
        for arm in SMOKE_ARMS
    }
    assert set(claims) == set(SMOKE_ARMS)
    assert len({claim.claim_path for claim in claims.values()}) == 4
    with pytest.raises(PermissionError, match="already consumed"):
        claim_smoke_invocation(
            context,
            arm="S0",
            claim_root=files.claim_root,
            require_root_owned=False,
            require_append_only=False,
        )


def test_smoke_runner_rejects_before_claim_model_cuda_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = {"claim": False, "model": False}
    monkeypatch.setattr(
        smoke_runner,
        "verify_smoke_authorization",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("smoke execution is not authorized")
        ),
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
    with pytest.raises(PermissionError, match="not authorized"):
        smoke_runner.run_smoke_training(
            arm="R3",
            config_path=tmp_path / "config",
            training_configuration_frozen_lock_path=tmp_path / "config-lock",
            smoke_plan_path=tmp_path / "plan",
            smoke_package_lock_path=tmp_path / "smoke-lock",
            smoke_authorization_path=None,
            private_smoke_dir=tmp_path / "private",
        )
    assert reached == {"claim": False, "model": False}


def test_smoke_runner_claim_failure_precedes_model_cuda_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _synthetic_smoke_authorization_files(tmp_path)
    context = _verify_fixture(files)
    rows = [
        {"rationale_active": index < 4}
        for index in range(SMOKE_EVENTS_PER_ARM)
    ]
    reached = {"model": False, "output": False}
    monkeypatch.setattr(
        smoke_runner,
        "verify_smoke_authorization",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        smoke_runner,
        "validate_training_configuration",
        lambda _config: None,
    )
    monkeypatch.setattr(
        smoke_runner,
        "_verify_frozen_smoke_package",
        lambda **_kwargs: type("PrivateData", (), {"rows": rows})(),
    )
    monkeypatch.setattr(
        smoke_runner,
        "claim_smoke_invocation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("smoke authorization already consumed")
        ),
    )
    monkeypatch.setattr(
        smoke_runner,
        "reserve_smoke_output_directory",
        lambda _claim: reached.__setitem__("output", True),
    )
    monkeypatch.setattr(
        smoke_runner,
        "verify_model_snapshot",
        lambda _config: reached.__setitem__("model", True),
    )
    with pytest.raises(PermissionError, match="already consumed"):
        smoke_runner.run_smoke_training(
            arm="R3",
            config_path=files.config_path,
            training_configuration_frozen_lock_path=(
                files.configuration_lock_path
            ),
            smoke_plan_path=files.plan_path,
            smoke_package_lock_path=files.smoke_lock_path,
            smoke_authorization_path=files.authorization_path,
            private_smoke_dir=tmp_path / "private",
        )
    assert reached == {"model": False, "output": False}


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "torch.float32"):
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        return math.prod(self.shape)


class _FakeSlice:
    def __init__(self, shape: list[int], dtype: str):
        self._shape = shape
        self._dtype = dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype


class _FakeSafeOpen:
    def __init__(self, tensors: dict[str, tuple[list[int], str]]):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return list(self._tensors)

    def get_slice(self, key: str):
        shape, dtype = self._tensors[key]
        return _FakeSlice(shape, dtype)


def _adapter_fixture(tmp_path: Path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"header")
    state = {
        "base_model.layer.lora_A.weight": _FakeTensor((2, 3)),
        "base_model.layer.lora_B.weight": _FakeTensor((4, 2)),
    }
    spec = smoke_runner._adapter_state_spec(
        state,
        expected_parameter_count=14,
    )
    tensors = {
        key: (value["shape"], value["dtype"]) for key, value in spec.items()
    }
    return adapter_dir, spec, tensors


def test_adapter_content_exact_match_passes(tmp_path: Path) -> None:
    adapter_dir, spec, tensors = _adapter_fixture(tmp_path)
    hashes = smoke_runner._adapter_artifact_hashes(
        adapter_dir,
        expected_tensor_spec=spec,
        expected_parameter_count=14,
        safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(tensors),
    )
    assert set(hashes) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


@pytest.mark.parametrize(
    "mutation",
    ["base_key", "missing", "extra", "shape", "dtype"],
)
def test_adapter_content_mismatch_hard_fails(
    tmp_path: Path,
    mutation: str,
) -> None:
    adapter_dir, spec, tensors = _adapter_fixture(tmp_path)
    changed = copy.deepcopy(tensors)
    if mutation == "base_key":
        changed["base_model.layer.weight"] = ([4, 4], "F32")
    elif mutation == "missing":
        changed.pop(next(iter(changed)))
    elif mutation == "extra":
        changed["base_model.other.lora_A.weight"] = ([1, 1], "F32")
    elif mutation == "shape":
        key = next(iter(changed))
        changed[key] = ([99, 1], changed[key][1])
    else:
        key = next(iter(changed))
        changed[key] = (changed[key][0], "BF16")
    with pytest.raises(RuntimeError, match="tensor keys|shape|dtype"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
            safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(changed),
        )


def test_invalid_safetensors_and_extra_weight_artifacts_fail(
    tmp_path: Path,
) -> None:
    adapter_dir, spec, _tensors = _adapter_fixture(tmp_path)

    def invalid_open(*_args, **_kwargs):
        raise ValueError("invalid safetensors")

    with pytest.raises(RuntimeError, match="invalid"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
            safe_open_fn=invalid_open,
        )
    (adapter_dir / "weights.safetensors").write_bytes(b"base")
    with pytest.raises(RuntimeError, match="undeclared weight"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
            safe_open_fn=invalid_open,
        )


def test_adapter_output_rejects_index_and_symlink(tmp_path: Path) -> None:
    adapter_dir, spec, tensors = _adapter_fixture(tmp_path)
    (adapter_dir / "adapter_model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="undeclared weight"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
            safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(tensors),
        )
    (adapter_dir / "adapter_model.safetensors.index.json").unlink()
    (adapter_dir / "link").symlink_to(adapter_dir / "adapter_config.json")
    with pytest.raises(RuntimeError, match="symlink"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
            safe_open_fn=lambda *_args, **_kwargs: _FakeSafeOpen(tensors),
        )


def test_real_safetensors_header_matches_expected_spec(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    state = {
        "base_model.layer.lora_A.weight": torch.zeros(2, 3),
        "base_model.layer.lora_B.weight": torch.zeros(4, 2),
    }
    safetensors_torch.save_file(
        state,
        str(adapter_dir / "adapter_model.safetensors"),
    )
    spec = smoke_runner._adapter_state_spec(
        state,
        expected_parameter_count=14,
    )
    smoke_runner._adapter_artifact_hashes(
        adapter_dir,
        expected_tensor_spec=spec,
        expected_parameter_count=14,
    )
    state_with_base = {
        **state,
        "base_model.layer.weight": torch.zeros(4, 4),
    }
    safetensors_torch.save_file(
        state_with_base,
        str(adapter_dir / "adapter_model.safetensors"),
    )
    with pytest.raises(RuntimeError, match="tensor keys"):
        smoke_runner._adapter_artifact_hashes(
            adapter_dir,
            expected_tensor_spec=spec,
            expected_parameter_count=14,
        )


def test_real_tiny_peft_adapter_matches_saved_tensor_spec(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=16,
            vocab_size=32,
        )
    )
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        ),
    )
    state = peft.get_peft_model_state_dict(model)
    expected_count = sum(int(tensor.numel()) for tensor in state.values())
    spec = smoke_runner._adapter_state_spec(
        state,
        expected_parameter_count=expected_count,
    )
    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    smoke_runner._adapter_artifact_hashes(
        adapter_dir,
        expected_tensor_spec=spec,
        expected_parameter_count=expected_count,
    )


def test_real_public_smoke_locks_are_frozen_but_not_authorized() -> None:
    report = json.loads(REAL_SMOKE_REPORT.read_text(encoding="utf-8"))
    lock = json.loads(REAL_SMOKE_FROZEN_LOCK.read_text(encoding="utf-8"))
    configuration = json.loads(
        REAL_TRAINING_CONFIGURATION_FROZEN_LOCK.read_text(encoding="utf-8")
    )
    assert lock["candidate_report_sha256"] == sha256_file(REAL_SMOKE_REPORT)
    assert lock["training_configuration_frozen_lock_sha256"] == sha256_file(
        REAL_TRAINING_CONFIGURATION_FROZEN_LOCK
    )
    assert report["score_target_histogram"] == {
        "1": 2,
        "2": 2,
        "3": 1,
        "4": 1,
        "5": 2,
    }
    assert report["rationale_active_events_by_arm"] == {
        "S0": 0,
        "R1": 4,
        "R2": 4,
        "R3": 4,
    }
    assert report["smoke_runtime_source_closure_file_count"] == 20
    assert configuration["runtime_source_closure_file_count"] == 16
    for artifact in (report, lock):
        assert artifact["reviewed_smoke_package_commit"] == (
            REVIEWED_SMOKE_PACKAGE_COMMIT
        )
        assert artifact["max_invocations_per_arm"] == 1
        assert artifact["read_once_authenticated_context_required"] is True
        assert artifact["atomic_per_arm_claim_required"] is True
        assert artifact["claim_directory_append_only_required"] is True
        assert artifact["adapter_tensor_content_audit_required"] is True
    for artifact in (report, lock, configuration):
        assert artifact["trust_anchor_install_allowed"] is False
        assert artifact["forward_backward_allowed"] is False
        assert artifact["smoke_training_allowed"] is False
        assert artifact["formal_training_allowed"] is False
        assert artifact["dev_accessed"] is False
        assert artifact["test_accessed"] is False
        assert artifact["training_used"] is False
