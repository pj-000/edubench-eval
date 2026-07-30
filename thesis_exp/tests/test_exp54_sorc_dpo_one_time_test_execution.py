from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from thesis_exp.exp54_rar_sft import collect_sorc_dpo_test_results as collector
from thesis_exp.exp54_rar_sft import sorc_dpo_test_execution_contract as contract
from thesis_exp.exp54_rar_sft.run_dev_inference import file_sha256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _synthetic_preregistration(
    repo_root: Path,
) -> dict[str, object]:
    arms: dict[str, dict[str, str]] = {}
    for arm in contract.ARMS:
        arms[arm] = {"role": "test"}
        for seed in contract.SEEDS:
            adapter_path = contract.checkpoint_path(
                repo_root=repo_root,
                arm=arm,
                seed=seed,
            )
            adapter_path.mkdir(parents=True)
            (adapter_path / "adapter_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            payload = f"{arm}|{seed}".encode()
            (adapter_path / "adapter_model.safetensors").write_bytes(payload)
            arms[arm][f"seed_{seed}_adapter_model_sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
            if arm == "P0_R3_SFT":
                _write_json(
                    adapter_path.parent / "trainer_state.json",
                    {
                        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
                        "arm": "R3",
                        "seed": seed,
                        "logical_epoch_number": 3,
                        "global_optimizer_step": 996,
                        "test_accessed": False,
                    },
                )
            else:
                _write_json(
                    adapter_path.parent / "result.json",
                    {
                        "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
                        "arm": arm,
                        "seed": seed,
                        "optimizer_steps": 27,
                        "dev_accessed": False,
                        "test_accessed": False,
                        "output_adapter_model_sha256": arms[arm][
                            f"seed_{seed}_adapter_model_sha256"
                        ],
                    },
                )
    return {"arms": arms}


def _prediction_rows(arm_index: int, seed: int) -> list[dict[str, object]]:
    gold = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    predictions = {
        0: [4, 4, 3, 4, 5, 1, 4, 3, 4, 5],
        1: [2, 3, 3, 4, 5, 1, 2, 3, 4, 5],
        2: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        3: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    }[arm_index]
    return [
        {
            "row_position": index,
            "record_id": f"record-{index}",
            "label_5": label,
            "parse_success": True,
            "prediction": {
                "score": predictions[index],
                "rationale": f"seed {seed} rationale",
            },
            "forced_completion": bool(index == 0 and arm_index == 3),
        }
        for index, label in enumerate(gold)
    ]


def _write_complete_grid(root: Path) -> dict[str, object]:
    preregistration: dict[str, object] = {"arms": {}}
    for arm_index, arm in enumerate(contract.ARMS):
        preregistration["arms"][arm] = {}
        for seed in contract.SEEDS:
            run_dir = root / arm.lower() / f"seed_{seed}"
            predictions_path = run_dir / "predictions.jsonl"
            protocol_path = run_dir / "protocol.json"
            _write_jsonl(
                predictions_path,
                _prediction_rows(arm_index, seed),
            )
            _write_json(
                protocol_path,
                {
                    "status": (
                        "EXP54_SORC_DPO_ONE_TIME_TEST_INFERENCE_COMPLETE"
                    ),
                    "protocol_id": "RAR_SFT_VLLM_COMPACT_JSON_V1",
                    "arm": arm,
                    "seed": seed,
                    "test_sha256": "a" * 64,
                    "test_git_blob_sha1": contract.EXPECTED_TEST_BLOB_SHA1,
                    "preregistration_sha256": (
                        contract.PREREGISTRATION_SHA256
                    ),
                    "test_rows": 10,
                    "generation": {
                        "do_sample": False,
                        "temperature": 0.0,
                        "max_new_tokens": 256,
                        "max_model_len": 1796,
                    },
                    "backend": {
                        "xgrammar_source_sha256": (
                            contract.EXPECTED_XGRAMMAR_SHA256
                        )
                    },
                    "checkpoint": {
                        "adapter_model_sha256": f"{arm_index}{seed}"
                    },
                    "dev_accessed": False,
                    "test_accessed": True,
                    "scientific_metrics_computed": False,
                },
            )
            _write_json(
                run_dir / "completion_receipt.json",
                {
                    "status": (
                        "EXP54_SORC_DPO_ONE_TIME_TEST_RUN_COMPLETE"
                    ),
                    "arm": arm,
                    "seed": seed,
                    "predictions_sha256": file_sha256(predictions_path),
                    "protocol_sha256": file_sha256(protocol_path),
                    "dev_accessed": False,
                    "test_accessed": True,
                    "scientific_metrics_computed": False,
                },
            )
            preregistration["arms"][arm][
                f"seed_{seed}_adapter_model_sha256"
            ] = f"{arm_index}{seed}"
    return preregistration


def test_checkpoint_grid_is_exact_and_tamper_fails(tmp_path: Path) -> None:
    preregistration = _synthetic_preregistration(tmp_path)
    rows = contract.validate_all_checkpoints(
        repo_root=tmp_path,
        preregistration=preregistration,
    )
    assert [row["run_key"] for row in rows] == list(
        contract.EXPECTED_RUN_KEYS
    )
    model_path = contract.checkpoint_path(
        repo_root=tmp_path,
        arm="P2_SORC_SCORE",
        seed=43,
    ) / "adapter_model.safetensors"
    model_path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="adapter SHA-256 differs"):
        contract.validate_all_checkpoints(
            repo_root=tmp_path,
            preregistration=preregistration,
        )


def test_materialization_uses_exact_git_blob(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    payload = b'{"record_id":"frozen"}\n'
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=repo,
        input=payload,
        check=True,
        capture_output=True,
    ).stdout.decode().strip()
    dirty_copy = repo / "test.jsonl"
    dirty_copy.write_bytes(b'{"record_id":"dirty"}\n')
    destination = tmp_path / "campaign/isolated/test.jsonl"
    binding = contract.materialize_test_blob(
        repo_root=repo,
        destination=destination,
        expected_blob_sha1=blob,
    )
    assert destination.read_bytes() == payload
    assert destination.read_bytes() != dirty_copy.read_bytes()
    assert binding["git_blob_sha1"] == blob
    with pytest.raises(FileExistsError):
        contract.materialize_test_blob(
            repo_root=repo,
            destination=destination,
            expected_blob_sha1=blob,
        )


def test_runtime_source_closure_rejects_dirty_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "runtime.py"
    artifact_name = "config.json"
    source = tmp_path / source_name
    artifact = tmp_path / artifact_name
    source.write_text("frozen\n", encoding="utf-8")
    artifact.write_text("{}\n", encoding="utf-8")
    lock_path = tmp_path / "runtime_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": (
                "exp54-sorc-dpo-one-time-test-runtime-lock-v1"
            ),
            "status": "FROZEN_FOR_ONE_TIME_TEST_EXECUTION",
            "source_sha256": {
                source_name: file_sha256(source),
            },
            "artifact_sha256": {
                artifact_name: file_sha256(artifact),
            },
            "runtime_versions": {
                "vllm": "0.10.0",
                "torch": "2.7.1+cu118",
                "cuda": "11.8",
                "xgrammar_source_sha256": (
                    contract.EXPECTED_XGRAMMAR_SHA256
                ),
            },
        },
    )
    monkeypatch.setattr(
        contract,
        "EXPECTED_RUNTIME_SOURCE_NAMES",
        {source_name},
    )
    monkeypatch.setattr(
        contract,
        "EXPECTED_RUNTIME_ARTIFACT_NAMES",
        {artifact_name},
    )
    contract.validate_runtime_source_closure(
        repo_root=tmp_path,
        lock_path=lock_path,
    )
    source.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source SHA-256 differs"):
        contract.validate_runtime_source_closure(
            repo_root=tmp_path,
            lock_path=lock_path,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "thesis_exp/src/edujudge/utils/io.py",
        "thesis_exp/src/edujudge/utils/text_norm.py",
    ],
)
def test_prompt_dependency_tamper_fails(
    tmp_path: Path,
    relative: str,
) -> None:
    lock = contract.read_object(contract.RUNTIME_LOCK_PATH)
    assert relative in contract.EXPECTED_RUNTIME_SOURCE_NAMES
    for name in (
        *lock["source_sha256"],
        *lock["artifact_sha256"],
    ):
        source = contract.REPO_ROOT / name
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    lock_path = tmp_path / "runtime_lock.json"
    shutil.copyfile(contract.RUNTIME_LOCK_PATH, lock_path)
    contract.validate_runtime_source_closure(
        repo_root=tmp_path,
        lock_path=lock_path,
    )
    target = tmp_path / relative
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime source SHA-256 differs"):
        contract.validate_runtime_source_closure(
            repo_root=tmp_path,
            lock_path=lock_path,
        )


def test_collector_does_not_parse_partial_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    _write_complete_grid(root)
    (root / "p3_joint_sorc/seed_44/completion_receipt.json").unlink()

    def forbidden_read(_path: Path) -> list[dict[str, object]]:
        raise AssertionError("partial prediction was parsed")

    monkeypatch.setattr(collector, "read_jsonl", forbidden_read)
    with pytest.raises(FileNotFoundError):
        collector._load_all_runs(root)


def test_completion_receipt_tamper_fails(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_complete_grid(root)
    contract.verify_completion_receipts(root)
    path = root / "p1_field_dpo/seed_42/predictions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash differs"):
        contract.verify_completion_receipts(root)


def test_metric_and_benefit_sign_conventions() -> None:
    gold = np.asarray([1, 2, 3, 4, 5])
    weak = np.asarray([4, 4, 3, 4, 5])
    strong = np.asarray([1, 2, 3, 4, 5])
    weak_metrics = collector._score_metrics(gold, weak)
    strong_metrics = collector._score_metrics(gold, strong)
    assert weak_metrics["L2H_count"] == 2
    assert weak_metrics["H2L_count"] == 0
    assert strong_metrics["QWK"] == pytest.approx(1.0)
    assert collector._benefit_delta(
        "MAE",
        baseline=weak_metrics["MAE"],
        treatment=strong_metrics["MAE"],
    ) > 0
    assert collector._benefit_delta(
        "Exact",
        baseline=weak_metrics["Exact"],
        treatment=strong_metrics["Exact"],
    ) > 0


def test_full_collection_waits_then_uses_frozen_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    output = tmp_path / "results"
    preregistration = _write_complete_grid(root)
    monkeypatch.setattr(collector, "BOOTSTRAP_REPLICATES", 100)
    monkeypatch.setattr(collector, "MINIMUM_VALID_REPLICATES", 90)
    original_loader = collector.load_preregistration
    monkeypatch.setattr(
        collector,
        "load_preregistration",
        lambda _path: {
            **original_loader(contract.PREREGISTRATION_PATH),
            "arms": preregistration["arms"],
        },
    )
    monkeypatch.setattr(
        collector,
        "validate_runtime_source_closure",
        lambda **_kwargs: {},
    )
    collector.collect(
        type(
            "Args",
            (),
            {
                "test_root": root,
                "output_dir": output,
                "preregistration": contract.PREREGISTRATION_PATH,
            },
        )()
    )
    final = json.loads(
        (output / "final_results.json").read_text(encoding="utf-8")
    )
    assert final["run_count"] == 12
    assert final["all_metrics_computed_per_seed_then_unweighted_mean"]
    assert final["holm_family_size"] == 6
    assert final["test_rerun_allowed"] is False
    assert final["contrasts"]["H1_FIELD_DPO"]["MAE"][
        "point_benefit"
    ] > 0
    assert final["contrasts"]["H1_FIELD_DPO"]["L2H_rate"][
        "point_benefit"
    ] > 0


def test_campaign_has_no_intermediate_metric_collection() -> None:
    path = (
        contract.REPO_ROOT
        / "thesis_exp/scripts/run_exp54_sorc_dpo_one_time_test.sh"
    )
    source = path.read_text(encoding="utf-8")
    receipts_position = source.index("receipts")
    collection_position = source.index("collect_sorc_dpo_test_results")
    assert receipts_position < collection_position
    assert "metrics.json" not in source
    assert "score_metrics" not in source
    assert "rationale_metrics" not in source
    assert "EDUBENCH_TEST_CAMPAIGN_ROOT" not in source
    assert source.index('mkdir "${CAMPAIGN_ROOT}"') < source.index(
        "materialize"
    )
    runner_source = (
        contract.REPO_ROOT
        / "thesis_exp/exp54_rar_sft/"
        "run_sorc_dpo_test_inference_vllm.py"
    ).read_text(encoding="utf-8")
    assert "train_rar_sft" not in runner_source
    assert "run_dev_inference" not in runner_source
    assert "validate_runtime_source_closure" in runner_source


def _interpretation_fixture() -> tuple[
    dict[str, dict[str, dict[str, object]]],
    dict[str, dict[str, float]],
]:
    bootstrap: dict[str, dict[str, dict[str, object]]] = {}
    for contrast_id, _baseline, _treatment in collector.CONTRASTS:
        bootstrap[contrast_id] = {}
        for endpoint in collector.INFERENTIAL_ENDPOINTS:
            bootstrap[contrast_id][endpoint] = {
                "minimum_valid_replicates_met": True,
                "point_benefit": 0.0,
                "ci95_low": -0.001,
                "ci95_high": 0.001,
            }
        for endpoint in collector.PRIMARY_ENDPOINTS:
            bootstrap[contrast_id][endpoint].update(
                {
                    "holm_family_resolved": True,
                    "holm_adjusted_p": 1.0,
                }
            )
        bootstrap[contrast_id]["forced_completion_increase"] = {
            "point_treatment_minus_baseline": 0.0,
            "ci95_low": -0.001,
            "ci95_high": 0.001,
        }
    aggregate = {
        arm: {
            "strict_parse_rate": 1.0,
            "forced_completion_rate": 0.0,
        }
        for arm in contract.ARMS
    }
    return bootstrap, aggregate


def test_unresolved_primary_cannot_produce_support() -> None:
    bootstrap, aggregate = _interpretation_fixture()
    bootstrap["H1_FIELD_DPO"]["L2H_rate"][
        "minimum_valid_replicates_met"
    ] = False
    preregistration = contract.load_preregistration()
    outcomes = collector._interpret(
        preregistration=preregistration,
        aggregate=aggregate,
        bootstrap=bootstrap,
    )
    assert outcomes[0]["classification"] == "UNRESOLVED"
    assert "L2H_rate" in outcomes[0]["unresolved_endpoints"]


def test_qwk_is_secondary_not_a_material_harm_guardrail() -> None:
    bootstrap, aggregate = _interpretation_fixture()
    qwk = bootstrap["H1_FIELD_DPO"]["QWK"]
    qwk.update(
        {
            "point_benefit": -0.5,
            "ci95_low": -0.6,
            "ci95_high": -0.4,
        }
    )
    outcomes = collector._interpret(
        preregistration=contract.load_preregistration(),
        aggregate=aggregate,
        bootstrap=bootstrap,
    )
    assert outcomes[0]["classification"] == "APPROXIMATELY_ZERO"
    assert outcomes[0]["guardrail_harms"] == []


def test_significant_negative_primary_is_not_approximately_zero() -> None:
    bootstrap, aggregate = _interpretation_fixture()
    mae = bootstrap["H1_FIELD_DPO"]["MAE"]
    mae.update(
        {
            "point_benefit": -0.005,
            "ci95_low": -0.009,
            "ci95_high": -0.001,
            "holm_adjusted_p": 0.01,
        }
    )
    outcomes = collector._interpret(
        preregistration=contract.load_preregistration(),
        aggregate=aggregate,
        bootstrap=bootstrap,
    )
    assert outcomes[0]["classification"] == "UNSUPPORTED"
