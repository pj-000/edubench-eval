"""Evaluate one frozen mechanism-control checkpoint on the existing test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_sorc_dpo_test_inference_vllm import (
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    OUTPUT_SCHEMA,
    _force_complete_budget_limited_json,
    _prepare_prompts,
)
from thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract import (
    EXPECTED_TEST_BLOB_SHA1,
    EXPECTED_XGRAMMAR_SHA256,
    file_sha256,
    git_blob_sha1,
    load_inference_configuration,
    parse_test_rows,
    regular_bytes,
    validate_runtime_source_closure,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


ARMS = ("R3_TOKENAVG", "P1_FULLSEQ", "P1_SYN_LR5E6")
SEEDS = (42, 43, 44)
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "mechanism_control_test_plan_v1.json"
)
DEFAULT_CHECKPOINT_LOCK = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "mechanism_control_formal_audit/checkpoint_lock.json"
)
DEFAULT_AUDIT_REPORT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "mechanism_control_formal_audit/audit_report.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_plan(
    *,
    plan_path: Path,
    repo_root: Path,
    artifact_root: Path,
    test_path: Path,
    output_root: Path,
    arm: str,
    seed: int,
    cuda_device_uuid: str | None,
) -> dict[str, Any]:
    plan = _read_object(plan_path)
    if (
        plan.get("status")
        != "FROZEN_POSTHOC_MECHANISM_TEST_ON_EXISTING_BENCHMARK"
        or plan.get("arms") != list(ARMS)
        or plan.get("seeds") != list(SEEDS)
        or int(plan.get("new_checkpoint_run_count", -1)) != 9
        or plan.get("old_predictions_regenerated") is not False
        or plan.get("test_adaptive_tuning_or_retraining") is not False
        or plan.get("training_allowed") is not False
        or plan.get("dev_allowed") is not False
        or plan.get("test_allowed") is not True
    ):
        raise PermissionError("mechanism-control test plan differs")
    paths = plan["server_paths"]
    expected = {
        "repo_root": repo_root,
        "artifact_root": artifact_root,
        "test_path": test_path,
        "output_root": output_root,
    }
    for key, path in expected.items():
        if str(path) != str(paths[key]):
            raise ValueError(f"mechanism-control test path differs: {key}")
    bindings = plan["bindings"]
    repository_bindings = {
        "runner_sha256": Path(__file__),
        "collector_sha256": (
            repo_root
            / "thesis_exp/exp54_rar_sft/"
            "collect_mechanism_control_test_results.py"
        ),
        "worker_sha256": (
            repo_root
            / "thesis_exp/scripts/"
            "run_exp54_mechanism_control_test_worker.sh"
        ),
        "runtime_lock_sha256": (
            repo_root
            / "thesis_exp/exp54_rar_sft/configs/"
            "sorc_dpo_one_time_test_runtime_lock_v1.json"
        ),
        "checkpoint_lock_sha256": DEFAULT_CHECKPOINT_LOCK,
        "checkpoint_audit_report_sha256": DEFAULT_AUDIT_REPORT,
    }
    for key, path in repository_bindings.items():
        if not path.is_file() or file_sha256(path) != str(bindings[key]):
            raise ValueError(f"mechanism-control test binding differs: {key}")
    if (
        git_blob_sha1(regular_bytes(test_path)) != EXPECTED_TEST_BLOB_SHA1
        or file_sha256(test_path) != plan["test_source"]["sha256"]
    ):
        raise ValueError("existing frozen test payload differs")
    assignment = plan["gpu_assignments_by_arm"][arm]
    if cuda_device_uuid is not None and (
        cuda_device_uuid != assignment["uuid"]
    ):
        raise ValueError("mechanism-control test CUDA UUID differs")
    generation = plan["generation"]
    if (
        generation.get("protocol_id") != "RAR_SFT_VLLM_COMPACT_JSON_V1"
        or generation.get("do_sample") is not False
        or float(generation.get("temperature", -1.0)) != 0.0
        or int(generation.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
        or int(generation.get("max_model_len", -1)) != MAX_MODEL_LEN
        or int(generation.get("max_num_seqs", -1)) != 128
        or float(generation.get("gpu_memory_utilization", -1.0)) != 0.9
        or generation.get("guided_decoding_backend") != "xgrammar"
        or int(plan.get("new_checkpoint_max_test_invocations", -1)) != 1
        or plan["execution_hardware_difference"].get("must_be_disclosed")
        is not True
    ):
        raise ValueError("mechanism-control generation contract differs")
    return plan


def _validate_gpu(
    torch: Any,
    *,
    expected_uuid: str,
) -> dict[str, Any]:
    cuda = torch.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("test inference requires one visible CUDA GPU")
    name = str(cuda.get_device_name(0))
    if name != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("test inference requires NVIDIA GeForce RTX 3090")
    uuids = cuda._raw_device_uuid_nvml()
    physical_index = int(cuda._get_nvml_device_index(0))
    if uuids is None or physical_index >= len(uuids):
        raise RuntimeError("test inference cannot resolve CUDA UUID")
    uuid = str(uuids[physical_index])
    if uuid != expected_uuid:
        raise RuntimeError("test inference CUDA UUID differs")
    free_bytes, total_bytes = cuda.mem_get_info(0)
    if int(free_bytes) < 22 * 1024**3:
        raise RuntimeError("assigned 3090 is not sufficiently idle")
    return {
        "name": name,
        "uuid": uuid,
        "free_memory_bytes_before_load": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
    }


def _validate_checkpoint(
    *,
    artifact_root: Path,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    lock = _read_object(DEFAULT_CHECKPOINT_LOCK)
    audit = _read_object(DEFAULT_AUDIT_REPORT)
    if (
        lock.get("status")
        != "MECHANISM_CONTROL_FORMAL_CHECKPOINTS_FROZEN"
        or lock.get("checkpoint_frozen") is not True
        or lock.get("training_complete") is not True
        or lock.get("dev_accessed") is not False
        or lock.get("test_accessed") is not False
        or lock.get("evaluation_started") is not False
        or audit.get("status")
        != (
            "MECHANISM_CONTROL_FORMAL_RESULT_AUDIT_PASS_"
            "EVALUATION_NOT_STARTED"
        )
        or file_sha256(DEFAULT_AUDIT_REPORT)
        != lock["audit_report_sha256"]
    ):
        raise ValueError("mechanism-control checkpoint freeze differs")
    key = f"{arm}:seed{seed}"
    directory = artifact_root / "mechanism_control_formal" / arm.lower()
    directory = directory / f"seed_{seed}"
    result_path = directory / "result.json"
    if (
        not result_path.is_file()
        or file_sha256(result_path) != lock["result_hashes"][key]
    ):
        raise ValueError(f"{key}: formal result hash differs")
    result = _read_object(result_path)
    adapter = (
        directory / "checkpoint-logical-epoch-3/adapter"
        if arm == "R3_TOKENAVG"
        else directory / "adapter"
    )
    config_path = adapter / "adapter_config.json"
    model_path = adapter / "adapter_model.safetensors"
    if (
        not config_path.is_file()
        or not model_path.is_file()
        or file_sha256(config_path)
        != result["output_adapter_config_sha256"]
        or file_sha256(model_path) != lock["adapter_hashes"][key]
        or file_sha256(model_path)
        != result["output_adapter_model_sha256"]
    ):
        raise ValueError(f"{key}: frozen adapter differs")
    return {
        "adapter_path": str(adapter),
        "adapter_config_sha256": file_sha256(config_path),
        "adapter_model_sha256": file_sha256(model_path),
        "formal_result_sha256": file_sha256(result_path),
        "checkpoint_lock_sha256": file_sha256(DEFAULT_CHECKPOINT_LOCK),
    }


def _preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = _validate_plan(
        plan_path=args.plan,
        repo_root=args.repo_root,
        artifact_root=args.artifact_root,
        test_path=args.test_path,
        output_root=args.output_root,
        arm=args.arm,
        seed=args.seed,
        cuda_device_uuid=args.cuda_device_uuid,
    )
    runtime_lock = validate_runtime_source_closure(
        repo_root=args.repo_root,
    )
    config = load_inference_configuration(
        repo_root=args.repo_root,
        runtime_lock=runtime_lock,
    )
    checkpoint = _validate_checkpoint(
        artifact_root=args.artifact_root,
        arm=args.arm,
        seed=args.seed,
    )
    output_dir = args.output_root / args.arm.lower() / f"seed_{args.seed}"
    if output_dir.exists():
        raise FileExistsError(f"test output already exists: {output_dir}")
    rows = parse_test_rows(regular_bytes(args.test_path))
    if (
        len(rows) != int(plan["test_source"]["rows"])
        or len({str(row["record_id"]) for row in rows}) != len(rows)
    ):
        raise ValueError("existing frozen test inventory differs")
    return plan, config, checkpoint, rows


def run(args: argparse.Namespace) -> None:
    started = time.time()
    plan, config, checkpoint, rows = _preflight(args)
    output_dir = args.output_root / args.arm.lower() / f"seed_{args.seed}"

    import torch
    import vllm
    import xgrammar
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    identity = _validate_gpu(
        torch,
        expected_uuid=args.cuda_device_uuid,
    )
    generation = plan["generation"]
    if (
        vllm.__version__ != "0.10.0"
        or torch.__version__ != "2.7.1+cu118"
        or torch.version.cuda != "11.8"
        or file_sha256(Path(xgrammar.__file__))
        != EXPECTED_XGRAMMAR_SHA256
    ):
        raise ValueError("frozen vLLM runtime version differs")
    model_path = str(config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = int(config["model"]["pad_token_id"])
    prepared = _prepare_prompts(tokenizer, rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        output_dir / "run_state.json",
        {
            "status": "MECHANISM_CONTROL_TEST_RUN_STARTED",
            "arm": args.arm,
            "seed": args.seed,
            "dev_accessed": False,
            "test_accessed": True,
        },
    )

    init_started = time.perf_counter()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        task="generate",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=float(
            generation["gpu_memory_utilization"]
        ),
        max_model_len=MAX_MODEL_LEN,
        enable_lora=True,
        max_lora_rank=int(config["lora"]["rank"]),
        max_num_seqs=int(generation["max_num_seqs"]),
        enforce_eager=True,
        disable_log_stats=True,
        guided_decoding_backend="xgrammar",
        guided_decoding_disable_any_whitespace=True,
    )
    engine_init_seconds = time.perf_counter() - init_started
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=args.seed,
        skip_special_tokens=False,
        guided_decoding=GuidedDecodingParams(
            json=OUTPUT_SCHEMA,
            disable_fallback=True,
            disable_any_whitespace=True,
        ),
    )
    generation_started = time.perf_counter()
    outputs = llm.generate(
        prompt_token_ids=[item["token_ids"] for item in prepared],
        sampling_params=sampling,
        use_tqdm=True,
        lora_request=LoRARequest(
            f"exp54-mechanism-{args.arm.lower()}-seed{args.seed}",
            1,
            checkpoint["adapter_path"],
        ),
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(prepared):
        raise RuntimeError("mechanism-control test row count differs")

    predictions = []
    for item, output in zip(prepared, outputs, strict=True):
        candidate = output.outputs[0]
        backend_text = candidate.text
        backend_token_ids = [int(value) for value in candidate.token_ids]
        forced_completion = False
        removed_characters = 0
        try:
            prediction = parse_review_json(backend_text)
            final_text = backend_text
            final_token_ids = backend_token_ids
        except ValueError:
            if (
                str(candidate.finish_reason) != "length"
                or len(backend_token_ids) != MAX_NEW_TOKENS
            ):
                raise
            (
                final_text,
                prediction,
                final_token_ids,
                removed_characters,
            ) = _force_complete_budget_limited_json(
                backend_text,
                tokenizer=tokenizer,
                max_tokens=MAX_NEW_TOKENS,
            )
            forced_completion = True
        row = item["row"]
        predictions.append(
            {
                "row_position": item["row_position"],
                "record_id": str(row["record_id"]),
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "prompt_token_count": len(item["token_ids"]),
                "prompt_token_ids_sha256": token_ids_sha256(
                    item["token_ids"]
                ),
                "raw_output": final_text,
                "raw_backend_output": backend_text,
                "output_token_ids_sha256": token_ids_sha256(final_token_ids),
                "parse_success": True,
                "parse_error": None,
                "prediction": prediction,
                "generated_token_count": len(final_token_ids),
                "backend_generated_token_count": len(backend_token_ids),
                "finish_reason": (
                    "forced_completion"
                    if forced_completion
                    else str(candidate.finish_reason)
                ),
                "backend_finish_reason": str(candidate.finish_reason),
                "forced_completion": forced_completion,
                "forced_completion_removed_characters": removed_characters,
            }
        )
    if [row["row_position"] for row in predictions] != list(
        range(len(rows))
    ):
        raise RuntimeError("mechanism-control test row order differs")
    if any(
        row["generated_token_count"] > MAX_NEW_TOKENS
        or row["finish_reason"] not in {"stop", "forced_completion"}
        for row in predictions
    ):
        raise RuntimeError("mechanism-control output exceeded frozen budget")
    final_checkpoint = _validate_checkpoint(
        artifact_root=args.artifact_root,
        arm=args.arm,
        seed=args.seed,
    )
    if final_checkpoint != checkpoint:
        raise RuntimeError("checkpoint changed during test inference")

    predictions_path = output_dir / "predictions.jsonl"
    protocol_path = output_dir / "protocol.json"
    write_jsonl(predictions_path, predictions)
    write_json(
        protocol_path,
        {
            "status": "MECHANISM_CONTROL_TEST_INFERENCE_COMPLETE",
            "protocol_id": "RAR_SFT_VLLM_COMPACT_JSON_V1",
            "arm": args.arm,
            "seed": args.seed,
            "test_git_blob_sha1": EXPECTED_TEST_BLOB_SHA1,
            "test_sha256": file_sha256(args.test_path),
            "test_rows": len(rows),
            "mechanism_test_plan_sha256": file_sha256(args.plan),
            "checkpoint": checkpoint,
            "model_path": model_path,
            "generation": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
            },
            "backend": {
                "name": "vllm",
                "version": vllm.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "xgrammar_source_sha256": EXPECTED_XGRAMMAR_SHA256,
                "compact_json_whitespace_disabled": True,
                "budget_boundary_policy": (
                    "truncate_rationale_prefix_and_force_json_close"
                ),
            },
            "gpu_identity": identity,
            "max_num_seqs": int(generation["max_num_seqs"]),
            "gpu_memory_utilization": float(
                generation["gpu_memory_utilization"]
            ),
            "engine_init_seconds": engine_init_seconds,
            "generation_seconds": generation_seconds,
            "elapsed_seconds": time.time() - started,
            "dev_accessed": False,
            "test_accessed": True,
            "scientific_metrics_computed": False,
            "old_predictions_regenerated": False,
            "inference_hardware_difference_disclosed": True,
        },
    )
    receipt = {
        "status": "MECHANISM_CONTROL_TEST_RUN_COMPLETE",
        "arm": args.arm,
        "seed": args.seed,
        "rows": len(rows),
        "predictions_sha256": file_sha256(predictions_path),
        "protocol_sha256": file_sha256(protocol_path),
        "dev_accessed": False,
        "test_accessed": True,
        "scientific_metrics_computed": False,
        "old_predictions_regenerated": False,
        "test_adaptive_tuning_or_retraining": False,
    }
    write_json(output_dir / "completion_receipt.json", receipt)
    write_json(
        output_dir / "run_state.json",
        {
            **receipt,
            "status": "MECHANISM_CONTROL_TEST_RUN_COMPLETE",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-device-uuid", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate frozen sources and checkpoint without loading CUDA",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.validate_only:
        _preflight(parsed_args)
        print("MECHANISM_CONTROL_TEST_PREFLIGHT_PASS")
    else:
        run(parsed_args)
