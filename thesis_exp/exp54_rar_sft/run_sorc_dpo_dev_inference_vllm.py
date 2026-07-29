"""Run one formal SORC-DPO adapter on frozen dev with the SFT vLLM protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_dev_inference import (
    DEV_PATH,
    REPO_ROOT,
    file_sha256,
    load_dev_rows,
    prepare_prompts,
    rationale_metrics,
    score_metrics,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.run_dev_inference_vllm import (
    OUTPUT_SCHEMA,
    _execution_metrics,
    _force_complete_budget_limited_json,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
)


ARMS = (
    "P1_FIELD_DPO",
    "P2_SORC_SCORE",
    "P3_JOINT_SORC",
    "P1_SYN_SEED42",
)
SEEDS = (42, 43, 44)
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_formal_runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_dev_runs_vllm"
)
DEFAULT_AUDIT_REPORT = (
    DEFAULT_TRAINING_ROOT / "formal_training_audit_report.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_adapter(
    *,
    arm: str,
    seed: int,
    training_root: Path,
    audit_report_path: Path,
) -> tuple[Path, dict[str, str]]:
    if arm == "P1_SYN_SEED42" and seed != 42:
        raise ValueError("P1-SYN is frozen for seed 42 only")
    result_path = training_root / arm.lower() / f"seed_{seed}/result.json"
    adapter_path = result_path.parent / "adapter"
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_model_path = adapter_path / "adapter_model.safetensors"
    for path in (result_path, adapter_config_path, adapter_model_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    result = _read_json(result_path)
    exact = {
        "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
        "arm": arm,
        "seed": seed,
        "optimizer_steps": 27,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for field, expected in exact.items():
        if result.get(field) != expected:
            raise ValueError(f"{arm}/{seed}: training result {field} differs")
    hashes = {
        "adapter_config": file_sha256(adapter_config_path),
        "adapter_model": file_sha256(adapter_model_path),
        "training_result": file_sha256(result_path),
    }
    if (
        hashes["adapter_config"] != result["output_adapter_config_sha256"]
        or hashes["adapter_model"] != result["output_adapter_model_sha256"]
    ):
        raise ValueError(f"{arm}/{seed}: adapter hash differs")
    audit = _read_json(audit_report_path)
    run_key = f"{arm}/seed_{seed}"
    if (
        audit.get("status") != "SORC_DPO_FORMAL_TRAINING_AUDIT_PASS"
        or audit.get("run_count") != 10
        or audit.get("dev_accessed") is not False
        or audit.get("test_accessed") is not False
        or audit["runs"].get(run_key, {}).get("status") != "PASS"
        or audit["runs"][run_key].get("adapter_model_sha256")
        != hashes["adapter_model"]
        or audit["runs"][run_key].get("result_sha256")
        != hashes["training_result"]
    ):
        raise ValueError(f"{arm}/{seed}: training audit binding differs")
    return adapter_path, hashes


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_root / args.arm.lower() / f"seed_{args.seed}"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"preference dev output exists: {output_dir}") from exc
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_SORC_DPO_DEV_VLLM_STARTED",
            "arm": args.arm,
            "seed": args.seed,
            "dev_accessed": True,
            "test_accessed": False,
        },
    )

    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    adapter_path, checkpoint_hashes = _validate_adapter(
        arm=args.arm,
        seed=args.seed,
        training_root=args.training_root,
        audit_report_path=args.audit_report,
    )
    rows = load_dev_rows()

    import torch
    import vllm
    import xgrammar
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    model_path = str(config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = int(config["model"]["pad_token_id"])
    prepared = prepare_prompts(tokenizer, rows)
    max_model_len = max(len(item["token_ids"]) for item in prepared) + int(
        config["generation"]["max_new_tokens"]
    )
    init_started = time.perf_counter()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        task="generate",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=True,
        max_lora_rank=int(config["lora"]["rank"]),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        disable_log_stats=True,
        guided_decoding_backend="xgrammar",
        guided_decoding_disable_any_whitespace=True,
    )
    engine_init_seconds = time.perf_counter() - init_started
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(config["generation"]["max_new_tokens"]),
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
            f"exp54-{args.arm.lower()}-seed{args.seed}",
            1,
            str(adapter_path),
        ),
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(prepared):
        raise RuntimeError("vLLM output row count differs")

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
            token_ids = backend_token_ids
        except ValueError:
            if (
                str(candidate.finish_reason) != "length"
                or len(backend_token_ids)
                != int(config["generation"]["max_new_tokens"])
            ):
                raise
            (
                final_text,
                prediction,
                token_ids,
                removed_characters,
            ) = _force_complete_budget_limited_json(
                backend_text,
                tokenizer=tokenizer,
                max_tokens=int(config["generation"]["max_new_tokens"]),
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
                "prompt_token_ids_sha256": token_ids_sha256(item["token_ids"]),
                "raw_output": final_text,
                "raw_backend_output": backend_text,
                "output_token_ids_sha256": token_ids_sha256(token_ids),
                "parse_success": True,
                "parse_error": None,
                "prediction": prediction,
                "rationale_token_count": len(
                    tokenizer.encode(
                        prediction["rationale"],
                        add_special_tokens=False,
                    )
                ),
                "generated_token_count": len(token_ids),
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
    predictions.sort(key=lambda row: int(row["row_position"]))
    if [row["row_position"] for row in predictions] != list(range(len(rows))):
        raise RuntimeError("vLLM prediction row order differs")
    if any(
        row["generated_token_count"]
        > int(config["generation"]["max_new_tokens"])
        or row["finish_reason"] not in {"stop", "forced_completion"}
        for row in predictions
    ):
        raise RuntimeError("vLLM output exceeded budget or did not finish safely")
    final_hashes = {
        "adapter_config": file_sha256(adapter_path / "adapter_config.json"),
        "adapter_model": file_sha256(
            adapter_path / "adapter_model.safetensors"
        ),
        "training_result": file_sha256(adapter_path.parent / "result.json"),
    }
    if final_hashes != checkpoint_hashes:
        raise RuntimeError("preference checkpoint changed during dev inference")

    execution = _execution_metrics(
        predictions,
        generation_seconds=generation_seconds,
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(
        output_dir / "protocol.json",
        {
            "status": "EXP54_SORC_DPO_DEV_VLLM_COMPLETE",
            "protocol_id": "RAR_SFT_VLLM_COMPACT_JSON_V1",
            "arm": args.arm,
            "seed": args.seed,
            "dev_path": str(DEV_PATH),
            "dev_sha256": file_sha256(DEV_PATH),
            "dev_rows": len(rows),
            "model_path": model_path,
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": checkpoint_hashes["adapter_config"],
            "adapter_model_sha256": checkpoint_hashes["adapter_model"],
            "training_result_sha256": checkpoint_hashes["training_result"],
            "training_audit_report_sha256": file_sha256(args.audit_report),
            "generation": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": int(config["generation"]["max_new_tokens"]),
            },
            "backend": {
                "name": "vllm",
                "version": vllm.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "xgrammar_source_sha256": file_sha256(Path(xgrammar.__file__)),
                "compact_json_whitespace_disabled": True,
                "budget_boundary_policy": (
                    "truncate_rationale_prefix_and_force_json_close"
                ),
                "forced_completion_count": execution[
                    "forced_completion_count"
                ],
            },
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "engine_init_seconds": engine_init_seconds,
            "execution": execution,
            "dev_accessed": True,
            "test_accessed": False,
            "elapsed_seconds": time.time() - started,
        },
    )
    write_json(
        output_dir / "metrics.json",
        {
            "status": "EXP54_SORC_DPO_DEV_METRICS_VLLM_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "score": score_metrics(predictions),
            "rationale": rationale_metrics(predictions, tokenizer),
            "execution": execution,
            "dev_accessed": True,
            "test_accessed": False,
        },
    )
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_SORC_DPO_DEV_VLLM_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "rows": len(rows),
            "generation_seconds": generation_seconds,
            "test_accessed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=DEFAULT_AUDIT_REPORT,
    )
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.validate_only:
        frozen_config = _read_object(DEFAULT_CONFIG)
        validate_training_configuration(frozen_config)
        frozen_adapter_path, frozen_hashes = _validate_adapter(
            arm=parsed_args.arm,
            seed=parsed_args.seed,
            training_root=parsed_args.training_root,
            audit_report_path=parsed_args.audit_report,
        )
        print(
            json.dumps(
                {
                    "status": "EXP54_SORC_DPO_DEV_VLLM_VALIDATION_PASS",
                    "arm": parsed_args.arm,
                    "seed": parsed_args.seed,
                    "adapter_path": str(frozen_adapter_path),
                    "checkpoint_hashes": frozen_hashes,
                    "dev_accessed": False,
                    "test_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        run(parsed_args)
