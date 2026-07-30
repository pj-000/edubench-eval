"""Run one frozen arm/seed on the isolated Exp54 one-time test blob."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_dev_inference import (
    file_sha256,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.run_dev_inference_vllm import (
    OUTPUT_SCHEMA,
    _force_complete_budget_limited_json,
)
from thesis_exp.exp54_rar_sft.sorc_dpo_test_execution_contract import (
    ARMS,
    EXPECTED_TEST_BLOB_SHA1,
    EXPECTED_XGRAMMAR_SHA256,
    PREREGISTRATION_PATH,
    PREREGISTRATION_SHA256,
    SEEDS,
    git_blob_sha1,
    load_preregistration,
    parse_test_rows,
    regular_bytes,
    validate_checkpoint,
)
from thesis_exp.exp54_rar_sft.training_contract import (
    apply_chat_template,
    prompt_messages,
    token_ids_sha256,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
)


MAX_MODEL_LEN = 1796
MAX_NEW_TOKENS = 256


def _prepare_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    max_prompt_tokens = MAX_MODEL_LEN - MAX_NEW_TOKENS
    for row_position, row in enumerate(rows):
        token_ids = list(
            apply_chat_template(
                tokenizer,
                prompt_messages(row),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if not token_ids or len(token_ids) > max_prompt_tokens:
            raise ValueError(
                "test prompt length is outside the frozen generation budget: "
                f"{row['record_id']}={len(token_ids)}"
            )
        prepared.append(
            {
                "row_position": row_position,
                "row": row,
                "token_ids": token_ids,
            }
        )
    return prepared


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_root / args.arm.lower() / f"seed_{args.seed}"
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"one-time test output exists: {output_dir}") from exc
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_SORC_DPO_ONE_TIME_TEST_RUN_STARTED",
            "arm": args.arm,
            "seed": args.seed,
            "dev_accessed": False,
            "test_accessed": True,
        },
    )

    preregistration = load_preregistration(args.preregistration)
    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    checkpoint = validate_checkpoint(
        repo_root=args.repo_root,
        preregistration=preregistration,
        arm=args.arm,
        seed=args.seed,
    )
    test_payload = regular_bytes(args.test_path)
    if git_blob_sha1(test_payload) != EXPECTED_TEST_BLOB_SHA1:
        raise ValueError("isolated test payload Git blob differs")
    rows = parse_test_rows(test_payload)

    import torch
    import vllm
    import xgrammar
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    if file_sha256(Path(xgrammar.__file__)) != EXPECTED_XGRAMMAR_SHA256:
        raise ValueError("XGrammar source hash differs")
    model_path = str(config["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = int(config["model"]["pad_token_id"])
    prepared = _prepare_prompts(tokenizer, rows)

    init_started = time.perf_counter()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        task="generate",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
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
            f"exp54-test-{args.arm.lower()}-seed{args.seed}",
            1,
            checkpoint["adapter_path"],
        ),
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(prepared):
        raise RuntimeError("vLLM test output row count differs")

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
                "prompt_token_ids_sha256": token_ids_sha256(item["token_ids"]),
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
    predictions.sort(key=lambda row: int(row["row_position"]))
    if [row["row_position"] for row in predictions] != list(range(len(rows))):
        raise RuntimeError("vLLM test prediction row order differs")
    if any(
        row["generated_token_count"] > MAX_NEW_TOKENS
        or row["finish_reason"] not in {"stop", "forced_completion"}
        for row in predictions
    ):
        raise RuntimeError("vLLM test output exceeded the frozen budget")

    final_checkpoint = validate_checkpoint(
        repo_root=args.repo_root,
        preregistration=preregistration,
        arm=args.arm,
        seed=args.seed,
    )
    if final_checkpoint != checkpoint:
        raise RuntimeError("checkpoint changed during one-time test inference")

    predictions_path = output_dir / "predictions.jsonl"
    protocol_path = output_dir / "protocol.json"
    write_jsonl(predictions_path, predictions)
    write_json(
        protocol_path,
        {
            "status": "EXP54_SORC_DPO_ONE_TIME_TEST_INFERENCE_COMPLETE",
            "protocol_id": "RAR_SFT_VLLM_COMPACT_JSON_V1",
            "arm": args.arm,
            "seed": args.seed,
            "test_git_blob_sha1": EXPECTED_TEST_BLOB_SHA1,
            "test_sha256": file_sha256(args.test_path),
            "test_rows": len(rows),
            "preregistration_sha256": PREREGISTRATION_SHA256,
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
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "engine_init_seconds": engine_init_seconds,
            "generation_seconds": generation_seconds,
            "elapsed_seconds": time.time() - started,
            "dev_accessed": False,
            "test_accessed": True,
            "scientific_metrics_computed": False,
        },
    )
    write_json(
        output_dir / "completion_receipt.json",
        {
            "status": "EXP54_SORC_DPO_ONE_TIME_TEST_RUN_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "rows": len(rows),
            "predictions_sha256": file_sha256(predictions_path),
            "protocol_sha256": file_sha256(protocol_path),
            "dev_accessed": False,
            "test_accessed": True,
            "scientific_metrics_computed": False,
        },
    )
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_SORC_DPO_ONE_TIME_TEST_RUN_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "rows": len(rows),
            "dev_accessed": False,
            "test_accessed": True,
            "scientific_metrics_computed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=PREREGISTRATION_PATH,
    )
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
