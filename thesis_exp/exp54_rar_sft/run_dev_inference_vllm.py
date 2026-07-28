"""Run one Exp54 checkpoint with vLLM and compact XGrammar JSON."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_dev_inference import (
    ARMS,
    DEV_PATH,
    EPOCHS,
    REPO_ROOT,
    SEEDS,
    file_sha256,
    load_dev_rows,
    prepare_prompts,
    rationale_metrics,
    score_metrics,
    write_json,
    write_jsonl,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
)


DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/dev_runs_vllm"
)
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "rationale"],
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
}
COMPACT_PREFIX_PATTERN = re.compile(
    r'^\{\s*"score"\s*:\s*([1-5])\s*,\s*'
    r'"rationale"\s*:\s*"',
    re.DOTALL,
)


def _force_complete_budget_limited_json(
    text: str,
    *,
    tokenizer: Any,
    max_tokens: int,
) -> tuple[str, dict[str, Any], list[int], int]:
    """Close a schema-valid JSON prefix without exceeding its token budget."""
    match = COMPACT_PREFIX_PATTERN.match(text)
    if match is None:
        raise ValueError(
            "budget-limited output is not the expected compact JSON prefix"
        )
    score = int(match.group(1))
    encoded_fragment = text[match.end() :]
    for end in range(len(encoded_fragment), -1, -1):
        candidate_fragment = encoded_fragment[:end]
        try:
            rationale = json.loads(f'"{candidate_fragment}"')
        except json.JSONDecodeError:
            continue
        completed_text = json.dumps(
            {"score": score, "rationale": rationale},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        completed_token_ids = [
            int(value)
            for value in tokenizer.encode(
                completed_text,
                add_special_tokens=False,
            )
        ]
        if len(completed_token_ids) <= max_tokens:
            return (
                completed_text,
                {"score": score, "rationale": rationale},
                completed_token_ids,
                len(encoded_fragment) - end,
            )
    raise ValueError(
        "no deterministic JSON completion fits the generation token budget"
    )


def _p95(values: list[int]) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.int64), 95))


def _execution_metrics(
    predictions: list[dict[str, Any]],
    *,
    generation_seconds: float,
) -> dict[str, Any]:
    import numpy as np

    generated_lengths = [
        int(row["generated_token_count"]) for row in predictions
    ]
    backend_generated_lengths = [
        int(row["backend_generated_token_count"])
        for row in predictions
    ]
    rationale_lengths = [
        int(row["rationale_token_count"]) for row in predictions
    ]
    finish_reasons = Counter(
        str(row["finish_reason"]) for row in predictions
    )
    generated_total = sum(generated_lengths)
    return {
        "rows": len(predictions),
        "strict_parse_rate": 1.0,
        "single_object_rate": 1.0,
        "valid_score_rate": 1.0,
        "rationale_string_rate": 1.0,
        "empty_rationale_rate": float(
            np.mean(
                [
                    row["prediction"]["rationale"] == ""
                    for row in predictions
                ]
            )
        ),
        "generated_tokens_mean": float(np.mean(generated_lengths)),
        "generated_tokens_p95": _p95(generated_lengths),
        "generated_tokens_max": max(generated_lengths),
        "backend_generated_tokens_max": max(
            backend_generated_lengths
        ),
        "generated_tokens_total": generated_total,
        "rationale_tokens_mean": float(np.mean(rationale_lengths)),
        "rationale_tokens_p95": _p95(rationale_lengths),
        "rationale_tokens_max": max(rationale_lengths),
        "max_token_hit_rate": float(
            np.mean(
                [
                    value >= 256
                    for value in backend_generated_lengths
                ]
            )
        ),
        "forced_completion_count": sum(
            bool(row["forced_completion"])
            for row in predictions
        ),
        "forced_completion_rate": float(
            np.mean(
                [
                    bool(row["forced_completion"])
                    for row in predictions
                ]
            )
        ),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "generation_seconds": generation_seconds,
        "generated_tokens_per_second": (
            generated_total / generation_seconds
        ),
        "structured_output_backend": "xgrammar",
        "compact_json_whitespace_disabled": True,
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = (
        args.output_root
        / args.arm.lower()
        / f"seed{args.seed}"
        / f"epoch{args.epoch}"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"vLLM dev output already exists: {output_dir}"
        ) from exc
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_DEV_INFERENCE_VLLM_STARTED",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
            "dev_accessed": True,
            "test_accessed": False,
        },
    )

    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    rows = load_dev_rows()
    adapter_path = (
        args.training_root
        / f"seed{args.seed}"
        / args.arm.lower()
        / f"checkpoint-logical-epoch-{args.epoch}"
        / "adapter"
    )
    state_path = adapter_path.parent / "trainer_state.json"
    if not adapter_path.is_dir() or not state_path.is_file():
        raise FileNotFoundError(adapter_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_state = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": args.arm,
        "seed": args.seed,
        "logical_epoch_number": args.epoch,
        "global_optimizer_step": args.epoch * 332,
        "dev_accessed": False,
        "test_accessed": False,
    }
    for key, value in expected_state.items():
        if state.get(key) != value:
            raise ValueError(f"checkpoint state differs at {key}")
    checkpoint_hashes = {
        "adapter_config": file_sha256(
            adapter_path / "adapter_config.json"
        ),
        "adapter_model": file_sha256(
            adapter_path / "adapter_model.safetensors"
        ),
        "trainer_state": file_sha256(state_path),
    }

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
    max_model_len = max(
        len(item["token_ids"]) for item in prepared
    ) + int(config["generation"]["max_new_tokens"])
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
        prompt_token_ids=[
            item["token_ids"] for item in prepared
        ],
        sampling_params=sampling,
        use_tqdm=True,
        lora_request=LoRARequest(
            f"exp54-{args.arm.lower()}-seed{args.seed}-"
            f"epoch{args.epoch}",
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
        backend_token_ids = [
            int(value) for value in candidate.token_ids
        ]
        forced_completion = False
        forced_completion_removed_characters = 0
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
                forced_completion_removed_characters,
            ) = _force_complete_budget_limited_json(
                backend_text,
                tokenizer=tokenizer,
                max_tokens=int(
                    config["generation"]["max_new_tokens"]
                ),
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
                "backend_generated_token_count": len(
                    backend_token_ids
                ),
                "finish_reason": (
                    "forced_completion"
                    if forced_completion
                    else str(candidate.finish_reason)
                ),
                "backend_finish_reason": str(
                    candidate.finish_reason
                ),
                "forced_completion": forced_completion,
                "forced_completion_removed_characters": (
                    forced_completion_removed_characters
                ),
            }
        )
    predictions.sort(key=lambda row: int(row["row_position"]))
    if [row["row_position"] for row in predictions] != list(
        range(len(rows))
    ):
        raise RuntimeError("vLLM prediction row order differs")
    if any(
        row["generated_token_count"]
        > int(config["generation"]["max_new_tokens"])
        or row["finish_reason"]
        not in {"stop", "forced_completion"}
        for row in predictions
    ):
        raise RuntimeError(
            "vLLM output exceeded token budget or did not finish safely"
        )
    final_checkpoint_hashes = {
        "adapter_config": file_sha256(
            adapter_path / "adapter_config.json"
        ),
        "adapter_model": file_sha256(
            adapter_path / "adapter_model.safetensors"
        ),
        "trainer_state": file_sha256(state_path),
    }
    if final_checkpoint_hashes != checkpoint_hashes:
        raise RuntimeError("checkpoint changed during vLLM dev")

    execution = _execution_metrics(
        predictions,
        generation_seconds=generation_seconds,
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(
        output_dir / "protocol.json",
        {
            "status": "EXP54_DEV_INFERENCE_VLLM_COMPLETE",
            "protocol_id": "RAR_SFT_VLLM_COMPACT_JSON_V1",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
            "checkpoint_global_step": args.epoch * 332,
            "dev_path": str(DEV_PATH),
            "dev_sha256": file_sha256(DEV_PATH),
            "dev_rows": len(rows),
            "model_path": model_path,
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": checkpoint_hashes[
                "adapter_config"
            ],
            "adapter_model_sha256": checkpoint_hashes[
                "adapter_model"
            ],
            "generation": {
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": int(
                    config["generation"]["max_new_tokens"]
                ),
            },
            "backend": {
                "name": "vllm",
                "version": vllm.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "xgrammar_source_sha256": file_sha256(
                    Path(xgrammar.__file__)
                ),
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
            "gpu_memory_utilization": (
                args.gpu_memory_utilization
            ),
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
            "status": "EXP54_DEV_METRICS_VLLM_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
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
            "status": "EXP54_DEV_INFERENCE_VLLM_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
            "rows": len(rows),
            "generation_seconds": generation_seconds,
            "test_accessed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--epoch", required=True, type=int, choices=EPOCHS)
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
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
