"""Train-only vLLM startup and structured-decoding benchmark for Exp54."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_inference_v2_train_smoke import (
    build_smoke_selection,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data"
)
DEFAULT_TRAINING_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    import torch
    import vllm
    import xgrammar
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    selected = build_smoke_selection(args.data_dir)[: args.rows]
    if len(selected) != args.rows or args.rows < 1:
        raise ValueError("invalid train-only benchmark row count")
    prompts = [
        [int(token_id) for token_id in row["prompt_token_ids"]]
        for row in selected
    ]
    config = _read_object(DEFAULT_CONFIG)
    model_path = str(config["model"]["local_path"])
    adapter_path = (
        args.training_root
        / "seed42/s0/checkpoint-logical-epoch-1/adapter"
    )
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)

    init_started = time.perf_counter()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        task="generate",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=42,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(len(row) for row in prompts) + 256,
        enable_lora=True,
        max_lora_rank=16,
        max_num_seqs=max(8, args.rows),
        enforce_eager=True,
        disable_log_stats=True,
        guided_decoding_backend="xgrammar",
        guided_decoding_disable_any_whitespace=True,
    )
    init_seconds = time.perf_counter() - init_started
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        seed=42,
        skip_special_tokens=False,
        guided_decoding=GuidedDecodingParams(
            json=OUTPUT_SCHEMA,
            disable_fallback=True,
            disable_any_whitespace=True,
        ),
    )
    generate_started = time.perf_counter()
    outputs = llm.generate(
        prompt_token_ids=prompts,
        sampling_params=sampling,
        use_tqdm=True,
        lora_request=LoRARequest(
            "exp54-s0-seed42-epoch1",
            1,
            str(adapter_path),
        ),
    )
    generation_seconds = time.perf_counter() - generate_started
    texts = [output.outputs[0].text for output in outputs]
    token_rows = [
        [int(value) for value in output.outputs[0].token_ids]
        for output in outputs
    ]
    parsed: list[dict[str, Any] | None] = []
    parse_errors: list[str | None] = []
    for text in texts:
        try:
            parsed.append(parse_review_json(text))
            parse_errors.append(None)
        except ValueError as exc:
            parsed.append(None)
            parse_errors.append(str(exc))
    finish_reasons = [
        str(output.outputs[0].finish_reason) for output in outputs
    ]
    generated_tokens = sum(len(row) for row in token_rows)
    _write_json(
        args.output_path,
        {
            "status": (
                "EXP54_VLLM_V2_TRAIN_ONLY_BENCHMARK_PASS"
                if all(value is not None for value in parsed)
                else "EXP54_VLLM_V2_TRAIN_ONLY_BENCHMARK_PARSE_FAILURE"
            ),
            "rows": args.rows,
            "train_only": True,
            "dev_accessed": False,
            "test_accessed": False,
            "strict_parse_rate": (
                sum(value is not None for value in parsed) / len(parsed)
            ),
            "all_scores_valid": all(
                value is not None
                and 1 <= int(value["score"]) <= 5
                for value in parsed
            ),
            "all_rationales_strings": all(
                value is not None
                and isinstance(value["rationale"], str)
                for value in parsed
            ),
            "parse_errors": parse_errors,
            "generated_token_counts": [
                len(row) for row in token_rows
            ],
            "first_token_ids": [row[:16] for row in token_rows],
            "last_token_ids": [row[-16:] for row in token_rows],
            "prompt_token_ids_sha256": _canonical_sha256(prompts),
            "output_token_ids_sha256": _canonical_sha256(token_rows),
            "output_bytes_sha256": _canonical_sha256(
                [text.encode("utf-8").hex() for text in texts]
            ),
            "finish_reasons": finish_reasons,
            "generated_tokens": generated_tokens,
            "engine_init_seconds": init_seconds,
            "generation_seconds": generation_seconds,
            "generated_tokens_per_second": (
                generated_tokens / generation_seconds
            ),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "runtime": {
                "vllm": vllm.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "xgrammar_module_sha256": hashlib.sha256(
                    Path(xgrammar.__file__).read_bytes()
                ).hexdigest(),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
