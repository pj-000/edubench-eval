"""Generate one train-only greedy R3 actual-failure inventory with vLLM.

The command performs no model import or GPU work unless --execute is supplied.
GPU execution additionally requires the operator to choose an idle permitted
GPU through CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    SEEDS,
    compact_json,
    load_train_rows,
    make_failure_row,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_dev_inference import prepare_prompts
from thesis_exp.exp54_rar_sft.run_dev_inference_vllm import (
    OUTPUT_SCHEMA,
    _force_complete_budget_limited_json,
)


TRAINING_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "actual_failure_bank"
)
CONFIG_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json"
)
PLAN_PATH = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/"
    "actual_failure_bank_generation.json"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    os.replace(temporary, path)


def checkpoint(seed: int) -> tuple[Path, dict[str, Any]]:
    root = (
        TRAINING_ROOT
        / f"seed{seed}"
        / "r3"
        / "checkpoint-logical-epoch-3"
    )
    adapter = root / "adapter"
    state_path = root / "trainer_state.json"
    if not adapter.is_dir() or not state_path.is_file():
        raise FileNotFoundError(root)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {
        "status": "EXP54_FORMAL_CHECKPOINT_UNEVALUATED",
        "arm": "R3",
        "seed": seed,
        "logical_epoch_number": 3,
        "global_optimizer_step": 996,
        "test_accessed": False,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"{state_path}: checkpoint differs at {key}")
    return adapter, {
        "adapter_config_sha256": sha256_file(
            adapter / "adapter_config.json"
        ),
        "adapter_model_sha256": sha256_file(
            adapter / "adapter_model.safetensors"
        ),
        "trainer_state_sha256": sha256_file(state_path),
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_train_rows()
    adapter, hashes = checkpoint(args.seed)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if (
        plan.get("split") != "train"
        or plan.get("arm") != "R3"
        or plan.get("checkpoint_epoch") != 3
        or plan.get("generation", {}).get("mode") != "greedy"
    ):
        raise ValueError("actual failure-bank plan differs")
    output = OUTPUT_ROOT / "private" / f"seed{args.seed}"
    return {
        "status": "ACTUAL_FAILURE_BANK_VLLM_DRY_RUN",
        "seed": args.seed,
        "arm": "R3",
        "epoch": 3,
        "train_rows": len(rows),
        "expected_outputs": len(rows),
        "generation_mode": "greedy",
        "adapter_relative_path": adapter.relative_to(REPO_ROOT).as_posix(),
        "adapter_hashes": hashes,
        "private_output_relative_path": output.relative_to(
            REPO_ROOT
        ).as_posix(),
        "gpu_used": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("EXP54_GPU_USER_CONFIRMED") != "1":
        raise RuntimeError(
            "EXP54_GPU_USER_CONFIRMED=1 is required after explicit user "
            "confirmation"
        )
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_gpu not in {"4", "6", "7"}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must name exactly one permitted GPU: 4, 6, "
            "or 7"
        )
    rows = load_train_rows()
    adapter, checkpoint_hashes = checkpoint(args.seed)
    output_dir = OUTPUT_ROOT / "private" / f"seed{args.seed}"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

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
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    max_model_len = max(
        len(item["token_ids"]) for item in prepared
    ) + max_new_tokens
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
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        seed=args.seed,
        skip_special_tokens=False,
        guided_decoding=GuidedDecodingParams(
            json=OUTPUT_SCHEMA,
            disable_fallback=True,
            disable_any_whitespace=True,
        ),
    )
    started = time.perf_counter()
    outputs = llm.generate(
        prompt_token_ids=[item["token_ids"] for item in prepared],
        sampling_params=sampling,
        use_tqdm=True,
        lora_request=LoRARequest(
            f"exp54-r3-seed{args.seed}-epoch3-failure-bank",
            1,
            str(adapter),
        ),
    )
    generation_seconds = time.perf_counter() - started
    if len(outputs) != len(prepared):
        raise RuntimeError("vLLM output row count differs")

    failure_rows = []
    raw_rows = []
    for item, output in zip(prepared, outputs, strict=True):
        candidate = output.outputs[0]
        backend_text = str(candidate.text)
        backend_ids = [int(value) for value in candidate.token_ids]
        forced_completion = False
        removed_characters = 0
        final_text = backend_text
        try:
            prediction = parse_review_json(backend_text)
        except ValueError:
            prediction = None
            if (
                str(candidate.finish_reason) == "length"
                and len(backend_ids) == max_new_tokens
            ):
                try:
                    (
                        final_text,
                        prediction,
                        _,
                        removed_characters,
                    ) = _force_complete_budget_limited_json(
                        backend_text,
                        tokenizer=tokenizer,
                        max_tokens=max_new_tokens,
                    )
                    forced_completion = True
                except ValueError:
                    prediction = None
        source = item["row"]
        failure_rows.append(
            make_failure_row(
                source=source,
                row_position=int(item["row_position"]),
                generator_seed=args.seed,
                adapter_sha256=checkpoint_hashes[
                    "adapter_model_sha256"
                ],
                generation_mode="greedy",
                rollout_seed=None,
                prediction=prediction,
                forced_completion=forced_completion,
            )
        )
        raw_rows.append(
            {
                "record_id": str(source["record_id"]),
                "row_position": int(item["row_position"]),
                "raw_backend_output": backend_text,
                "materialized_output": final_text,
                "finish_reason": str(candidate.finish_reason),
                "backend_generated_tokens": len(backend_ids),
                "forced_completion": forced_completion,
                "forced_completion_removed_characters": removed_characters,
            }
        )
    failure_rows.sort(key=lambda row: int(row["row_position"]))
    raw_rows.sort(key=lambda row: int(row["row_position"]))
    if [row["row_position"] for row in failure_rows] != list(
        range(len(rows))
    ):
        raise RuntimeError("failure bank row order differs")
    if checkpoint(args.seed)[1] != checkpoint_hashes:
        raise RuntimeError("checkpoint changed during generation")

    failure_path = output_dir / "failure_rows.jsonl"
    raw_path = output_dir / "raw_generations.jsonl"
    write_jsonl(failure_path, failure_rows)
    write_jsonl(raw_path, raw_rows)
    class_counts = Counter(row["error_class"] for row in failure_rows)
    report = {
        "status": "ACTUAL_FAILURE_BANK_GREEDY_SEED_COMPLETE",
        "seed": args.seed,
        "arm": "R3",
        "epoch": 3,
        "rows": len(failure_rows),
        "class_counts": dict(sorted(class_counts.items())),
        "forced_completion_count": sum(
            bool(row["forced_completion"]) for row in failure_rows
        ),
        "failure_rows_sha256": sha256_file(failure_path),
        "raw_generations_sha256": sha256_file(raw_path),
        "checkpoint_hashes": checkpoint_hashes,
        "generation_seconds": generation_seconds,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "xgrammar_source_sha256": sha256_file(Path(xgrammar.__file__)),
        "gpu_visible_devices": visible_gpu,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(OUTPUT_ROOT / "runs" / f"seed{args.seed}_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = execute(args) if args.execute else dry_run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
