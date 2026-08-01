"""Run one Exp54 checkpoint under the reviewed-candidate V2 decoder."""

from __future__ import annotations

import argparse
import json
import time
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
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    DEFAULT_PROTOCOL_CONFIG,
    constrained_greedy_decode,
    load_v2_protocol,
    prepare_v2_runtime,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
    verify_model_snapshot,
)


DEFAULT_TRAINING_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2/dev_runs_v2"
)
DIRECT_RUNTIME_BATCH_SIZE = 40


def _p95(values: list[int]) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.int64), 95))


def execution_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    diagnostics = [row["decoding_diagnostics"] for row in predictions]
    generated_lengths = [
        int(item["generated_token_count"]) for item in diagnostics
    ]
    rationale_lengths = [
        int(row["rationale_token_count"]) for row in predictions
    ]
    active_steps = sum(
        int(item["active_generation_steps"]) for item in diagnostics
    )
    intervention_steps = sum(
        int(item["grammar_intervention_steps"]) for item in diagnostics
    )
    blocked_steps = sum(
        int(item["unconstrained_top1_blocked_steps"])
        for item in diagnostics
    )
    removed_mass_weighted_sum = sum(
        float(item["removed_probability_mass_mean"])
        * int(item["active_generation_steps"])
        for item in diagnostics
    )
    count = len(predictions)
    return {
        "rows": count,
        "strict_parse_rate": float(
            np.mean([row["parse_success"] for row in predictions])
        ),
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
        "rationale_tokens_mean": float(np.mean(rationale_lengths)),
        "rationale_tokens_p95": _p95(rationale_lengths),
        "rationale_tokens_max": max(rationale_lengths),
        "max_token_hit_rate": float(
            np.mean(
                [
                    item["completion_at_max_token_boundary"]
                    for item in diagnostics
                ]
            )
        ),
        "forced_completion_rate": float(
            np.mean(
                [item["forced_completion"] for item in diagnostics]
            )
        ),
        "grammar_intervention_step_rate": (
            intervention_steps / active_steps
        ),
        "illegal_unconstrained_top1_rate": (
            blocked_steps / active_steps
        ),
        "removed_probability_mass_mean": (
            removed_mass_weighted_sum / active_steps
        ),
    }


def run_inference_v2(args: argparse.Namespace) -> None:
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
            f"V2 dev output already exists; refusing duplicate run: "
            f"{output_dir}"
        ) from exc
    write_json(
        output_dir / "run_state.json",
        {
            "status": "EXP54_DEV_INFERENCE_V2_STARTED",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
            "execution_mode": "operator_direct",
            "test_accessed": False,
        },
    )

    protocol = load_v2_protocol()
    protocol_reference_batch_size = int(
        protocol["generation"]["formal_batch_size"]
    )
    batch_size = DIRECT_RUNTIME_BATCH_SIZE
    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = config["model"]["pad_token_id"]
    prepared = prepare_prompts(tokenizer, rows)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    grammar_path = REPO_ROOT / protocol["grammar"]["path"]
    grammar_payload = grammar_path.read_bytes()
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=int(model.config.vocab_size),
        protocol=protocol,
        grammar_payload=grammar_payload,
    )
    by_position: dict[int, dict[str, Any]] = {}
    ordered_for_batches = sorted(
        prepared,
        key=lambda item: len(item["token_ids"]),
    )
    for start in range(
        0,
        len(ordered_for_batches),
        batch_size,
    ):
        batch = ordered_for_batches[start : start + batch_size]
        decoded = constrained_greedy_decode(
            model,
            tokenizer,
            [item["token_ids"] for item in batch],
            protocol=protocol,
            runtime=runtime,
            device="cuda:0",
        )
        for item, result in zip(batch, decoded):
            row = item["row"]
            prediction = parse_review_json(result.text)
            rationale_token_count = len(
                tokenizer.encode(
                    prediction["rationale"],
                    add_special_tokens=False,
                )
            )
            by_position[item["row_position"]] = {
                "row_position": item["row_position"],
                "record_id": str(row["record_id"]),
                "label_5": int(row["label_5"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "prompt_token_count": len(item["token_ids"]),
                "prompt_token_ids_sha256": token_ids_sha256(
                    item["token_ids"]
                ),
                "raw_output": result.text,
                "output_token_ids_sha256": token_ids_sha256(
                    result.token_ids
                ),
                "parse_success": True,
                "parse_error": None,
                "prediction": prediction,
                "rationale_token_count": rationale_token_count,
                "decoding_diagnostics": result.to_dict()["diagnostics"],
            }
        completed = min(
            start + len(batch),
            len(ordered_for_batches),
        )
        elapsed = time.time() - started
        rows_per_second = completed / elapsed if elapsed > 0 else 0.0
        remaining = len(ordered_for_batches) - completed
        eta_seconds = (
            remaining / rows_per_second
            if rows_per_second > 0
            else 0.0
        )
        print(
            f"[Exp54 dev V2] {args.arm}/seed{args.seed}/"
            f"epoch{args.epoch}: "
            f"{completed}/{len(ordered_for_batches)} "
            f"({completed / len(ordered_for_batches) * 100:.1f}%), "
            f"ETA {eta_seconds / 60:.1f} min",
            flush=True,
        )
    predictions = [
        by_position[position] for position in range(len(rows))
    ]
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
        raise RuntimeError("checkpoint artifacts changed during V2 dev")
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    execution = execution_metrics(predictions)
    run_record = {
        "status": "EXP54_DEV_INFERENCE_V2_COMPLETE",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": file_sha256(DEFAULT_PROTOCOL_CONFIG),
        "grammar_sha256": protocol["grammar"]["sha256"],
        "decoder_source_sha256": file_sha256(
            Path(__file__).resolve().parent / "structured_decoder_v2.py"
        ),
        "arm": args.arm,
        "seed": args.seed,
        "epoch": args.epoch,
        "checkpoint_global_step": args.epoch * 332,
        "dev_path": str(DEV_PATH),
        "dev_sha256": file_sha256(DEV_PATH),
        "dev_rows": len(rows),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "adapter_config_sha256": checkpoint_hashes["adapter_config"],
        "adapter_model_sha256": checkpoint_hashes["adapter_model"],
        "generation": protocol["generation"],
        "backend": protocol["backend"],
        "batch_size": batch_size,
        "protocol_reference_batch_size": protocol_reference_batch_size,
        "execution_mode": "operator_direct",
        "duplicate_run_protection": "existing_output_directory",
        "max_prompt_tokens": max(
            len(item["token_ids"]) for item in prepared
        ),
        "execution": execution,
        "dev_accessed": True,
        "test_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "protocol.json", run_record)
    write_json(
        output_dir / "metrics.json",
        {
            "status": "EXP54_DEV_METRICS_V2_COMPLETE",
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
            "status": "EXP54_DEV_INFERENCE_V2_COMPLETE",
            "arm": args.arm,
            "seed": args.seed,
            "epoch": args.epoch,
            "execution_mode": "operator_direct",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference_v2(args)


if __name__ == "__main__":
    main()
