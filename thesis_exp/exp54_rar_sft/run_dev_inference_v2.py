"""Run one Exp54 checkpoint under the reviewed-candidate V2 decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def _p95(values: list[int]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.int64), 95))


def execution_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
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
    protocol = load_v2_protocol(args.protocol_config)
    if protocol["selection_and_access"]["formal_v2_dev_allowed"]:
        raise ValueError(
            "candidate config must remain not-authorized in repository"
        )
    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
    rows = load_dev_rows()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
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
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=int(model.config.vocab_size),
        protocol=protocol,
    )
    by_position: dict[int, dict[str, Any]] = {}
    ordered_for_batches = sorted(
        prepared,
        key=lambda item: len(item["token_ids"]),
    )
    for start in range(0, len(ordered_for_batches), args.batch_size):
        batch = ordered_for_batches[start : start + args.batch_size]
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
        print(
            f"[Exp54 dev V2] {args.arm}/seed{args.seed}/"
            f"epoch{args.epoch}: "
            f"{min(start + len(batch), len(ordered_for_batches))}/"
            f"{len(ordered_for_batches)}",
            flush=True,
        )
    predictions = [
        by_position[position] for position in range(len(rows))
    ]
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    execution = execution_metrics(predictions)
    run_record = {
        "status": "EXP54_DEV_INFERENCE_V2_COMPLETE",
        "protocol_id": protocol["protocol_id"],
        "protocol_config_sha256": file_sha256(args.protocol_config),
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
        "adapter_config_sha256": file_sha256(
            adapter_path / "adapter_config.json"
        ),
        "adapter_model_sha256": file_sha256(
            adapter_path / "adapter_model.safetensors"
        ),
        "generation": protocol["generation"],
        "backend": protocol["backend"],
        "batch_size": args.batch_size,
        "max_prompt_tokens": max(
            len(item["token_ids"]) for item in prepared
        ),
        "execution": execution,
        "dev_accessed": True,
        "test_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "protocol.json", run_record)
    write_json(
        args.output_dir / "metrics.json",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--epoch", required=True, type=int, choices=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/formal_runs"
        ),
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_CONFIG,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    run_inference_v2(args)


if __name__ == "__main__":
    main()
