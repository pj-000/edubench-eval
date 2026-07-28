"""Check V2 repeated-run determinism on one real train-only checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.run_inference_v2_train_smoke import (
    build_smoke_selection,
    file_sha256,
)
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    DEFAULT_PROTOCOL_CONFIG,
    constrained_greedy_decode,
    load_v2_protocol,
    prepare_v2_runtime,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
    verify_model_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict[str, Any]) -> None:
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


def run_probe(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    protocol = load_v2_protocol(args.protocol_config)
    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
    selected = build_smoke_selection(args.data_dir)
    row = next(
        item
        for item in selected
        if (
            item["arm"] == "S0"
            and item["seed"] == 42
            and item["epoch"] == 1
        )
    )
    adapter_path = (
        args.training_root
        / "seed42/s0/checkpoint-logical-epoch-1/adapter"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["local_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = config["model"]["pad_token_id"]
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["local_path"],
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
    first = constrained_greedy_decode(
        model,
        tokenizer,
        [row["prompt_token_ids"]],
        protocol=protocol,
        runtime=runtime,
        device="cuda:0",
    )[0]
    second = constrained_greedy_decode(
        model,
        tokenizer,
        [row["prompt_token_ids"]],
        protocol=protocol,
        runtime=runtime,
        device="cuda:0",
    )[0]
    parse_review_json(first.text)
    parse_review_json(second.text)
    byte_identical = (
        first.text.encode("utf-8") == second.text.encode("utf-8")
    )
    token_identical = first.token_ids == second.token_ids
    diagnostics_identical = (
        first.diagnostics == second.diagnostics
    )
    if not (byte_identical and token_identical and diagnostics_identical):
        raise ValueError("real-checkpoint repeated V2 inference differs")
    write_json(
        args.output_path,
        {
            "status": "EXP54_INFERENCE_V2_REAL_DETERMINISM_PROBE_PASS",
            "schema_version": "exp54-inference-v2-determinism-probe-v1",
            "protocol_config_sha256": file_sha256(args.protocol_config),
            "probe_source_sha256": file_sha256(Path(__file__).resolve()),
            "adapter_model_sha256": file_sha256(
                adapter_path / "adapter_model.safetensors"
            ),
            "arm": "S0",
            "seed": 42,
            "logical_epoch": 1,
            "repetitions": 2,
            "output_bytes_identical": True,
            "output_token_ids_identical": True,
            "diagnostics_identical": True,
            "strict_parse_success_both": True,
            "generated_text_public": False,
            "token_ids_public": False,
            "row_identifier_public": False,
            "train_only": True,
            "dev_accessed": False,
            "test_accessed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data"
        ),
    )
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
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    run_probe(parse_args())


if __name__ == "__main__":
    main()
