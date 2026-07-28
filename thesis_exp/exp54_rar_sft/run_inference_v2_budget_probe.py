"""Run the locked 256-token V2 completion probe without data access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
from thesis_exp.exp54_rar_sft.structured_decoder_v2 import (
    DEFAULT_PROTOCOL_CONFIG,
    constrained_greedy_decode,
    file_sha256,
    load_v2_protocol,
    prepare_v2_runtime,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    verify_model_snapshot,
)


class ContinuationPolicy:
    """Synthetic policy that prefers an illegal EOS, then endless content."""

    def __init__(
        self,
        *,
        vocab_size: int,
        prefix_tokens: list[int],
        continuation_token_id: int,
        eos_token_id: int,
    ):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.prefix_tokens = prefix_tokens
        self.continuation_token_id = continuation_token_id
        self.eos_token_id = eos_token_id
        self.call_index = 0

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        use_cache: bool,
    ) -> Any:
        import torch

        batch_size = int(input_ids.shape[0])
        logits = torch.full(
            (batch_size, 1, self.config.vocab_size),
            -20.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        preferred = (
            self.prefix_tokens[self.call_index]
            if self.call_index < len(self.prefix_tokens)
            else self.continuation_token_id
        )
        logits[:, :, preferred] = 5.0
        logits[:, :, self.eos_token_id] = 10.0
        self.call_index += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


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
    from transformers import AutoTokenizer

    protocol = load_v2_protocol(args.protocol_config)
    if protocol["generation"]["max_new_tokens"] != 256:
        raise ValueError("budget probe requires the locked 256-token budget")
    config = _read_object(DEFAULT_CONFIG)
    verify_model_snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["local_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = config["model"]["pad_token_id"]
    prefix_tokens = tokenizer.encode(
        '{"score":2,"rationale":"',
        add_special_tokens=False,
    )
    continuation_tokens = tokenizer.encode("a", add_special_tokens=False)
    if len(continuation_tokens) != 1:
        raise ValueError("locked tokenizer does not encode probe content once")
    policy = ContinuationPolicy(
        vocab_size=protocol["model_and_tokenizer"]["model_vocab_size"],
        prefix_tokens=prefix_tokens,
        continuation_token_id=continuation_tokens[0],
        eos_token_id=protocol["model_and_tokenizer"]["eos_token_id"],
    )
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=policy.config.vocab_size,
        protocol=protocol,
    )
    decoded = constrained_greedy_decode(
        policy,
        tokenizer,
        [[1]],
        protocol=protocol,
        runtime=runtime,
        device="cuda:0",
    )[0]
    prediction = parse_review_json(decoded.text)
    diagnostics = decoded.to_dict()["diagnostics"]
    if prediction["score"] != 2:
        raise ValueError("budget probe changed the selected score")
    if not prediction["rationale"] or set(prediction["rationale"]) != {"a"}:
        raise ValueError("budget probe added non-probe semantic content")
    if (
        diagnostics["generated_token_count"] != 256
        or not diagnostics["forced_completion"]
        or not diagnostics["completion_at_max_token_boundary"]
    ):
        raise ValueError("budget probe did not close at locked boundary")
    write_json(
        args.output_path,
        {
            "status": "EXP54_INFERENCE_V2_LOCKED_BUDGET_PROBE_PASS",
            "schema_version": "exp54-inference-v2-budget-probe-v1",
            "protocol_config_sha256": file_sha256(args.protocol_config),
            "probe_source_sha256": file_sha256(Path(__file__).resolve()),
            "generated_token_count": diagnostics[
                "generated_token_count"
            ],
            "forced_completion": diagnostics["forced_completion"],
            "completion_at_max_token_boundary": diagnostics[
                "completion_at_max_token_boundary"
            ],
            "score_preserved": True,
            "strict_parse_success": True,
            "unconstrained_top1_blocked_steps": diagnostics[
                "unconstrained_top1_blocked_steps"
            ],
            "generated_text_public": False,
            "token_ids_public": False,
            "data_accessed": False,
            "dev_accessed": False,
            "test_accessed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
