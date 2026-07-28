"""Run a deterministic train-only structural smoke for Exp54 V2 inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import DEFAULT_TRAIN
from thesis_exp.exp54_rar_sft.inference_contract import parse_review_json
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
ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
EPOCHS = (1, 2, 3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _selection_key(
    row: dict[str, Any],
    *,
    target_label: int,
    target_language: str,
    target_active: bool,
) -> tuple[Any, ...]:
    return (
        int(int(row["score_target"]) != target_label),
        int(str(row["language"]) != target_language),
        int(bool(row["rationale_active"]) != target_active),
        str(row["base_event_id"]),
    )


def build_smoke_selection(data_dir: Path) -> list[dict[str, Any]]:
    train_rows = read_jsonl(DEFAULT_TRAIN)
    metadata = {
        str(row["record_id"]): {
            "label": int(row["label_5"]),
            "language": str(row["language"]),
        }
        for row in train_rows
    }
    if len(metadata) != len(train_rows):
        raise ValueError("train metadata contains duplicate record IDs")
    prompt_cache_rows = read_jsonl(data_dir / "shared_prompt_cache.jsonl")
    prompts = {
        str(row["prompt_cache_id"]): row for row in prompt_cache_rows
    }
    if len(prompts) != len(prompt_cache_rows):
        raise ValueError("prompt cache contains duplicate IDs")

    selected = []
    group_index = 0
    for arm in ARMS:
        for seed in SEEDS:
            manifest_path = (
                data_dir / f"training_manifest_{arm.lower()}_seed{seed}.jsonl"
            )
            manifest = read_jsonl(manifest_path)
            for epoch in EPOCHS:
                candidates = [
                    dict(row, **metadata[str(row["record_id"])])
                    for row in manifest
                    if int(row["epoch_number"]) == epoch
                ]
                if not candidates:
                    raise ValueError("empty train-only smoke group")
                target_label = group_index % 5 + 1
                target_language = "zh" if group_index % 2 == 0 else "en"
                target_active = arm != "S0" and group_index % 2 == 0
                chosen = min(
                    candidates,
                    key=lambda row: _selection_key(
                        row,
                        target_label=target_label,
                        target_language=target_language,
                        target_active=target_active,
                    ),
                )
                prompt = prompts[str(chosen["prompt_cache_id"])]
                if (
                    prompt["record_id"] != chosen["record_id"]
                    or prompt["prompt_token_ids_sha256"]
                    != chosen["prompt_token_ids_sha256"]
                ):
                    raise ValueError("manifest and prompt-cache binding differs")
                selected.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        "label": int(chosen["score_target"]),
                        "language": str(chosen["language"]),
                        "rationale_active": bool(
                            chosen["rationale_active"]
                        ),
                        "base_event_id": str(chosen["base_event_id"]),
                        "record_id": str(chosen["record_id"]),
                        "prompt_cache_id": str(chosen["prompt_cache_id"]),
                        "prompt_token_ids": [
                            int(value)
                            for value in prompt["prompt_token_ids"]
                        ],
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": file_sha256(manifest_path),
                    }
                )
                group_index += 1
    expected_groups = {
        (arm, seed, epoch)
        for arm in ARMS
        for seed in SEEDS
        for epoch in EPOCHS
    }
    actual_groups = {
        (row["arm"], row["seed"], row["epoch"]) for row in selected
    }
    if actual_groups != expected_groups or len(selected) != 36:
        raise ValueError("train-only smoke group closure differs")
    if {row["label"] for row in selected} != {1, 2, 3, 4, 5}:
        raise ValueError("train-only smoke does not cover all labels")
    if {row["language"] for row in selected} != {"en", "zh"}:
        raise ValueError("train-only smoke does not cover both languages")
    if {row["rationale_active"] for row in selected} != {False, True}:
        raise ValueError("train-only smoke does not cover activity states")
    return selected


def _adapter_path(
    training_root: Path,
    row: dict[str, Any],
) -> Path:
    return (
        training_root
        / f"seed{row['seed']}"
        / str(row["arm"]).lower()
        / f"checkpoint-logical-epoch-{row['epoch']}"
        / "adapter"
    )


def run_train_only_smoke(args: argparse.Namespace) -> None:
    started = time.time()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = load_v2_protocol(args.protocol_config)
    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
    selected = build_smoke_selection(args.data_dir)

    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = config["model"]["pad_token_id"]
    first_adapter = _adapter_path(args.training_root, selected[0])
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(
        base,
        str(first_adapter),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    runtime = prepare_v2_runtime(
        tokenizer,
        model_vocab_size=int(model.config.vocab_size),
        protocol=protocol,
    )
    private_results = []
    aggregate = {
        "rows": 0,
        "parse_success_count": 0,
        "forced_completion_count": 0,
        "max_boundary_count": 0,
        "active_generation_steps": 0,
        "grammar_intervention_steps": 0,
        "unconstrained_top1_blocked_steps": 0,
        "removed_probability_mass_weighted_sum": 0.0,
    }
    current_adapter = first_adapter
    for index, row in enumerate(selected):
        adapter = _adapter_path(args.training_root, row)
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
        if adapter != current_adapter:
            state = load_file(
                str(adapter / "adapter_model.safetensors"),
                device="cpu",
            )
            result = set_peft_model_state_dict(model, state)
            if any(
                "lora_" in key
                for key in getattr(result, "missing_keys", ())
            ):
                raise ValueError("adapter reload reported missing LoRA keys")
            if getattr(result, "unexpected_keys", None):
                raise ValueError("adapter reload reported unexpected keys")
            current_adapter = adapter
        decoded = constrained_greedy_decode(
            model,
            tokenizer,
            [row["prompt_token_ids"]],
            protocol=protocol,
            runtime=runtime,
            device="cuda:0",
        )[0]
        prediction = parse_review_json(decoded.text)
        diagnostics = decoded.to_dict()["diagnostics"]
        aggregate["rows"] += 1
        aggregate["parse_success_count"] += 1
        aggregate["forced_completion_count"] += int(
            diagnostics["forced_completion"]
        )
        aggregate["max_boundary_count"] += int(
            diagnostics["completion_at_max_token_boundary"]
        )
        aggregate["active_generation_steps"] += int(
            diagnostics["active_generation_steps"]
        )
        aggregate["grammar_intervention_steps"] += int(
            diagnostics["grammar_intervention_steps"]
        )
        aggregate["unconstrained_top1_blocked_steps"] += int(
            diagnostics["unconstrained_top1_blocked_steps"]
        )
        aggregate["removed_probability_mass_weighted_sum"] += (
            float(diagnostics["removed_probability_mass_mean"])
            * int(diagnostics["active_generation_steps"])
        )
        private_results.append(
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "epoch": row["epoch"],
                "label": row["label"],
                "language": row["language"],
                "rationale_active": row["rationale_active"],
                "base_event_id": row["base_event_id"],
                "record_id": row["record_id"],
                "prompt_cache_id": row["prompt_cache_id"],
                "adapter_path": str(adapter),
                "adapter_model_sha256": file_sha256(
                    adapter / "adapter_model.safetensors"
                ),
                "output_text": decoded.text,
                "prediction": prediction,
                "diagnostics": diagnostics,
            }
        )
        print(
            f"[Exp54 V2 train smoke] {index + 1}/36 "
            f"{row['arm']}/seed{row['seed']}/epoch{row['epoch']}",
            flush=True,
        )
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True)
    private_path = args.output_dir / "private_train_smoke_results.jsonl"
    write_jsonl(private_path, private_results)
    steps = aggregate["active_generation_steps"]
    public_report = {
        "status": "EXP54_INFERENCE_V2_TRAIN_ONLY_SMOKE_COMPLETE",
        "schema_version": "exp54-inference-v2-train-smoke-v1",
        "protocol_config_sha256": file_sha256(args.protocol_config),
        "grammar_sha256": protocol["grammar"]["sha256"],
        "decoder_source_sha256": file_sha256(
            Path(__file__).resolve().parent / "structured_decoder_v2.py"
        ),
        "runner_source_sha256": file_sha256(Path(__file__).resolve()),
        "train_source_sha256": file_sha256(DEFAULT_TRAIN),
        "shared_prompt_cache_sha256": file_sha256(
            args.data_dir / "shared_prompt_cache.jsonl"
        ),
        "private_results_sha256": file_sha256(private_path),
        "coverage": {
            "rows": len(selected),
            "arms": sorted({row["arm"] for row in selected}),
            "seeds": sorted({row["seed"] for row in selected}),
            "logical_epochs": sorted(
                {row["epoch"] for row in selected}
            ),
            "labels": sorted({row["label"] for row in selected}),
            "languages": sorted(
                {row["language"] for row in selected}
            ),
            "rationale_activity_states": sorted(
                {row["rationale_active"] for row in selected}
            ),
            "unique_arm_seed_epoch_groups": len(
                {
                    (row["arm"], row["seed"], row["epoch"])
                    for row in selected
                }
            ),
        },
        "execution": {
            "strict_parse_rate": (
                aggregate["parse_success_count"] / aggregate["rows"]
            ),
            "forced_completion_rate": (
                aggregate["forced_completion_count"] / aggregate["rows"]
            ),
            "max_token_hit_rate": (
                aggregate["max_boundary_count"] / aggregate["rows"]
            ),
            "grammar_intervention_step_rate": (
                aggregate["grammar_intervention_steps"] / steps
            ),
            "illegal_unconstrained_top1_rate": (
                aggregate["unconstrained_top1_blocked_steps"] / steps
            ),
            "removed_probability_mass_mean": (
                aggregate["removed_probability_mass_weighted_sum"] / steps
            ),
        },
        "privacy": {
            "row_level_selection_public": False,
            "record_or_event_ids_public": False,
            "train_prompts_public": False,
            "generated_text_public": False,
            "score_or_rationale_semantics_inspected": False,
        },
        "dev_accessed": False,
        "test_accessed": False,
        "scientific_metrics_computed": False,
        "formal_v2_dev_allowed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "public_train_smoke_report.json", public_report)


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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    run_train_only_smoke(parse_args())


if __name__ == "__main__":
    main()
