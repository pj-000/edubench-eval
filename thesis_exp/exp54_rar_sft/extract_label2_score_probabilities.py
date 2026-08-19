"""Extract five-way score probabilities from frozen Exp54 R3/P1 adapters.

This is a forward-only diagnostic.  It never reads test, generates a
rationale, computes gradients, updates parameters, or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.run_dev_inference import (
    DEV_PATH,
    load_dev_rows,
    prepare_prompts,
)
from thesis_exp.exp54_rar_sft.train_rar_sft import (
    DEFAULT_CONFIG,
    _read_object,
    validate_training_configuration,
    verify_model_snapshot,
)
from thesis_exp.exp54_rar_sft.training_contract import token_ids_sha256


ARMS = ("R3", "P1_FIELD_DPO")
SEEDS = (42, 43, 44)
SCORES = (1, 2, 3, 4, 5)
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_INVENTORY = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "label2_identification_audit/inventory_report.json"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_ARTIFACT_ROOT / "label2_identification_audit/private/score_probabilities"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
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
    temporary.replace(path)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def longest_common_token_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("canonical score sequences must be nonempty")
    limit = min(len(sequence) for sequence in sequences)
    end = 0
    while end < limit and len({sequence[end] for sequence in sequences}) == 1:
        end += 1
    if end == 0:
        raise ValueError("canonical score sequences have no shared token prefix")
    if end == limit:
        raise ValueError("a canonical score continuation is empty")
    return list(sequences[0][:end])


@dataclass(frozen=True)
class CanonicalScorePlan:
    full_token_ids: dict[int, list[int]]
    common_prefix_ids: list[int]
    continuation_ids: dict[int, list[int]]


def build_canonical_score_plan(tokenizer: Any) -> CanonicalScorePlan:
    texts = {score: f'{{"score":{score}' for score in SCORES}
    full = {
        score: [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        for score, text in texts.items()
    }
    for score, token_ids in full.items():
        if tokenizer.decode(token_ids, skip_special_tokens=False) != texts[score]:
            raise ValueError(f"canonical score {score} does not round-trip")
    common = longest_common_token_prefix(list(full.values()))
    continuations = {
        score: token_ids[len(common) :] for score, token_ids in full.items()
    }
    if any(not token_ids for token_ids in continuations.values()):
        raise ValueError("canonical score continuation must be nonempty")
    return CanonicalScorePlan(
        full_token_ids=full,
        common_prefix_ids=common,
        continuation_ids=continuations,
    )


def normalize_logprob_row(values: list[float]) -> list[float]:
    if len(values) != len(SCORES) or any(not math.isfinite(x) for x in values):
        raise ValueError("five finite score log-probabilities are required")
    maximum = max(values)
    denominator = sum(math.exp(value - maximum) for value in values)
    return [math.exp(value - maximum) / denominator for value in values]


def _score_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: list[list[int]],
    plan: CanonicalScorePlan,
    device: str,
) -> list[list[float]]:
    """Return sequence log-probabilities for canonical scores 1..5."""
    import torch

    if not prompt_token_ids:
        return []
    contexts = [tokens + plan.common_prefix_ids for tokens in prompt_token_ids]
    continuation_lengths = {
        len(value) for value in plan.continuation_ids.values()
    }
    if continuation_lengths == {1}:
        encoded = tokenizer.pad(
            {"input_ids": contexts},
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(
                **encoded,
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        candidate_ids = torch.tensor(
            [plan.continuation_ids[score][0] for score in SCORES],
            dtype=torch.long,
            device=log_probs.device,
        )
        selected = log_probs.index_select(-1, candidate_ids)
        return [
            [float(value) for value in row]
            for row in selected.detach().cpu().tolist()
        ]

    expanded: list[list[int]] = []
    metadata: list[tuple[int, int]] = []
    max_continuation = max(continuation_lengths)
    for row_index, context in enumerate(contexts):
        for score in SCORES:
            continuation = plan.continuation_ids[score]
            expanded.append(context + continuation)
            metadata.append((row_index, score))
    encoded = tokenizer.pad(
        {"input_ids": expanded},
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(
            **encoded,
            use_cache=False,
            logits_to_keep=max_continuation + 1,
        ).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    result = [[float("nan")] * len(SCORES) for _ in contexts]
    for expanded_index, (row_index, score) in enumerate(metadata):
        continuation = plan.continuation_ids[score]
        start = int(log_probs.shape[1]) - len(continuation) - 1
        positions = torch.arange(
            start,
            start + len(continuation),
            device=log_probs.device,
        )
        token_ids = torch.tensor(
            continuation,
            dtype=torch.long,
            device=log_probs.device,
        )
        value = log_probs[expanded_index, positions, token_ids].sum()
        result[row_index][score - 1] = float(value.detach().cpu().item())
    return result


def _artifact_paths(
    *,
    artifact_root: Path,
    arm: str,
    seed: int,
) -> tuple[Path, Path]:
    if arm == "R3":
        adapter = (
            artifact_root
            / f"formal_runs/seed{seed}/r3/"
            "checkpoint-logical-epoch-3/adapter"
        )
        predictions = (
            artifact_root
            / f"dev_runs_vllm/r3/seed{seed}/epoch3/predictions.jsonl"
        )
    elif arm == "P1_FIELD_DPO":
        adapter = (
            artifact_root
            / "preference_lr5e6_followup/train/"
            f"p1_field_dpo/seed_{seed}/adapter"
        )
        predictions = (
            artifact_root
            / "preference_lr5e6_followup/dev/"
            f"p1_field_dpo/seed_{seed}/predictions.jsonl"
        )
    else:
        raise ValueError(f"unsupported arm: {arm}")
    return adapter, predictions


def _expected_inventory_group(arm: str) -> str:
    return (
        "r3_epoch3_dev_predictions"
        if arm == "R3"
        else "p1_lr5e6_dev_predictions"
    )


def _expected_adapter_group(arm: str) -> str:
    return "r3_epoch3_adapter_sha256" if arm == "R3" else "p1_lr5e6_adapter_sha256"


def _load_predictions(path: Path, expected_hash: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_hash:
        raise ValueError("frozen prediction hash mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 664:
        raise ValueError("frozen prediction row count differs")
    return rows


def run(args: argparse.Namespace) -> None:
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise ValueError("arm or seed is outside the frozen set")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
        raise RuntimeError("exactly one CUDA device must be made visible")
    if "," in str(os.environ["CUDA_VISIBLE_DEVICES"]):
        raise RuntimeError("each extraction process must see exactly one GPU")
    output_dir = args.output_root / args.arm.lower() / f"seed_{args.seed}"
    if output_dir.exists():
        raise FileExistsError(output_dir)

    config = _read_object(DEFAULT_CONFIG)
    validate_training_configuration(config)
    verify_model_snapshot(config)
    inventory = _load_object(args.inventory)
    if inventory.get("status") != "INVENTORY_COMPLETE_LOGITS_MISSING":
        raise ValueError("inventory is not the expected frozen precursor")
    if inventory.get("test_accessed_by_inventory") is not False:
        raise ValueError("inventory test boundary differs")

    adapter_path, predictions_path = _artifact_paths(
        artifact_root=args.artifact_root,
        arm=args.arm,
        seed=args.seed,
    )
    adapter_model = adapter_path / "adapter_model.safetensors"
    adapter_config = adapter_path / "adapter_config.json"
    if any(not path.is_file() or path.is_symlink() for path in (adapter_model, adapter_config)):
        raise FileNotFoundError(adapter_path)
    server = inventory["server_read_only_inventory"]
    expected_adapter_hash = server[_expected_adapter_group(args.arm)][str(args.seed)]
    if sha256_file(adapter_model) != expected_adapter_hash:
        raise ValueError("frozen adapter hash mismatch")
    expected_prediction = server[_expected_inventory_group(args.arm)][str(args.seed)]
    prediction_rows = _load_predictions(
        predictions_path,
        str(expected_prediction["sha256"]),
    )
    dev_rows = load_dev_rows()
    if sha256_file(DEV_PATH) != _load_object(
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/label2_identification_audit_v1.json"
    )["locked_inputs"]["dev"]["sha256"]:
        raise ValueError("locked dev hash mismatch")
    for position, (dev_row, prediction) in enumerate(
        zip(dev_rows, prediction_rows, strict=True)
    ):
        if (
            int(prediction["row_position"]) != position
            or str(prediction["record_id"]) != str(dev_row["record_id"])
            or int(prediction["label_5"]) != int(dev_row["label_5"])
            or prediction.get("parse_success") is not True
        ):
            raise ValueError(f"frozen prediction differs at row {position}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    started = time.time()
    model_path = config["model"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = int(config["model"]["pad_token_id"])
    plan = build_canonical_score_plan(tokenizer)
    prepared = prepare_prompts(tokenizer, dev_rows)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("forward-only model unexpectedly has trainable parameters")

    output_dir.mkdir(parents=True, exist_ok=False)
    rows_by_position: dict[int, dict[str, Any]] = {}
    ordered = sorted(prepared, key=lambda item: len(item["token_ids"]))
    for start in range(0, len(ordered), args.batch_size):
        batch = ordered[start : start + args.batch_size]
        raw_logprobs = _score_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=[item["token_ids"] for item in batch],
            plan=plan,
            device="cuda:0",
        )
        for item, logprob_values in zip(batch, raw_logprobs, strict=True):
            probabilities = normalize_logprob_row(logprob_values)
            prediction = prediction_rows[item["row_position"]]
            row = item["row"]
            rows_by_position[item["row_position"]] = {
                "row_position": int(item["row_position"]),
                "record_id": str(row["record_id"]),
                "question_key": str(row["question_key"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
                "label_5": int(row["label_5"]),
                "human_mean_5": float(row["human_mean_5"]),
                "human_1_5": float(row["human_1_5"]),
                "human_2_5": float(row["human_2_5"]),
                "human_3_5": float(row["human_3_5"]),
                "parsed_score": int(prediction["prediction"]["score"]),
                "forced_completion": bool(prediction["forced_completion"]),
                "prompt_token_ids_sha256": token_ids_sha256(item["token_ids"]),
                "canonical_score_option_logprobs": {
                    str(score): logprob_values[score - 1] for score in SCORES
                },
                "canonical_score_option_probabilities": {
                    str(score): probabilities[score - 1] for score in SCORES
                },
                "forced_choice_score": int(
                    max(SCORES, key=lambda score: probabilities[score - 1])
                ),
            }
        completed = min(start + len(batch), len(ordered))
        print(
            f"LABEL2_SCORE_PROB_PROGRESS arm={args.arm} seed={args.seed} "
            f"rows={completed}/{len(ordered)}",
            flush=True,
        )
    output_rows = [rows_by_position[index] for index in range(len(dev_rows))]
    probabilities_path = output_dir / "score_probabilities.jsonl"
    write_jsonl(probabilities_path, output_rows)
    report = {
        "status": "LABEL2_SCORE_PROBABILITY_EXTRACTION_COMPLETE",
        "arm": args.arm,
        "seed": args.seed,
        "rows": len(output_rows),
        "label_2_rows": sum(row["label_5"] == 2 for row in output_rows),
        "score_probability_rows": sum(
            abs(
                sum(row["canonical_score_option_probabilities"].values()) - 1.0
            )
            < 1e-6
            for row in output_rows
        ),
        "adapter_model_sha256": sha256_file(adapter_model),
        "adapter_config_sha256": sha256_file(adapter_config),
        "input_predictions_sha256": sha256_file(predictions_path),
        "output_score_probabilities_sha256": sha256_file(probabilities_path),
        "extractor_source_sha256": sha256_file(Path(__file__)),
        "inventory_report_sha256": sha256_file(args.inventory),
        "dev_sha256": sha256_file(DEV_PATH),
        "canonical_full_token_ids_sha256": {
            str(score): token_ids_sha256(plan.full_token_ids[score])
            for score in SCORES
        },
        "canonical_common_prefix_token_ids_sha256": token_ids_sha256(
            plan.common_prefix_ids
        ),
        "canonical_continuation_lengths": {
            str(score): len(plan.continuation_ids[score]) for score in SCORES
        },
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "forward_only": True,
        "grad_enabled": torch.is_grad_enabled(),
        "training_started": False,
        "test_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    if report["score_probability_rows"] != len(output_rows):
        raise RuntimeError("score probability normalization failed")
    write_json(output_dir / "report.json", report)
    print(
        f"LABEL2_SCORE_PROBABILITY_EXTRACTION_PASS arm={args.arm} "
        f"seed={args.seed}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
