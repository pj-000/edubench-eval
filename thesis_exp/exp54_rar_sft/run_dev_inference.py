"""Run deterministic Exp54 dev inference for one frozen LoRA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thesis_exp.exp54_rar_sft.inference_contract import (
    GENERATION_KWARGS,
    parse_review_json,
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
    verify_model_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_PATH = (
    REPO_ROOT
    / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
EXPECTED_DEV_ROWS = 664
ARMS = ("S0", "R1", "R2", "R3")
SEEDS = (42, 43, 44)
EPOCHS = (1, 2, 3)
LABELS = (1, 2, 3, 4, 5)


def file_sha256(path: Path) -> str:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def load_dev_rows() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in DEV_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_DEV_ROWS:
        raise ValueError("Exp54 dev row count differs")
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if any(not value for value in record_ids):
        raise ValueError("Exp54 dev contains an empty record ID")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Exp54 dev contains duplicate record IDs")
    if any(int(row["label_5"]) not in LABELS for row in rows):
        raise ValueError("Exp54 dev label is outside 1-5")
    return rows


def kendall_tau_b(gold: list[int], pred: list[int]) -> float:
    if len(gold) < 2:
        return 0.0
    table = np.zeros((5, 5), dtype=np.int64)
    for gold_value, pred_value in zip(gold, pred):
        table[gold_value - 1, pred_value - 1] += 1
    concordant = 0
    discordant = 0
    for i in range(5):
        for j in range(5):
            count = int(table[i, j])
            concordant += count * int(table[i + 1 :, j + 1 :].sum())
            discordant += count * int(table[i + 1 :, :j].sum())
    tied_gold = sum(
        int(count) * (int(count) - 1) // 2
        for count in table.sum(axis=1)
    )
    tied_pred = sum(
        int(count) * (int(count) - 1) // 2
        for count in table.sum(axis=0)
    )
    tied_both = sum(
        int(count) * (int(count) - 1) // 2
        for count in table.reshape(-1)
    )
    denominator = math.sqrt(
        (concordant + discordant + tied_gold - tied_both)
        * (concordant + discordant + tied_pred - tied_both)
    )
    return (
        float((concordant - discordant) / denominator)
        if denominator
        else 0.0
    )


def score_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(not row["parse_success"] for row in rows):
        raise ValueError("strict parse failure prevents score evaluation")
    gold = np.asarray([int(row["label_5"]) for row in rows], dtype=np.int64)
    pred = np.asarray(
        [int(row["prediction"]["score"]) for row in rows],
        dtype=np.int64,
    )
    low = gold <= 2
    metrics: dict[str, Any] = {
        "n": len(rows),
        "Exact": float(np.mean(pred == gold)),
        "MAE": float(np.mean(np.abs(pred - gold))),
        "Kendall": kendall_tau_b(gold.tolist(), pred.tolist()),
        "Signed_Bias": float(np.mean(pred - gold)),
        "L2H_count": int(np.sum(low & (pred >= 4))),
        "L2H_rate": float(np.mean(pred[low] >= 4)),
        "low_n": int(low.sum()),
        "parse_success_rate": 1.0,
    }
    for label in LABELS:
        selected = gold == label
        metrics[f"Label_{label}_n"] = int(selected.sum())
        metrics[f"Label_{label}_correct_count"] = int(
            np.sum(selected & (pred == label))
        )
        metrics[f"Recall_{label}"] = float(
            np.mean(pred[selected] == label)
        )
    metrics["confusion_matrix"] = [
        [
            int(np.sum((gold == gold_label) & (pred == pred_label)))
            for pred_label in LABELS
        ]
        for gold_label in LABELS
    ]
    return metrics


def rationale_metrics(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    rationales = [
        str(row["prediction"]["rationale"]) for row in rows
    ]
    token_lengths = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in rationales
    ]
    nonempty = [bool(text.strip()) for text in rationales]
    score_leak = []
    language_match = []
    for row, rationale in zip(rows, rationales):
        lowered = rationale.lower()
        score = int(row["prediction"]["score"])
        score_leak.append(
            f'"score":{score}' in lowered.replace(" ", "")
            or f"score {score}" in lowered
            or f"评分{score}" in rationale.replace(" ", "")
            or f"{score}分" in rationale.replace(" ", "")
        )
        cjk = sum("\u4e00" <= char <= "\u9fff" for char in rationale)
        latin = sum(char.isascii() and char.isalpha() for char in rationale)
        language = str(row["language"])
        language_match.append(
            (cjk > 0 and cjk >= latin * 0.05)
            if language == "zh"
            else (latin > 0 and cjk <= latin * 0.05)
        )
    return {
        "nonempty_rationale_rate": float(np.mean(nonempty)),
        "empty_rationale_count": int(len(rows) - sum(nonempty)),
        "mean_rationale_tokens": float(np.mean(token_lengths)),
        "median_rationale_tokens": float(np.median(token_lengths)),
        "max_rationale_tokens": max(token_lengths),
        "explicit_score_leakage_count": int(sum(score_leak)),
        "explicit_score_leakage_rate": float(np.mean(score_leak)),
        "language_consistency_rate": float(np.mean(language_match)),
    }


def prepare_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for row_position, row in enumerate(rows):
        token_ids = list(
            apply_chat_template(
                tokenizer,
                prompt_messages(row),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if not token_ids or len(token_ids) > 1792:
            raise ValueError(
                f"dev prompt length is outside frozen generation budget: "
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


def run_inference(args: argparse.Namespace) -> None:
    started = time.time()
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
    by_position: dict[int, dict[str, Any]] = {}
    ordered_for_batches = sorted(
        prepared,
        key=lambda item: len(item["token_ids"]),
    )
    for start in range(0, len(ordered_for_batches), args.batch_size):
        batch = ordered_for_batches[start : start + args.batch_size]
        encoded = tokenizer.pad(
            {"input_ids": [item["token_ids"] for item in batch]},
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.to("cuda:0")
            for key, value in encoded.items()
        }
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **GENERATION_KWARGS,
                pad_token_id=config["model"]["pad_token_id"],
                eos_token_id=config["model"]["eos_token_id"],
            )
        new_tokens = generated[:, input_width:]
        texts = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
        for item, text in zip(batch, texts):
            row = item["row"]
            try:
                prediction = parse_review_json(text)
                parse_success = True
                parse_error = None
            except (TypeError, ValueError) as exc:
                prediction = None
                parse_success = False
                parse_error = str(exc)
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
                "raw_output": text,
                "parse_success": parse_success,
                "parse_error": parse_error,
                "prediction": prediction,
            }
        print(
            f"[Exp54 dev] {args.arm}/seed{args.seed}/epoch{args.epoch}: "
            f"{min(start + len(batch), len(ordered_for_batches))}/"
            f"{len(ordered_for_batches)}",
            flush=True,
        )
    predictions = [
        by_position[position] for position in range(len(rows))
    ]
    parse_failures = [
        row for row in predictions if not row["parse_success"]
    ]
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    protocol = {
        "status": "EXP54_DEV_INFERENCE_COMPLETE",
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
        "generation": GENERATION_KWARGS,
        "batch_size": args.batch_size,
        "max_prompt_tokens": max(
            len(item["token_ids"]) for item in prepared
        ),
        "parse_failure_count": len(parse_failures),
        "dev_accessed": True,
        "test_accessed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(args.output_dir / "protocol.json", protocol)
    if parse_failures:
        write_json(
            args.output_dir / "parse_failures.json",
            {
                "count": len(parse_failures),
                "record_ids": [
                    row["record_id"] for row in parse_failures
                ],
            },
        )
        raise RuntimeError(
            f"strict dev parse failed for {len(parse_failures)} rows"
        )
    metrics = {
        "status": "EXP54_DEV_METRICS_COMPLETE",
        "arm": args.arm,
        "seed": args.seed,
        "epoch": args.epoch,
        "score": score_metrics(predictions),
        "rationale": rationale_metrics(predictions, tokenizer),
        "dev_accessed": True,
        "test_accessed": False,
    }
    write_json(args.output_dir / "metrics.json", metrics)


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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    run_inference(args)


if __name__ == "__main__":
    main()
