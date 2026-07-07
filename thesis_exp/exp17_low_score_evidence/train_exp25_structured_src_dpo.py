"""Train or predict with Exp25 Structured SRC-DPO.

This trainer implements weighted length-normalized DPO over same-schema
reason/score consistency pairs. It intentionally stays independent of
LLaMA-Factory internals because Exp25 needs per-pair weights and
length-normalized response log probabilities.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean, messages_for  # noqa: E402
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_json  # noqa: E402


DEFAULT_MODEL = "/home/jpang/models/modelscope/Qwen/Qwen3-4B"
DEFAULT_INIT_ADAPTER = "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
DEFAULT_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "data/edubench_r7h_score_mismatch_only_train.json"
)
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_OUT_DIR = Path("saves/edubench/qwen3-4b/exp25_src_score_mismatch_r2c")
DEFAULT_PRED_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "dev_predictions/exp25_src_score_mismatch_r2c"
)


def require_training_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from peft import PeftModel
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:  # pragma: no cover - server dependency path
        raise SystemExit(f"Missing training dependency: {exc}") from exc
    return torch, F, PeftModel, pad_sequence, DataLoader, (AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def assistant_content(message: dict[str, Any]) -> str:
    return clean(message.get("content"))


def chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message['role']}:\n{message['content']}")
    parts.append("assistant:\n")
    return "\n\n".join(parts)


def sequence_features(tokenizer: Any, prompt_text: str, target_text: str, max_length: int) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_ids = tokenizer(target_text + eos, add_special_tokens=False).input_ids
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_length:
        keep_prompt = max(1, max_length - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        if len(prompt_ids) + len(target_ids) > max_length:
            target_ids = target_ids[: max(1, max_length - len(prompt_ids))]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


@dataclass
class EncodedItem:
    chosen: dict[str, list[int]]
    rejected: dict[str, list[int]]
    weight: float
    pair_id: str
    negative_type: str


class SRCDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace):
        self.items: list[EncodedItem] = []
        for row in rows:
            prompt = chat_prompt(tokenizer, row["messages"])
            chosen = sequence_features(tokenizer, prompt, assistant_content(row["chosen"]), args.cutoff_len)
            rejected = sequence_features(tokenizer, prompt, assistant_content(row["rejected"]), args.cutoff_len)
            self.items.append(
                EncodedItem(
                    chosen=chosen,
                    rejected=rejected,
                    weight=float(row.get("pair_weight") or 1.0),
                    pair_id=clean(row.get("pair_id")),
                    negative_type=clean(row.get("negative_type")),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedItem:
        return self.items[idx]


def collate_items(batch: list[EncodedItem], tokenizer: Any, torch: Any, pad_sequence: Any) -> dict[str, Any]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_seq(name: str) -> tuple[Any, Any, Any]:
        ids = [torch.tensor(getattr(item, name)["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(getattr(item, name)["labels"], dtype=torch.long) for item in batch]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
        label_ids = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(pad_id).long()
        return input_ids, label_ids, attention_mask

    chosen_ids, chosen_labels, chosen_mask = collate_seq("chosen")
    rejected_ids, rejected_labels, rejected_mask = collate_seq("rejected")
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_mask,
        "weights": torch.tensor([item.weight for item in batch], dtype=torch.float32),
    }


def avg_logp_and_nll(model: Any, batch: dict[str, Any], prefix: str, F: Any) -> tuple[Any, Any]:
    outputs = model(
        input_ids=batch[f"{prefix}_input_ids"],
        attention_mask=batch[f"{prefix}_attention_mask"],
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch[f"{prefix}_labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) * mask
    token_count = mask.sum(dim=-1).clamp(min=1)
    avg_logp = token_logps.sum(dim=-1) / token_count
    nll = -token_logps.sum() / token_count.sum().clamp(min=1)
    return avg_logp, nll


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def load_tokenizer_and_models(args: argparse.Namespace, trainable: bool) -> tuple[Any, Any, Any | None, Any]:
    torch, _F, PeftModel, _pad, _DataLoader, hf = require_training_deps()
    AutoModelForCausalLM, AutoTokenizer, _scheduler = hf
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter_name_or_path, is_trainable=trainable)
    model.config.use_cache = False
    if trainable and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    ref_model = None
    if trainable:
        ref_base = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        ref_model = PeftModel.from_pretrained(ref_base, args.ref_adapter_name_or_path or args.adapter_name_or_path, is_trainable=False)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    device = next(model.parameters()).device
    return tokenizer, model, ref_model, device


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, F, _PeftModel, pad_sequence, DataLoader, hf = require_training_deps()
    _AutoModel, _AutoTokenizer, get_linear_schedule_with_warmup = hf
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_json_array(args.data)
    if args.max_train_examples > 0:
        rows = rows[: args.max_train_examples]
    tokenizer, model, ref_model, device = load_tokenizer_and_models(args, trainable=True)
    dataset = SRCDataset(rows, tokenizer, args)
    item_weights = [item.weight for item in dataset.items]
    negative_counts = Counter(clean(row.get("negative_type")) for row in rows)
    trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_param_count = sum(param.numel() for param in model.parameters())
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_items(batch, tokenizer, torch, pad_sequence),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, args.warmup_steps),
        num_training_steps=args.max_steps,
    )
    model.train()
    start = time.time()
    step = 0
    micro_step = 0
    epoch = 0
    last_metrics: dict[str, float] = {}
    initial_mean_delta_step1: float | None = None
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        epoch += 1
        for batch in loader:
            batch = move_batch(batch, device)
            weights = batch["weights"].to(device=device, dtype=torch.float32)
            chosen_logp, chosen_nll = avg_logp_and_nll(model, batch, "chosen", F)
            rejected_logp, _rejected_nll = avg_logp_and_nll(model, batch, "rejected", F)
            with torch.no_grad():
                ref_chosen_logp, _ = avg_logp_and_nll(ref_model, batch, "chosen", F)
                ref_rejected_logp, _ = avg_logp_and_nll(ref_model, batch, "rejected", F)
            delta = (chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
            src_loss = (weights * F.softplus(-args.beta * delta)).sum() / weights.sum().clamp(min=1.0)
            loss = src_loss + args.pref_ftx * chosen_nll
            (loss / args.gradient_accumulation_steps).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps != 0:
                continue
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_metrics = {
                "loss": float(loss.detach().cpu()),
                "src_loss": float(src_loss.detach().cpu()),
                "chosen_nll": float(chosen_nll.detach().cpu()),
                "mean_delta": float(delta.detach().mean().cpu()),
                "mean_weight": float(weights.detach().mean().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            if step == 1 and initial_mean_delta_step1 is None:
                initial_mean_delta_step1 = last_metrics["mean_delta"]
            if step == 1 or step % args.logging_steps == 0 or step >= args.max_steps:
                elapsed = time.time() - start
                eta = elapsed / max(step, 1) * max(args.max_steps - step, 0)
                print(
                    "[exp25-src] "
                    f"{args.run_name} step {step}/{args.max_steps} epoch={epoch} "
                    f"loss={last_metrics['loss']:.4f} src={last_metrics['src_loss']:.4f} "
                    f"chosen_nll={last_metrics['chosen_nll']:.4f} delta={last_metrics['mean_delta']:.4f} "
                    f"lr={last_metrics['lr']:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    cuda_peak_mb = None
    cuda_device_name = ""
    if torch.cuda.is_available():
        cuda_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        cuda_device_name = torch.cuda.get_device_name(0)
    summary = {
        "run_name": args.run_name,
        "data": str(args.data),
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "negative_type_counts": dict(sorted(negative_counts.items())),
        "max_steps": args.max_steps,
        "completed_steps": step,
        "learning_rate": args.learning_rate,
        "cutoff_len": args.cutoff_len,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "beta": args.beta,
        "pref_ftx": args.pref_ftx,
        "length_normalized_logp": True,
        "initial_mean_delta_step1": initial_mean_delta_step1,
        "mean_weight_all_pairs": sum(item_weights) / max(len(item_weights), 1),
        "max_weight_all_pairs": max(item_weights) if item_weights else None,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "cuda_device_name": cuda_device_name,
        "cuda_memory_allocated_peak_mb": cuda_peak_mb,
        "last_metrics": last_metrics,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = args.summary_dir / f"{args.run_name}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    return summary


def predict(args: argparse.Namespace) -> dict[str, Any]:
    torch, _F, _PeftModel, _pad, _DataLoader, _hf = require_training_deps()
    tokenizer, model, _ref_model, device = load_tokenizer_and_models(args, trainable=False)
    model.eval()
    rows = read_jsonl(args.dev_jsonl)
    if args.max_predict_examples > 0:
        rows = rows[: args.max_predict_examples]
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.prediction_dir / "generated_predictions.jsonl"
    start = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            prompt = chat_prompt(tokenizer, messages_for(row))
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            f.write(json.dumps({"predict": text}, ensure_ascii=False) + "\n")
            if idx == 1 or idx % args.predict_log_steps == 0 or idx == len(rows):
                elapsed = time.time() - start
                eta = elapsed / max(idx, 1) * max(len(rows) - idx, 0)
                print(
                    f"[exp25-src] {args.run_name} generated {idx}/{len(rows)} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
    result = {
        "run_name": args.run_name,
        "rows": len(rows),
        "prediction_file": str(output_path),
        "elapsed_seconds": time.time() - start,
    }
    write_json(args.prediction_dir / "predict_results.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/predict Exp25 Structured SRC-DPO.")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-name-or-path", default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--ref-adapter-name-or-path", default="")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/training_summaries"),
    )
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--pref-ftx", type=float, default=0.05)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-predict-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--predict-log-steps", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        result = train(args)
    else:
        if not args.adapter_name_or_path:
            args.adapter_name_or_path = str(args.output_dir)
        result = predict(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
