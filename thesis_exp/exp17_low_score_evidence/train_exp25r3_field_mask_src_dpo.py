"""Exp25R3 loss-scale and field-masked SRC-DPO sanity trainer.

This is a small overfit diagnostic, not a formal dev experiment. It checks
whether DPO preference margins can be increased on train-only Exp25 pairs
when beta, pref_ftx, and token masks are changed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean  # noqa: E402
from thesis_exp.exp17_low_score_evidence.train_exp25_structured_src_dpo import (  # noqa: E402
    assistant_content,
    chat_prompt,
    load_json_array,
    load_tokenizer_and_models,
    require_training_deps,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json  # noqa: E402


DEFAULT_MODEL = "/home/jpang/models/modelscope/Qwen/Qwen3-4B"
DEFAULT_INIT_ADAPTER = "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
DEFAULT_SCORE_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "data/edubench_r7h_score_mismatch_only_train.json"
)
DEFAULT_MIXED_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "data/edubench_r7h_structured_src_dpo_train.json"
)
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25r3_loss_scale_sanity_seed42")


FIELD_RE = re.compile(r'"(?P<field>reason|major_failures|failure_tag_source|score_cap|score)":')


@dataclass
class EncodedPair:
    chosen: dict[str, list[int]]
    rejected: dict[str, list[int]]
    weight: float
    pair_id: str
    negative_type: str
    risk_type: str


def choose_fields(negative_type: str) -> set[str]:
    if negative_type in {"score_mismatch_same_reason", "high_protection_score_mismatch"}:
        return {"score_cap", "score"}
    if negative_type == "reason_mismatch_same_score":
        return {"reason"}
    if negative_type == "low_failure_erasure_counterfactual":
        return {"major_failures", "failure_tag_source", "score_cap", "score"}
    return {"reason", "major_failures", "failure_tag_source", "score_cap", "score"}


def value_end(text: str, start: int) -> int:
    quote = False
    escape = False
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                quote = False
            continue
        if ch == '"':
            quote = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            if depth == 0:
                return idx
            depth -= 1
        elif ch == "," and depth == 0:
            return idx
    return len(text)


def selected_char_spans(target_text: str, fields: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in FIELD_RE.finditer(target_text):
        field = match.group("field")
        if field not in fields:
            continue
        start = match.start()
        end = value_end(target_text, match.end())
        spans.append((start, end))
    return spans


def overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < b and end > a for a, b in spans)


def sequence_features(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    max_length: int,
    logp_mode: str,
    selected_fields: set[str],
) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    target_encoded = tokenizer(
        target_text + eos,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    target_ids = target_encoded.input_ids
    offsets = target_encoded.offset_mapping
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]
        offsets = [(0, 0)]
    target_labels = list(target_ids)
    if logp_mode == "field":
        spans = selected_char_spans(target_text, selected_fields)
        masked_labels: list[int] = []
        for token_id, offset in zip(target_ids, offsets):
            if offset == (0, 0) or not overlaps(offset, spans):
                masked_labels.append(-100)
            else:
                masked_labels.append(token_id)
        if not any(label != -100 for label in masked_labels):
            masked_labels = target_labels
        target_labels = masked_labels
    if len(prompt_ids) + len(target_ids) > max_length:
        keep_prompt = max(1, max_length - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        if len(prompt_ids) + len(target_ids) > max_length:
            keep_target = max(1, max_length - len(prompt_ids))
            target_ids = target_ids[:keep_target]
            target_labels = target_labels[:keep_target]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_labels
    return {"input_ids": input_ids, "labels": labels}


class FieldMaskedDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, args: argparse.Namespace):
        self.items: list[EncodedPair] = []
        for row in rows:
            negative_type = clean(row.get("negative_type"))
            fields = choose_fields(negative_type)
            prompt = chat_prompt(tokenizer, row["messages"])
            chosen = sequence_features(
                tokenizer,
                prompt,
                assistant_content(row["chosen"]),
                args.cutoff_len,
                args.logp_mode,
                fields,
            )
            rejected = sequence_features(
                tokenizer,
                prompt,
                assistant_content(row["rejected"]),
                args.cutoff_len,
                args.logp_mode,
                fields,
            )
            self.items.append(
                EncodedPair(
                    chosen=chosen,
                    rejected=rejected,
                    weight=float(row.get("pair_weight") or 1.0),
                    pair_id=clean(row.get("pair_id")),
                    negative_type=negative_type,
                    risk_type=clean(row.get("risk_type")),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> EncodedPair:
        return self.items[idx]


def collate_pairs(batch: list[EncodedPair], tokenizer: Any, torch: Any, pad_sequence: Any) -> dict[str, Any]:
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
        "pair_ids": [item.pair_id for item in batch],
        "negative_types": [item.negative_type for item in batch],
        "risk_types": [item.risk_type for item in batch],
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
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.examples_per_negative_type > 0:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[clean(row.get("negative_type"))].append(row)
        selected: list[dict[str, Any]] = []
        for key in sorted(buckets):
            selected.extend(buckets[key][: args.examples_per_negative_type])
        return selected
    if args.max_train_examples > 0:
        return rows[: args.max_train_examples]
    return rows


def metric_summary(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in key_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        deltas = [float(row["dpo_delta"]) for row in values]
        model_margins = [float(row["model_margin"]) for row in values]
        item: dict[str, Any] = {field: value for field, value in zip(key_fields, key)}
        item.update(
            {
                "n": len(values),
                "dpo_pref_acc": sum(1 for value in deltas if value > 0) / len(deltas),
                "raw_pref_acc": sum(1 for value in model_margins if value > 0) / len(model_margins),
                "mean_delta": sum(deltas) / len(deltas),
                "mean_model_margin": sum(model_margins) / len(model_margins),
            }
        )
        out.append(item)
    return out


def evaluate(model: Any, ref_model: Any, loader: Any, device: Any, F: Any, torch: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            meta = {key: batch[key] for key in ["pair_ids", "negative_types", "risk_types"]}
            batch = move_batch(batch, device)
            chosen_logp, _ = avg_logp_and_nll(model, batch, "chosen", F)
            rejected_logp, _ = avg_logp_and_nll(model, batch, "rejected", F)
            ref_chosen_logp, _ = avg_logp_and_nll(ref_model, batch, "chosen", F)
            ref_rejected_logp, _ = avg_logp_and_nll(ref_model, batch, "rejected", F)
            model_margin = chosen_logp - rejected_logp
            ref_margin = ref_chosen_logp - ref_rejected_logp
            delta = model_margin - ref_margin
            for idx, pair_id in enumerate(meta["pair_ids"]):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "negative_type": meta["negative_types"][idx],
                        "risk_type": meta["risk_types"][idx],
                        "model_margin": float(model_margin[idx].detach().float().cpu()),
                        "ref_margin": float(ref_margin[idx].detach().float().cpu()),
                        "dpo_delta": float(delta[idx].detach().float().cpu()),
                    }
                )
    model.train()
    return rows


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, F, _PeftModel, pad_sequence, DataLoader, hf = require_training_deps()
    _AutoModel, _AutoTokenizer, get_linear_schedule_with_warmup = hf
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = select_rows(load_json_array(args.data), args)
    tokenizer, model, ref_model, device = load_tokenizer_and_models(args, trainable=True)
    dataset = FieldMaskedDataset(rows, tokenizer, args)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, torch, pad_sequence),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, torch, pad_sequence),
    )
    before_rows = evaluate(model, ref_model, loader, device, F, torch)
    before_summary = metric_summary(before_rows, ["negative_type"])
    before_overall = metric_summary(before_rows, ["risk_type"])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.max_steps)
    model.train()
    step = 0
    micro_step = 0
    start = time.time()
    first_batch_delta_before: float | None = None
    first_batch_delta_after: float | None = None
    last_metrics: dict[str, float] = {}
    first_batch_cache: dict[str, Any] | None = None
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in train_loader:
            if first_batch_cache is None:
                first_batch_cache = move_batch({key: value.clone() if hasattr(value, "clone") else value for key, value in batch.items()}, device)
                with torch.no_grad():
                    c0, _ = avg_logp_and_nll(model, first_batch_cache, "chosen", F)
                    r0, _ = avg_logp_and_nll(model, first_batch_cache, "rejected", F)
                    rc0, _ = avg_logp_and_nll(ref_model, first_batch_cache, "chosen", F)
                    rr0, _ = avg_logp_and_nll(ref_model, first_batch_cache, "rejected", F)
                    first_batch_delta_before = float(((c0 - r0) - (rc0 - rr0)).mean().detach().float().cpu())
            batch = move_batch(batch, device)
            weights = batch["weights"].to(device=device, dtype=torch.float32)
            chosen_logp, chosen_nll = avg_logp_and_nll(model, batch, "chosen", F)
            rejected_logp, _ = avg_logp_and_nll(model, batch, "rejected", F)
            with torch.no_grad():
                ref_chosen_logp, _ = avg_logp_and_nll(ref_model, batch, "chosen", F)
                ref_rejected_logp, _ = avg_logp_and_nll(ref_model, batch, "rejected", F)
            delta = (chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
            dpo_loss = (weights * F.softplus(-(args.beta * delta - args.margin))).sum() / weights.sum().clamp(min=1.0)
            loss = dpo_loss + args.pref_ftx * chosen_nll
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
            if step == 1 and first_batch_cache is not None:
                with torch.no_grad():
                    c1, _ = avg_logp_and_nll(model, first_batch_cache, "chosen", F)
                    r1, _ = avg_logp_and_nll(model, first_batch_cache, "rejected", F)
                    rc1, _ = avg_logp_and_nll(ref_model, first_batch_cache, "chosen", F)
                    rr1, _ = avg_logp_and_nll(ref_model, first_batch_cache, "rejected", F)
                    first_batch_delta_after = float(((c1 - r1) - (rc1 - rr1)).mean().detach().float().cpu())
            last_metrics = {
                "loss": float(loss.detach().float().cpu()),
                "dpo_loss": float(dpo_loss.detach().float().cpu()),
                "chosen_nll": float(chosen_nll.detach().float().cpu()),
                "mean_delta": float(delta.detach().mean().float().cpu()),
                "lr": float(scheduler.get_last_lr()[0]),
            }
            if step == 1 or step % args.logging_steps == 0 or step >= args.max_steps:
                elapsed = time.time() - start
                eta = elapsed / max(step, 1) * max(args.max_steps - step, 0)
                print(
                    f"[exp25r3] {args.config_name} step {step}/{args.max_steps} "
                    f"loss={last_metrics['loss']:.4f} dpo={last_metrics['dpo_loss']:.4f} "
                    f"delta={last_metrics['mean_delta']:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    after_rows = evaluate(model, ref_model, loader, device, F, torch)
    after_summary = metric_summary(after_rows, ["negative_type"])
    after_overall = metric_summary(after_rows, ["risk_type"])
    overall_before = metric_summary(before_rows, ["constant"])[0] if False else None
    before_acc = sum(1 for row in before_rows if row["dpo_delta"] > 0) / max(len(before_rows), 1)
    after_acc = sum(1 for row in after_rows if row["dpo_delta"] > 0) / max(len(after_rows), 1)
    before_delta = sum(float(row["dpo_delta"]) for row in before_rows) / max(len(before_rows), 1)
    after_delta = sum(float(row["dpo_delta"]) for row in after_rows) / max(len(after_rows), 1)
    summary = {
        "config_name": args.config_name,
        "data": str(args.data),
        "rows": len(rows),
        "negative_type_counts": dict(Counter(item.negative_type for item in dataset.items)),
        "logp_mode": args.logp_mode,
        "beta": args.beta,
        "margin": args.margin,
        "pref_ftx": args.pref_ftx,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "before_dpo_pref_acc": before_acc,
        "after_dpo_pref_acc": after_acc,
        "acc_gain": after_acc - before_acc,
        "before_mean_delta": before_delta,
        "after_mean_delta": after_delta,
        "mean_delta_gain": after_delta - before_delta,
        "first_batch_delta_before": first_batch_delta_before,
        "first_batch_delta_after": first_batch_delta_after,
        "first_batch_delta_gain": None
        if first_batch_delta_before is None or first_batch_delta_after is None
        else first_batch_delta_after - first_batch_delta_before,
        "last_metrics": last_metrics,
        "elapsed_seconds": time.time() - start,
    }
    out_dir = args.out_dir
    run_dir = out_dir / "run_summaries"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / f"{args.config_name}.json", summary)
    write_csv(run_dir / f"{args.config_name}_before_by_negative_type.csv", before_summary)
    write_csv(run_dir / f"{args.config_name}_after_by_negative_type.csv", after_summary)
    write_csv(run_dir / f"{args.config_name}_before_by_risk_type.csv", before_overall)
    write_csv(run_dir / f"{args.config_name}_after_by_risk_type.csv", after_overall)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp25R3 field-masked SRC-DPO sanity trainer.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-name-or-path", default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--ref-adapter-name-or-path", default="")
    parser.add_argument("--data", type=Path, default=DEFAULT_SCORE_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--logp-mode", choices=["full", "field"], default="full")
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--pref-ftx", type=float, default=0.05)
    parser.add_argument("--max-train-examples", type=int, default=32)
    parser.add_argument("--examples-per-negative-type", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.ref_adapter_name_or_path:
        args.ref_adapter_name_or_path = args.adapter_name_or_path
    print(json.dumps(train(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
