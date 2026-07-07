"""Diagnose whether Exp25 adapters learned train-side DPO preferences.

This script does not train, generate text, or read test labels. It scores
chosen/rejected responses from Exp25 train-only DPO pairs under the trained
adapter and the reference init adapter, then reports whether chosen is
preferred overall and by negative/risk type.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean  # noqa: E402
from thesis_exp.exp17_low_score_evidence.train_exp25_structured_src_dpo import (  # noqa: E402
    SRCDataset,
    avg_logp_and_nll,
    collate_items,
    load_json_array,
    require_training_deps,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_MODEL = "/home/jpang/models/modelscope/Qwen/Qwen3-4B"
DEFAULT_REF_ADAPTER = "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25r2_train_preference_diagnosis_seed42")
DEFAULT_RUNS = [
    {
        "run_name": "exp25_src_score_mismatch_r2c",
        "adapter": "saves/edubench/qwen3-4b/exp25_src_score_mismatch_r2c",
        "data": (
            "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
            "data/edubench_r7h_score_mismatch_only_train.json"
        ),
        "ref_adapter": DEFAULT_REF_ADAPTER,
    },
    {
        "run_name": "exp25_src_mixed_r2c",
        "adapter": "saves/edubench/qwen3-4b/exp25_src_mixed_r2c",
        "data": (
            "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
            "data/edubench_r7h_structured_src_dpo_train.json"
        ),
        "ref_adapter": DEFAULT_REF_ADAPTER,
    },
]


def parse_run_spec(spec: str) -> dict[str, str]:
    parts = spec.split(":", 3)
    if len(parts) not in {3, 4}:
        raise ValueError("Run spec must be run_name:adapter:data[:ref_adapter]")
    run_name, adapter, data = parts[:3]
    ref_adapter = parts[3] if len(parts) == 4 else DEFAULT_REF_ADAPTER
    return {"run_name": run_name, "adapter": adapter, "data": data, "ref_adapter": ref_adapter}


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    denom = sum(weights)
    if not values or denom <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / denom


def group_summary(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(field, "")) for field in key_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items()):
        deltas = [float(row["dpo_delta"]) for row in items]
        model_margins = [float(row["model_margin"]) for row in items]
        weights = [float(row["pair_weight"]) for row in items]
        row: dict[str, Any] = {field: value for field, value in zip(key_fields, key)}
        row.update(
            {
                "n": len(items),
                "dpo_pref_acc": mean([1.0 if value > 0 else 0.0 for value in deltas]),
                "raw_pref_acc": mean([1.0 if value > 0 else 0.0 for value in model_margins]),
                "weighted_dpo_pref_acc": weighted_mean([1.0 if value > 0 else 0.0 for value in deltas], weights),
                "mean_dpo_delta": mean(deltas),
                "p10_dpo_delta": quantile(deltas, 0.10),
                "p50_dpo_delta": quantile(deltas, 0.50),
                "p90_dpo_delta": quantile(deltas, 0.90),
                "mean_model_margin": mean(model_margins),
            }
        )
        out.append(row)
    return out


def load_eval_adapter(model_name_or_path: str, adapter_path: str, bf16: bool) -> tuple[Any, Any, Any, Any]:
    torch, F, PeftModel, _pad_sequence, _DataLoader, hf = require_training_deps()
    AutoModelForCausalLM, AutoTokenizer, _scheduler = hf
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if bf16 else torch.float16
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device, F


def release_model(model: Any) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def score_adapter(args: argparse.Namespace, adapter_path: str, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    torch, _F, _PeftModel, pad_sequence, DataLoader, _hf = require_training_deps()
    tokenizer, model, device, F = load_eval_adapter(args.model_name_or_path, adapter_path, args.bf16)
    dataset = SRCDataset(rows, tokenizer, args)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_items(batch, tokenizer, torch, pad_sequence),
    )
    scores: dict[str, dict[str, float]] = {}
    start = time.time()
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            pair_ids = [item.pair_id for item in dataset.items[seen : seen + len(batch["weights"])]]
            seen += len(pair_ids)
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            chosen_logp, _ = avg_logp_and_nll(model, batch, "chosen", F)
            rejected_logp, _ = avg_logp_and_nll(model, batch, "rejected", F)
            chosen_values = chosen_logp.detach().float().cpu().tolist()
            rejected_values = rejected_logp.detach().float().cpu().tolist()
            for pair_id, chosen_value, rejected_value in zip(pair_ids, chosen_values, rejected_values):
                scores[pair_id] = {
                    "chosen_logp": float(chosen_value),
                    "rejected_logp": float(rejected_value),
                    "margin": float(chosen_value - rejected_value),
                }
            if batch_idx == 1 or batch_idx % args.log_steps == 0 or seen >= len(dataset):
                elapsed = time.time() - start
                eta = elapsed / max(seen, 1) * max(len(dataset) - seen, 0)
                print(
                    f"[exp25r2] scored {seen}/{len(dataset)} with {adapter_path} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
    release_model(model)
    return scores


def diagnose_run(args: argparse.Namespace, run: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_json_array(Path(run["data"]))
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    print(f"[exp25r2] run={run['run_name']} rows={len(rows)}", flush=True)
    model_scores = score_adapter(args, run["adapter"], rows)
    ref_scores = score_adapter(args, run["ref_adapter"], rows)

    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        pair_id = clean(row.get("pair_id"))
        model = model_scores[pair_id]
        ref = ref_scores[pair_id]
        model_margin = model["margin"]
        ref_margin = ref["margin"]
        delta = model_margin - ref_margin
        detail_rows.append(
            {
                "run_name": run["run_name"],
                "pair_id": pair_id,
                "negative_type": clean(row.get("negative_type")),
                "risk_type": clean(row.get("risk_type")),
                "gold_label": row.get("gold_label", ""),
                "rejected_score": row.get("rejected_score", ""),
                "ordinal_distance": row.get("ordinal_distance", ""),
                "pair_weight": float(row.get("pair_weight") or 1.0),
                "model_chosen_logp": model["chosen_logp"],
                "model_rejected_logp": model["rejected_logp"],
                "model_margin": model_margin,
                "ref_margin": ref_margin,
                "dpo_delta": delta,
                "raw_pref_correct": model_margin > 0,
                "dpo_pref_correct": delta > 0,
            }
        )
    summary = group_summary(detail_rows, ["run_name"])[0]
    summary.update(
        {
            "adapter": run["adapter"],
            "ref_adapter": run["ref_adapter"],
            "data": run["data"],
            "unique_source_samples": len({clean(row.get("source_sample_id")) for row in rows}),
        }
    )
    return summary, detail_rows


def make_decision(summary_rows: list[dict[str, Any]], by_negative: list[dict[str, Any]]) -> dict[str, Any]:
    low_accuracy = [row for row in summary_rows if float(row.get("dpo_pref_acc", 0.0)) < 0.60]
    learned = [row for row in summary_rows if float(row.get("dpo_pref_acc", 0.0)) >= 0.75]
    low_failure_rows = [
        row
        for row in by_negative
        if row.get("negative_type") == "low_failure_erasure_counterfactual"
        and float(row.get("dpo_pref_acc", 0.0)) < 0.60
    ]
    if low_accuracy:
        recommendation = "fix_loss_beta_steps_or_trainer_before_data_expansion"
        reason = "At least one Exp25 run has train DPO preference accuracy below 0.60."
    elif low_failure_rows:
        recommendation = "increase_specific_low_failure_supervision_before_more_dpo"
        reason = "Overall train preference is not the only issue; low-failure counterfactual pairs remain weak."
    elif len(learned) == len(summary_rows):
        recommendation = "preference_learned_but_not_generalized_expand_hidden_failure_data"
        reason = "Train preferences are learned, so dev D1 failure points to data coverage/generalization."
    else:
        recommendation = "ambiguous_train_preference_strength_tune_and_expand"
        reason = "Train preferences are partially learned but not strong enough for a clear attribution."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "train_preference_thresholds": {
            "low_accuracy_cutoff": 0.60,
            "learned_cutoff": 0.75,
        },
        "test_read": False,
        "train_only_dpo_pairs_used": True,
        "no_training": True,
        "raw_predictions_committed": False,
    }


def write_report(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    by_negative: list[dict[str, Any]],
    by_risk: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Exp25R2 Train Preference Diagnosis",
        "",
        "This diagnostic checks whether Exp25 trained adapters prefer chosen over rejected on train-only DPO pairs.",
        "It does not train, generate text, or read test labels.",
        "",
        "## Overall",
        "",
        "| run | n | dpo pref acc | raw pref acc | mean delta | p50 delta | unique source samples |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run_name']}` | {row['n']} | {fmt(float(row['dpo_pref_acc']))} | "
            f"{fmt(float(row['raw_pref_acc']))} | {fmt(float(row['mean_dpo_delta']))} | "
            f"{fmt(float(row['p50_dpo_delta']))} | {row['unique_source_samples']} |"
        )
    lines.extend(["", "## By Negative Type", "", "| run | negative_type | n | dpo pref acc | mean delta |", "|---|---|---:|---:|---:|"])
    for row in by_negative:
        lines.append(
            f"| `{row['run_name']}` | `{row['negative_type']}` | {row['n']} | "
            f"{fmt(float(row['dpo_pref_acc']))} | {fmt(float(row['mean_dpo_delta']))} |"
        )
    lines.extend(["", "## By Risk Type", "", "| run | risk_type | n | dpo pref acc | mean delta |", "|---|---|---:|---:|---:|"])
    for row in by_risk:
        lines.append(
            f"| `{row['run_name']}` | `{row['risk_type']}` | {row['n']} | "
            f"{fmt(float(row['dpo_pref_acc']))} | {fmt(float(row['mean_dpo_delta']))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- recommendation: `{decision['recommendation']}`",
            f"- reason: {decision['reason']}",
            "",
            "## Guardrails",
            "",
            "- No test split is read.",
            "- No training is performed.",
            "- Human reason remains only in assistant targets from train-only DPO pairs.",
            "- Raw generated predictions are not written by this diagnostic.",
        ]
    )
    write_text(out_dir / "reports" / "exp25r2_train_preference_diagnosis_report.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_specs = [parse_run_spec(spec) for spec in args.run] if args.run else DEFAULT_RUNS
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for spec in run_specs:
        summary, details = diagnose_run(args, spec)
        summary_rows.append(summary)
        detail_rows.extend(details)
    by_negative = group_summary(detail_rows, ["run_name", "negative_type"])
    by_risk = group_summary(detail_rows, ["run_name", "risk_type"])
    by_score_pair = group_summary(detail_rows, ["run_name", "gold_label", "rejected_score"])
    decision = make_decision(summary_rows, by_negative)

    write_csv(args.out_dir / "tables" / "exp25r2_train_pair_preference_summary.csv", summary_rows)
    write_csv(args.out_dir / "tables" / "exp25r2_train_pair_preference_by_negative_type.csv", by_negative)
    write_csv(args.out_dir / "tables" / "exp25r2_train_pair_preference_by_risk_type.csv", by_risk)
    write_csv(args.out_dir / "tables" / "exp25r2_train_pair_preference_by_score_pair.csv", by_score_pair)
    if args.write_pair_details:
        write_csv(args.out_dir / "tables" / "exp25r2_train_pair_preference_details.csv", detail_rows)
    write_json(args.out_dir / "decision" / "exp25r2_train_preference_decision.json", decision)
    write_report(args.out_dir, summary_rows, by_negative, by_risk, decision)
    return {
        "runs": len(summary_rows),
        "recommendation": decision["recommendation"],
        "out_dir": str(args.out_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Exp25 train-side preference learning.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec: run_name:adapter:data[:ref_adapter]. Defaults to the two Exp25 R2C runs.",
    )
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--write-pair-details", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
