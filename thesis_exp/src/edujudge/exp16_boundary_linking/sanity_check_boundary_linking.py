"""Sanity checks for Exp16A boundary linking."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from thesis_exp.src.edujudge.exp16_boundary_linking import EXP16_SANITY_DIR, ensure_exp16_dirs
from thesis_exp.src.edujudge.exp16_boundary_linking.data import (
    BoundaryLinkingCollator,
    BoundaryLinkingDataset,
    SimpleBoundaryTokenizer,
    default_data_paths,
    load_samples,
    make_sample,
)
from thesis_exp.src.edujudge.exp16_boundary_linking.model import BoundaryLinkingOrdinalModel
from thesis_exp.src.edujudge.exp16_boundary_linking.train_boundary_linking import run_training
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


def _add(rows: list[dict[str, Any]], check: str, passed: bool, details: Any = "") -> None:
    rows.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})


def run_sanity(output_dir: Path, variant: str = "qmr_meta") -> list[dict[str, Any]]:
    ensure_exp16_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = default_data_paths()
    samples = load_samples(paths["train"], variant=variant, boundary_fields=None, limit=8)
    tokenizer = SimpleBoundaryTokenizer()
    collator = BoundaryLinkingCollator(tokenizer, max_length_quality=128, max_length_boundary=96)
    batch = collator(samples[:4])
    model = BoundaryLinkingOrdinalModel.from_model_name("__tiny_random__", variant=variant)
    model.eval()
    with torch.no_grad():
        outputs = model(
            quality_input_ids=batch["quality_input_ids"],
            quality_attention_mask=batch["quality_attention_mask"],
            boundary_input_ids=batch["boundary_input_ids"],
            boundary_attention_mask=batch["boundary_attention_mask"],
        )

    rows: list[dict[str, Any]] = []
    logits = outputs["logits"]
    tau = outputs["thresholds_tau"]
    probs = outputs["probs"]
    _add(rows, "logits shape [B,4]", tuple(logits.shape) == (4, 4), tuple(logits.shape))
    _add(rows, "tau shape [B,4]", tuple(tau.shape) == (4, 4), tuple(tau.shape))
    tau_order = bool(torch.all(tau[:, 0] < tau[:, 1]) and torch.all(tau[:, 1] < tau[:, 2]) and torch.all(tau[:, 2] < tau[:, 3]))
    _add(rows, "tau strictly increasing", tau_order, tau.detach().cpu().tolist()[0])
    prob_order = bool(torch.all(probs[:, 0] >= probs[:, 1]) and torch.all(probs[:, 1] >= probs[:, 2]) and torch.all(probs[:, 2] >= probs[:, 3]))
    _add(rows, "probs monotonic nonincreasing", prob_order, probs.detach().cpu().tolist()[0])

    raw_rows = read_jsonl(paths["train"])
    base = dict(raw_rows[0])
    changed = dict(raw_rows[0])
    changed["answer"] = "A deliberately changed answer for Exp16A sanity."
    sample_a = make_sample(base, variant=variant)
    sample_b = make_sample(changed, variant=variant)
    _add(rows, "changed answer changes quality_text", sample_a["quality_text"] != sample_b["quality_text"])
    _add(rows, "changed answer leaves boundary_text unchanged", sample_a["boundary_text"] == sample_b["boundary_text"])
    same_boundary_batch = collator([sample_a, sample_b])
    with torch.no_grad():
        same_boundary_outputs = model(
            quality_input_ids=same_boundary_batch["quality_input_ids"],
            quality_attention_mask=same_boundary_batch["quality_attention_mask"],
            boundary_input_ids=same_boundary_batch["boundary_input_ids"],
            boundary_attention_mask=same_boundary_batch["boundary_attention_mask"],
        )
    tau_delta = torch.max(torch.abs(same_boundary_outputs["thresholds_tau"][0] - same_boundary_outputs["thresholds_tau"][1])).item()
    _add(rows, "same boundary_text gives matching tau", tau_delta < 1e-6, tau_delta)
    leak_free = all(not sample["answer"] or sample["answer"] not in sample["boundary_text"] for sample in samples)
    _add(rows, "boundary_text does not contain answer", leak_free)

    dry_run_dir = output_dir / "dry_run"
    args = Namespace(
        model_name_or_path="__tiny_random__",
        train_path=paths["train"],
        dev_path=paths["dev"],
        test_path=paths["test"],
        output_dir=dry_run_dir,
        variant=variant,
        boundary_fields="",
        max_length_quality=128,
        max_length_boundary=96,
        batch_size=2,
        grad_accum_steps=1,
        epochs=1.0,
        learning_rate=5e-4,
        weight_decay=0.0,
        seed=42,
        class_weights="",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        freeze_encoder=False,
        eval_every_epoch=True,
        save_best_by="dev_mae",
        max_train_steps=2,
        max_train_samples=8,
        max_eval_samples=8,
        trust_remote_code=False,
        local_files_only=False,
    )
    result = run_training(args)
    _add(rows, "dry-run completed", result.get("status") == "COMPLETED", result)
    _add(rows, "dry-run metrics_dev exists", (dry_run_dir / "metrics_dev.json").exists(), relpath(dry_run_dir / "metrics_dev.json"))
    _add(
        rows,
        "dry-run predictions_dev exists",
        (dry_run_dir / "predictions_dev.jsonl").exists(),
        relpath(dry_run_dir / "predictions_dev.jsonl"),
    )
    if (dry_run_dir / "predictions_dev.jsonl").exists():
        pred_rows = read_jsonl(dry_run_dir / "predictions_dev.jsonl")
        required = {
            "boundary_key",
            "quality_score_s",
            "tau1",
            "tau2",
            "tau3",
            "tau4",
            "margin_tau2",
            "margin_tau3",
            "is_low_to_high",
        }
        present = set(pred_rows[0]) if pred_rows else set()
        _add(rows, "predictions contain boundary key plus s/tau/margins/risk fields", required.issubset(present), sorted(required - present))
    write_csv(output_dir / "sanity_check_boundary_linking.csv", rows)
    lines = ["# Exp16A Boundary Linking Sanity", ""]
    lines.extend(f"- {row['check']}: {row['status']} ({row['details']})" for row in rows)
    write_text(output_dir / "sanity_check_boundary_linking.md", "\n".join(lines))
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise SystemExit(f"Exp16A sanity failed. See {relpath(output_dir / 'sanity_check_boundary_linking.md')}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Exp16A boundary-linking sanity checks.")
    parser.add_argument("--output_dir", type=Path, default=EXP16_SANITY_DIR)
    parser.add_argument("--variant", default="qmr_meta")
    args = parser.parse_args(argv)
    rows = run_sanity(args.output_dir, variant=args.variant)
    print(f"Exp16A sanity PASS ({len(rows)} checks).")


if __name__ == "__main__":
    main()
