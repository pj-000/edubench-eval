"""Post-result diagnostic for checkpointing/RNG parity; never trains or reads test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.model import make_cbrd_dual_head_classifier
from thesis_exp.exp57_cbrd.preflight import config


TRACE_PATH = OUTPUT_ROOT.parent / "exp51_hmsa" / "runs" / "hmsa_lambda1" / "seed_42" / "training_trace_first64.json"


def max_abs(left: Any, right: Any) -> float:
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max())


def run(model_path: str, *, deterministic: bool) -> dict[str, Any]:
    import torch

    from thesis_exp.src.edujudge.exp02.train_ce_baseline import load_model_and_tokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))["trace"]
    wanted = [trace[0]["record_ids"], trace[1]["record_ids"]]
    rows = {str(row["record_id"]): row for row in model_rows("train")}
    batches = [[rows[str(record_id)] for record_id in ids] for ids in wanted]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    set_seed(42)
    cfg = config(model_path, bf16="true")
    cfg.gradient_checkpointing = True
    base, tokenizer, model_mode = load_model_and_tokenizer(cfg)
    if model_mode != "sequence_classification":
        raise RuntimeError(model_mode)
    device = torch.device("cuda")
    model = make_cbrd_dual_head_classifier(base.to(device)).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    encoded_batches: list[tuple[dict[str, Any], Any, Any]] = []
    for batch in batches:
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
        targets = torch.tensor([row["soft_target_5"] for row in batch], dtype=torch.float32, device=device)
        encoded_batches.append((encoded, labels, targets))

    selected_backbone = next(
        parameter
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "layers." in name and parameter.numel() <= 5_000_000
    )

    def capture(variant: str) -> dict[str, Any]:
        set_seed(42)
        model.zero_grad(set_to_none=True)
        first_encoded, first_labels, first_targets = encoded_batches[0]
        first = model(
            **first_encoded,
            labels=first_labels,
            soft_targets=first_targets,
            aux_route=variant,
            route_loss_scale=1.0 / 32.0,
        )
        first_objective = cbrd_objective(first, first_labels, first_targets, variant=variant)
        first_loss = first_objective["optimization_loss"]
        (first_loss / 32.0).backward()
        gradients = {
            "hard_head": model.hard_head.weight.grad.detach().clone(),
            "soft_head": model.soft_head.weight.grad.detach().clone(),
            "backbone": selected_backbone.grad.detach().clone(),
        }
        second_encoded, second_labels, second_targets = encoded_batches[1]
        second = model(
            **second_encoded,
            labels=second_labels,
            soft_targets=second_targets,
            aux_route=variant,
            route_loss_scale=1.0 / 32.0,
        )
        second_objective = cbrd_objective(second, second_labels, second_targets, variant=variant)
        return {
            "first_loss": float(first_loss.detach().cpu()),
            "second_loss_after_first_backward": float(second_objective["optimization_loss"].detach().cpu()),
            "gradients": gradients,
        }

    ordinary = capture("ordinary_hmsa")
    routed = capture("routed_hmsa")
    differences = {
        name: max_abs(ordinary["gradients"][name], routed["gradients"][name])
        for name in ordinary["gradients"]
    }
    report = {
        "status": "DIAGNOSTIC_COMPLETE",
        "deterministic": deterministic,
        "gradient_checkpointing": True,
        "ordinary_first_loss": ordinary["first_loss"],
        "routed_first_loss": routed["first_loss"],
        "first_loss_abs_difference": abs(ordinary["first_loss"] - routed["first_loss"]),
        "ordinary_second_loss_after_first_backward": ordinary["second_loss_after_first_backward"],
        "routed_second_loss_after_first_backward": routed["second_loss_after_first_backward"],
        "second_loss_abs_difference": abs(
            ordinary["second_loss_after_first_backward"]
            - routed["second_loss_after_first_backward"]
        ),
        "gradient_max_abs_differences_after_first_backward": differences,
        "interpretation_scope": "post-result implementation diagnosis only; no optimizer step and no test access",
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    suffix = "deterministic" if deterministic else "training_kernels"
    output = OUTPUT_ROOT / "audit" / f"posthoc_checkpoint_identity_{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.model_name_or_path, deterministic=args.deterministic), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
