"""Real-model directional audit for every CBRD hidden-gradient route."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import (
    load_rows,
    model_rows,
    shuffled_residual_audit,
)
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.method import target_state
from thesis_exp.exp57_cbrd.model import (
    head_contract,
    make_cbrd_dual_head_classifier,
    module_hash,
)
from thesis_exp.exp57_cbrd.preflight import EXPECTED_SEED42_HEAD_HASH, config


SOFT_VARIANTS = (
    "ordinary_hmsa",
    "routed_hmsa",
    "consensus_only",
    "residual_only",
    "sign_flipped",
    "shuffled_residual",
    "detached_soft",
)


def _max_abs(left: Any, right: Any) -> float:
    return float((left - right).abs().max())


def _relative(left: Any, right: Any) -> float:
    return _max_abs(left, right) / max(float(left.abs().max()), 1e-12)


def run(model_path: str, *, loss_scale: float = 1.0) -> dict[str, Any]:
    import torch

    from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
        load_model_and_tokenizer,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The real-model mechanism audit requires CUDA")
    if not loss_scale > 0.0:
        raise ValueError("loss_scale must be positive")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(42)

    raw_rows = load_rows("train")
    converted_rows = model_rows("train")
    converted_by_id = {str(row["record_id"]): row for row in converted_rows}
    shuffle = shuffled_residual_audit(raw_rows)
    mapping_by_id = {
        str(row["recipient_record_id"]): row for row in shuffle["mapping"]
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in converted_rows:
        row_id = str(row["record_id"])
        state = target_state(int(row["label_5"]), row["soft_target_5"])
        mapping = mapping_by_id[row_id]
        if state not in selected and bool(mapping["effectively_changed"]):
            selected[state] = row
        if set(selected) == {"zero", "down", "up"}:
            break
    if set(selected) != {"zero", "down", "up"}:
        raise RuntimeError(f"Could not select all relation states: {sorted(selected)}")
    rows = [selected[state] for state in ("zero", "down", "up")]
    shuffled_targets = [
        [value / 3.0 for value in mapping_by_id[str(row["record_id"])]["shuffled_target_thirds"]]
        for row in rows
    ]

    cfg = config(model_path, bf16="true")
    base, tokenizer, model_mode = load_model_and_tokenizer(cfg)
    if model_mode != "sequence_classification":
        raise RuntimeError(model_mode)
    if module_hash(base.score) != EXPECTED_SEED42_HEAD_HASH:
        raise AssertionError("Mechanism audit did not reproduce the Exp51 BF16 head")
    device = torch.device("cuda")
    model = make_cbrd_dual_head_classifier(base.to(device)).to(device).eval()
    contract = head_contract(model)
    encoded = tokenizer(
        [row["text"] for row in rows],
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long, device=device)
    targets = torch.tensor(
        [row["soft_target_5"] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    shuffled_target_tensor = torch.tensor(
        shuffled_targets,
        dtype=torch.float32,
        device=device,
    )

    def capture(variant: str) -> dict[str, Any]:
        set_seed(42)
        model.zero_grad(set_to_none=True)
        outputs = model(
            **encoded,
            labels=labels,
            soft_targets=targets,
            residual_targets=shuffled_target_tensor,
            aux_route=variant,
            route_loss_scale=loss_scale,
            retain_backbone_hidden=True,
        )
        objective = cbrd_objective(
            outputs,
            labels,
            targets,
            variant=variant,
            residual_targets=shuffled_target_tensor,
        )
        (objective["optimization_loss"] * loss_scale).backward()
        return {
            "hidden": outputs["backbone_hidden"].grad.detach().float().cpu().clone(),
            "hard_head": {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.hard_head.named_parameters()
            },
            "soft_head": {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.soft_head.named_parameters()
            },
        }

    captured = {variant: capture(variant) for variant in SOFT_VARIANTS}
    hard_hidden = captured["detached_soft"]["hidden"]
    auxiliary = {
        variant: captured[variant]["hidden"] - hard_hidden
        for variant in SOFT_VARIANTS
    }

    batch_size = len(rows)
    non_pad = (encoded["input_ids"] != model.config.pad_token_id).to(torch.int32)
    token_indices = torch.arange(
        encoded["input_ids"].shape[-1],
        device=device,
        dtype=torch.int32,
    )
    last_indices = (token_indices * non_pad).argmax(-1)

    def analytic_residual(target_values: Any) -> Any:
        hard = torch.nn.functional.one_hot(labels, num_classes=5).float()
        selected_gradient = loss_scale * (
            (hard - target_values.float()) @ model.soft_head.weight.detach().float()
        ) / float(batch_size)
        full = torch.zeros_like(captured["ordinary_hmsa"]["hidden"], device=device)
        full = full.to(dtype=torch.bfloat16)
        row_indices = torch.arange(batch_size, device=device)
        full[row_indices, last_indices] = selected_gradient.to(torch.bfloat16)
        return full.float().cpu()

    original_residual = analytic_residual(targets)
    shuffled_residual = analytic_residual(shuffled_target_tensor)
    comparisons = {
        "identity_hook": _max_abs(
            auxiliary["ordinary_hmsa"],
            auxiliary["routed_hmsa"],
        ),
        "full_equals_consensus_plus_residual": _max_abs(
            auxiliary["ordinary_hmsa"],
            auxiliary["consensus_only"] + auxiliary["residual_only"],
        ),
        "residual_matches_analytic": _max_abs(
            auxiliary["residual_only"],
            original_residual,
        ),
        "sign_flip_matches_consensus_minus_residual": _max_abs(
            auxiliary["sign_flipped"],
            auxiliary["consensus_only"] - auxiliary["residual_only"],
        ),
        "shuffled_matches_consensus_plus_shuffled_residual": _max_abs(
            auxiliary["shuffled_residual"],
            auxiliary["consensus_only"] + shuffled_residual,
        ),
        "detached_auxiliary_is_zero": float(auxiliary["detached_soft"].abs().max()),
        "unanimous_residual_is_zero": float(original_residual[0].abs().max()),
    }
    relative_comparisons = {
        "full_equals_consensus_plus_residual": _relative(
            auxiliary["ordinary_hmsa"],
            auxiliary["consensus_only"] + auxiliary["residual_only"],
        ),
        "sign_flip_matches_consensus_minus_residual": _relative(
            auxiliary["sign_flipped"],
            auxiliary["consensus_only"] - auxiliary["residual_only"],
        ),
        "shuffled_matches_consensus_plus_shuffled_residual": _relative(
            auxiliary["shuffled_residual"],
            auxiliary["consensus_only"] + shuffled_residual,
        ),
    }
    head_gradient_differences: dict[str, float] = {}
    for variant in SOFT_VARIANTS:
        for name, ordinary_gradient in captured["ordinary_hmsa"]["soft_head"].items():
            head_gradient_differences[f"soft_head.{variant}.{name}"] = _max_abs(
                ordinary_gradient,
                captured[variant]["soft_head"][name],
            )
        for name, ordinary_gradient in captured["ordinary_hmsa"]["hard_head"].items():
            head_gradient_differences[f"hard_head.{variant}.{name}"] = _max_abs(
                ordinary_gradient,
                captured[variant]["hard_head"][name],
            )

    absolute_tolerance = 5e-4
    relative_tolerance = 0.02
    checks = {
        "identity_hook_exact": comparisons["identity_hook"] == 0.0,
        "component_algebra": max(comparisons.values()) <= absolute_tolerance,
        "relative_component_algebra": max(relative_comparisons.values()) <= relative_tolerance,
        "all_soft_variant_head_gradients_identical": max(head_gradient_differences.values()) == 0.0,
        "historical_head_hash_matches": contract["hard_head_hash"] == EXPECTED_SEED42_HEAD_HASH,
        "all_three_relation_states_present": [
            target_state(int(row["label_5"]), row["soft_target_5"]) for row in rows
        ] == ["zero", "down", "up"],
        "no_test_access": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "implementation": "single_soft_ce_hidden_gradient_hook",
        "loss_scale": loss_scale,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "rows": [
            {
                "record_id": row["record_id"],
                "hard_label": row["label_5"],
                "state": target_state(int(row["label_5"]), row["soft_target_5"]),
                "target": row["soft_target_5"],
                "shuffled_residual_target": shuffled_targets[index],
                "shuffle_effectively_changed": mapping_by_id[str(row["record_id"])]["effectively_changed"],
            }
            for index, row in enumerate(rows)
        ],
        "comparisons": comparisons,
        "relative_comparisons": relative_comparisons,
        "head_gradient_differences": head_gradient_differences,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "checks": checks,
        "test_access_count": 0,
    }
    output_name = (
        "stage0_real_qwen3_mechanism_routes.json"
        if loss_scale == 1.0
        else "stage1_real_qwen3_accumulation_scale_routes.json"
    )
    output = OUTPUT_ROOT / "audit" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    args = parser.parse_args()
    report = run(args.model_name_or_path, loss_scale=args.loss_scale)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
