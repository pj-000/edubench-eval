"""No-outcome real-model/state preflight for the Exp64 implementation.

This check uses one already-observed Exp63 checkpoint only to validate state
mapping, shared scaling and the exact AdamW finite-difference machinery.  It
does not call ``optimizer.step`` and does not evaluate development or test
outcomes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.train import load_model
from thesis_exp.exp63_same_state_counterfactual import ARTIFACT_ROOT as EXP63_ARTIFACT_ROOT
from thesis_exp.exp63_same_state_counterfactual.counterfactual import (
    collect_components,
    config_for,
    scalar_geometry,
)
from thesis_exp.exp63_same_state_counterfactual.runtime import (
    hash_strings,
    ordered_rows,
    sha256_file,
)
from thesis_exp.exp64_optimizer_state_residual import OUTPUT_ROOT
from thesis_exp.exp64_optimizer_state_residual.mechanics import (
    exact_adamw_displacement,
    fixed_denominator_attributable_displacement,
    l2_norm,
    shared_scale_from_norms,
    subtract_displacements,
)
from thesis_exp.exp64_optimizer_state_residual.state import optimizer_contract, restore_rng
from thesis_exp.src.edujudge.exp02.train_ce_baseline import make_dataloader, set_seed


EXP63_SEEDS = (67, 68, 69, 70, 71)
EXP63_STAGES = (2, 5, 8)
ARMS = (
    "blocked",
    "full_residual",
    "parallel_only",
    "orthogonal_only",
    "sign_flipped_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=EXP63_SEEDS, default=67)
    parser.add_argument("--stage", type=int, choices=EXP63_STAGES, default=2)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "preflight" / "existing_exp63_seed67_epoch2.json",
    )
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _component(
    name: str,
    arm: str,
    common: Any,
    residual: Any | None,
    coefficient: float,
) -> Any:
    if residual is None or not name.startswith("backbone.") or arm == "blocked":
        return common
    if arm == "full_residual":
        return common + residual
    if arm == "parallel_only":
        return common + coefficient * common
    if arm == "orthogonal_only":
        return common + residual - coefficient * common
    if arm == "sign_flipped_residual":
        return common - residual
    raise ValueError(arm)


def _candidate_norms(
    common: dict[str, Any], residual: dict[str, Any], coefficient: float
) -> dict[str, float]:
    result = {}
    for arm in ARMS:
        squared = 0.0
        for name, common_value in common.items():
            value = _component(name, arm, common_value, residual.get(name), coefficient)
            squared += float(value.detach().double().square().sum())
        result[arm] = math.sqrt(max(0.0, squared))
    return result


def _candidate(
    common: dict[str, Any],
    residual: dict[str, Any],
    coefficient: float,
    arm: str,
    scale: float,
) -> dict[str, Any]:
    return {
        name: _component(name, arm, value, residual.get(name), coefficient).mul(scale)
        for name, value in common.items()
    }


def _all_finite(tensors: dict[str, Any]) -> bool:
    import torch

    return all(bool(torch.isfinite(value).all()) for value in tensors.values())


def main() -> None:
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Exp64 real-model preflight requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    checkpoint = args.checkpoint or (
        EXP63_ARTIFACT_ROOT / f"seed_{args.seed}" / f"after_epoch_{args.stage}.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "EXP63_COMPLETE_STAGE_STATE_V1":
        raise RuntimeError("Exp64 preflight expected an Exp63 complete checkpoint")
    set_seed(args.seed)
    config = config_for(
        args.seed,
        args.model_name_or_path,
        OUTPUT_ROOT / "preflight" / "scratch",
        checkpoint.parent,
    )
    model, tokenizer, _, _ = load_model(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    rows = model_rows("train")
    epoch_rows = ordered_rows(rows, args.seed, args.stage + 1)
    expected_ids = [str(value) for value in payload["next_window_record_ids"]]
    row_by_id = {str(row["record_id"]): row for row in epoch_rows}
    window_rows = [row_by_id[record_id] for record_id in expected_ids]
    if hash_strings(expected_ids) != payload["next_window_sha256"]:
        raise RuntimeError("Exp64 preflight checkpoint-window hash mismatch")
    loader = make_dataloader(window_rows, tokenizer, config, "train", shuffle=False)
    restore_rng(torch, device, payload["rng"])
    common, residual, window = collect_components(
        model, loader, device, config.gradient_accumulation_steps
    )
    if window["record_ids"] != expected_ids:
        raise RuntimeError("Exp64 preflight consumed a different next window")
    geometry = scalar_geometry(common, residual)
    norms = _candidate_norms(common, residual, geometry["projection_coefficient"])
    scale = shared_scale_from_norms(norms, target_norm=0.95)
    scaled_norms = {name: value * scale for name, value in norms.items()}

    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    states, groups = optimizer_contract(model, payload["optimizer"], None)
    blocked_gradient = _candidate(
        common, residual, geometry["projection_coefficient"], "blocked", scale
    )
    blocked_step = exact_adamw_displacement(parameters, blocked_gradient, states, groups)
    attributable: dict[str, dict[str, Any]] = {}
    exact_norms: dict[str, float] = {}
    fixed_norms: dict[str, float] = {}
    for arm in ARMS[1:]:
        gradient = _candidate(common, residual, geometry["projection_coefficient"], arm, scale)
        candidate_step = exact_adamw_displacement(parameters, gradient, states, groups)
        difference = subtract_displacements(candidate_step, blocked_step)
        exact_norms[arm] = l2_norm(difference)
        fixed = fixed_denominator_attributable_displacement(
            blocked_gradient, gradient, states, groups
        )
        fixed_norms[arm] = l2_norm(fixed)
        if arm in ("full_residual", "parallel_only", "orthogonal_only"):
            attributable[arm] = {
                name: value.detach().cpu() for name, value in difference.items()
            }
        del gradient, candidate_step, difference, fixed
        gc.collect()
        torch.cuda.empty_cache()

    interaction = {
        name: attributable["full_residual"][name]
        - attributable["parallel_only"][name]
        - attributable["orthogonal_only"][name]
        for name in attributable["full_residual"]
    }
    interaction_norm = l2_norm(interaction)
    nonbackbone_max_difference = 0.0
    full_gradient = _candidate(
        common, residual, geometry["projection_coefficient"], "full_residual", scale
    )
    for name in common:
        if name.startswith("backbone."):
            continue
        nonbackbone_max_difference = max(
            nonbackbone_max_difference,
            float((full_gradient[name] - blocked_gradient[name]).abs().max()),
        )
    checks = {
        "checkpoint_rng_restored_before_components": True,
        "next_window_exact": window["record_ids"] == expected_ids,
        "five_candidates_present": set(norms) == set(ARMS),
        "one_shared_scale": 0.0 < scale <= 1.0,
        "all_scaled_norms_at_most_0_95": max(scaled_norms.values()) <= 0.9500005,
        "all_clip_coefficients_one": max(scaled_norms.values()) < 1.0,
        "nonbackbone_common_update_identical": nonbackbone_max_difference == 0.0,
        "exact_displacements_finite": all(math.isfinite(value) for value in exact_norms.values()),
        "fixed_displacements_finite": all(math.isfinite(value) for value in fixed_norms.values()),
        "exact_full_nonzero": exact_norms["full_residual"] > 0.0,
        "interaction_finite": math.isfinite(interaction_norm),
        "optimizer_step_calls": 0 == 0,
        "dev_outcomes_read": 0 == 0,
        "test_access_count": 0 == 0,
    }
    result = {
        "status": "EXP64_EXISTING_STATE_NO_OUTCOME_PREFLIGHT_PASS"
        if all(checks.values())
        else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args.seed,
        "stage": args.stage,
        "window": window,
        "geometry": geometry,
        "candidate_raw_norms": norms,
        "shared_scale": scale,
        "candidate_scaled_norms": scaled_norms,
        "exact_attributable_norms": exact_norms,
        "fixed_denominator_attributable_norms": fixed_norms,
        "adamw_nonadditivity_norm": interaction_norm,
        "nonbackbone_max_gradient_difference": nonbackbone_max_difference,
        "checks": checks,
        "model_outcomes_read": 0,
        "optimizer_step_calls": 0,
        "test_access_count": 0,
    }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "EXP64_EXISTING_STATE_NO_OUTCOME_PREFLIGHT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
