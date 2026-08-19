"""Audit whether Exp63 gradients change when checkpoint RNG is restored.

This is a read-only scientific-integrity audit.  It never calls
``optimizer.step`` and never reads the test split.  For one frozen seed-stage
state it reproduces the legacy component-collection flow, repeats collection
after restoring the complete checkpoint RNG snapshot, and compares exact
tensor hashes.  Any hash difference conservatively requires a formal Exp63
rerun under the original protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Must be set before CUDA libraries are initialized.  Exact hash comparison is
# scientifically meaningful only under deterministic CUDA linear algebra.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.train import evaluate, load_model
from thesis_exp.exp63_same_state_counterfactual import (
    ARMS,
    ARTIFACT_ROOT,
    OUTPUT_ROOT,
    SEEDS,
    STAGE_EPOCHS,
)
from thesis_exp.exp63_same_state_counterfactual.counterfactual import (
    candidate_tensor,
    collect_components,
    config_for,
    relevant_metrics,
    restore_saved_training_rng,
    scalar_geometry,
)
from thesis_exp.exp63_same_state_counterfactual.runtime import (
    hash_strings,
    ordered_rows,
    sha256_file,
    write_json,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    make_dataloader,
    set_seed,
)


def tensor_map_sha256(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def candidate_hashes(
    common: dict[str, Any], residual: dict[str, Any], coefficient: float
) -> dict[str, str]:
    result: dict[str, str] = {}
    for arm in ARMS:
        candidate = {
            name: candidate_tensor(
                name, arm, common_value, residual.get(name), coefficient
            )
            for name, common_value in common.items()
        }
        result[arm] = tensor_map_sha256(candidate)
    return result


def maximum_absolute_difference(
    left: dict[str, Any], right: dict[str, Any]
) -> float:
    if set(left) != set(right):
        raise RuntimeError("Compared tensor maps have different parameter names")
    maximum = 0.0
    for name in left:
        difference = (left[name].double() - right[name].double()).abs()
        if difference.numel():
            maximum = max(maximum, float(difference.max()))
    return maximum


def metrics_sha256(metrics: dict[str, Any]) -> str:
    payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--stage_epoch", type=int, required=True, choices=STAGE_EPOCHS)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    artifact_dir = args.artifact_dir or ARTIFACT_ROOT / f"seed_{args.seed}"
    output = args.output or (
        OUTPUT_ROOT
        / "rng_audit"
        / f"seed_{args.seed}"
        / f"after_epoch_{args.stage_epoch}.json"
    )
    checkpoint_path = artifact_dir / f"after_epoch_{args.stage_epoch}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("seed") != args.seed or payload.get("epoch") != args.stage_epoch:
        raise RuntimeError("Checkpoint identity mismatch")

    train_rows = model_rows("train")
    dev_rows = model_rows("dev")
    config = config_for(args.seed, args.model_name_or_path, output.parent, artifact_dir)
    next_rows = ordered_rows(train_rows, args.seed, args.stage_epoch + 1)
    window_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    window_rows = next_rows[:window_size]
    window_ids = [str(row["record_id"]) for row in window_rows]
    if window_ids != payload["next_window_record_ids"]:
        raise RuntimeError("Audit window differs from frozen checkpoint contract")

    set_seed(args.seed)
    model, tokenizer, model_mode, head_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The complete Exp63 RNG audit requires CUDA")
    model.to(device)
    model.gradient_checkpointing_enable()
    model.load_state_dict(payload["model"], strict=True)
    window_loader = make_dataloader(window_rows, tokenizer, config, "train", shuffle=False)

    # Legacy path: use the RNG state left by set_seed + model construction.
    legacy_common, legacy_residual, legacy_window = collect_components(
        model, window_loader, device, config.gradient_accumulation_steps
    )
    legacy_geometry = scalar_geometry(legacy_common, legacy_residual)
    legacy_candidates = candidate_hashes(
        legacy_common, legacy_residual, legacy_geometry["projection_coefficient"]
    )
    model.load_state_dict(payload["model"], strict=True)
    dev_loader = make_dataloader(dev_rows, tokenizer, config, "dev", shuffle=False)
    legacy_probe = relevant_metrics(evaluate(model, dev_loader, device, "dev").metrics)

    # Corrected path: restore the exact complete checkpoint RNG immediately
    # before component collection.
    model.load_state_dict(payload["model"], strict=True)
    restore_saved_training_rng(torch, device, payload)
    restored_common, restored_residual, restored_window = collect_components(
        model, window_loader, device, config.gradient_accumulation_steps
    )
    restored_geometry = scalar_geometry(restored_common, restored_residual)
    restored_candidates = candidate_hashes(
        restored_common, restored_residual, restored_geometry["projection_coefficient"]
    )
    model.load_state_dict(payload["model"], strict=True)
    restored_probe = relevant_metrics(evaluate(model, dev_loader, device, "dev").metrics)

    hashes = {
        "legacy": {
            "common": tensor_map_sha256(legacy_common),
            "residual": tensor_map_sha256(legacy_residual),
            "candidates": legacy_candidates,
            "pre_update_probe": metrics_sha256(legacy_probe),
        },
        "restored_checkpoint_rng": {
            "common": tensor_map_sha256(restored_common),
            "residual": tensor_map_sha256(restored_residual),
            "candidates": restored_candidates,
            "pre_update_probe": metrics_sha256(restored_probe),
        },
    }
    exact_checks = {
        "window_hash_equal": legacy_window["record_ids_sha256"]
        == restored_window["record_ids_sha256"]
        == hash_strings(window_ids),
        "common_hash_equal": hashes["legacy"]["common"]
        == hashes["restored_checkpoint_rng"]["common"],
        "residual_hash_equal": hashes["legacy"]["residual"]
        == hashes["restored_checkpoint_rng"]["residual"],
        "candidate_hashes_equal": legacy_candidates == restored_candidates,
        "projection_coefficient_equal": legacy_geometry["projection_coefficient"]
        == restored_geometry["projection_coefficient"],
        "pre_update_probe_hash_equal": hashes["legacy"]["pre_update_probe"]
        == hashes["restored_checkpoint_rng"]["pre_update_probe"],
        "test_access_count_zero": True,
    }
    exact_pass = all(exact_checks.values())
    result = {
        "status": "RNG_INVARIANT_EXACT" if exact_pass else "RNG_CHANGES_GRADIENTS_RERUN_REQUIRED",
        "seed": args.seed,
        "stage_epoch": args.stage_epoch,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "format": payload.get("format"),
        },
        "model_mode": model_mode,
        "head_contract": head_contract,
        "window_sha256": hash_strings(window_ids),
        "hashes": hashes,
        "exact_checks": exact_checks,
        "diagnostic_differences": {
            "common_max_abs": maximum_absolute_difference(
                legacy_common, restored_common
            ),
            "residual_max_abs": maximum_absolute_difference(
                legacy_residual, restored_residual
            ),
            "projection_coefficient_abs": abs(
                legacy_geometry["projection_coefficient"]
                - restored_geometry["projection_coefficient"]
            ),
        },
        "legacy_geometry": legacy_geometry,
        "restored_geometry": restored_geometry,
        "legacy_probe": legacy_probe,
        "restored_probe": restored_probe,
        "decision_rule": (
            "All component, candidate and probe hashes must match exactly to retain "
            "the original Exp63 formal results without rerunning them."
        ),
        "determinism_contract": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
